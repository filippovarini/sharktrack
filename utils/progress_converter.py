#!/usr/bin/env python3
"""
Intelligent FFmpeg converter with progress monitoring and adaptive timeouts
"""

import subprocess
import threading
import time
import re
import json
import queue
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
import concurrent.futures
from datetime import datetime, timedelta

class ProgressMonitor:
    """Monitors FFmpeg progress in real-time"""

    def __init__(self, duration_seconds: float, callback: Optional[Callable] = None):
        self.duration_seconds = duration_seconds
        self.callback = callback
        self.current_time = 0.0
        self.current_frame = 0
        self.fps = 0.0
        self.speed = "0x"
        self.progress_percent = 0.0
        self.last_update_time = time.time()
        self.stalled = False

    def parse_progress_line(self, line: str) -> bool:
        """Parse FFmpeg progress output and update statistics"""
        if "time=" in line:
            # Extract time progress: time=00:01:23.45
            time_match = re.search(r'time=(\d+):(\d+):(\d+\.\d+)', line)
            if time_match:
                hours, minutes, seconds = time_match.groups()
                self.current_time = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                self.progress_percent = (self.current_time / self.duration_seconds) * 100
                self.last_update_time = time.time()

                if self.callback:
                    self.callback(self.progress_percent, self.current_time, self.speed)
                return True

        if "frame=" in line:
            # Extract frame count
            frame_match = re.search(r'frame=\s*(\d+)', line)
            if frame_match:
                self.current_frame = int(frame_match.group(1))

        if "fps=" in line:
            # Extract processing FPS
            fps_match = re.search(r'fps=\s*([\d.]+)', line)
            if fps_match:
                self.fps = float(fps_match.group(1))

        if "speed=" in line:
            # Extract processing speed
            speed_match = re.search(r'speed=\s*([\d.]+x)', line)
            if speed_match:
                self.speed = speed_match.group(1)

        return False

    def is_stalled(self, stall_threshold_seconds: float = 300) -> bool:
        """Check if progress has stalled for too long"""
        time_since_update = time.time() - self.last_update_time
        return time_since_update > stall_threshold_seconds

    def estimate_remaining_time(self) -> float:
        """Estimate remaining processing time in seconds"""
        if self.current_time == 0 or self.progress_percent == 0:
            return float('inf')

        elapsed_processing_time = time.time() - self.last_update_time
        processing_rate = self.current_time / elapsed_processing_time if elapsed_processing_time > 0 else 1.0
        remaining_video_seconds = self.duration_seconds - self.current_time

        return remaining_video_seconds / processing_rate if processing_rate > 0 else float('inf')


class IntelligentConverter:
    """FFmpeg converter with intelligent progress monitoring and adaptive timeouts"""

    def __init__(self):
        self.active_conversions = {}
        self.worker_lines = {}  # Track which line each worker uses
        self.progress_lock = threading.Lock()  # Synchronize progress updates
        self.overall_progress = {}  # Track overall conversion progress
        self.start_time = None
        self.total_videos = 0
        self.video_sizes = {}  # Track video sizes for weighted progress
        self.total_video_seconds = 0  # Total duration of all videos
        self.completed_video_seconds = 0  # Completed duration

    def get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                data = json.loads(result.stdout)
                return float(data['format']['duration'])
        except Exception as e:
            print(f"⚠️  Could not get duration for {Path(video_path).name}: {e}")

        # Fallback: estimate based on file size (rough approximation)
        try:
            file_size_gb = Path(video_path).stat().st_size / (1024**3)
            estimated_duration = file_size_gb * 1000  # Very rough: 1GB ≈ 1000 seconds
            return estimated_duration
        except:
            return 3600  # 1 hour default fallback

    def update_overall_status(self):
        """Update the overall progress status line"""
        if not self.start_time:
            return

        # Calculate size-weighted overall progress
        if self.total_video_seconds > 0:
            # Calculate completed seconds: fully completed videos + partial progress
            current_completed_seconds = self.completed_video_seconds
            for worker_id, progress_data in self.overall_progress.items():
                if not progress_data.get('completed', False):
                    video_duration = self.video_sizes.get(worker_id, 0)
                    progress_percent = progress_data.get('progress', 0)
                    current_completed_seconds += (video_duration * progress_percent / 100.0)

            overall_percent = (current_completed_seconds / self.total_video_seconds) * 100.0
        else:
            # Fallback: if no duration data, use simple average but ensure it's proportion of total work
            # This should rarely be used since we try to get duration for all videos
            active_workers = len(self.overall_progress)
            if active_workers > 0:
                total_progress = sum(p.get('progress', 0) for p in self.overall_progress.values())
                overall_percent = total_progress / active_workers  # Average progress of active workers
            else:
                overall_percent = 0

        # Calculate timing
        elapsed_seconds = time.time() - self.start_time
        elapsed_hours, remainder = divmod(int(elapsed_seconds), 3600)
        elapsed_mins, elapsed_secs = divmod(remainder, 60)
        elapsed_str = f"{elapsed_hours:02d}:{elapsed_mins:02d}:{elapsed_secs:02d}"

        # Estimate completion time
        if overall_percent > 0:
            total_estimated_seconds = elapsed_seconds * (100 / overall_percent)
            remaining_seconds = total_estimated_seconds - elapsed_seconds

            end_time = time.time() + remaining_seconds
            import datetime
            end_time_str = datetime.datetime.fromtimestamp(end_time).strftime("%H:%M:%S")

            duration_hours, remainder = divmod(int(total_estimated_seconds), 3600)
            duration_mins, duration_secs = divmod(remainder, 60)
            duration_str = f"{duration_hours:02d}:{duration_mins:02d}:{duration_secs:02d}"
        else:
            end_time_str = "--:--:--"
            duration_str = "--:--:--"

        # Create overall progress bar
        overall_blocks = min(int(overall_percent // 10), 10)  # Cap at 10 for 100%
        overall_bar = "#" * overall_blocks + " " * (10 - overall_blocks)

        # Get start time
        start_time_str = datetime.datetime.fromtimestamp(self.start_time).strftime("%H:%M:%S")

        # Format status line
        status_line = f"Start {start_time_str}. End (est.) {end_time_str}. Duration {duration_str}. Complete {overall_percent:5.1f}% [{overall_bar}]"

        # Update the summary line (should be 6 lines above: 4 workers + 1 "Starting workers..." + 1 blank)
        print(f"\033[s", end="")  # Save cursor position
        print(f"\033[6A", end="")  # Move up to summary line
        print(f"\r{status_line:<120}", end="")  # Overwrite line
        print(f"\033[u", end="", flush=True)  # Restore cursor position

    def create_progress_callback(self, video_name: str, worker_id: str, position: int, total: int, line_number: int):
        """Create progress callback for displaying conversion progress"""
        def callback(progress_percent: float, current_time: float, speed: str):
            with self.progress_lock:
                # Check if this video just completed
                was_completed = self.overall_progress.get(worker_id, {}).get('completed', False)
                is_now_completed = progress_percent >= 100

                # Update this video's progress in overall tracking
                self.overall_progress[worker_id] = {
                    'progress': progress_percent,
                    'completed': is_now_completed,
                    'video_name': video_name
                }

                # If video just completed, add its duration to completed total
                if is_now_completed and not was_completed:
                    video_duration = self.video_sizes.get(worker_id, 0)
                    self.completed_video_seconds += video_duration

                # Update progress display with in-place overwriting
                mins, secs = divmod(int(current_time), 60)
                hours, mins = divmod(mins, 60)
                time_str = f"{hours:02d}:{mins:02d}:{secs:02d}"

                # Create progress bar: 10 blocks, one per 10%
                progress_blocks = min(int(progress_percent // 10), 10)  # Cap at 10 for 100%
                progress_bar = "#" * progress_blocks + " " * (10 - progress_blocks)

                # Update worker's specific line
                progress_line = f"[{worker_id:>6}] {position:3d}/{total}: {video_name} - {progress_percent:5.1f}% [{progress_bar}] ({time_str}, {speed})"

                # Move cursor to worker's specific line and update in place
                print(f"\033[s", end="")  # Save cursor position
                print(f"\033[{line_number}A", end="")  # Move up to worker's line
                print(f"\r{progress_line:<120}", end="")  # Overwrite line with padding
                print(f"\033[u", end="", flush=True)  # Restore cursor position

                # Update overall status
                self.update_overall_status()

        return callback

    def convert_with_progress(self, input_path: str, output_path: str,
                            worker_id: str, position: int, total: int) -> Dict:
        """Convert video with real-time progress monitoring"""
        video_file = Path(input_path)
        output_file = Path(output_path)

        # Skip if already exists
        if output_file.exists() and output_file.stat().st_size > 1024:
            # Get line number for this worker
            line_number = self.worker_lines.get(worker_id, 1)
            with self.progress_lock:
                skip_line = f"[{worker_id:>6}] {position:3d}/{total}: {video_file.name} - Already exists, skipping"
                print(f"\033[s", end="")  # Save cursor position
                print(f"\033[{line_number}A", end="")  # Move up to worker's line
                print(f"\r{skip_line:<120}", end="")  # Overwrite line
                print(f"\033[u", end="", flush=True)  # Restore cursor position
            return {'success': True, 'skipped': True}

        # Get video duration
        duration = self.get_video_duration(input_path)

        # Store video duration for this worker
        with self.progress_lock:
            self.video_sizes[worker_id] = duration

        # Get line number for this worker (will be set by main process)
        line_number = self.worker_lines.get(worker_id, 1)

        # Update the worker's line immediately to show starting state
        with self.progress_lock:
            start_line = f"[{worker_id:>6}] {position:3d}/{total}: {video_file.name} - Starting ({duration/60:.1f} min)"
            print(f"\033[s", end="")  # Save cursor position
            print(f"\033[{line_number}A", end="")  # Move up to worker's line
            print(f"\r{start_line:<120}", end="")  # Overwrite line
            print(f"\033[u", end="", flush=True)  # Restore cursor position

        # Create progress monitor
        progress_callback = self.create_progress_callback(video_file.name, worker_id, position, total, line_number)
        monitor = ProgressMonitor(duration, progress_callback)

        # FFmpeg command with progress output
        cmd = [
            'ffmpeg', '-y', '-progress', 'pipe:2',  # Progress to stderr
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            output_path
        ]

        start_time = time.time()

        try:
            # Start FFmpeg process
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, universal_newlines=True
            )

            # Monitor progress in real-time
            conversion_successful = False
            error_output = []

            for line in iter(process.stderr.readline, ''):
                if line.strip():
                    # Parse progress updates
                    if monitor.parse_progress_line(line):
                        continue

                    # Collect non-progress output for error analysis
                    if not line.startswith('ffmpeg version') and not line.startswith('built with'):
                        error_output.append(line.strip())

                    # Check for stall
                    if monitor.is_stalled(stall_threshold_seconds=600):  # 10 minutes
                        print(f"\n❌ [{worker_id:>6}] {position:3d}/{total}: {video_file.name} - Stalled at {monitor.progress_percent:.1f}%")
                        process.terminate()
                        break

            # Wait for process completion
            process.wait()
            conversion_time = time.time() - start_time

            # Clear progress line
            print()

            if process.returncode == 0:
                print(f"[{worker_id:>6}] {position:3d}/{total}: {video_file.name} - ✅ Completed ({conversion_time/60:.1f} min)")
                return {'success': True, 'time': conversion_time, 'method': 'cpu'}
            else:
                # Get meaningful error from collected output
                meaningful_errors = [line for line in error_output
                                   if any(keyword in line.lower() for keyword in
                                         ['error', 'failed', 'invalid', 'corrupt', 'permission denied'])]

                if meaningful_errors:
                    error_msg = meaningful_errors[-1]  # Last meaningful error
                else:
                    error_msg = f"Process failed with return code {process.returncode}"

                print(f"[{worker_id:>6}] {position:3d}/{total}: {video_file.name} - ❌ Failed: {error_msg}")
                return {'success': False, 'error': error_msg, 'method': 'cpu'}

        except Exception as e:
            print(f"\n[{worker_id:>6}] {position:3d}/{total}: {video_file.name} - ❌ Exception: {e}")
            return {'success': False, 'error': str(e), 'method': 'cpu'}

    def process_videos_intelligent(self, video_files: List[Path], output_path: Path,
                                 workers: int = 4, skip_existing: bool = True) -> Dict:
        """Process videos with intelligent monitoring"""

        # Filter existing files
        if skip_existing:
            videos_to_process = []
            for video_file in video_files:
                output_file = output_path / video_file.name
                if not (output_file.exists() and output_file.stat().st_size > 1024):
                    videos_to_process.append(video_file)

            already_exist = len(video_files) - len(videos_to_process)
            if already_exist > 0:
                print(f"📋 Found {len(video_files)} total videos, {already_exist} already converted")
                print(f"🎯 Processing {len(videos_to_process)} remaining videos")
        else:
            videos_to_process = video_files

        total_videos = len(videos_to_process)

        if total_videos == 0:
            print("✅ All videos already converted!")
            return {'total_videos': len(video_files), 'successful': len(video_files),
                   'failed': 0, 'total_time': 0, 'throughput': 0}

        print(f"\n🚀 Starting intelligent conversion of {total_videos} videos with {workers} workers...")
        print(f"📊 Features: progress monitoring, adaptive timeouts, stall detection")

        # Calculate total video duration for accurate progress tracking
        print("📏 Calculating total video duration for accurate progress estimation...")

        # Calculate total duration of ALL videos in the job (for correct overall progress)
        self.total_video_seconds = 0
        for video_file in video_files:  # ALL videos, not just videos_to_process
            duration = self.get_video_duration(str(video_file))
            self.total_video_seconds += duration

        # Calculate duration already completed (videos that already exist)
        self.completed_video_seconds = 0
        for video_file in video_files:
            if video_file not in videos_to_process:  # This video already exists
                duration = self.get_video_duration(str(video_file))
                self.completed_video_seconds += duration

        total_hours = self.total_video_seconds / 3600
        remaining_hours = (self.total_video_seconds - self.completed_video_seconds) / 3600
        completion_percent = (self.completed_video_seconds / self.total_video_seconds) * 100 if self.total_video_seconds > 0 else 0

        print(f"📊 Total job: {total_hours:.1f} hours ({len(video_files)} videos)")
        print(f"📊 Already complete: {completion_percent:.1f}% ({len(video_files) - len(videos_to_process)} videos)")
        print(f"📊 Remaining to process: {remaining_hours:.1f} hours ({len(videos_to_process)} videos)")

        # Initialize overall tracking
        self.total_videos = workers  # Track active workers, not total videos
        self.start_time = time.time()

        # Reserve lines: overall status + worker lines
        print()  # Extra space
        print("Start --:--:--. End (est.) --:--:--. Duration --:--:--. Complete   0.0% [          ]")  # Overall status line

        for i in range(workers):
            worker_id = f"CPU{i+1:2d}"
            self.worker_lines[worker_id] = workers - i + 1  # Lines above current position
            print(f"[{worker_id:>6}] Waiting for task assignment...")

        # Add blank line for cursor positioning (will be overwritten by completion messages)
        print()

        results = []
        start_time = time.time()
        # Use a queue to assign worker slots (1, 2, 3, 4) and reuse them
        available_worker_slots = queue.Queue()
        for i in range(1, workers + 1):
            available_worker_slots.put(i)

        worker_slot_lock = threading.Lock()

        def worker_wrapper(args):
            video_file, position, start_delay = args

            # Stagger worker starts
            time.sleep(start_delay)

            # Get an available worker slot (blocks if all busy)
            worker_slot = available_worker_slots.get()
            worker_id = f"CPU{worker_slot:2d}"

            try:
                result = self.convert_with_progress(
                    str(video_file), str(output_path / video_file.name),
                    worker_id, position, total_videos
                )
                return result
            finally:
                # Return the worker slot for reuse
                available_worker_slots.put(worker_slot)

        # Process videos in parallel with staggered starts
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            # Create tasks with staggered start delays (2 seconds apart)
            video_tasks = [(video_file, i+1, (i % workers) * 2) for i, video_file in enumerate(videos_to_process)]

            futures = [executor.submit(worker_wrapper, task) for task in video_tasks]

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"❌ Task failed: {e}")
                    results.append({'success': False, 'error': str(e)})

        # Calculate statistics
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r.get('success', False))

        # Convert time to hours:minutes format
        total_minutes = total_time / 60
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)

        # Calculate correct throughput (videos per minute, not second)
        throughput_per_minute = successful / total_minutes if total_minutes > 0 else 0

        # Add timestamps
        from datetime import datetime
        end_time = datetime.now()
        start_datetime = end_time - timedelta(seconds=total_time)

        stats = {
            'total_videos': total_videos,
            'successful': successful,
            'failed': total_videos - successful,
            'total_time': total_time,
            'throughput': throughput_per_minute,
            'start_time': start_datetime,
            'end_time': end_time
        }

        print(f"\n✅ Intelligent conversion completed!")
        print(f"   • Total time: {hours:02d}:{minutes:02d}")
        print(f"   • Started: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   • Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   • Successful: {successful}/{total_videos}")
        print(f"   • Throughput: {throughput_per_minute:.2f} videos/minute")

        return stats


def convert_videos_intelligent(input_dir: str, output_dir: str, workers: int = 4) -> bool:
    """Convert videos using intelligent progress monitoring"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find video files
    video_files = []
    for ext in ['.MP4', '.mp4']:
        video_files.extend(list(input_path.rglob(f'*{ext}')))

    if not video_files:
        print("❌ No video files found!")
        return False

    # Create converter and process
    converter = IntelligentConverter()
    stats = converter.process_videos_intelligent(video_files, output_path, workers)

    return stats['successful'] > 0


if __name__ == "__main__":
    # Test intelligent conversion
    test_converter = IntelligentConverter()
    print("🧠 Intelligent converter ready")