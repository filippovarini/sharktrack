#!/usr/bin/env python3
"""
GPU-Accelerated Video Converter for BRUV Processing
Uses NVIDIA NVENC hardware encoding for massive performance improvements
"""

import subprocess
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import concurrent.futures
import queue

class GPUAcceleratedConverter:
    """
    Hybrid CPU+GPU video conversion system
    """

    def __init__(self, cpu_workers: int = 4, gpu_workers: int = 2):
        self.cpu_workers = cpu_workers
        self.gpu_workers = gpu_workers
        self.gpu_available = self._check_gpu_support()

        # Separate queues for CPU and GPU tasks
        self.cpu_queue = queue.Queue()
        self.gpu_queue = queue.Queue()

        print(f"🎯 Converter initialized:")
        print(f"   • CPU workers: {cpu_workers}")
        print(f"   • GPU workers: {gpu_workers if self.gpu_available else 0} (NVENC)")
        print(f"   • GPU available: {self.gpu_available}")

    def _check_gpu_support(self) -> bool:
        """Check if GPU encoding is available"""
        try:
            # Test NVENC availability
            result = subprocess.run([
                'ffmpeg', '-f', 'lavfi', '-i', 'testsrc=duration=1:size=320x240:rate=1',
                '-c:v', 'h264_nvenc', '-f', 'null', '-'
            ], capture_output=True, timeout=10)

            return result.returncode == 0
        except:
            return False

    def get_gpu_command(self, input_path: str, output_path: str) -> List[str]:
        """Generate GPU-accelerated FFmpeg command"""
        return [
            'ffmpeg', '-y',
            '-hwaccel', 'cuda',                    # Hardware decode acceleration
            '-i', input_path,
            '-c:v', 'h264_nvenc',                  # GPU encoding
            '-preset', 'fast',                     # NVENC preset (fast/medium/slow)
            '-crf', '23',                          # Quality setting
            '-c:a', 'aac',                         # Audio codec
            '-b:a', '128k',                        # Audio bitrate
            '-movflags', '+faststart',             # Optimize for streaming
            output_path
        ]

    def get_cpu_command(self, input_path: str, output_path: str) -> List[str]:
        """Generate CPU-only FFmpeg command (fallback)"""
        return [
            'ffmpeg', '-y',
            '-i', input_path,
            '-c:v', 'libx264',                     # CPU encoding
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            output_path
        ]

    def convert_single_video_gpu(self, video_file: Path, output_path: Path,
                                worker_id: int, position: int, total: int) -> dict:
        """Convert single video using GPU acceleration"""
        output_file = output_path / video_file.name

        # Skip if already exists
        if output_file.exists() and output_file.stat().st_size > 1024:
            print(f"[GPU {worker_id:2d}] {position:3d}/{total}: {video_file.name} - Already exists, skipping")
            return {'success': True, 'skipped': True, 'method': 'gpu'}

        print(f"[GPU {worker_id:2d}] {position:3d}/{total}: {video_file.name} - Starting GPU conversion")

        cmd = self.get_gpu_command(str(video_file), str(output_file))

        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            conversion_time = time.time() - start_time

            if result.returncode == 0:
                print(f"[GPU {worker_id:2d}] {position:3d}/{total}: {video_file.name} - ✅ Completed ({conversion_time:.1f}s)")
                return {'success': True, 'method': 'gpu', 'time': conversion_time}
            else:
                print(f"[GPU {worker_id:2d}] {position:3d}/{total}: {video_file.name} - ❌ GPU failed, trying CPU")
                # Fallback to CPU
                return self.convert_single_video_cpu(video_file, output_path, f"GPU{worker_id}→CPU", position, total)

        except subprocess.TimeoutExpired:
            print(f"[GPU {worker_id:2d}] {position:3d}/{total}: {video_file.name} - ⏰ GPU timeout, trying CPU")
            return self.convert_single_video_cpu(video_file, output_path, f"GPU{worker_id}→CPU", position, total)
        except Exception as e:
            print(f"[GPU {worker_id:2d}] {position:3d}/{total}: {video_file.name} - ❌ GPU error: {e}")
            return {'success': False, 'method': 'gpu', 'error': str(e)}

    def convert_single_video_cpu(self, video_file: Path, output_path: Path,
                                worker_id: str, position: int, total: int) -> dict:
        """Convert single video using CPU (fallback)"""
        output_file = output_path / video_file.name

        print(f"[CPU {worker_id:>2}] {position:3d}/{total}: {video_file.name} - Starting CPU conversion")

        cmd = self.get_cpu_command(str(video_file), str(output_file))

        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            conversion_time = time.time() - start_time

            if result.returncode == 0:
                print(f"[CPU {worker_id:>2}] {position:3d}/{total}: {video_file.name} - ✅ Completed ({conversion_time:.1f}s)")
                return {'success': True, 'method': 'cpu', 'time': conversion_time}
            else:
                # Extract meaningful error from stderr (skip version banner)
                stderr_lines = result.stderr.split('\n')
                meaningful_errors = [line for line in stderr_lines
                                   if any(keyword in line.lower() for keyword in
                                         ['error', 'failed', 'invalid', 'corrupt', 'permission denied', 'no such file'])]

                if meaningful_errors:
                    error_msg = meaningful_errors[-1][:100]  # Last meaningful error, truncated
                else:
                    error_msg = f"Process failed with return code {result.returncode}"

                print(f"[CPU {worker_id:>2}] {position:3d}/{total}: {video_file.name} - ❌ Failed: {error_msg}")
                return {'success': False, 'method': 'cpu', 'error': error_msg}

        except subprocess.TimeoutExpired:
            print(f"[CPU {worker_id:>2}] {position:3d}/{total}: {video_file.name} - ⏰ Timed out")
            return {'success': False, 'method': 'cpu', 'error': 'Timeout'}
        except Exception as e:
            print(f"[CPU {worker_id:>2}] {position:3d}/{total}: {video_file.name} - ❌ Error: {e}")
            return {'success': False, 'method': 'cpu', 'error': str(e)}

    def process_videos_hybrid(self, video_files: List[Path], output_path: Path, skip_existing: bool = True) -> Dict:
        """
        Process videos using hybrid CPU+GPU approach
        """
        # Filter out already existing files if requested
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
            print(f"✅ All videos already converted!")
            return {'total_videos': len(video_files), 'successful': len(video_files), 'failed': 0,
                   'gpu_successes': 0, 'cpu_successes': 0, 'total_time': 0, 'throughput': 0}

        print(f"\n🚀 Starting hybrid CPU+GPU conversion of {total_videos} videos...")

        if self.gpu_available:
            print(f"🎮 GPU acceleration enabled - expected 3-5x speedup")
        else:
            print(f"⚠️  GPU not available - falling back to CPU only")

        # Create task assignments using only videos that need processing
        gpu_tasks = []
        cpu_tasks = []

        if self.gpu_available:
            # Assign 70% of videos to GPU, 30% to CPU
            gpu_count = int(total_videos * 0.7)
            gpu_tasks = [(videos_to_process[i], i+1) for i in range(gpu_count)]
            cpu_tasks = [(videos_to_process[i], i+1) for i in range(gpu_count, total_videos)]
        else:
            # All videos go to CPU
            cpu_tasks = [(videos_to_process[i], i+1) for i in range(total_videos)]

        print(f"📋 Task distribution: {len(gpu_tasks)} GPU, {len(cpu_tasks)} CPU")

        # Show time estimates after task distribution
        if total_videos > 0:
            if self.gpu_available and len(gpu_tasks) > 0:
                # GPU is typically 3-5x faster, so adjust estimate
                effective_workers = len(cpu_tasks) + (len(gpu_tasks) * 4)  # GPU tasks count as 4x CPU
                worker_label = f"{self.cpu_workers}x CPU + {self.gpu_workers}x GPU"
            else:
                effective_workers = self.cpu_workers
                worker_label = f"{self.cpu_workers}x CPU"

            estimated_time_hours = (total_videos * 2) / (effective_workers * 60)  # 2 min per video estimate
            estimated_minutes = estimated_time_hours * 60
            hours = int(estimated_minutes // 60)
            minutes = int(estimated_minutes % 60)

            # Calculate completion time
            from datetime import datetime, timedelta
            completion_time = datetime.now() + timedelta(hours=hours, minutes=minutes)

            print(f"   • Estimated duration: {hours:02d}:{minutes:02d} ({worker_label})")
            print(f"   • Estimated completion time: {completion_time.strftime('%Y-%m-%d %H:%M')}")

        results = []
        start_time = time.time()

        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.cpu_workers + self.gpu_workers) as executor:
            futures = []

            # Submit GPU tasks
            for i, (video_file, position) in enumerate(gpu_tasks):
                worker_id = (i % self.gpu_workers) + 1
                future = executor.submit(
                    self.convert_single_video_gpu,
                    video_file, output_path, worker_id, position, total_videos
                )
                futures.append(future)

            # Submit CPU tasks
            for i, (video_file, position) in enumerate(cpu_tasks):
                worker_id = (i % self.cpu_workers) + 1
                future = executor.submit(
                    self.convert_single_video_cpu,
                    video_file, output_path, str(worker_id), position, total_videos
                )
                futures.append(future)

            # Collect results
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
        gpu_successes = sum(1 for r in results if r.get('success', False) and r.get('method') == 'gpu')
        cpu_successes = sum(1 for r in results if r.get('success', False) and r.get('method') == 'cpu')

        # Calculate average conversion times
        gpu_times = [r.get('time', 0) for r in results if r.get('method') == 'gpu' and r.get('success')]
        cpu_times = [r.get('time', 0) for r in results if r.get('method') == 'cpu' and r.get('success')]

        stats = {
            'total_videos': total_videos,
            'successful': successful,
            'failed': total_videos - successful,
            'gpu_successes': gpu_successes,
            'cpu_successes': cpu_successes,
            'total_time': total_time,
            'avg_gpu_time': sum(gpu_times) / len(gpu_times) if gpu_times else 0,
            'avg_cpu_time': sum(cpu_times) / len(cpu_times) if cpu_times else 0,
            'throughput': successful / total_time if total_time > 0 else 0
        }

        print(f"\n✅ Hybrid conversion completed!")
        print(f"   • Total time: {total_time/60:.1f} minutes")
        print(f"   • Successful: {successful}/{total_videos}")
        print(f"   • GPU conversions: {gpu_successes} (avg: {stats['avg_gpu_time']:.1f}s)")
        print(f"   • CPU conversions: {cpu_successes} (avg: {stats['avg_cpu_time']:.1f}s)")
        print(f"   • Throughput: {stats['throughput']:.2f} videos/second")

        if gpu_times and cpu_times:
            speedup = stats['avg_cpu_time'] / stats['avg_gpu_time']
            print(f"   • GPU speedup: {speedup:.1f}x faster than CPU")

        return stats


def convert_videos_gpu_accelerated(input_dir: str, output_dir: str,
                                  cpu_workers: int = 8, gpu_workers: int = 2) -> bool:
    """
    Convert videos with GPU acceleration
    """
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
    converter = GPUAcceleratedConverter(cpu_workers=cpu_workers, gpu_workers=gpu_workers)
    stats = converter.process_videos_hybrid(video_files, output_path)

    return stats['successful'] > 0


if __name__ == "__main__":
    # Test GPU acceleration
    test_converter = GPUAcceleratedConverter()
    print(f"GPU support test completed")