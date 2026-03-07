"""
Deployment/Retrieval Detection for BRUV Videos
Detects unstable periods (deployment/retrieval) using background motion analysis
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional


class DeploymentDetector:
    """
    Detects deployment and retrieval phases in BRUV videos by analyzing
    background stability and camera motion.
    """

    def __init__(self, stability_threshold: float = 0.15, min_stable_duration: int = 60,
                 sample_rate: int = 15):
        """
        Args:
            stability_threshold: Motion threshold (0-1). Higher = more lenient. Default 0.15
            min_stable_duration: Minimum seconds of stability to consider "deployed". Default 60s
            sample_rate: Seconds between sampled frames. Default 15s
        """
        self.stability_threshold = stability_threshold
        self.min_stable_duration = min_stable_duration
        self.sample_rate = sample_rate

    def analyze_video_stability(self, video_path: str, sample_rate: int = None) -> Tuple[int, int, List[float]]:
        """
        Analyze entire video to detect stable recording period.

        Args:
            video_path: Path to video file
            sample_rate: Seconds between samples (default: use self.sample_rate)

        Returns:
            Tuple of (start_frame, end_frame, motion_scores)
            - start_frame: First frame of stable period
            - end_frame: Last frame of stable period
            - motion_scores: List of motion scores for visualization
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps == 0 or total_frames == 0:
            print(f"Warning: Could not read video properties for {video_path}")
            return 0, total_frames, []

        frame_skip = int(fps * sample_rate)
        motion_scores = []
        frame_indices = []

        prev_frame = None
        frame_idx = 0

        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                break

            # Convert to grayscale and resize for speed
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)

            if prev_frame is not None:
                motion_score = self._compute_motion_score(prev_frame, small)
                motion_scores.append(motion_score)
                frame_indices.append(frame_idx)

            prev_frame = small
            frame_idx += frame_skip

        cap.release()

        if len(motion_scores) == 0:
            print("Warning: No motion scores computed")
            return 0, total_frames, []

        # Find stable period
        start_frame, end_frame = self._find_stable_period(
            motion_scores, frame_indices, fps, sample_rate
        )

        total_sec = total_frames / fps
        stable_sec = (end_frame - start_frame) / fps
        print(f"  Stable: {start_frame/fps:.0f}s to {end_frame/fps:.0f}s "
              f"({stable_sec:.0f}s of {total_sec:.0f}s)")

        return start_frame, end_frame, motion_scores

    def _compute_motion_score(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """
        Compute normalized motion score between two frames.

        Returns:
            float: Motion score between 0 (stable) and 1 (high motion)
        """
        # Frame difference
        diff = cv2.absdiff(prev_frame, curr_frame)

        # Threshold to get binary motion mask
        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Calculate percentage of pixels with motion
        motion_pixels = np.count_nonzero(motion_mask)
        total_pixels = motion_mask.size
        motion_ratio = motion_pixels / total_pixels

        # Also consider magnitude of change
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)

        # Combined score (weighted average)
        score = (motion_ratio * 0.6) + (mean_diff / 255.0 * 0.3) + (max_diff / 255.0 * 0.1)

        return float(score)

    def _find_stable_period(self, motion_scores: List[float], frame_indices: List[int],
                           fps: float, sample_rate: int) -> Tuple[int, int]:
        """
        Find the longest continuous stable period in the video.

        Uses two-phase approach:
        1. Find deployment start: first sustained low-motion period
        2. Find retrieval start: work backwards from end looking for sharp onset
           of high-magnitude motion (>retrieval_threshold). Distinguishes real
           retrieval (scores 0.3-0.8) from animal activity (scores 0.1-0.2).

        Returns:
            Tuple of (start_frame, end_frame)
        """
        if len(motion_scores) == 0:
            return 0, 0

        # Smooth motion scores with moving average (3 samples = 45s at 15s rate)
        window_size = min(3, len(motion_scores))
        smoothed = np.convolve(motion_scores, np.ones(window_size)/window_size, mode='same')

        # Find stable regions (below threshold)
        is_stable = smoothed < self.stability_threshold

        # Find continuous stable periods
        stable_periods = []
        start_idx = None

        for i, stable in enumerate(is_stable):
            if stable and start_idx is None:
                start_idx = i
            elif not stable and start_idx is not None:
                duration = (i - start_idx) * sample_rate
                if duration >= self.min_stable_duration:
                    stable_periods.append((start_idx, i))
                start_idx = None

        # Handle case where video ends during stable period
        if start_idx is not None:
            duration = (len(is_stable) - start_idx) * sample_rate
            if duration >= self.min_stable_duration:
                stable_periods.append((start_idx, len(is_stable)))

        if not stable_periods:
            # No stable period found, return middle section as best guess
            print("  Warning: No stable period detected, using middle 80% of video")
            total_duration = len(frame_indices)
            margin = int(total_duration * 0.1)
            return frame_indices[margin], frame_indices[-margin] if len(frame_indices) > margin else frame_indices[-1]

        # Find deployment start from longest stable period
        longest_period = max(stable_periods, key=lambda p: p[1] - p[0])
        start_idx = longest_period[0]
        start_frame = frame_indices[start_idx]

        # Find retrieval: work backwards from end looking for sustained high motion
        end_idx = self._find_retrieval_onset(motion_scores, smoothed, sample_rate)

        if end_idx is not None and end_idx > start_idx:
            end_frame = frame_indices[end_idx] if end_idx < len(frame_indices) else frame_indices[-1]
        else:
            # No retrieval detected — stable period extends to end of video
            end_frame = frame_indices[-1]

        return start_frame, end_frame

    def _find_retrieval_onset(self, motion_scores: List[float], smoothed: np.ndarray,
                              sample_rate: int) -> Optional[int]:
        """
        Find retrieval onset by scanning backwards from end of video.

        After retrieval, the camera may sit still on the boat producing low-motion
        samples at the very end. This method skips any trailing quiet period to
        find the high-motion retrieval block. Scans the last 50% of samples.

        Real retrieval has high-magnitude motion (mean >0.25, or max >0.4).
        Animal activity produces lower scores (0.1-0.2). Requires at least
        min_retrieval_samples (3) above-threshold samples in the block.

        Returns:
            Index of first retrieval sample, or None if no retrieval detected.
        """
        retrieval_mag_threshold = 0.25  # mean motion for real retrieval
        retrieval_max_threshold = 0.4   # or max motion for real retrieval
        min_retrieval_samples = 3       # at least 3 samples (~45s)
        n = len(motion_scores)

        if n < min_retrieval_samples + 1:
            return None

        # Scan the last 50% of the video for a retrieval block
        search_start = max(0, int(n * 0.50))

        # Find all above-threshold samples in the search region
        above = [i for i in range(search_start, n)
                 if motion_scores[i] > self.stability_threshold]

        if len(above) < min_retrieval_samples:
            return None

        # Find contiguous runs of above-threshold samples
        runs = []
        run_start = above[0]
        prev = above[0]
        for idx in above[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                runs.append((run_start, prev))
                run_start = idx
                prev = idx
        runs.append((run_start, prev))

        # Find the longest run (or merge runs separated by at most 1 gap sample)
        merged = [runs[0]]
        for run in runs[1:]:
            last = merged[-1]
            if run[0] - last[1] <= 2:  # allow 1-sample gap (30s)
                merged[-1] = (last[0], run[1])
            else:
                merged.append(run)

        # Take the longest merged block
        best = max(merged, key=lambda r: r[1] - r[0] + 1)
        block_start, block_end = best
        block_len = block_end - block_start + 1

        if block_len < min_retrieval_samples:
            return None

        # Check magnitude: real retrieval vs animal activity
        block_scores = motion_scores[block_start:block_end + 1]
        mean_block = np.mean(block_scores)
        max_block = np.max(block_scores)

        if mean_block < retrieval_mag_threshold and max_block < retrieval_max_threshold:
            return None

        # Retrieval confirmed — return the start of the block
        return block_start

    def should_skip_frame(self, frame_number: int, fps: float, start_frame: int, end_frame: int) -> bool:
        """
        Determine if a frame should be skipped based on stable period.

        Args:
            frame_number: Current frame number
            fps: Video FPS
            start_frame: First frame of stable period
            end_frame: Last frame of stable period

        Returns:
            bool: True if frame should be skipped
        """
        return frame_number < start_frame or frame_number > end_frame

    def get_stable_times(self, video_path: str) -> dict:
        """
        Get stable period timing for a video.

        Returns:
            dict with keys: stable_start_sec, stable_end_sec, stable_duration_sec,
                           total_duration_sec, deploy_sec, retrieve_sec
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if fps == 0 or total_frames == 0:
            print(f"Warning: Could not read video, returning no times")
            return {
                'stable_start_sec': 0, 'stable_end_sec': 0,
                'stable_duration_sec': 0, 'total_duration_sec': 0,
                'deploy_sec': 0, 'retrieve_sec': 0,
            }

        start_frame, end_frame, _ = self.analyze_video_stability(video_path)

        total_sec = total_frames / fps
        stable_start = start_frame / fps
        stable_end = end_frame / fps

        return {
            'stable_start_sec': round(stable_start, 1),
            'stable_end_sec': round(stable_end, 1),
            'stable_duration_sec': round(stable_end - stable_start, 1),
            'total_duration_sec': round(total_sec, 1),
            'deploy_sec': round(stable_start, 1),
            'retrieve_sec': round(total_sec - stable_end, 1),
        }

    def get_skip_times(self, video_path: str) -> Tuple[float, float]:
        """
        Get skip times in seconds for command line usage.
        Kept for backwards compatibility with app.py.

        Returns:
            Tuple of (skip_start_seconds, skip_end_seconds)
        """
        times = self.get_stable_times(video_path)
        return times['deploy_sec'], times['retrieve_sec']


def analyze_deployment_batch(video_paths: List[str], output_file: Optional[str] = None,
                             checkpoint_file: Optional[str] = None) -> List[dict]:
    """
    Analyze multiple videos and compute stable recording durations.
    Checkpoints after each video for resumability.

    Args:
        video_paths: List of video file paths
        output_file: Optional CSV file to save results
        checkpoint_file: Optional JSON checkpoint for resuming

    Returns:
        List of dicts with timing info per video
    """
    import json

    detector = DeploymentDetector()
    results = []
    completed = set()

    # Load checkpoint
    if checkpoint_file and Path(checkpoint_file).exists():
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
        results = checkpoint.get('results', [])
        completed = {r['video_path'] for r in results}
        print(f"Resuming from checkpoint: {len(completed)} videos done")

    for i, video_path in enumerate(video_paths):
        if video_path in completed:
            continue

        print(f"\n[{len(completed)+1}/{len(video_paths)}] {Path(video_path).name}")
        try:
            times = detector.get_stable_times(video_path)
            times['video_path'] = video_path
            results.append(times)
            completed.add(video_path)
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'video_path': video_path,
                'stable_start_sec': 0, 'stable_end_sec': 0,
                'stable_duration_sec': 0, 'total_duration_sec': 0,
                'deploy_sec': 0, 'retrieve_sec': 0,
            })
            completed.add(video_path)

        # Checkpoint
        if checkpoint_file and len(completed) % 5 == 0:
            with open(checkpoint_file, 'w') as f:
                json.dump({'results': results}, f)

    # Final save
    if checkpoint_file:
        with open(checkpoint_file, 'w') as f:
            json.dump({'results': results}, f)

    if output_file:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python deployment_detector.py <video_path_or_directory>")
        sys.exit(1)

    target = sys.argv[1]
    detector = DeploymentDetector()

    if Path(target).is_file():
        times = detector.get_stable_times(target)
        print(f"\nStable period: {times['stable_start_sec']}s to {times['stable_end_sec']}s")
        print(f"Stable duration: {times['stable_duration_sec']}s")
        print(f"Deployment: {times['deploy_sec']}s, Retrieval: {times['retrieve_sec']}s")
    elif Path(target).is_dir():
        videos = sorted(str(p) for p in Path(target).rglob("*.MP4"))
        if not videos:
            videos = sorted(str(p) for p in Path(target).rglob("*.mp4"))
        print(f"Found {len(videos)} videos")
        analyze_deployment_batch(
            videos,
            output_file=str(Path(target) / "stable_durations.csv"),
            checkpoint_file=str(Path(target) / "stable_durations_checkpoint.json"),
        )
    else:
        print(f"Not found: {target}")
