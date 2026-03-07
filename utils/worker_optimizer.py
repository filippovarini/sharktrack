#!/usr/bin/env python3
"""
Worker Optimization System
Automatically determines optimal number of parallel workers for BRUV processing
"""

import os
import time
import psutil
import subprocess
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
import shutil

class WorkerOptimizer:
    """Automatically determine optimal worker count for video processing"""

    def __init__(self, test_video_path: Optional[str] = None):
        self.test_video_path = test_video_path
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> Dict:
        """Get system hardware information"""
        return {
            'cpu_cores': multiprocessing.cpu_count(),
            'cpu_logical': psutil.cpu_count(logical=True),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'gpu_available': self._check_gpu_available(),
            'disk_type': self._detect_disk_type()
        }

    def _check_gpu_available(self) -> bool:
        """Check if CUDA GPU is available"""
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def _detect_disk_type(self) -> str:
        """Detect if storage is SSD or HDD"""
        try:
            # Check if root filesystem is on SSD
            result = subprocess.run(['lsblk', '-d', '-o', 'name,rota'],
                                  capture_output=True, text=True)
            if '0' in result.stdout:  # 0 = SSD, 1 = HDD
                return 'SSD'
            else:
                return 'HDD'
        except:
            return 'Unknown'

    def get_theoretical_optimum(self) -> Dict:
        """Calculate theoretical optimal worker count based on system specs"""
        cpu_cores = self.system_info['cpu_cores']
        memory_gb = self.system_info['memory_gb']
        gpu_available = self.system_info['gpu_available']
        disk_type = self.system_info['disk_type']

        # Base recommendations
        recommendations = {
            'conservative': max(1, cpu_cores // 2),
            'balanced': cpu_cores,
            'aggressive': cpu_cores + 2
        }

        # Adjust for memory constraints (assume 2GB per worker)
        max_memory_workers = int(memory_gb // 2)

        # Adjust for disk type
        disk_multiplier = 1.5 if disk_type == 'SSD' else 0.75

        # Adjust for GPU availability
        gpu_multiplier = 1.2 if gpu_available else 1.0

        # Apply constraints
        for key in recommendations:
            workers = recommendations[key]
            workers = min(workers, max_memory_workers)
            workers = int(workers * disk_multiplier * gpu_multiplier)
            workers = max(1, min(workers, 16))  # Cap at 16 workers
            recommendations[key] = workers

        return {
            'system_info': self.system_info,
            'recommendations': recommendations,
            'memory_constraint': max_memory_workers,
            'reasoning': self._get_reasoning(recommendations)
        }

    def _get_reasoning(self, recommendations: Dict) -> str:
        """Provide reasoning for worker recommendations"""
        reasoning = []

        reasoning.append(f"CPU cores: {self.system_info['cpu_cores']}")
        reasoning.append(f"Memory: {self.system_info['memory_gb']:.1f} GB")
        reasoning.append(f"GPU: {'Available' if self.system_info['gpu_available'] else 'Not available'}")
        reasoning.append(f"Storage: {self.system_info['disk_type']}")

        reasoning.append(f"Conservative: {recommendations['conservative']} workers (safest)")
        reasoning.append(f"Balanced: {recommendations['balanced']} workers (recommended)")
        reasoning.append(f"Aggressive: {recommendations['aggressive']} workers (maximum performance)")

        return "; ".join(reasoning)

    def benchmark_workers(self, test_videos: List[str],
                         worker_counts: List[int] = None) -> Dict:
        """Benchmark different worker counts with actual video processing"""

        if worker_counts is None:
            theoretical = self.get_theoretical_optimum()
            worker_counts = [1,
                           theoretical['recommendations']['conservative'],
                           theoretical['recommendations']['balanced'],
                           theoretical['recommendations']['aggressive']]
            # Remove duplicates and sort
            worker_counts = sorted(list(set(worker_counts)))

        results = {}

        print(f"🧪 Benchmarking worker counts: {worker_counts}")
        print(f"📹 Test videos: {len(test_videos)}")

        for worker_count in worker_counts:
            print(f"\n⚡ Testing {worker_count} workers...")

            start_time = time.time()
            success_rate = self._run_processing_test(test_videos, worker_count)
            end_time = time.time()

            processing_time = end_time - start_time
            throughput = len(test_videos) / processing_time if processing_time > 0 else 0

            # Monitor system resources during test
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent

            results[worker_count] = {
                'processing_time': processing_time,
                'throughput': throughput,  # videos per second
                'success_rate': success_rate,
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'efficiency': throughput / worker_count  # throughput per worker
            }

            print(f"   Time: {processing_time:.1f}s, "
                  f"Throughput: {throughput:.2f} videos/sec, "
                  f"Success: {success_rate:.1%}")

        # Find optimal worker count
        optimal = self._find_optimal_workers(results)

        return {
            'benchmark_results': results,
            'optimal_workers': optimal,
            'recommendation': self._generate_recommendation(results, optimal)
        }

    def _run_processing_test(self, test_videos: List[str], worker_count: int) -> float:
        """Run actual processing test with specified worker count"""

        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            successful_conversions = 0

            # Simulate parallel processing (simplified)
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = []

                for video_path in test_videos:
                    future = executor.submit(self._process_single_video, video_path, temp_dir)
                    futures.append(future)

                for future in concurrent.futures.as_completed(futures):
                    try:
                        success = future.result(timeout=300)  # 5 minute timeout per video
                        if success:
                            successful_conversions += 1
                    except Exception as e:
                        print(f"   Video processing failed: {e}")

            return successful_conversions / len(test_videos) if test_videos else 0.0

    def _process_single_video(self, video_path: str, output_dir: str) -> bool:
        """Process a single video (simplified version for testing)"""
        try:
            output_file = Path(output_dir) / f"test_{Path(video_path).stem}.mp4"

            # Simple ffmpeg conversion for testing
            cmd = [
                'ffmpeg', '-i', video_path,
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28',
                '-t', '10',  # Only process first 10 seconds for testing
                '-y', str(output_file)
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=120)
            return result.returncode == 0

        except Exception:
            return False

    def _find_optimal_workers(self, results: Dict) -> int:
        """Analyze results to find optimal worker count"""

        best_efficiency = 0
        optimal_workers = 1

        for worker_count, metrics in results.items():
            # Consider throughput, efficiency, and success rate
            if metrics['success_rate'] > 0.8:  # Minimum 80% success rate
                score = metrics['efficiency'] * metrics['success_rate']

                # Penalize high resource usage
                if metrics['cpu_usage'] > 90 or metrics['memory_usage'] > 85:
                    score *= 0.8

                if score > best_efficiency:
                    best_efficiency = score
                    optimal_workers = worker_count

        return optimal_workers

    def _generate_recommendation(self, results: Dict, optimal: int) -> str:
        """Generate human-readable recommendation"""

        optimal_metrics = results[optimal]

        recommendation = f"""
🎯 Optimal Worker Count: {optimal}

📊 Performance Metrics:
• Processing time: {optimal_metrics['processing_time']:.1f} seconds
• Throughput: {optimal_metrics['throughput']:.2f} videos/second
• Success rate: {optimal_metrics['success_rate']:.1%}
• CPU usage: {optimal_metrics['cpu_usage']:.1f}%
• Memory usage: {optimal_metrics['memory_usage']:.1f}%
• Efficiency: {optimal_metrics['efficiency']:.3f} videos/sec/worker

💡 For 398 videos, estimated processing time: {398 / optimal_metrics['throughput'] / 3600:.1f} hours
"""

        return recommendation.strip()

    def quick_test(self, sample_videos: List[str] = None, max_test_videos: int = 5) -> Dict:
        """Quick optimization test with sample videos"""

        if sample_videos is None:
            # Find sample videos automatically
            sample_videos = self._find_sample_videos(max_test_videos)

        if not sample_videos:
            # Fallback to theoretical optimization
            print("⚠️  No test videos found, using theoretical optimization")
            return self.get_theoretical_optimum()

        # Limit test videos for speed
        test_videos = sample_videos[:max_test_videos]

        print(f"🚀 Quick worker optimization test")
        print(f"📹 Using {len(test_videos)} sample videos")

        return self.benchmark_workers(test_videos)

    def _find_sample_videos(self, max_videos: int) -> List[str]:
        """Find sample videos for testing"""
        video_extensions = ['.MP4', '.mp4', '.MOV', '.mov', '.AVI', '.avi']
        sample_videos = []

        # Check current directory and input_videos for sample videos
        search_paths = [
            '.',
            'input_videos',
        ]

        for search_path in search_paths:
            if not Path(search_path).exists():
                continue

            for ext in video_extensions:
                videos = list(Path(search_path).rglob(f'*{ext}'))
                sample_videos.extend(videos[:max_videos])

                if len(sample_videos) >= max_videos:
                    break

            if len(sample_videos) >= max_videos:
                break

        return [str(v) for v in sample_videos[:max_videos]]


def auto_optimize_workers(input_dir: str = None, quick_test: bool = True) -> int:
    """
    Convenience function to automatically determine optimal workers
    """
    optimizer = WorkerOptimizer()

    if quick_test and input_dir:
        # Find sample videos from input directory
        sample_videos = []
        for ext in ['.MP4', '.mp4']:
            videos = list(Path(input_dir).rglob(f'*{ext}'))
            sample_videos.extend(videos[:3])  # Just 3 videos for quick test

        if sample_videos:
            result = optimizer.benchmark_workers([str(v) for v in sample_videos],
                                               worker_counts=[1, 2, 4, 6])
            print(result['recommendation'])
            return result['optimal_workers']

    # Fallback to theoretical optimization
    theoretical = optimizer.get_theoretical_optimum()
    print(f"🧠 Theoretical optimization:")
    print(f"   Conservative: {theoretical['recommendations']['conservative']} workers")
    print(f"   Balanced: {theoretical['recommendations']['balanced']} workers")
    print(f"   Aggressive: {theoretical['recommendations']['aggressive']} workers")
    print(f"💡 Recommended: {theoretical['recommendations']['balanced']} workers")

    return theoretical['recommendations']['balanced']


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Optimize worker count for BRUV processing')
    parser.add_argument('--input-dir', help='Directory containing sample videos')
    parser.add_argument('--quick', action='store_true', help='Quick test with few videos')
    parser.add_argument('--full-benchmark', action='store_true', help='Full benchmark test')

    args = parser.parse_args()

    if args.full_benchmark:
        optimizer = WorkerOptimizer()
        result = optimizer.quick_test()
        print(result['recommendation'])
    else:
        optimal_workers = auto_optimize_workers(args.input_dir, args.quick)
        print(f"\n🎯 Use --workers {optimal_workers} for optimal performance")