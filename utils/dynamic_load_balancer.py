#!/usr/bin/env python3
"""
Dynamic Load Balancer for BRUV Video Processing
Automatically adjusts worker count and queue management based on real-time system metrics
"""

import time
import psutil
import threading
import queue
from typing import Dict, List, Callable, Any, Optional
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
import logging

@dataclass
class SystemMetrics:
    """Real-time system performance metrics"""
    cpu_usage_per_core: List[float]
    cpu_usage_avg: float
    memory_usage: float
    disk_io_read: float
    disk_io_write: float
    gpu_usage: Optional[float] = None
    active_workers: int = 0
    queued_tasks: int = 0

class DynamicLoadBalancer:
    """
    Intelligent load balancer that adjusts worker count and task distribution
    based on real-time system performance
    """

    def __init__(self,
                 initial_workers: int = 4,
                 min_workers: int = 1,
                 max_workers: int = 16,
                 target_cpu_usage: float = 85.0,
                 adjustment_interval: float = 5.0):

        self.initial_workers = initial_workers
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_cpu_usage = target_cpu_usage
        self.adjustment_interval = adjustment_interval

        # Task management
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.active_workers = initial_workers

        # Performance tracking
        self.metrics_history = []
        self.adjustment_history = []

        # Control flags
        self.running = False
        self.monitor_thread = None
        self.executor = None

        # Performance counters
        self.tasks_completed = 0
        self.total_processing_time = 0.0
        self.last_adjustment_time = time.time()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_system_metrics(self) -> SystemMetrics:
        """Collect real-time system performance metrics"""

        # CPU metrics
        cpu_per_core = psutil.cpu_percent(percpu=True, interval=0.1)
        cpu_avg = psutil.cpu_percent(interval=0.1)

        # Memory metrics
        memory = psutil.virtual_memory()

        # Disk I/O metrics
        disk_io = psutil.disk_io_counters()
        disk_read = disk_io.read_bytes / (1024**2)  # MB
        disk_write = disk_io.write_bytes / (1024**2)  # MB

        # GPU metrics (if available)
        gpu_usage = self._get_gpu_usage()

        return SystemMetrics(
            cpu_usage_per_core=cpu_per_core,
            cpu_usage_avg=cpu_avg,
            memory_usage=memory.percent,
            disk_io_read=disk_read,
            disk_io_write=disk_write,
            gpu_usage=gpu_usage,
            active_workers=self.active_workers,
            queued_tasks=self.task_queue.qsize()
        )

    def _get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage if NVIDIA GPU available"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu',
                                   '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        return None

    def calculate_optimal_workers(self, metrics: SystemMetrics) -> int:
        """
        Calculate optimal worker count based on current system metrics
        """

        # Base calculation on CPU usage
        if metrics.cpu_usage_avg < self.target_cpu_usage * 0.7:
            # CPU underutilized - can add workers
            suggested_workers = min(self.active_workers + 1, self.max_workers)
        elif metrics.cpu_usage_avg > self.target_cpu_usage * 1.1:
            # CPU overutilized - reduce workers
            suggested_workers = max(self.active_workers - 1, self.min_workers)
        else:
            # CPU usage in target range
            suggested_workers = self.active_workers

        # Memory constraint check
        memory_workers = int((100 - metrics.memory_usage) / 8)  # ~8% memory per worker
        suggested_workers = min(suggested_workers, memory_workers)

        # I/O bottleneck detection
        if self._detect_io_bottleneck(metrics):
            # Reduce workers if I/O bound
            suggested_workers = max(suggested_workers - 1, self.min_workers)

        # Queue depth consideration
        if metrics.queued_tasks > suggested_workers * 3:
            # Large queue - might need more workers
            suggested_workers = min(suggested_workers + 1, self.max_workers)
        elif metrics.queued_tasks < suggested_workers and metrics.cpu_usage_avg < 60:
            # Small queue and low CPU - reduce workers
            suggested_workers = max(suggested_workers - 1, self.min_workers)

        return max(self.min_workers, min(suggested_workers, self.max_workers))

    def _detect_io_bottleneck(self, metrics: SystemMetrics) -> bool:
        """Detect if system is I/O bound rather than CPU bound"""

        # Simple heuristic: low CPU but high disk activity
        if len(self.metrics_history) < 3:
            return False

        recent_metrics = self.metrics_history[-3:]
        avg_cpu = sum(m.cpu_usage_avg for m in recent_metrics) / len(recent_metrics)

        # If CPU is low but we have consistent load, likely I/O bound
        return avg_cpu < 50 and metrics.queued_tasks > 0

    def adjust_workers(self, new_worker_count: int) -> bool:
        """
        Dynamically adjust the number of active workers
        """

        if new_worker_count == self.active_workers:
            return False

        old_count = self.active_workers
        self.active_workers = new_worker_count

        # Restart executor with new worker count
        if self.executor:
            # Don't shutdown immediately - let current tasks finish
            self.executor.shutdown(wait=False)

        self.executor = ThreadPoolExecutor(max_workers=self.active_workers)

        self.adjustment_history.append({
            'timestamp': time.time(),
            'old_workers': old_count,
            'new_workers': new_worker_count,
            'reason': f"CPU: {self.metrics_history[-1].cpu_usage_avg:.1f}%" if self.metrics_history else "Initial"
        })

        self.logger.info(f"🔧 Adjusted workers: {old_count} → {new_worker_count}")
        return True

    def monitor_and_adjust(self):
        """
        Background thread that continuously monitors system and adjusts workers
        """

        while self.running:
            try:
                # Collect metrics
                metrics = self.get_system_metrics()
                self.metrics_history.append(metrics)

                # Keep only recent history (last 10 minutes)
                if len(self.metrics_history) > 120:  # 120 * 5s = 10 minutes
                    self.metrics_history = self.metrics_history[-120:]

                # Check if adjustment is needed
                current_time = time.time()
                if current_time - self.last_adjustment_time > self.adjustment_interval:

                    optimal_workers = self.calculate_optimal_workers(metrics)

                    if optimal_workers != self.active_workers:
                        self.adjust_workers(optimal_workers)
                        self.last_adjustment_time = current_time

                # Log current status
                if len(self.metrics_history) % 12 == 0:  # Every minute
                    self.log_status(metrics)

                time.sleep(self.adjustment_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring thread: {e}")
                time.sleep(self.adjustment_interval)

    def log_status(self, metrics: SystemMetrics):
        """Log current system status"""

        throughput = self.tasks_completed / (time.time() - self.start_time) if hasattr(self, 'start_time') else 0

        self.logger.info(
            f"📊 Workers: {self.active_workers} | "
            f"CPU: {metrics.cpu_usage_avg:.1f}% | "
            f"RAM: {metrics.memory_usage:.1f}% | "
            f"Queue: {metrics.queued_tasks} | "
            f"Throughput: {throughput:.2f} tasks/sec"
        )

    def start_processing(self, task_function: Callable, task_list: List[Any]) -> List[Any]:
        """
        Start dynamic load-balanced processing of task list
        """

        self.running = True
        self.start_time = time.time()

        # Add all tasks to queue
        for task in task_list:
            self.task_queue.put(task)

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor_and_adjust, daemon=True)
        self.monitor_thread.start()

        # Start initial executor
        self.executor = ThreadPoolExecutor(max_workers=self.active_workers)

        # Process tasks with dynamic worker adjustment
        results = []
        active_futures = set()

        try:
            while not self.task_queue.empty() or active_futures:

                # Submit new tasks up to worker limit
                while len(active_futures) < self.active_workers and not self.task_queue.empty():
                    try:
                        task = self.task_queue.get_nowait()
                        future = self.executor.submit(task_function, task)
                        active_futures.add(future)
                    except queue.Empty:
                        break

                # Check completed futures
                completed_futures = set()
                for future in active_futures:
                    if future.done():
                        try:
                            result = future.result()
                            results.append(result)
                            self.tasks_completed += 1
                        except Exception as e:
                            self.logger.error(f"Task failed: {e}")
                        completed_futures.add(future)

                # Remove completed futures
                active_futures -= completed_futures

                # Brief pause to prevent busy waiting
                time.sleep(0.1)

        finally:
            # Cleanup
            self.running = False
            if self.executor:
                self.executor.shutdown(wait=True)
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)

        return results

    def get_performance_report(self) -> Dict:
        """Generate performance report"""

        if not self.metrics_history:
            return {}

        total_time = time.time() - self.start_time if hasattr(self, 'start_time') else 0
        avg_cpu = sum(m.cpu_usage_avg for m in self.metrics_history) / len(self.metrics_history)
        avg_memory = sum(m.memory_usage for m in self.metrics_history) / len(self.metrics_history)

        return {
            'total_processing_time': total_time,
            'tasks_completed': self.tasks_completed,
            'average_throughput': self.tasks_completed / total_time if total_time > 0 else 0,
            'average_cpu_usage': avg_cpu,
            'average_memory_usage': avg_memory,
            'worker_adjustments': len(self.adjustment_history),
            'final_worker_count': self.active_workers,
            'adjustment_history': self.adjustment_history[-10:]  # Last 10 adjustments
        }


def process_with_dynamic_balancing(task_function: Callable,
                                  task_list: List[Any],
                                  initial_workers: int = 4,
                                  target_cpu_usage: float = 85.0) -> tuple:
    """
    Convenience function to process tasks with dynamic load balancing

    Returns: (results, performance_report)
    """

    balancer = DynamicLoadBalancer(
        initial_workers=initial_workers,
        target_cpu_usage=target_cpu_usage,
        min_workers=max(1, initial_workers // 2),
        max_workers=initial_workers * 2
    )

    results = balancer.start_processing(task_function, task_list)
    report = balancer.get_performance_report()

    return results, report


# Integration example for video processing
def process_videos_with_balancing(video_paths: List[str],
                                 processing_function: Callable) -> tuple:
    """
    Process videos with dynamic load balancing
    """

    def video_task_wrapper(video_path):
        start_time = time.time()
        result = processing_function(video_path)
        processing_time = time.time() - start_time
        return {
            'video_path': video_path,
            'result': result,
            'processing_time': processing_time
        }

    return process_with_dynamic_balancing(
        video_task_wrapper,
        video_paths,
        initial_workers=8,  # Start with current detection
        target_cpu_usage=80.0  # Target 80% CPU usage
    )


if __name__ == "__main__":
    # Example usage
    def dummy_task(item):
        import random
        time.sleep(random.uniform(0.5, 2.0))  # Simulate variable processing time
        return f"Processed {item}"

    # Test with dummy tasks
    test_tasks = list(range(50))
    results, report = process_with_dynamic_balancing(dummy_task, test_tasks)

    print("🎯 Dynamic Load Balancing Test Results:")
    print(f"Tasks completed: {report['tasks_completed']}")
    print(f"Average throughput: {report['average_throughput']:.2f} tasks/sec")
    print(f"Average CPU usage: {report['average_cpu_usage']:.1f}%")
    print(f"Worker adjustments: {report['worker_adjustments']}")