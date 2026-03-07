#!/usr/bin/env python3
"""
Resumable Processing System for BRUV Video Analysis
Enables stopping and resuming large batch jobs without losing progress
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

@dataclass
class TaskStatus:
    """Individual task processing status"""
    task_id: str
    input_path: str
    output_path: str
    status: str  # pending, processing, completed, failed, skipped
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    worker_id: Optional[str] = None
    file_size: Optional[int] = None
    checksum: Optional[str] = None

@dataclass
class BatchProgress:
    """Overall batch processing progress"""
    batch_id: str
    input_directory: str
    output_directory: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    skipped_tasks: int
    start_time: float
    last_update_time: float
    estimated_completion_time: Optional[float] = None
    current_workers: int = 8
    delete_policy: str = "ask-each"

class ResumableProcessor:
    """
    Manages resumable batch processing with progress persistence
    """

    def __init__(self,
                 batch_name: str,
                 input_dir: str,
                 output_dir: str,
                 progress_dir: str = "./progress",
                 auto_save_interval: int = 30):

        self.batch_name = batch_name
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.progress_dir = Path(progress_dir)
        self.auto_save_interval = auto_save_interval

        # Create progress directory
        self.progress_dir.mkdir(parents=True, exist_ok=True)

        # Progress files
        self.batch_id = self._generate_batch_id()
        self.progress_file = self.progress_dir / f"{self.batch_id}_progress.json"
        self.tasks_file = self.progress_dir / f"{self.batch_id}_tasks.json"

        # State
        self.tasks: Dict[str, TaskStatus] = {}
        self.batch_progress: Optional[BatchProgress] = None
        self.last_save_time = time.time()

    def _generate_batch_id(self) -> str:
        """Generate unique batch ID based on input parameters"""
        content = f"{self.batch_name}_{self.input_dir}_{self.output_dir}_{datetime.now().date()}"
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum for file integrity verification"""
        if not file_path.exists():
            return ""

        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def discover_tasks(self, video_extensions: List[str] = None) -> List[TaskStatus]:
        """
        Discover all video files and create task list
        """
        if video_extensions is None:
            video_extensions = ['.MP4', '.mp4', '.MOV', '.mov', '.AVI', '.avi']

        tasks = []

        print(f"🔍 Discovering video files in {self.input_dir}")

        for ext in video_extensions:
            video_files = list(self.input_dir.rglob(f'*{ext}'))

            for video_file in video_files:
                # Generate task ID from relative path
                rel_path = video_file.relative_to(self.input_dir)
                task_id = str(rel_path).replace(os.sep, '_').replace('.', '_')

                # Determine output path
                output_path = self.output_dir / "converted" / video_file.name

                task = TaskStatus(
                    task_id=task_id,
                    input_path=str(video_file),
                    output_path=str(output_path),
                    status="pending",
                    file_size=video_file.stat().st_size if video_file.exists() else None,
                    checksum=self._calculate_file_checksum(video_file)
                )

                tasks.append(task)

        print(f"📋 Discovered {len(tasks)} video files")
        return tasks

    def load_existing_progress(self) -> bool:
        """
        Load existing progress from previous run
        """
        if not self.progress_file.exists() or not self.tasks_file.exists():
            return False

        try:
            # Load batch progress
            with open(self.progress_file, 'r') as f:
                progress_data = json.load(f)
                self.batch_progress = BatchProgress(**progress_data)

            # Load task statuses
            with open(self.tasks_file, 'r') as f:
                tasks_data = json.load(f)
                self.tasks = {
                    task_id: TaskStatus(**task_data)
                    for task_id, task_data in tasks_data.items()
                }

            print(f"📂 Loaded existing progress:")
            print(f"   • Batch ID: {self.batch_progress.batch_id}")
            print(f"   • Total tasks: {self.batch_progress.total_tasks}")
            print(f"   • Completed: {self.batch_progress.completed_tasks}")
            print(f"   • Failed: {self.batch_progress.failed_tasks}")
            print(f"   • Remaining: {self.get_remaining_task_count()}")

            return True

        except Exception as e:
            print(f"⚠️  Failed to load existing progress: {e}")
            return False

    def initialize_new_batch(self, delete_policy: str = "ask-each") -> List[TaskStatus]:
        """
        Initialize a new batch processing session
        """
        tasks = self.discover_tasks()

        # Create task status dictionary
        self.tasks = {task.task_id: task for task in tasks}

        # Create batch progress
        self.batch_progress = BatchProgress(
            batch_id=self.batch_id,
            input_directory=str(self.input_dir),
            output_directory=str(self.output_dir),
            total_tasks=len(tasks),
            completed_tasks=0,
            failed_tasks=0,
            skipped_tasks=0,
            start_time=time.time(),
            last_update_time=time.time(),
            delete_policy=delete_policy
        )

        # Save initial progress
        self.save_progress()

        print(f"🚀 Initialized new batch:")
        print(f"   • Batch ID: {self.batch_id}")
        print(f"   • Total tasks: {len(tasks)}")
        print(f"   • Progress saved to: {self.progress_file}")

        return tasks

    def get_pending_tasks(self) -> List[TaskStatus]:
        """Get list of tasks that still need processing"""
        return [task for task in self.tasks.values() if task.status == "pending"]

    def get_failed_tasks(self) -> List[TaskStatus]:
        """Get list of failed tasks that could be retried"""
        return [task for task in self.tasks.values() if task.status == "failed"]

    def get_remaining_task_count(self) -> int:
        """Get count of tasks remaining (pending + failed)"""
        return len([t for t in self.tasks.values() if t.status in ["pending", "failed"]])

    def mark_task_started(self, task_id: str, worker_id: str = None):
        """Mark task as started"""
        if task_id in self.tasks:
            self.tasks[task_id].status = "processing"
            self.tasks[task_id].start_time = time.time()
            self.tasks[task_id].worker_id = worker_id

    def mark_task_completed(self, task_id: str, output_path: str = None):
        """Mark task as completed"""
        if task_id in self.tasks:
            self.tasks[task_id].status = "completed"
            self.tasks[task_id].end_time = time.time()
            if output_path:
                self.tasks[task_id].output_path = output_path

            self.batch_progress.completed_tasks += 1
            self._auto_save()

    def mark_task_failed(self, task_id: str, error_message: str = None):
        """Mark task as failed"""
        if task_id in self.tasks:
            self.tasks[task_id].status = "failed"
            self.tasks[task_id].end_time = time.time()
            self.tasks[task_id].error_message = error_message
            self.tasks[task_id].retry_count += 1

            self.batch_progress.failed_tasks += 1
            self._auto_save()

    def mark_task_skipped(self, task_id: str, reason: str = None):
        """Mark task as skipped (e.g., output already exists)"""
        if task_id in self.tasks:
            self.tasks[task_id].status = "skipped"
            self.tasks[task_id].end_time = time.time()
            self.tasks[task_id].error_message = reason

            self.batch_progress.skipped_tasks += 1
            self._auto_save()

    def verify_completed_tasks(self) -> List[str]:
        """
        Verify that completed tasks actually have valid output files
        Returns list of task IDs that need to be re-processed
        """
        invalid_tasks = []

        for task_id, task in self.tasks.items():
            if task.status == "completed":
                output_path = Path(task.output_path)

                # Check if output file exists and has reasonable size
                if not output_path.exists():
                    invalid_tasks.append(task_id)
                    task.status = "pending"
                    self.batch_progress.completed_tasks -= 1
                elif output_path.stat().st_size < 1024:  # Less than 1KB
                    invalid_tasks.append(task_id)
                    task.status = "pending"
                    self.batch_progress.completed_tasks -= 1

        if invalid_tasks:
            print(f"⚠️  Found {len(invalid_tasks)} completed tasks with invalid outputs")
            self.save_progress()

        return invalid_tasks

    def _auto_save(self):
        """Auto-save progress periodically"""
        current_time = time.time()
        if current_time - self.last_save_time > self.auto_save_interval:
            self.save_progress()
            self.last_save_time = current_time

    def save_progress(self):
        """Save current progress to disk"""
        try:
            # Update batch progress
            self.batch_progress.last_update_time = time.time()

            # Calculate estimated completion time
            if self.batch_progress.completed_tasks > 0:
                elapsed_time = time.time() - self.batch_progress.start_time
                avg_time_per_task = elapsed_time / self.batch_progress.completed_tasks
                remaining_tasks = self.get_remaining_task_count()
                estimated_remaining = remaining_tasks * avg_time_per_task
                self.batch_progress.estimated_completion_time = time.time() + estimated_remaining

            # Save batch progress
            with open(self.progress_file, 'w') as f:
                json.dump(asdict(self.batch_progress), f, indent=2)

            # Save task statuses
            tasks_dict = {task_id: asdict(task) for task_id, task in self.tasks.items()}
            with open(self.tasks_file, 'w') as f:
                json.dump(tasks_dict, f, indent=2)

        except Exception as e:
            print(f"⚠️  Failed to save progress: {e}")

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current progress summary"""
        if not self.batch_progress:
            return {}

        remaining = self.get_remaining_task_count()
        elapsed_time = time.time() - self.batch_progress.start_time

        # Calculate completion percentage
        completion_pct = (self.batch_progress.completed_tasks / self.batch_progress.total_tasks) * 100

        # Calculate ETA
        eta_str = "Unknown"
        if self.batch_progress.estimated_completion_time:
            eta_seconds = self.batch_progress.estimated_completion_time - time.time()
            if eta_seconds > 0:
                eta_hours = eta_seconds / 3600
                eta_str = f"{eta_hours:.1f} hours"

        # Calculate throughput
        throughput = self.batch_progress.completed_tasks / elapsed_time if elapsed_time > 0 else 0

        return {
            'batch_id': self.batch_progress.batch_id,
            'total_tasks': self.batch_progress.total_tasks,
            'completed': self.batch_progress.completed_tasks,
            'failed': self.batch_progress.failed_tasks,
            'skipped': self.batch_progress.skipped_tasks,
            'remaining': remaining,
            'completion_percentage': completion_pct,
            'elapsed_time_hours': elapsed_time / 3600,
            'estimated_time_remaining': eta_str,
            'throughput_per_hour': throughput * 3600,
            'current_workers': self.batch_progress.current_workers
        }

    def print_progress_summary(self):
        """Print formatted progress summary"""
        summary = self.get_progress_summary()

        print(f"\n📊 Batch Processing Progress")
        print(f"   • Batch ID: {summary['batch_id']}")
        print(f"   • Progress: {summary['completed']}/{summary['total_tasks']} ({summary['completion_percentage']:.1f}%)")
        print(f"   • Completed: {summary['completed']}")
        print(f"   • Failed: {summary['failed']}")
        print(f"   • Skipped: {summary['skipped']}")
        print(f"   • Remaining: {summary['remaining']}")
        print(f"   • Elapsed: {summary['elapsed_time_hours']:.1f} hours")
        print(f"   • ETA: {summary['estimated_time_remaining']}")
        print(f"   • Throughput: {summary['throughput_per_hour']:.1f} videos/hour")

    def cleanup_progress_files(self):
        """Remove progress files after successful completion"""
        try:
            if self.progress_file.exists():
                self.progress_file.unlink()
            if self.tasks_file.exists():
                self.tasks_file.unlink()
            print(f"🧹 Cleaned up progress files")
        except Exception as e:
            print(f"⚠️  Failed to cleanup progress files: {e}")


def resume_or_start_batch(batch_name: str,
                         input_dir: str,
                         output_dir: str,
                         delete_policy: str = "ask-each") -> ResumableProcessor:
    """
    Convenience function to either resume existing batch or start new one
    """

    processor = ResumableProcessor(batch_name, input_dir, output_dir)

    # Try to load existing progress
    if processor.load_existing_progress():
        print(f"🔄 Resuming existing batch")

        # Verify completed tasks are still valid
        invalid_tasks = processor.verify_completed_tasks()
        if invalid_tasks:
            print(f"🔧 Reset {len(invalid_tasks)} invalid completed tasks")

        processor.print_progress_summary()

        # Ask user if they want to continue
        remaining = processor.get_remaining_task_count()
        if remaining > 0:
            response = input(f"\nContinue processing {remaining} remaining tasks? (y/N): ")
            if response.lower() != 'y':
                print("❌ Resumption cancelled")
                return None
        else:
            print("✅ All tasks already completed!")
            return processor

    else:
        print(f"🚀 Starting new batch")
        processor.initialize_new_batch(delete_policy)

    return processor


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) < 3:
        print("Usage: python resumable_processor.py <batch_name> <input_dir> [output_dir]")
        sys.exit(1)
    processor = resume_or_start_batch(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3] if len(sys.argv) > 3 else "./batch_output"
    )

    if processor:
        pending_tasks = processor.get_pending_tasks()
        print(f"Ready to process {len(pending_tasks)} tasks")
        processor.print_progress_summary()