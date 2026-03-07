import threading
import queue
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from collections import defaultdict
import torch
from .species_classifier import SpeciesClassifier
from .time_processor import string_to_ms
from .image_processor import extract_frame_at_time


@dataclass
class ClassificationTask:
    """Single detection to be classified"""
    track_id: int
    track_metadata: str
    video_path: str
    time: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    confidence: float
    row_index: int  # Original dataframe index for updating results


@dataclass
class BatchResult:
    """Result from batch classification"""
    row_index: int
    species: Optional[str]
    classification_confidence: float


class FrameCache:
    """LRU cache for video frames to avoid redundant extractions"""
    def __init__(self, max_size: int = 50):
        self.cache: Dict[str, np.ndarray] = {}
        self.access_order: List[str] = []
        self.max_size = max_size

    def get_frame(self, video_path: str, time: str) -> np.ndarray:
        cache_key = f"{video_path}:{time}"

        if cache_key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(cache_key)
            self.access_order.append(cache_key)
            return self.cache[cache_key]

        # Extract frame
        frame = extract_frame_at_time(video_path, string_to_ms(time))

        # Add to cache
        self.cache[cache_key] = frame
        self.access_order.append(cache_key)

        # Evict oldest if over limit
        while len(self.cache) > self.max_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]

        return frame


class ParallelSpeciesClassifier:
    """Parallel species classifier with batch processing"""

    def __init__(self, species_classifier: SpeciesClassifier, batch_size: int = 16, queue_size: int = 1000):
        self.species_classifier = species_classifier
        self.batch_size = batch_size
        self.task_queue = queue.Queue(maxsize=queue_size)
        self.result_queue = queue.Queue()
        self.frame_cache = FrameCache()
        self.shutdown_event = threading.Event()
        self.worker_thread = None
        self.video_path_prefix = None

    def start(self, video_path_prefix: str):
        """Start the background classification worker"""
        self.video_path_prefix = video_path_prefix
        self.shutdown_event.clear()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print(f"Started parallel species classifier with batch size {self.batch_size}")

    def stop(self):
        """Stop the background worker and return all pending results"""
        self.shutdown_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=10)

        # Collect any remaining results
        results = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def submit_task(self, task: ClassificationTask):
        """Submit a detection for classification"""
        try:
            self.task_queue.put(task, timeout=1)
        except queue.Full:
            print("Warning: Classification queue full, dropping task")

    def get_results(self) -> List[BatchResult]:
        """Get all available classification results"""
        results = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def _worker_loop(self):
        """Background worker that processes classification tasks in batches"""
        print("Species classifier worker started")

        while not self.shutdown_event.is_set():
            # Collect batch of tasks
            batch_tasks = []
            deadline = time.time() + 0.1  # 100ms batch collection window

            while len(batch_tasks) < self.batch_size and time.time() < deadline:
                try:
                    task = self.task_queue.get(timeout=0.05)
                    batch_tasks.append(task)
                except queue.Empty:
                    continue

            if not batch_tasks:
                continue

            try:
                # Process batch
                results = self._process_batch(batch_tasks)

                # Submit results
                for result in results:
                    self.result_queue.put(result)

            except Exception as e:
                print(f"Error in classification batch: {e}")
                # Submit failed results
                for task in batch_tasks:
                    self.result_queue.put(BatchResult(
                        row_index=task.row_index,
                        species=None,
                        classification_confidence=0.0
                    ))

        print("Species classifier worker stopped")

    def _process_batch(self, tasks: List[ClassificationTask]) -> List[BatchResult]:
        """Process a batch of classification tasks"""
        if not self.species_classifier:
            return [BatchResult(task.row_index, None, 0.0) for task in tasks]

        # Group tasks by frame to minimize video reads
        frame_groups = defaultdict(list)
        for task in tasks:
            frame_key = f"{task.video_path}:{task.time}"
            frame_groups[frame_key].append(task)

        results = []

        for frame_key, frame_tasks in frame_groups.items():
            # Extract frame once per group
            video_path = str(Path(self.video_path_prefix) / frame_tasks[0].video_path)
            frame = self.frame_cache.get_frame(video_path, frame_tasks[0].time)

            # Prepare batch of crops
            crops = []
            task_indices = []

            for task in frame_tasks:
                # Extract crop
                crop = frame[task.ymin:task.ymax, task.xmin:task.xmax]
                if crop.size > 0:  # Valid crop
                    crops.append(crop)
                    task_indices.append(task)

            if not crops:
                # No valid crops in this frame
                for task in frame_tasks:
                    results.append(BatchResult(task.row_index, None, 0.0))
                continue

            # Batch inference
            try:
                batch_results = self._classify_batch(crops)

                # Map results back to tasks
                for task, (confidence, species) in zip(task_indices, batch_results):
                    results.append(BatchResult(
                        row_index=task.row_index,
                        species=species,
                        classification_confidence=confidence
                    ))

                # Handle any tasks that didn't get crops
                processed_indices = {task.row_index for task in task_indices}
                for task in frame_tasks:
                    if task.row_index not in processed_indices:
                        results.append(BatchResult(task.row_index, None, 0.0))

            except Exception as e:
                print(f"Batch classification failed: {e}")
                for task in frame_tasks:
                    results.append(BatchResult(task.row_index, None, 0.0))

        return results

    def _classify_batch(self, crops: List[np.ndarray]) -> List[Tuple[float, Optional[str]]]:
        """Run batch inference on a list of crops"""
        if not crops:
            return []

        # Preprocess all crops
        processed_crops = []
        for crop in crops:
            try:
                processed_crop = self.species_classifier.transform(crop)
                processed_crops.append(processed_crop)
            except Exception:
                # If preprocessing fails, add a dummy tensor
                processed_crops.append(torch.zeros(3, 200, 400))

        # Stack into batch tensor
        batch_tensor = torch.stack(processed_crops).to(self.species_classifier.device)

        # Batch inference
        with torch.no_grad():
            outputs = self.species_classifier.model(batch_tensor)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            confidences, preds = torch.max(outputs, 1)

        # Convert results
        results = []
        for i in range(len(crops)):
            confidence = confidences[i].item()
            predicted_class = self.species_classifier.classes[preds[i].item()]

            if confidence < self.species_classifier.confidence_threshold:
                predicted_class = None

            results.append((confidence, predicted_class))

        return results