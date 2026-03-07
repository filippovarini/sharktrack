"""
Training Frame Extractor for SharkTrack

Extracts multiple frames from validated tracks for classifier training.
Uses the full video to get better training data than single thumbnails.
"""

import os
import cv2
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import random


@dataclass
class TrackInfo:
    """Information about a validated track"""
    track_id: int
    video_path: str
    video_name: str
    species: str
    start_time_ms: int
    end_time_ms: int
    bbox: Optional[Tuple[int, int, int, int]] = None  # xmin, ymin, xmax, ymax
    confidence: float = 0.0


class TrainingFrameExtractor:
    """Extract training frames from videos based on validated tracks"""

    def __init__(self,
                 frames_per_track: int = 20,
                 frame_size: Tuple[int, int] = (224, 224),
                 crop_padding: float = 0.2,
                 min_frame_interval_ms: int = 200):
        """
        Args:
            frames_per_track: Number of frames to extract per track
            frame_size: Output frame size (width, height)
            crop_padding: Extra padding around bbox (fraction of bbox size)
            min_frame_interval_ms: Minimum time between extracted frames
        """
        self.frames_per_track = frames_per_track
        self.frame_size = frame_size
        self.crop_padding = crop_padding
        self.min_frame_interval_ms = min_frame_interval_ms

    def extract_training_data(self,
                              validation_csv: str,
                              output_csv: str,
                              video_base_dir: str,
                              output_dir: str,
                              progress_callback: Callable = None) -> Dict:
        """
        Extract training frames from videos based on validation results.

        Args:
            validation_csv: Path to validation CSV (with species labels)
            output_csv: Path to output.csv (with track bboxes and times)
            video_base_dir: Base directory for videos
            output_dir: Where to save extracted frames (organized by class)
            progress_callback: Function(current, total, message) for progress

        Returns:
            Dict with extraction statistics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load validation data
        val_df = pd.read_csv(validation_csv)

        # Load detection output for timing/bbox info
        det_df = pd.read_csv(output_csv)

        # Get validated tracks with species labels
        tracks = self._get_validated_tracks(val_df, det_df, video_base_dir)

        if not tracks:
            return {"error": "No validated tracks found", "extracted": 0}

        # Group tracks by video for efficient processing
        tracks_by_video = {}
        for track in tracks:
            if track.video_path not in tracks_by_video:
                tracks_by_video[track.video_path] = []
            tracks_by_video[track.video_path].append(track)

        total_tracks = len(tracks)
        processed = 0
        stats = {
            "total_tracks": total_tracks,
            "total_frames": 0,
            "species": {},
            "videos_processed": 0,
            "errors": []
        }

        # Process each video
        for video_path, video_tracks in tracks_by_video.items():
            if progress_callback:
                progress_callback(
                    processed, total_tracks,
                    f"Processing {Path(video_path).name} ({len(video_tracks)} tracks)"
                )

            try:
                frames_extracted = self._extract_from_video(
                    video_path, video_tracks, output_path
                )
                stats["total_frames"] += frames_extracted
                stats["videos_processed"] += 1

                # Update species stats
                for track in video_tracks:
                    if track.species not in stats["species"]:
                        stats["species"][track.species] = {"tracks": 0, "frames": 0}
                    stats["species"][track.species]["tracks"] += 1
                    stats["species"][track.species]["frames"] += frames_extracted // len(video_tracks)

            except Exception as e:
                stats["errors"].append(f"{video_path}: {str(e)}")

            processed += len(video_tracks)

        if progress_callback:
            progress_callback(total_tracks, total_tracks, "Extraction complete")

        return stats

    def _get_validated_tracks(self,
                              val_df: pd.DataFrame,
                              det_df: pd.DataFrame,
                              video_base_dir: str) -> List[TrackInfo]:
        """Parse validation and detection data into TrackInfo objects"""
        tracks = []

        # Get species from validation CSV
        # Expected columns: thumbnail, true_detection, species (or similar)
        species_col = None
        for col in ['species', 'label', 'class', 'validated_species']:
            if col in val_df.columns:
                species_col = col
                break

        if not species_col:
            print("Warning: No species column found in validation CSV")
            return tracks

        # Get track_id from thumbnail filename or track_id column
        for _, row in val_df.iterrows():
            # Skip if not a true detection
            true_det = str(row.get('true_detection', '')).lower()
            if true_det not in ['true', 'yes', '1', 't']:
                continue

            species = row.get(species_col, '')
            if not species or pd.isna(species):
                continue

            # Clean species name for folder
            species = str(species).strip().replace(' ', '_').replace('.', '')

            # Get track_id from thumbnail name or column
            track_id = None
            if 'track_id' in row:
                track_id = int(row['track_id'])
            elif 'thumbnail' in row:
                thumb = str(row['thumbnail'])
                # Try to extract track_id from thumbnail filename
                # Format might be: "123-species.jpg" or "track_123.jpg"
                try:
                    track_id = int(thumb.split('-')[0].split('_')[-1].split('.')[0])
                except:
                    pass

            if track_id is None:
                continue

            # Find this track in detection output
            track_dets = det_df[det_df['track_id'] == track_id]
            if track_dets.empty:
                continue

            # Get timing and bbox info
            video_name = track_dets.iloc[0].get('video_name', '')
            video_path_rel = track_dets.iloc[0].get('video_path', '')

            # Build full video path
            if video_path_rel:
                video_path = str(Path(video_base_dir) / video_path_rel)
            else:
                video_path = str(Path(video_base_dir) / video_name)

            if not os.path.exists(video_path):
                # Try adding extension
                for ext in ['.MP4', '.mp4', '.MOV', '.mov', '.AVI', '.avi']:
                    test_path = video_path + ext if not video_path.endswith(ext) else video_path
                    if os.path.exists(test_path):
                        video_path = test_path
                        break

            # Get time range
            times = track_dets['time'].tolist()
            start_time = self._time_to_ms(min(times))
            end_time = self._time_to_ms(max(times))

            # Get representative bbox (use middle detection)
            mid_idx = len(track_dets) // 2
            mid_det = track_dets.iloc[mid_idx]
            bbox = (
                int(mid_det.get('xmin', 0)),
                int(mid_det.get('ymin', 0)),
                int(mid_det.get('xmax', 0)),
                int(mid_det.get('ymax', 0))
            )

            tracks.append(TrackInfo(
                track_id=track_id,
                video_path=video_path,
                video_name=video_name,
                species=species,
                start_time_ms=start_time,
                end_time_ms=end_time,
                bbox=bbox,
                confidence=float(mid_det.get('conf', 0))
            ))

        return tracks

    def _time_to_ms(self, time_str) -> int:
        """Convert time string (HH:MM:SS or MM:SS) to milliseconds"""
        if isinstance(time_str, (int, float)):
            return int(time_str * 1000)

        try:
            parts = str(time_str).split(':')
            if len(parts) == 3:
                h, m, s = parts
                return int((int(h) * 3600 + int(m) * 60 + float(s)) * 1000)
            elif len(parts) == 2:
                m, s = parts
                return int((int(m) * 60 + float(s)) * 1000)
            else:
                return int(float(time_str) * 1000)
        except:
            return 0

    def _extract_from_video(self,
                           video_path: str,
                           tracks: List[TrackInfo],
                           output_dir: Path) -> int:
        """Extract frames for all tracks from a single video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_extracted = 0

        try:
            for track in tracks:
                # Create output directory for this species
                species_dir = output_dir / track.species
                species_dir.mkdir(exist_ok=True)

                # Calculate frame times to extract
                duration_ms = track.end_time_ms - track.start_time_ms
                if duration_ms < 100:
                    duration_ms = 2000  # Default to 2 seconds if track too short

                # Distribute frames across track duration
                num_frames = min(self.frames_per_track, duration_ms // self.min_frame_interval_ms)
                num_frames = max(num_frames, 3)  # At least 3 frames

                frame_times = np.linspace(
                    track.start_time_ms,
                    track.end_time_ms,
                    num_frames
                ).astype(int)

                for i, time_ms in enumerate(frame_times):
                    cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
                    ret, frame = cap.read()

                    if not ret:
                        continue

                    # Crop around bbox with padding
                    if track.bbox and all(track.bbox):
                        cropped = self._crop_with_padding(frame, track.bbox)
                    else:
                        cropped = frame

                    # Resize to target size
                    resized = cv2.resize(cropped, self.frame_size)

                    # Save frame
                    frame_name = f"{track.video_name}_track{track.track_id:04d}_frame{i:03d}.jpg"
                    output_path = species_dir / frame_name
                    cv2.imwrite(str(output_path), resized)
                    total_extracted += 1

        finally:
            cap.release()

        return total_extracted

    def _crop_with_padding(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop frame around bbox with padding"""
        h, w = frame.shape[:2]
        xmin, ymin, xmax, ymax = bbox

        # Add padding
        box_w = xmax - xmin
        box_h = ymax - ymin
        pad_x = int(box_w * self.crop_padding)
        pad_y = int(box_h * self.crop_padding)

        # Expand bbox with padding, clamped to frame bounds
        x1 = max(0, xmin - pad_x)
        y1 = max(0, ymin - pad_y)
        x2 = min(w, xmax + pad_x)
        y2 = min(h, ymax + pad_y)

        return frame[y1:y2, x1:x2]

    def extract_from_single_track(self,
                                  video_path: str,
                                  track_id: int,
                                  species: str,
                                  start_time_ms: int,
                                  end_time_ms: int,
                                  output_dir: str,
                                  bbox: Tuple[int, int, int, int] = None) -> int:
        """
        Extract frames from a single track (for on-demand extraction).

        Returns number of frames extracted.
        """
        track = TrackInfo(
            track_id=track_id,
            video_path=video_path,
            video_name=Path(video_path).stem,
            species=species,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            bbox=bbox
        )

        return self._extract_from_video(video_path, [track], Path(output_dir))


def extract_training_frames(validation_csv: str,
                           output_csv: str,
                           video_dir: str,
                           output_dir: str,
                           frames_per_track: int = 20,
                           progress_callback: Callable = None) -> Dict:
    """
    Convenience function to extract training frames.

    Args:
        validation_csv: Path to validation results CSV
        output_csv: Path to detection output.csv
        video_dir: Base directory containing videos
        output_dir: Where to save training frames (organized by class)
        frames_per_track: How many frames to extract per track
        progress_callback: Optional progress callback function

    Returns:
        Extraction statistics dict
    """
    extractor = TrainingFrameExtractor(frames_per_track=frames_per_track)
    return extractor.extract_training_data(
        validation_csv=validation_csv,
        output_csv=output_csv,
        video_base_dir=video_dir,
        output_dir=output_dir,
        progress_callback=progress_callback
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract training frames from videos")
    parser.add_argument("--validation", "-v", required=True, help="Path to validation CSV")
    parser.add_argument("--output-csv", "-o", required=True, help="Path to output.csv")
    parser.add_argument("--video-dir", "-d", required=True, help="Video directory")
    parser.add_argument("--out-dir", "-O", required=True, help="Output directory for frames")
    parser.add_argument("--frames", "-f", type=int, default=20, help="Frames per track")

    args = parser.parse_args()

    def progress(current, total, msg):
        print(f"[{current}/{total}] {msg}")

    stats = extract_training_frames(
        validation_csv=args.validation,
        output_csv=args.output_csv,
        video_dir=args.video_dir,
        output_dir=args.out_dir,
        frames_per_track=args.frames,
        progress_callback=progress
    )

    print("\nExtraction complete:")
    print(f"  Total tracks: {stats['total_tracks']}")
    print(f"  Total frames: {stats['total_frames']}")
    print(f"  Videos processed: {stats['videos_processed']}")
    if stats.get('errors'):
        print(f"  Errors: {len(stats['errors'])}")
        for err in stats['errors'][:5]:
            print(f"    - {err}")
