import pandas as pd
import sys
import cv2
import os
from pathlib import Path
import hashlib
import numpy as np
sys.path.append("utils")
from time_processor import string_to_ms
from config import configs
from image_processor import draw_bboxes, annotate_image, extract_frame_at_time
from path_resolver import compute_frames_output_path, remove_input_prefix_from_video_path
from species_classifier import SpeciesClassifier
import time as t


SHARKTRACK_COLUMNS = ["video_path", "video_name", "frame", "time", "xmin", "ymin", "xmax", "ymax", "w", "h", "confidence", "label", "track_metadata", "track_id"]

classes_mapping = ["elasmobranch"]

# Global deduplication cache
_detection_cache = {}

def compute_detection_hash(image, bbox):
    """
    Compute a hash for a detection to identify duplicates.
    Uses perceptual hash of frame + bbox coordinates.

    Args:
        image: numpy array of the frame
        bbox: tuple of (xmin, ymin, xmax, ymax)

    Returns:
        str: hash string for this detection
    """
    # Simple difference hash (dHash) for perceptual image hashing
    # Resize to 8x8 for speed
    small_image = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY) if len(small_image.shape) == 3 else small_image

    # Compute horizontal gradient
    diff = gray[:, 1:] > gray[:, :-1]

    # Convert boolean array to hash string
    image_hash = ''.join(['1' if v else '0' for row in diff for v in row])

    # Combine with bbox coordinates (rounded to avoid floating point differences)
    xmin, ymin, xmax, ymax = bbox
    bbox_str = f"{int(xmin)}_{int(ymin)}_{int(xmax)}_{int(ymax)}"

    # Create combined hash
    combined = f"{image_hash}_{bbox_str}"
    return hashlib.md5(combined.encode()).hexdigest()

def is_duplicate_detection(video_path, image, bbox, tolerance=2):
    """
    Check if this detection is a duplicate of a previously saved one.

    Args:
        video_path: path to video (used as cache key scope)
        image: numpy array of the frame
        bbox: tuple of (xmin, ymin, xmax, ymax)
        tolerance: pixel tolerance for bbox matching (default 2px)

    Returns:
        bool: True if duplicate, False if unique
    """
    detection_hash = compute_detection_hash(image, bbox)
    cache_key = f"{video_path}_{detection_hash}"

    if cache_key in _detection_cache:
        return True

    # Also check for near-duplicate bboxes (within tolerance pixels)
    xmin, ymin, xmax, ymax = bbox
    video_detections = [k for k in _detection_cache.keys() if k.startswith(video_path)]

    for existing_key in video_detections:
        # Extract bbox from existing key
        try:
            parts = existing_key.split('_')
            if len(parts) >= 5:
                existing_bbox = tuple(map(int, parts[-4:]))
                ex_xmin, ex_ymin, ex_xmax, ex_ymax = existing_bbox

                # Check if bboxes are within tolerance
                if (abs(xmin - ex_xmin) <= tolerance and
                    abs(ymin - ex_ymin) <= tolerance and
                    abs(xmax - ex_xmax) <= tolerance and
                    abs(ymax - ex_ymax) <= tolerance):
                    return True
        except (ValueError, IndexError):
            continue

    # Not a duplicate, add to cache
    _detection_cache[cache_key] = True
    return False

def clear_detection_cache():
    """Clear the deduplication cache (call when starting a new batch)"""
    global _detection_cache
    _detection_cache = {}
   
def extract_frame_results(frame_results, tracking):
    boxes = frame_results.boxes.xyxy.cpu().tolist()
    tracks = frame_results.boxes.id
    track_ids = tracks.int().cpu().tolist() if tracks is not None else []
    confidences = frame_results.boxes.conf.cpu().tolist()
    classes = frame_results.boxes.cls.cpu().tolist()

    return zip(boxes, confidences, classes, track_ids) if tracking else zip(boxes, confidences, classes)


# ============================================================================
# RAW DETECTION LOGGING
# ============================================================================
# These functions log ALL detections before tracking/filtering for QC purposes

RAW_DETECTION_COLUMNS = [
    "video_path", "video_name", "frame", "time",
    "xmin", "ymin", "xmax", "ymax", "w", "h",
    "confidence", "label",
    "has_track_id", "track_id"
]

def extract_raw_detections(video_path, input_path, frame_results, frame_id, time):
    """
    Extract ALL detections from a frame, regardless of tracking status.

    This captures detections that may be filtered out by the tracker
    (e.g., due to new_track_thresh) for later QC review.

    Args:
        video_path: Path to video file
        input_path: Base input path for relative path computation
        frame_results: YOLO results object
        frame_id: Frame number
        time: Timestamp string

    Returns:
        list: List of detection dicts
    """
    boxes = frame_results.boxes.xyxy.cpu().tolist()
    tracks = frame_results.boxes.id
    track_ids = tracks.int().cpu().tolist() if tracks is not None else [None] * len(boxes)
    confidences = frame_results.boxes.conf.cpu().tolist()
    classes = frame_results.boxes.cls.cpu().tolist()

    # Pad track_ids if shorter than boxes (happens when some detections don't get tracked)
    while len(track_ids) < len(boxes):
        track_ids.append(None)

    raw_rows = []
    relative_video_path = remove_input_prefix_from_video_path(video_path, input_path)

    for box, confidence, cls, track_id in zip(boxes, confidences, classes, track_ids):
        row = {
            "video_path": relative_video_path,
            "video_name": os.path.basename(video_path),
            "frame": frame_id,
            "time": time,
            "xmin": box[0],
            "ymin": box[1],
            "xmax": box[2],
            "ymax": box[3],
            "w": frame_results.orig_shape[1],
            "h": frame_results.orig_shape[0],
            "confidence": confidence,
            "label": classes_mapping[int(cls)],
            "has_track_id": track_id is not None,
            "track_id": track_id if track_id is not None else -1,
        }
        raw_rows.append(row)

    return raw_rows


def save_raw_detections(raw_detections, output_folder):
    """
    Save raw detections to a separate CSV for QC purposes.

    This file contains ALL detections, including those that:
    - Didn't meet new_track_thresh to start a track
    - Were in gaps between tracked frames
    - Were filtered out by postprocessing

    Args:
        raw_detections: List of detection dicts from extract_raw_detections
        output_folder: Output folder path
    """
    if not raw_detections:
        return

    df = pd.DataFrame(raw_detections)
    output_path = os.path.join(output_folder, "raw_detections.csv")

    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_csv(output_path, index=False)

    # Log summary
    n_total = len(df)
    n_tracked = df['has_track_id'].sum()
    n_untracked = n_total - n_tracked
    print(f"  📝 Raw detections logged: {len(raw_detections)} new ({n_untracked} untracked)")

def extract_sightings(video_path, input_path, frame_results, frame_id, time, **kwargs):
    is_tracking = kwargs.get("tracking", False)
    track_results = extract_frame_results(frame_results, is_tracking)

    frame_results_rows = []

    relative_video_path = remove_input_prefix_from_video_path(video_path, input_path)
    
    for box, confidence, cls, *rest in track_results:
          preprocess_track_id = next(iter(rest)) if is_tracking else kwargs["track_id"]
          track_metadata = f"{video_path}/{preprocess_track_id}"
          row = {
              "video_path": relative_video_path,
              "video_name": os.path.basename(video_path),
              "frame": frame_id,
              "time": time,
              "track_id": preprocess_track_id,
              "xmin": box[0],
              "ymin": box[1],
              "xmax": box[2],
              "ymax": box[3],
              "h": frame_results.orig_shape[0],
              "w": frame_results.orig_shape[1],
              "confidence": confidence,
              "label": classes_mapping[int(cls)],
              "track_metadata": track_metadata,
          }

          directories = Path(relative_video_path).parent.parts
          for i, folder in enumerate(directories):
             if folder:
              row[f"folder{i+1}"] = folder

          frame_results_rows.append(row)

    return frame_results_rows

def save_analyst_output(video_path, model_results, out_folder, next_track_index, **kwargs):
  results_df = model_results
  tracks_found = 0

  if not results_df.empty:
    postprocessed_results = postprocess(results_df, kwargs["fps"], next_track_index)

    if not postprocessed_results.empty:
      directory_columns = [c for c in postprocessed_results.columns if c.startswith("folder")]
      postprocessed_results = postprocessed_results[SHARKTRACK_COLUMNS + directory_columns]
      assert all([c in postprocessed_results.columns for c in SHARKTRACK_COLUMNS])

      frame_output_path = compute_frames_output_path(video_path, kwargs["input"], out_folder, kwargs["is_chapters"])
      frame_output_path.mkdir(exist_ok=True, parents=True)
      write_max_conf(postprocessed_results, frame_output_path, kwargs["input"], kwargs.get("species_classifier", None))

      concat_df(postprocessed_results, os.path.join(out_folder, "output.csv"))
      new_next_track_index = postprocessed_results["track_id"].max() + 1
      tracks_found = new_next_track_index - next_track_index
      next_track_index = new_next_track_index
  
  print(f"Found {tracks_found} tracks!")

  overview_row = {"video_path": video_path, "tracks_found": tracks_found}
  concat_df(pd.DataFrame([overview_row]), os.path.join(out_folder, configs["overview_filename"]))

  return next_track_index

def save_peek_output(video_path, frame_results, out_folder, next_track_index, **kwargs):
  # Save peek frames
  frames_save_dir = compute_frames_output_path(video_path, kwargs["input"], out_folder, kwargs["is_chapters"])
  frames_save_dir.mkdir(exist_ok=True)

  if len(frame_results[0].boxes.xyxy.cpu().tolist()) > 0:
      plot = frame_results[0].plot(line_width=2)
      img = annotate_image(
        plot, 
        f"Video: {video_path}",
        f"Track ID: {next_track_index}",
        f"Time: {kwargs['time']}")
      cv2.imwrite(str(frames_save_dir / f"{next_track_index}.jpg"), img)

      # Save sightings in csv
      sightings = extract_sightings(video_path, kwargs["input"], frame_results[0], kwargs["frame_id"], kwargs["time"], **{"track_id": next_track_index})
      df = pd.DataFrame(sightings)
      out_path = os.path.join(out_folder, "output.csv")
      concat_df(df, out_path)

      next_track_index += 1

  return next_track_index

def concat_df(df, output_path):
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        if not existing_df.empty:
          df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(output_path, index=False)

def postprocess(results, fps, next_track_index):
    """
    results: pd.DataFrame with columns SHARKTRACK_COLUMNS
    """
    length_thresh = fps
    motion_thresh = 0.08   # Motion is the max(x,y) movement of the centre of the bounding box wrt the frame size
    max_conf_thresh = 0.7

    results["cx"] = (results["xmin"] + results["xmax"]) / 2
    results["cy"] = (results["ymin"] + results["ymax"]) / 2

    grouped = results.groupby("track_metadata")
    results["track_life"] = grouped["track_metadata"].transform("count")
    results["max_conf"] = grouped["confidence"].transform("max")
    results["motion_x"] = (grouped["cx"].transform("max") - grouped["cx"].transform("min")) / results["w"]
    results["motion_y"] = (grouped["cy"].transform("max") - grouped["cy"].transform("min")) / results["h"]

    might_be_false_positive = (results["track_life"] < length_thresh) | (results[["motion_x", "motion_y"]].max(axis=1) < motion_thresh) 
    false_positive = (might_be_false_positive & (results["max_conf"] < max_conf_thresh))

    # Set CopyOnWrite, according to https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy:~:text=2%0A40%20%203-,Returning%20a%20view%20versus%20a%20copy,-%23
    pd.options.mode.copy_on_write = True
    filtered_results = results.loc[~false_positive]
    filtered_results["track_id"] = filtered_results.groupby("track_id").ngroup(ascending=True) + next_track_index

    return filtered_results

def write_max_conf(postprocessed_results: pd.DataFrame, out_folder: Path, video_path_prefix: str, species_classifier: SpeciesClassifier = None):
  """
  Saves annotated images with the maximum confidence detection for each track
  Uses parallel classification when species_classifier is provided
  """
  start_time = t.time()

  # Clear deduplication cache for this video
  # Note: This ensures each video starts fresh, preventing false positives across videos
  clear_detection_cache()

  max_conf_detections_idx = postprocessed_results.groupby("track_metadata")["confidence"].idxmax()
  max_conf_detections_df = postprocessed_results.loc[max_conf_detections_idx]

  if species_classifier:
    # Use parallel classification
    write_max_conf_parallel(postprocessed_results, max_conf_detections_df, out_folder, video_path_prefix, species_classifier)
  else:
    # Original sequential processing without classification
    write_max_conf_sequential(max_conf_detections_df, out_folder, video_path_prefix)

  elapsed_time = t.time() - start_time
  print(f"✅ annotated images in {elapsed_time:.2f} seconds")


def write_max_conf_sequential(max_conf_detections_df: pd.DataFrame, out_folder: Path, video_path_prefix: str):
  """Sequential processing without species classification"""
  duplicates_skipped = 0

  for _, row in max_conf_detections_df.iterrows():
    video_short_path = row["video_path"]
    video_path = Path(video_path_prefix) / video_short_path
    time = row["time"]
    image = extract_frame_at_time(str(video_path), string_to_ms(time))
    label = row["label"]
    confidence = row["confidence"]

    # Check for duplicates
    bbox = row[["xmin", "ymin", "xmax", "ymax"]].values
    if is_duplicate_detection(str(video_path), image, tuple(bbox)):
      duplicates_skipped += 1
      print(f"  Skipped duplicate: track {row['track_id']} in {video_short_path}")
      continue

    img = draw_bboxes(image, [bbox], [f"{row['track_id']}: {confidence*100:.0f}%"])
    img = annotate_image(img,  f"Video: {video_short_path or row['video_name']}", f"Track ID: {row['track_id']}", f"Time: {time}")

    output_image_id = f"{row['track_id']}-{label}.jpg"
    output_path = str(out_folder / output_image_id)
    cv2.imwrite(output_path, img)

  if duplicates_skipped > 0:
    print(f"  ⚠️  Skipped {duplicates_skipped} duplicate detections")


def write_max_conf_parallel(postprocessed_results: pd.DataFrame, max_conf_detections_df: pd.DataFrame,
                           out_folder: Path, video_path_prefix: str, species_classifier: SpeciesClassifier):
  """Parallel processing with species classification"""

  # Import here to avoid circular import
  from .parallel_classifier import ParallelSpeciesClassifier, ClassificationTask

  # Initialize parallel classifier
  parallel_classifier = ParallelSpeciesClassifier(species_classifier, batch_size=16)
  parallel_classifier.start(video_path_prefix)

  classification_start = t.time()

  try:
    # Submit all classification tasks
    for idx, row in max_conf_detections_df.iterrows():
      task = ClassificationTask(
        track_id=row["track_id"],
        track_metadata=row["track_metadata"],
        video_path=row["video_path"],
        time=row["time"],
        xmin=int(row["xmin"]),
        ymin=int(row["ymin"]),
        xmax=int(row["xmax"]),
        ymax=int(row["ymax"]),
        confidence=row["confidence"],
        row_index=idx
      )
      parallel_classifier.submit_task(task)

    # Process results as they come in while also handling image generation
    processed_tracks = set()
    total_tracks = len(max_conf_detections_df)

    # Copy dataframe for safe modification
    results_df = postprocessed_results.copy()

    while len(processed_tracks) < total_tracks:
      # Get classification results
      batch_results = parallel_classifier.get_results()

      for result in batch_results:
        if result.row_index in processed_tracks:
          continue

        row = max_conf_detections_df.loc[result.row_index]

        # Update classification results
        if result.species:
          label = result.species
          classification_confidence = result.classification_confidence
        else:
          label = configs["unclassifiable"]
          classification_confidence = result.classification_confidence

        # Update all detections for this track
        track_mask = results_df["track_metadata"] == row["track_metadata"]
        results_df.loc[track_mask, "label"] = label
        results_df.loc[track_mask, "classification_confidence"] = classification_confidence

        # Generate annotated image
        generate_annotated_image(row, out_folder, video_path_prefix, label, classification_confidence)

        processed_tracks.add(result.row_index)

        if len(processed_tracks) % 10 == 0:
          print(f"Processed {len(processed_tracks)}/{total_tracks} classifications")

      # Small delay to prevent busy waiting
      t.sleep(0.01)

    # Update original dataframe
    postprocessed_results.update(results_df)

  finally:
    # Clean shutdown
    remaining_results = parallel_classifier.stop()
    print(f"Parallel classification completed. {len(remaining_results)} results remaining in queue.")

  classification_time = t.time() - classification_start
  print(f"⚡ Species classification completed in {classification_time:.2f} seconds")


def generate_annotated_image(row: pd.Series, out_folder: Path, video_path_prefix: str, label: str, confidence: float):
  """Generate a single annotated image"""
  video_short_path = row["video_path"]
  video_path = Path(video_path_prefix) / video_short_path
  time = row["time"]
  image = extract_frame_at_time(str(video_path), string_to_ms(time))

  # Check for duplicates
  bbox = row[["xmin", "ymin", "xmax", "ymax"]].values
  if is_duplicate_detection(str(video_path), image, tuple(bbox)):
    print(f"  Skipped duplicate: track {row['track_id']} in {video_short_path}")
    return False  # Indicate skipped

  display_confidence = confidence if confidence > 0 else row["confidence"]
  img = draw_bboxes(image, [bbox],
                   [f"{row['track_id']}: {display_confidence*100:.0f}%"])
  img = annotate_image(img, f"Video: {video_short_path or row['video_name']}",
                      f"Track ID: {row['track_id']}", f"Time: {time}")

  output_image_id = f"{row['track_id']}-{label}.jpg"
  output_path = str(out_folder / output_image_id)
  cv2.imwrite(output_path, img)
  return True  # Indicate saved


def resume_previous_run(output_path: Path):
  processed_videos = set()
  tracks_found = 0
  try:
    df = pd.read_csv(os.path.join(output_path, configs["overview_filename"]))
    processed_videos = set(df["video_path"].values.tolist())
    tracks_found = df["tracks_found"].sum()
  except Exception as e:
    print("Starting new SharkTrack analysis")
  
  return tracks_found, processed_videos
      