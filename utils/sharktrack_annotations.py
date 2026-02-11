import pandas as pd
import sys
import cv2
import os
from pathlib import Path
sys.path.append("utils")
from time_processor import string_to_ms
from config import configs
from image_processor import draw_bboxes, annotate_image, extract_frame_at_time
from path_resolver import compute_frames_output_path, remove_input_prefix_from_video_path
from species_classifier import SpeciesClassifier
import time as t


SHARKTRACK_COLUMNS = ["video_path", "video_name", "frame", "time", "xmin", "ymin", "xmax", "ymax", "w", "h", "confidence", "label", "track_metadata", "track_id"]

classes_mapping = ["elasmobranch"]
   
def extract_frame_results(frame_results, tracking):
    boxes = frame_results.boxes.xyxy.cpu().tolist()
    tracks = frame_results.boxes.id
    track_ids = tracks.int().cpu().tolist() if tracks is not None else []
    confidences = frame_results.boxes.conf.cpu().tolist()
    classes = frame_results.boxes.cls.cpu().tolist()

    return zip(boxes, confidences, classes, track_ids) if tracking else zip(boxes, confidences, classes)

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
  
  print(f"✅ Found {tracks_found} tracks!")

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
  """
  start_time = t.time()
  max_conf_detections_idx = postprocessed_results.groupby("track_metadata")["confidence"].idxmax()
  max_conf_detections_df = postprocessed_results.loc[max_conf_detections_idx]

  for _, row in max_conf_detections_df.iterrows():
    video_short_path = row["video_path"]
    video_path = Path(video_path_prefix) / video_short_path
    time = row["time"]
    image = extract_frame_at_time(str(video_path), string_to_ms(time))
    label = row["label"]
    confidence = row["confidence"]

    if species_classifier:
      confidence, species = species_classifier(row, image)
      label = species or configs["unclassifiable"]
      postprocessed_results.loc[postprocessed_results.track_metadata == row["track_metadata"], "label"] = label
      postprocessed_results.loc[postprocessed_results.track_metadata == row["track_metadata"], "classification_confidence"] = confidence
      print(f" species classification time: {elapsed_time:.2f} seconds")

    img = draw_bboxes(image, [row[["xmin", "ymin", "xmax", "ymax"]].values], [f"{row['track_id']}: {(confidence or row['confidence'])*100:.0f}%"])
    img = annotate_image(img,  f"Video: {video_short_path or row['video_name']}", f"Track ID: {row['track_id']}", f"Time: {time}")

    output_image_id = f"{row['track_id']}-{label}.jpg"
    output_path = str(out_folder / output_image_id)
    cv2.imwrite(output_path, img)
  elapsed_time = t.time() - start_time
  


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
      