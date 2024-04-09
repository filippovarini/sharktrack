import pandas as pd
import sys
import cv2
import os
sys.path.append("compute_output")
from utils import format_time, unformat_time
from viame_annotations import max_conf2viame, add_metadata_row
from image_processor import draw_bbox, annotate_image


SHARKTRACK_COLUMNS = ["chapter_path", "frame", "time", "track_metadata", "track_id", "xmin", "ymin", "xmax", "ymax", "confidence", "class"]

classes_mapping = ['elasmobranch']

def extract_frame_results(frame_results):
    boxes = frame_results.boxes.xyxy.cpu().tolist()
    tracks = frame_results.boxes.id
    track_ids = tracks.int().cpu().tolist() if tracks is not None else []
    confidences = frame_results.boxes.conf.cpu().tolist()
    classes = frame_results.boxes.cls.cpu().tolist()
    plot = frame_results.plot(conf=False, labels=False, line_width=1)

    return zip(boxes, track_ids, confidences, classes), plot

def build_chapter_output(chapter_id, chapter_results, fps, out_folder, next_track_index):
  """
  Turns ultralytics.Results into MOT format
  Postprocesses the results
  Saves Maximum Detection Confidence images
  Saves VIAME-annotations for cleaning
  """
  data = []
  max_conf_images = {}

  orig_shape = None

  for frame_id, frame_results in enumerate(chapter_results):
      time = format_time(frame_id / fps)

      if orig_shape is None:
        orig_shape = frame_results.orig_shape

      track_results, frame_plot = extract_frame_results(frame_results)

      for box, chapter_track_id, confidence, cls in track_results:
          track_metadata = f"{chapter_id}/{chapter_track_id}"
          row = {
              "chapter_path": chapter_id,
              "frame": frame_id,
              "time": time,
              "track_metadata": track_metadata,
              "chapter_track_id": chapter_track_id,
              "xmin": box[0],
              "ymin": box[1],
              "xmax": box[2],
              "ymax": box[3],
              "h": orig_shape[0],
              "w": orig_shape[1],
              "confidence": confidence,
              "class": classes_mapping[int(cls)],
          }
          data.append(row)
          if track_metadata not in max_conf_images or max_conf_images[track_metadata]["confidence"] < confidence:
              max_conf_images[track_metadata] = {
                  "confidence": confidence,
                  "image": frame_plot,
              }

  results_df = pd.DataFrame(data)

  if not results_df.empty:
    postprocessed_results = postprocess(results_df, fps, next_track_index)
    print(f"Removed {results_df['track_metadata'].nunique() - postprocessed_results['track_metadata'].nunique()} tracks with postprocessing!")

    if not postprocessed_results.empty:
      postprocessed_results = postprocessed_results[SHARKTRACK_COLUMNS]
      concat_df(postprocessed_results, os.path.join(out_folder, "output.csv"))
      write_max_conf(postprocessed_results, max_conf_images, out_folder)
      next_track_index = postprocessed_results["track_id"].max() + 1

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
    filtered_results["track_id"] = filtered_results.groupby("chapter_track_id").ngroup(ascending=True) + next_track_index

    return filtered_results

def write_max_conf(poostprocessed_results, max_conf_image, out_folder):
  """
  Saves annotated images with the maximum confidence detection for each track
  """
  max_conf_detections_idx = poostprocessed_results.groupby("track_metadata")["confidence"].idxmax()
  max_conf_detections_df = poostprocessed_results.loc[max_conf_detections_idx]

  # 2. For each of it, edit the plot by making the track more visible and save it
  for _, row in max_conf_detections_df.iterrows():
    video = row["chapter_path"]
    time = row["time"]
    confidence = row["confidence"]
    plot = max_conf_image[row["track_metadata"]]["image"]
    assert confidence == max_conf_image[row["track_metadata"]]["confidence"]

    img = annotate_image(plot, video, time, confidence)

    # 3. Make focused track more visible
    max_conf_bbox = row[["xmin", "ymin", "xmax", "ymax"]].values
    img = draw_bbox(img, max_conf_bbox)

    # save
    output_image_id = f"{row["track_id"]}.jpg"
    output_path = os.path.join(out_folder, output_image_id)
    cv2.imwrite(output_path, img)