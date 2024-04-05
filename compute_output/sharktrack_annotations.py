import pandas as pd
import sys
import cv2
import os
sys.path.append("compute_output")
from utils import format_time, unformat_time
from viame_annotations import max_conf2viame, add_metadata_row
from image_processor import draw_bboxes, annotate_image


SHARKTRACK_COLUMNS = ["chapter_path", "frame", "time", "track_metadata", "track_id", "xmin", "ymin", "xmax", "ymax", "confidence", "class"]

classes_mapping = ['elasmobranch']

def extract_frame_results(frame_results):
    boxes = frame_results.boxes.xyxy.cpu().tolist()
    tracks = frame_results.boxes.id
    track_ids = tracks.int().cpu().tolist() if tracks is not None else []
    confidences = frame_results.boxes.conf.cpu().tolist()
    classes = frame_results.boxes.cls.cpu().tolist()

    return zip(boxes, track_ids, confidences, classes)

def build_chapter_output(chapter_id, chapter_results, fps, out_folder, next_track_index):
  """
  Turns ultralytics.Results into MOT format
  Postprocesses the results
  Saves Maximum Detection Confidence images
  Saves VIAME-annotations for cleaning
  """
  data = []
  max_conf_images = {}

  for frame_id, frame_results in enumerate(chapter_results):
      time = format_time(frame_id / fps)

      for box, chapter_track_id, confidence, cls in extract_frame_results(frame_results):
          track_metadata = f"{chapter_id}/{chapter_track_id}"
          row = {
              "chapter_path": chapter_id,
              "frame": frame_id,
              "time": time,
              "track_metadata": track_metadata,
              "xmin": box[0],
              "ymin": box[1],
              "xmax": box[2],
              "ymax": box[3],
              "confidence": confidence,
              "class": classes_mapping[int(cls)],
          }
          data.append(row)
          if track_metadata not in max_conf_images or max_conf_images[track_metadata]["confidence"] < confidence:
              max_conf_images[track_metadata] = {
                  "confidence": confidence,
                  "image": frame_results.orig_img,
                  "time": time,
                  "video": chapter_id
              }

  df = pd.DataFrame(data, columns=SHARKTRACK_COLUMNS)
  total_tracks = df["track_metadata"].nunique()
  postprocessed_results = postprocess(df, fps, next_track_index)
  filtered_tracks = postprocessed_results["track_metadata"].nunique()
  print(f"Removed {total_tracks - filtered_tracks} with postprocessing")

  if not postprocessed_results.empty:
    concat_df(postprocessed_results, os.path.join(out_folder, "output.csv"))
    write_max_conf(postprocessed_results, max_conf_images, out_folder)
    build_annotation_cleaning_structure(postprocessed_results, out_folder, fps, next_track_index)
    next_track_index = postprocessed_results["track_id"].max() + 1

  return next_track_index

def concat_df(df, output_path):
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(output_path, index=False)

def postprocess(chapter_sharktrack_df, fps, next_track_index):
    """
        1. Extracts tracks that last for less than 1s (5frames)
        2. Removes the track if the max confidence is less than MAX_CONF_THRESHOLD
    """
    MAX_CONF_THRESHOLD = 0.8
    DURATION_THRESH = fps # 1 s

    track_counts = chapter_sharktrack_df["track_metadata"].value_counts()
    max_conf = chapter_sharktrack_df.groupby("track_metadata")["confidence"].max()
    valid_tracks = track_counts[(track_counts >= DURATION_THRESH) | (max_conf > MAX_CONF_THRESHOLD)].index
    filtered_df = chapter_sharktrack_df[chapter_sharktrack_df["track_metadata"].isin(valid_tracks)]

    # assign track id
    # To update the sliced dataset without copying (save up memory) and avoid
    # causing SettingWithCopyWarning, deactivate it
    pd.options.mode.chained_assignment = None
    filtered_df['track_id'] = filtered_df.groupby('track_metadata').ngroup() + next_track_index
    pd.options.mode.chained_assignment = 'warn'

    return filtered_df

def write_max_conf(chapter_sharktrack_df, max_conf_image, out_folder):
  """
  Saves annotated images with the maximum confidence detection for each track
  """
  det_folder = os.path.join(out_folder, "detections")
  os.makedirs(det_folder, exist_ok=True)
  for track_metadata in max_conf_image:
      track_id_filter = chapter_sharktrack_df[chapter_sharktrack_df["track_metadata"] == track_metadata]["track_id"]
      if track_id_filter.empty:
          # postprocessing removed it
          continue
      track_id = track_id_filter.iloc[0]
      output_image_id = f"{track_id}.jpg"
      output_path = os.path.join(det_folder, output_image_id)
      image = max_conf_image[track_metadata]["image"]
      time = max_conf_image[track_metadata]["time"]
      video = max_conf_image[track_metadata]["video"]
      conf = max_conf_image[track_metadata]["confidence"]
      image = annotate_image(image, video, time, conf)
      cv2.imwrite(output_path, image)
    
def extract_track_max_conf_detection(sharktrack_df):
  """
  Given the sharktrack_output in csv format, return the maximum confidence detection for each track, uniquely identified by video/chapter/track_id
  """
  max_conf_detection = (
     sharktrack_df
      .groupby(["track_metadata"], as_index = False)
      .apply(lambda x: x.loc[x["confidence"].idxmax()])
      .sort_values('track_id')
      .reset_index(drop=True)
    )
  return max_conf_detection

def build_annotation_cleaning_structure(sharktrack_output, out_folder, fps, next_track_index):
  max_conf_detections = extract_track_max_conf_detection(sharktrack_output)
  # add bounding boxes to detections
  detections_path = os.path.join(out_folder, "detections")
  save_track_max_conf_frame(sharktrack_output, max_conf_detections, detections_path)
  
  # construct viame df
  viame_df = max_conf2viame(max_conf_detections, fps, next_track_index)
  viame_path = os.path.join(out_folder, "viame.csv")
  if not os.path.exists(viame_path):
    viame_df = add_metadata_row(viame_df, fps)
  concat_df(viame_df, viame_path)


def save_track_max_conf_frame(sharktrack_output, max_conf_detections, det_folder):
  # For each detection, save the image to the detection folder
  for _, row in max_conf_detections.iterrows():
    time = row["time"]
    unique_track_id = row["track_id"]
    chapter_path = row["chapter_path"]
    max_conf_bbox = row[["xmin", "ymin", "xmax", "ymax"]].values

    # Draw the rectangles
    img_path = os.path.join(det_folder, f"{unique_track_id}.jpg")
    img = cv2.imread(img_path)
    all_bboxes = sharktrack_output[(sharktrack_output["chapter_path"] == chapter_path) & (sharktrack_output["time"] == time)][["xmin", "ymin", "xmax", "ymax"]].values
    img = draw_bboxes(img, all_bboxes, max_conf_bbox)

    cv2.imwrite(img_path, img)
