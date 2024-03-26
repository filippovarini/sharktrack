import pandas as pd
import sys
import cv2
import os
sys.path.append("compute_output")
from utils import format_time, unformat_time
from viame_annotations import max_conf2viame
from image_processor import draw_bboxes, annotate_image


SHARKTRACK_COLUMNS = ["chapter_path", "frame", "time", "track_metadata", "track_id", "xmin", "ymin", "xmax", "ymax", "confidence", "class"]

classes_mapping = ['shark', 'ray']

def extract_frame_results(frame_results):
    boxes = frame_results.boxes.xyxy.cpu().tolist()
    tracks = frame_results.boxes.id
    track_ids = tracks.int().cpu().tolist() if tracks is not None else []
    confidences = frame_results.boxes.conf.cpu().tolist()
    classes = frame_results.boxes.cls.cpu().tolist()

    return zip(boxes, track_ids, confidences, classes)

def yolo2sharktrack(chapter_id, chapter_results, fps, out_folder, track_count):
  """
  Given  in ultralytics.Results format, save the results to csv_file using the MOT format
  """
  data = []

  max_conf_image = {}

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
          if track_metadata not in max_conf_image or max_conf_image[track_metadata]["confidence"] < confidence:
              max_conf_image[track_metadata] = {
                  "confidence": confidence,
                  "image": frame_results.plot()
              }

  df = pd.DataFrame(data, columns=SHARKTRACK_COLUMNS)

  postprocessed_results = postprocess(df, fps, track_count)
  postprocessed_results.to_csv(os.path.join(out_folder, "output.csv"), index=False)

  write_max_conf(postprocessed_results, max_conf_image, out_folder)

  return postprocessed_results["track_id"].max()

def postprocess(chapter_sharktrack_df, fps, track_count):
    """
        1. Extracts tracks that last for less than 1s (5frames)
        2. Removes the track if the max confidence is less than MAX_CONF_THRESHOLD
    """
    MAX_CONF_THRESHOLD = 0.8
    DURATION_THRESH = 5 if fps == 5 else 2

    track_counts = chapter_sharktrack_df["track_metadata"].value_counts()
    max_conf = chapter_sharktrack_df.groupby("track_metadata")["confidence"].max()
    valid_tracks = track_counts[(track_counts >= DURATION_THRESH) | (max_conf > MAX_CONF_THRESHOLD)].index
    filtered_df = chapter_sharktrack_df[chapter_sharktrack_df["track_metadata"].isin(valid_tracks)]

    # assign track id
    # To update the sliced dataset without copying (save up memory) and avoid
    # causing SettingWithCopyWarning, deactivate it
    pd.options.mode.chained_assignment = None
    filtered_df['track_id'] = filtered_df.groupby('track_metadata').ngroup() + track_count
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
      cv2.imwrite(output_path, max_conf_image[track_metadata]["image"])
    
#TODO: viame

# def extract_track_max_conf_detection(sharktrack_df):
#   """
#   Given the sharktrack_output in csv format, return the maximum confidence detection for each track, uniquely identified by video/chapter/track_id
#   """
#   max_conf_detection = (
#      sharktrack_df
#       .groupby(["track_metadata"], as_index = False)
#       .apply(lambda x: x.loc[x["confidence"].idxmax()])
#       .sort_values('track_id')
#       .reset_index(drop=True)
#     )
#   return max_conf_detection


# def read_bboxes(sharktrack_output, chapter_path, time):
#   """
#   Given the sharktrack_output in dataframe format, return the bounding boxes for a given video, chapter, and time
#   """
#   rows = sharktrack_output[(sharktrack_output["chapter_path"] == chapter_path) & (sharktrack_output["time"] == time)]
#   return rows[["xmin", "ymin", "xmax", "ymax"]].values


# def save_track_max_conf_frame(sharktrack_output, max_conf_detections, video_base_folder, det_folder):
#   # For each detection, save the image to the detection folder
#   for index, row in max_conf_detections.iterrows():
#     chapter_path = row["chapter_path"]
#     time = row["time"]
#     unique_track_id = row["track_id"]
#     max_conf_bbox = row[["xmin", "ymin", "xmax", "ymax"]].values

#     # Get original image
#     video_path = os.path.join(video_base_folder, chapter_path)
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise ValueError(f"Could not open video file {video_path}")
#     cap.set(cv2.CAP_PROP_POS_MSEC, unformat_time(time))
#     _, img = cap.read()
#     cap.release()

#     # Draw the rectangles
#     img = draw_bboxes(img, read_bboxes(sharktrack_output, chapter_path, time), max_conf_bbox)
#     img = annotate_image(img, chapter_path, time)

#     output_image_id = f"{unique_track_id}.jpg"
#     output_path = os.path.join(det_folder, output_image_id)
#     cv2.imwrite(output_path, img)


# def build_detection_folder(sharktrack_output, video_base_folder, out_folder, fps):
#   max_conf_detections = extract_track_max_conf_detection(sharktrack_output)

#   assert os.path.exists(out_folder), f"Output folder {out_folder} does not exist"
#   det_folder = os.path.join(out_folder, "detections")
#   os.makedirs(det_folder, exist_ok=True)
  
#   save_track_max_conf_frame(sharktrack_output, max_conf_detections, video_base_folder, det_folder)
#   viame_df = max_conf2viame(max_conf_detections, fps)

#   viame_csv = os.path.join(out_folder, "viame.csv")
#   viame_df.to_csv(viame_csv, index=False)