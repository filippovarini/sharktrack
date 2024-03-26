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

def yolo2sharktrack(chapter_id, chapter_results, fps):
  """
  Given  in ultralytics.Results format, save the results to csv_file using the MOT format
  """
  data = []

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

  df = pd.DataFrame(data, columns=SHARKTRACK_COLUMNS)
  return df


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


def read_bboxes(sharktrack_output, chapter_path, time):
  """
  Given the sharktrack_output in dataframe format, return the bounding boxes for a given video, chapter, and time
  """
  rows = sharktrack_output[(sharktrack_output["chapter_path"] == chapter_path) & (sharktrack_output["time"] == time)]
  return rows[["xmin", "ymin", "xmax", "ymax"]].values


def save_track_max_conf_frame(sharktrack_output, max_conf_detections, video_base_folder, det_folder):
  # For each detection, save the image to the detection folder
  for index, row in max_conf_detections.iterrows():
    chapter_path = row["chapter_path"]
    time = row["time"]
    unique_track_id = row["track_id"]
    max_conf_bbox = row[["xmin", "ymin", "xmax", "ymax"]].values

    # Get original image
    video_path = os.path.join(video_base_folder, chapter_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, unformat_time(time))
    _, img = cap.read()
    cap.release()

    # Draw the rectangles
    img = draw_bboxes(img, read_bboxes(sharktrack_output, chapter_path, time), max_conf_bbox)
    img = annotate_image(img, chapter_path, time)

    output_image_id = f"{unique_track_id}.jpg"
    output_path = os.path.join(det_folder, output_image_id)
    cv2.imwrite(output_path, img)


def build_detection_folder(sharktrack_output, video_base_folder, out_folder, fps):
  max_conf_detections = extract_track_max_conf_detection(sharktrack_output)

  assert os.path.exists(out_folder), f"Output folder {out_folder} does not exist"
  det_folder = os.path.join(out_folder, "detections")
  os.makedirs(det_folder, exist_ok=True)
  
  save_track_max_conf_frame(sharktrack_output, max_conf_detections, video_base_folder, det_folder)
  viame_df = max_conf2viame(max_conf_detections, fps)

  viame_csv = os.path.join(out_folder, "viame.csv")
  viame_df.to_csv(viame_csv, index=False)