import pandas as pd
import sys
import cv2
import os
sys.path.append("compute_output")
from utils import format_time, unformat_time
from viame_annotations import max_conf2viame
from image_processor import draw_bboxes


SHARKTRACK_COLUMNS = ["video", "chapter", "frame", "time", "track_metadata", "track_id", "xmin", "ymin", "xmax", "ymax", "confidence", "class"]

def extract_frame_results(frame_results):
    boxes = frame_results.boxes.xyxy.cpu().tolist()
    tracks = frame_results.boxes.id
    track_ids = tracks.int().cpu().tolist() if tracks is not None else []
    confidences = frame_results.boxes.conf.cpu().tolist()
    classes = frame_results.boxes.cls.cpu().tolist()

    return zip(boxes, track_ids, confidences, classes)

def yolo2sharktrack(yolo_results, fps):
  """
  Given yolo_results in ultralytics.Results format, save the results to csv_file using the MOT format
  """
  data = []
  track_metadata_to_id = {}

  for chapter_id, chapter_results in yolo_results.items():
      for frame_id, frame_results in enumerate(chapter_results):
          video = chapter_id.split("/")[0]
          chapter = chapter_id.split("/")[1]
          time = format_time(frame_id / fps)

          for box, chapter_track_id, confidence, cls in extract_frame_results(frame_results):
              track_metadata = f"{video}/{chapter}/{chapter_track_id}"
              if track_metadata not in track_metadata_to_id:
                  track_metadata_to_id[track_metadata] = len(track_metadata_to_id)  

              unique_track_id = track_metadata_to_id[track_metadata]  
              
              row = {
                  "video": video,
                  "chapter": chapter,
                  "frame": frame_id,
                  "time": time,
                  "track_metadata": track_metadata,
                  "track_id": unique_track_id,
                  "xmin": box[0],
                  "ymin": box[1],
                  "xmax": box[2],
                  "ymax": box[3],
                  "confidence": confidence,
                  "class": cls,
              }
              data.append(row)

  df = pd.DataFrame(data, columns=SHARKTRACK_COLUMNS)
  return df


def extract_track_max_conf_detection(sharktrack_df):
  """
  Given the sharktrack_output in csv format, return the maximum confidence detection for each track, uniquely identified by video/chapter/track_id
  """
  max_conf_detection = sharktrack_df.groupby(["video", "chapter", "track_id"]).apply(lambda x: x.loc[x["confidence"].idxmax()])
  max_conf_detection = max_conf_detection.reset_index(drop=True)
  return max_conf_detection


def read_bboxes(sharktrack_output, video, chapter, time):
  """
  Given the sharktrack_output in dataframe format, return the bounding boxes for a given video, chapter, and time
  """
  rows = sharktrack_output[(sharktrack_output["video"] == video) & (sharktrack_output["chapter"] == chapter) & (sharktrack_output["time"] == time)]
  return rows[["xmin", "ymin", "xmax", "ymax"]].values


def save_track_max_conf_frame(sharktrack_output, max_conf_detections, video_base_folder, det_folder):
  # For each detection, save the image to the detection folder
  for index, row in max_conf_detections.iterrows():
    video = row["video"]
    chapter = row["chapter"]
    time = row["time"]
    unique_track_id = row["track_id"]
    max_conf_bbox = row[["xmin", "ymin", "xmax", "ymax"]].values

    # Get original image
    video_path = os.path.join(video_base_folder, video, chapter)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, unformat_time(time))
    _, img = cap.read()
    cap.release()

    # Draw the rectangles
    img = draw_bboxes(img, read_bboxes(sharktrack_output, video, chapter, time), max_conf_bbox)

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