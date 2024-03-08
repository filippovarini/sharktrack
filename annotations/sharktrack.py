import pandas as pd
import sys
import cv2
import os
sys.path.append('annotations')
from utils import format_time, unformat_time
from viame import max_conf2viame


SHARKTRACK_COLUMNS = ['video', 'chapter', 'frame', 'unique_frame_id', 'time', 'track_id', 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']

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
  frames_seen = 0

  for chapter_id, chapter_results in yolo_results.items():
      for frame_id, frame_results in enumerate(chapter_results):
          frames_seen += 1
          video = chapter_id.split('/')[0]
          chapter = chapter_id.split('/')[1]
          time = format_time(frame_id / fps)

          for box, track_id, confidence, cls in extract_frame_results(frame_results):
              row = {
                  'video': video,
                  'chapter': chapter,
                  'frame': frame_id,
                  'unique_frame_id': frames_seen,
                  'time': time,
                  'track_id': track_id,
                  'xmin': box[0],
                  'ymin': box[1],
                  'xmax': box[2],
                  'ymax': box[3],
                  'confidence': confidence,
                  'class': cls,
              }
              data.append(row)

  df = pd.DataFrame(data, columns=SHARKTRACK_COLUMNS)
  return df


def extract_track_max_conf_detection(sharktrack_df):
  """
  Given the sharktrack_output in csv format, return the maximum confidence detection for each track, uniquely identified by video/chapter/track_id
  """
  max_conf_detection = sharktrack_df.groupby(['video', 'chapter', 'track_id']).apply(lambda x: x.loc[x['confidence'].idxmax()])
  max_conf_detection = max_conf_detection.reset_index(drop=True)
  max_conf_detection = max_conf_detection.sort_values(by=['video', 'chapter', 'frame'])
  max_conf_detection = max_conf_detection.reset_index(drop=True)
  return max_conf_detection


def save_track_max_conf_frame(max_conf_detections, org_base_dir, det_folder):
  # For each detection, save the image to the detection folder
  for index, row in max_conf_detections.iterrows():
    video = row['video']
    chapter = row['chapter']
    time = row['time']
    # global unique frame id -> for 2 max conf detections in the same frame we
    # will save only one image
    frame_id = row['unique_frame_id'] 

    # Get original image
    video_path = os.path.join(org_base_dir, video, chapter)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, unformat_time(time))
    _, img = cap.read()
    cap.release()

    output_image_id = f'{frame_id}.jpg'
    output_path = os.path.join(det_folder, output_image_id)
    cv2.imwrite(output_path, img)


def build_detection_folder(max_conf_detections, org_base_dir, out_folder, fps):
  assert os.path.exists(out_folder), f'Output folder {out_folder} does not exist'
  det_folder = os.path.join(out_folder, 'detections')
  os.makedirs(det_folder, exist_ok=True)
  
  save_track_max_conf_frame(max_conf_detections, org_base_dir, det_folder)
  viame_df = max_conf2viame(max_conf_detections, fps)

  viame_csv = os.path.join(out_folder, 'viame.csv')
  viame_df.to_csv(viame_csv, index=False)