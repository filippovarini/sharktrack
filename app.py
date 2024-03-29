#%%
from compute_output.sharktrack_annotations import build_chapter_output
from scripts.reformat_gopro import valid_video
from argparse import ArgumentParser
from ultralytics import YOLO
import pandas as pd
import shutil
import torch
import cv2
import os

#%%
class Model():
  def __init__(self, videos_folder, max_video_cnt, stereo_prefix, output_path, mobile=False):
    """
    Args:
      mobile (bool): Whether to use lightweight model developed to run quickly on CPU
    
    Model types:
    | Type    |  Model  | Fps  |
    |---------|---------|------|
    | mobile  | Yolov8n | 2fps |
    | analyst | Yolov8s | 5fps |
    """
    self.videos_folder = videos_folder
    self.max_video_cnt = max_video_cnt
    self.stereo_prefix = stereo_prefix
    self.output_path = output_path

    mobile_model = "/vol/biomedic3/bglocker/ugproj2324/fv220/dev/SharkTrack-Dev/models/yolov8_n_mvd2_50/best.pt"
    analyst_model = "models/analyst.pt"
    assert not mobile
    if mobile:
      self.model_path = mobile_model
      self.tracker_path = "botsort.yaml"
      self.fps = 2
    else:
      self.model_path = analyst_model
      self.tracker_path = "./trackers/tracker_5fps.yaml"
      self.fps = 5
    
    # Static Hyperparameters
    self.conf_threshold = 0.2
    self.iou_association_threshold = 0.5
    self.imgsz = 640
    self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # config
    self.next_track_index = 0
  
  def _get_frame_skip(self, chapter_path):
    cap = cv2.VideoCapture(chapter_path)  
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = round(actual_fps / self.fps)
    return frame_skip

  def save_chapter_results(self, chapter_id, yolo_results):
    print(f"Saving results for {chapter_id}...")
    next_track_index = build_chapter_output(chapter_id, yolo_results, self.fps, self.output_path, self.next_track_index)
    self.next_track_index = next_track_index
  
  def track(self, chapter_path):
    print(f"Processing video: {chapter_path}... on device {self.device}")
    model = YOLO(self.model_path)

    results = model.track(
      chapter_path,
      conf=self.conf_threshold,
      iou=self.iou_association_threshold,
      imgsz=self.imgsz,
      tracker=self.tracker_path,
      vid_stride=self._get_frame_skip(chapter_path),
      verbose=False,
      stream=True,
      device=self.device
    )

    return results

  def run(self):
    # remove previous predictions first
    if os.path.exists(self.output_path):
      shutil.rmtree(self.output_path)

    os.makedirs(self.output_path)

    self.videos_folder = self.videos_folder
    processed_videos = []

    for (root, _, files) in os.walk(self.videos_folder):
      for file in files:
        if len(processed_videos) == self.max_video_cnt:
          break
        stereo_filter = self.stereo_prefix is None or file.startswith(self.stereo_prefix)
        if valid_video(file) and stereo_filter and 'AXA_2023-0452204/LGX050008.mp4' in os.path.join(root, file):
          chapter_path = os.path.join(root, file)
          chapter_id = chapter_path.replace(f"{self.videos_folder}/"
                                            , "")
          chapter_results = self.track(chapter_path)
          self.save_chapter_results(chapter_id, chapter_results)
          processed_videos.append(chapter_path)

    if len(processed_videos) == 0:
      print("No chapters found in the given folder")
      print("Please ensure the folder structure resembles the following:")
      print("videos_folder")
      print("├── video1")
      print("│   ├── chapter1.mp4")
      print("│   ├── chapter2.mp4")
      print("└── video2")
      print("    ├── chapter1.mp4")
      print("    ├── chapter2.mp4")

    return processed_videos

  def get_results(self):
    # 2. From the results construct VIAME
    return self.results

def main(video_path, max_video_cnt, stereo_prefix, output_path='./output'):
  model = Model(
    video_path,
    max_video_cnt,
    stereo_prefix,
    output_path
  )
  results = model.run()
  
  # 1. Run tracker with configs
  # 2. From the results construct VIAME

#%%
if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--input_root", type=str, required=True, help="Path to the video file")
  parser.add_argument("--stereo_prefix", type=str, help="Prefix to filter stereo videos")
  parser.add_argument("--max_videos", type=int, default=1000, help="Maximum videos to process")
  parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for the results")
  args = parser.parse_args()
  main(args.input_root, args.max_videos, args.stereo_prefix, args.output_dir)
# %%
