from compute_output.sharktrack_annotations import build_chapter_output
from scripts.reformat_gopro import valid_video
from argparse import ArgumentParser
from ultralytics import YOLO
import shutil
import torch
import cv2
import os
import av
import torch
import av.datasets

class Model():
  def __init__(self, videos_folder, max_video_cnt, stereo_prefix, output_path, mobile=False):
    """
    Args:
      mobile (bool): Whether to use lightweight model developed to run quickly on CPU
    
    Model types:
    | Type    |  Model  | Fps  | F1   | MOTA |
    |---------|---------|------|------|------|
    | mobile  | Yolov8n | 1fps | 0.83 | 0.39 |
    | analyst | Yolov8s | 5fps | 0.85 | 0.74 |
    """
    self.videos_folder = videos_folder
    self.max_video_cnt = max_video_cnt
    self.stereo_prefix = stereo_prefix
    self.output_path = output_path

    mobile_model = "models/mobile.pt"
    analyst_model = "models/analyst.pt"
    if mobile:
      print("Using mobile model...")
      self.model_path = mobile_model
      self.tracker_path = "trackers/tracker_1fps.yaml"
      self.run_tracker = self.track_frames
      self.fps = 1
    else:
      print("Using analyst model...")
      self.model_path = analyst_model
      self.tracker_path = "trackers/tracker_5fps.yaml"
      self.run_tracker = self.track_video
      self.fps = 5
    
    # Static Hyperparameters
    self.conf_threshold = 0.25
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
    assert next_track_index is not None, f"Error saving results for {chapter_id}"
    self.next_track_index = next_track_index
  
  def track_frames(self, chapter_path):
    """
    Tracks keyframes using PyAv to overcome the GoPro audio format issue.
    """
    print(f"Processing video: {chapter_path}... on device {self.device}")
    model = YOLO(self.model_path)
    results = []

    content = av.datasets.curated(chapter_path)
    
    with av.open(content) as container:
      video_stream = container.streams.video[0]         # take only video stream
      video_stream.codec_context.skip_frame = 'NONKEY'  # and only keyframes (1fps)

      for frame in container.decode(video_stream):
        frame_results = model.track(
          source=frame.to_image(),
          conf=self.conf_threshold,
          iou=self.iou_association_threshold,
          imgsz=self.imgsz,
          tracker=self.tracker_path,
          verbose=False,
          persist=True,
        )
        results.append(frame_results[0])

    return results

  def track_video(self, chapter_path):
    """
    Uses ultralytics built-in tracker to automatically track a video with OpenCV.
    This is faster but it fails with GoPro Audio format, requiring reformatting.
    """
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
        if valid_video(file) and stereo_filter:
          chapter_path = os.path.join(root, file)
          chapter_id = chapter_path.replace(f"{self.videos_folder}/", "")
          chapter_results = self.run_tracker(chapter_path)
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
      return

    return processed_videos

def main(video_path, max_video_cnt, stereo_prefix, output_path='./output', mobile=False):
  model = Model(
    video_path,
    max_video_cnt,
    stereo_prefix,
    output_path,
    mobile
  )
  model.run()
  

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--input_root", type=str, required=True, help="Path to the video folder")
  parser.add_argument("--stereo_prefix", type=str, help="Prefix to filter stereo BRUVS")
  parser.add_argument("--max_videos", type=int, default=1000, help="Maximum videos to process")
  parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for the results")
  parser.add_argument("--mobile", action="store_true", help="Use mobile model: 50% faster, slightly less accurate than humans")
  args = parser.parse_args()
  main(args.input_root, args.max_videos, args.stereo_prefix, args.output_dir, args.mobile)