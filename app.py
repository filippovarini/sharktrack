#%%
from compute_output.sharktrack_annotations import yolo2sharktrack
from scripts.reformat_gopro import valid_video
from argparse import ArgumentParser
from ultralytics import YOLO
import pandas as pd
import shutil
import cv2
import os

#%%
class Model():
  def __init__(self, mobile=False):
    """
    Args:
      mobile (bool): Whether to use lightweight model developed to run quickly on CPU
    
    Model types:
    | Type    |  Model  | Fps  |
    |---------|---------|------|
    | mobile  | Yolov8n | 2fps |
    | analyst | Yolov8s | 5fps |
    """
    mobile_model = "/vol/biomedic3/bglocker/ugproj2324/fv220/dev/SharkTrack-Dev/models/yolov8_n_mvd2_50/best.pt"
    analyst_model = "models/analyst.pt"
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

    # config
    self.sharktrack_results_name = 'output.csv'
    self.track_count = 0

  
  def _get_frame_skip(self, video_path):
    cap = cv2.VideoCapture(video_path)  
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = round(actual_fps / self.fps)
    return frame_skip

  
  def save_chapter_results(self, chapter_id, yolo_results, output_path="./output"):
    output_csv_path = os.path.join(output_path, self.sharktrack_results_name)
    max_track_id = yolo2sharktrack(chapter_id, yolo_results, self.fps, output_path, self.track_count)
    self.track_count = max_track_id + 1


  def save(self, output_path="./output"):
    assert os.path.exists(output_path), f"Output path {output_path} does not exist"
    output_csv_path = os.path.join(output_path, self.sharktrack_results_name)
    assert os.path.exists(output_csv_path), f"Output csv {output_csv_path} does not exist"

    sharktrack_results = pd.read_csv(output_csv_path)
    
    print(f"Postprocessing results...")

    sharktrack_results = self._postprocess(sharktrack_results)
    if sharktrack_results.empty:
      print("No detections found in the given folder")
      return
    build_detection_folder(sharktrack_results, self.videos_folder, output_path, self.fps)
    sharktrack_results.to_csv(output_csv_path, index=False)
  
  def track(self, video_path):
    print(f"Processing video: {video_path}...")
    model = YOLO(self.model_path)

    results = model.track(
      video_path,
      conf=self.conf_threshold,
      iou=self.iou_association_threshold,
      imgsz=self.imgsz,
      tracker=self.tracker_path,
      vid_stride=self._get_frame_skip(video_path),
      verbose=False,
      stream=True,
    )

    return results

  def run(self, videos_folder, stereo_prefix, max_video_cnt=1000, output_path="./output"):
    # remove previous predictions first
    if os.path.exists(output_path):
      shutil.rmtree(output_path)

    os.makedirs(output_path)

    self.videos_folder = videos_folder
    processed_videos = []

    for (root, _, files) in os.walk(videos_folder):
      for file in files:
        if len(processed_videos) == max_video_cnt:
          break
        stereo_filter = stereo_prefix is None or file.startswith(stereo_prefix)
        if valid_video(file) and stereo_filter:
          chapter_path = os.path.join(root, file)
          chapter_id = chapter_path.replace(f"{videos_folder}/"
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

    # self.save()

    return processed_videos

  def get_results(self):
    # 2. From the results construct VIAME
    return self.results

def main(video_path, stereo_prefix=None, max_video_cnt=1000):
  model = Model()
  results = model.run(video_path, stereo_prefix, max_video_cnt)
  
  # 1. Run tracker with configs
  # 2. From the results construct VIAME

#%%
if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--input_root", type=str, required=True, help="Path to the video file")
  parser.add_argument("--stereo_prefix", type=str, help="Prefix to filter stereo videos")
  args = parser.parse_args()
  main(args.input_root, args.stereo_prefix)
# %%
