#%%
from compute_output.sharktrack_annotations import yolo2sharktrack, extract_track_max_conf_detection, build_detection_folder
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

  
  def _get_frame_skip(self, video_path):
    cap = cv2.VideoCapture(video_path)  
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = round(actual_fps / self.fps)
    return frame_skip

  def _assign_track_id(self, filtered_results):
        """
        Given the postprocessed results, assign a unique track_id
        """
        # To update the sliced dataset without copying (save up memory) and avoid
        # causing SettingWithCopyWarning, deactivate it
        pd.options.mode.chained_assignment = None
        filtered_results['track_id'] = filtered_results.groupby('track_metadata').ngroup()
        pd.options.mode.chained_assignment = 'warn'
        return filtered_results


  def _postprocess(self, results):
      """
      Implements the following postprocessing steps:
      5fps:
          1. Extracts tracks that last for less than 1s (5frames)
          2. Removes the track if the max confidence is less than MAX_CONF_THRESHOLD
      """
      MAX_CONF_THRESHOLD = 0.8
      DURATION_THRESH = 5 if self.fps == 5 else 2

      track_counts = results["track_metadata"].value_counts()
      max_conf = results.groupby("track_metadata")["confidence"].max()
      valid_tracks = track_counts[(track_counts >= DURATION_THRESH) | (max_conf > MAX_CONF_THRESHOLD)].index
      filtered_df = results[results["track_metadata"].isin(valid_tracks)]

      return self._assign_track_id(filtered_df)
  
  def save_chapter_results(self, chapter_id, yolo_results, output_path="./output"):
    output_csv_path = os.path.join(output_path, self.sharktrack_results_name)
    combined_results = yolo2sharktrack(chapter_id, yolo_results, self.fps)

    if os.path.exists(output_csv_path):
      current_results = pd.read_csv(output_csv_path)
      combined_results = pd.concat([current_results, combined_results])
    
    print(f"Saving results to {output_csv_path}...")
    combined_results.to_csv(output_csv_path, index=False)


  def save(self, output_path="./output"):
    assert os.path.exists(output_path), f"Output path {output_path} does not exist"
    output_csv_path = os.path.join(output_path, self.sharktrack_results_name)
    assert os.path.exists(output_csv_path), f"Output csv {output_csv_path} does not exist"

    sharktrack_results = pd.read_csv(output_csv_path)
    if sharktrack_results.empty:
      print("No detections found in the given folder")
      return
    
    print(f"Postprocessing results...")
    sharktrack_results = self._postprocess(sharktrack_results)
    sharktrack_results.to_csv(output_csv_path, index=False)

    # Construct Detections Folder
    build_detection_folder(sharktrack_results, self.videos_folder, output_path, self.fps)

  
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


  def run(self, videos_folder, stereo=False, output_path="./output"):
    # remove previous predictions first
    if os.path.exists(output_path):
      shutil.rmtree(output_path)

    os.makedirs(output_path)

    self.videos_folder = videos_folder
    processed_videos = []

    for video in os.listdir(videos_folder):
      video_path = os.path.join(videos_folder, video)
      if os.path.isdir(video_path):
        for chapter in os.listdir(video_path):
          stereo_filter = not stereo or "LGX" in chapter # pick only left camera
          if chapter.endswith(".mp4") and stereo_filter: 
            chapter_id = os.path.join(video, chapter)
            chapter_path = os.path.join(videos_folder, chapter_id)
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

    self.save()

    return processed_videos

  def get_results(self):
    # 2. From the results construct VIAME
    return self.results

def main(video_path, stereo):
  model = Model()
  results = model.run(video_path, stereo=stereo)
  
  # 1. Run tracker with configs
  # 2. From the results construct VIAME

#%%
if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--video_path", type=str, required=True, help="Path to the video file")
  parser.add_argument("--stereo", action='store_true', help="Whether folder contains stereo BRUVS (LGX/RGX)")
  args = parser.parse_args()
  main(args.video_path, args.stereo)
# %%
