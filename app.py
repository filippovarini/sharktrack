from utils.sharktrack_annotations import build_chapter_output
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

    self.model_path = "models/sharktrack.pt"

    if mobile:
      print("Using mobile model...")
      self.tracker_path = "trackers/tracker_3fps.yaml"
      self.run_tracker = self.track_video
      self.fps = 3
    else:
      print("Using analyst model...")
      self.tracker_path = "trackers/tracker_5fps.yaml"
      self.run_tracker = self.track_video
      self.fps = 5
    
    # Static Hyperparameters
    self.conf_threshold = 0.25
    self.iou_association_threshold = 0.5
    self.imgsz = 640
    self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    self.model_args = {
          "conf": self.conf_threshold,
          "iou": self.iou_association_threshold,
          "imgsz": self.imgsz,
          "tracker": self.tracker_path,
          "verbose": False,
          "device":self.device,
          "persist": True
      }

    # config
    self.next_track_index = 0
  
  def _get_frame_skip(self, chapter_path):
    cap = cv2.VideoCapture(chapter_path)  
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = round(actual_fps / self.fps)
    return frame_skip

  def save_chapter_results(self, chapter_id, yolo_results):
    next_track_index = build_chapter_output(chapter_id, yolo_results, self.fps, self.output_path, self.next_track_index)
    assert next_track_index is not None, f"Error saving results for {chapter_id}"
    self.next_track_index = next_track_index

  def track_video(self, chapter_path):
    """
    Uses ultralytics built-in tracker to automatically track a video with OpenCV.
    This is faster but it fails with GoPro Audio format, requiring reformatting.
    """
    print(f"Processing video: {chapter_path} on device {self.device}. Might take some time...")
    model = YOLO(self.model_path)

    results = model.track(
      chapter_path,
      **self.model_args,
      vid_stride=self._get_frame_skip(chapter_path),
      stream=True,
    )

    return results

  def live_track(self, chapter_path, output_folder='./output/'):
    """
    Plot the live tracking video for debugging purposes
    """
    print("Live tracking video...")
    assert valid_video(chapter_path), f"⚠️ Live Tracking is heavy so it can be done on only one video at a time. Please provide a valid video path."
    cap = cv2.VideoCapture(chapter_path)

    filename = chapter_path.split('/')[-1]
    FILE_OUTPUT = f"{output_folder}{filename.split('.')[0]}_tracked.avi"
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter(FILE_OUTPUT, cv2.VideoWriter_fourcc('M','J','P','G'), self.fps, (frame_width, frame_height))

    model = YOLO(self.model_path)

    while cap.isOpened():
      success, frame = cap.read()

      if success:
        results = model.track(frame, **self.model_args)
        annotated_frame = results[0].plot()

        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
      else:
        break

    cap.release()
    cv2.destroyAllWindows()


  def run(self):
    elasmodetections = [
    "2020_RS_073",
      "2020_RS_076",
      "2020_RS_078",
      "2020_RS_001",
      "2020_RS_005",
      "2020_RS_005",
      "2020_RS_040",
      "2020_RS_050",
      "2020_RS_051",
      "2020_RS_054",
      "2020_RS_055",
      "2020_RS_056",
      "2020_RS_057",
      "2020_RS_060",
      "2020_RS_063",
      "2020_RS_069",
      "2020_RS_071",
      "2020_RS_075",
      "2020_RS_095",
      "2020_RS_101",
      "2020_RS_106",
      "2020_RS_109",
      "2020_RS_111",
      "2020_RS_004",
      "2020_RS_043",
      "2020_RS_064",
      "2020_RS_071",
      "2020_RS_074",
      "2020_RS_092",
      "2020_RS_096",
      "2020_RS_098",
      "2020_RS_109",
      "2020_RS_109",
      "2020_RS_119",
      "2020_RS_121",
      "2020_RS_123",
      "2020_RS_081",
      "2020_RS_109"
    ]
    self.videos_folder = self.videos_folder
    processed_videos = []

    if valid_video(self.videos_folder):
      print("processing only one video...")
      chapter_results = self.run_tracker(self.videos_folder)
      self.save_chapter_results(self.videos_folder, chapter_results)
      processed_videos.append(self.videos_folder)
    else:
      for (root, _, files) in os.walk(self.videos_folder):
        for file in files:
          if len(processed_videos) == self.max_video_cnt:
            break
          stereo_filter = self.stereo_prefix is None or file.startswith(self.stereo_prefix)
          if valid_video(file) and stereo_filter:
            chapter_path = os.path.join(root, file)
            if not any([f in chapter_path for f in elasmodetections]):
              print(f"{chapter_path} unprocessed!")
              continue
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
  
def convert_abs_path(path):
  if path and not os.path.isabs(path):
    path = os.path.abspath(path)
  return path


def main(video_path, max_video_cnt, stereo_prefix, output_path='./output', mobile=False, live=False):
  video_path = convert_abs_path(video_path)
  output_path = convert_abs_path(output_path)
  
  # remove previous predictions first
  if os.path.exists(output_path):
    shutil.rmtree(output_path)
  os.makedirs(output_path)

  model = Model(
    video_path,
    max_video_cnt,
    stereo_prefix,
    output_path,
    mobile
  )
  if live:
    model.live_track(video_path)
  else:
    model.run()

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--input", type=str, default="input_videos", help="Path to the video folder")
  parser.add_argument("--stereo_prefix", type=str, help="Prefix to filter stereo BRUVS")
  parser.add_argument("--limit", type=int, default=1000, help="Maximum videos to process")
  parser.add_argument("--output", type=str, default="./output", help="Output directory for the results")
  parser.add_argument("--mobile", action="store_true", help="Use mobile model: 50% faster, slightly less accurate than humans")
  parser.add_argument("--live", action="store_true", help="Show live tracking video for debugging purposes")
  args = parser.parse_args()

  # avoid duplicate libraries exception caused by numpy installation
  os.environ["KMP_DUPLICATE_LIB_OK"]="True"

  main(args.input, args.limit, args.stereo_prefix, args.output, args.mobile, args.live)