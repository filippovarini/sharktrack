from utils.sharktrack_annotations import build_chapter_output
from utils.image_processor import annotate_image
from utils.time_processor import format_time
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
  def __init__(self, videos_folder, max_video_cnt, stereo_prefix, output_path, peek=False, device_override=None, conf=0.25):
    self.videos_folder = videos_folder
    self.max_video_cnt = max_video_cnt
    self.stereo_prefix = stereo_prefix
    self.output_path = output_path

    self.model_path = "models/sharktrack.pt"

    self.model_args = {
      "conf": conf,
      "iou": 0.5,
      "device": torch.device(device_override) if device_override else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
      "verbose": False,
    }

    if peek:
      print("NOTE: You are using the model 'peek' mode. Please be aware of the following:")
      print("  ✅ Runs significantly faster")
      print("  ⛔️ You won't be able to compute MaxN, but only find interesting frames")
      print("  💡 You can safely ignore the ffmpeg warnings below")
      print("-" * 20)
      print("")
      self.inference_type = self.keyframe_detection
      self.model_args["imgsz"] = 320
    else:
      self.inference_type = self.track_video
      self.fps = 5
      self.model_args["tracker"] = "trackers/tracker_3fps.yaml"
      self.model_args["persist"] = True
      self.model_args["imgsz"] = 640
      
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

  def keyframe_detection(self, chapter_path):
    """
    Tracks keyframes using PyAv to overcome the GoPro audio format issue.
    """
    print(f"Processing video: {chapter_path} on device {self.model_args['device']}")
    peek_dir = os.path.join(self.output_path, "peek_frames")
    os.makedirs(peek_dir, exist_ok=True)

    model = YOLO(self.model_path)

    content = av.datasets.curated(chapter_path)
    
    with av.open(content) as container:
      video_stream = container.streams.video[0]         # take only video stream
      video_stream.codec_context.skip_frame = 'NONKEY'  # and only keyframes (1fps)
      frame_idx = 0
      for frame in container.decode(video_stream):
        frame_idx += 1
        print(f"  processing frame {frame_idx}...", end="\r")
        frame_results = model(
          source=frame.to_image(),
          **self.model_args
        )
        if len(frame_results[0].boxes.xyxy.cpu().tolist()) > 0:
          plot = frame_results[0].plot(line_width=2)
          time = format_time(float(frame.pts * video_stream.time_base))
          img = annotate_image(plot, chapter_path, time, conf=None)
          cv2.imwrite(os.path.join(peek_dir, f"{os.path.basename(chapter_path)}_{self.next_track_index}.jpg"), img)
          self.next_track_index += 1
        
  def track_video(self, chapter_path):
    """
    Uses ultralytics built-in tracker to automatically track a video with OpenCV.
    This is faster but it fails with GoPro Audio format, requiring reformatting.
    """
    print(f"Processing video: {chapter_path} on device {self.model_args['device']}. Might take some time...")
    model = YOLO(self.model_path)

    results = model.track(
      chapter_path,
      **self.model_args,
      vid_stride=self._get_frame_skip(chapter_path),
      stream=True,
    )

    chapter_id = chapter_path.replace(f"{self.videos_folder}/", "")
    self.save_chapter_results(chapter_id, results)

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
    self.videos_folder = self.videos_folder
    processed_videos = []

    if valid_video(self.videos_folder):
      self.inference_type(self.videos_folder)
      processed_videos.append(self.videos_folder)
    else:
      for (root, _, files) in os.walk(self.videos_folder):
        for file in files:
          if len(processed_videos) == self.max_video_cnt:
            break
          stereo_filter = self.stereo_prefix is None or file.startswith(self.stereo_prefix)
          if valid_video(file) and stereo_filter:
            chapter_path = os.path.join(root, file)
            self.inference_type(chapter_path)
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


def main(video_path, max_video_cnt, stereo_prefix, output_path='./output', peek=False, live=False, conf=0.25, device_override=None):
  video_path = convert_abs_path(video_path)
  output_path = convert_abs_path(output_path)

  
  if os.path.exists(output_path):
    shutil.rmtree(output_path)
  os.makedirs(output_path)

  model = Model(
    video_path,
    max_video_cnt,
    stereo_prefix,
    output_path,
    peek,
    device_override=device_override,
    conf=conf
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
  parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
  parser.add_argument("--output", type=str, default="./output", help="Output directory for the results")
  parser.add_argument("--peek", action="store_true", help="Use peek mode: 5x faster but only finds interesting frames, without tracking/computing MaxN")
  parser.add_argument("--live", action="store_true", help="Show live tracking video for debugging purposes")
  args = parser.parse_args()

  # avoid duplicate libraries exception caused by numpy installation
  os.environ["KMP_DUPLICATE_LIB_OK"]="True"

  main(args.input, args.limit, args.stereo_prefix, args.output, args.peek, args.live, args.conf)