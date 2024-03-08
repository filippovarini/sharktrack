#%% Cell1
from ultralytics import YOLO
from annotations.sharktrack import yolo2sharktrack, extract_track_max_conf_detection, build_detection_folder
import cv2
import os

#%% Cell2
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
    mobile_model = '/vol/biomedic3/bglocker/ugproj2324/fv220/dev/SharkTrack-Dev/models/yolov8_n_mvd2_50/best.pt'
    analyst_model = '/vol/biomedic3/bglocker/ugproj2324/fv220/dev/SharkTrack-Dev/models/p2v5_new/weights/best.pt'

    if mobile:
      self.model_path = mobile_model
      self.tracker_path = 'botsort.yaml'
      self.device = 'cpu'
      self.fps = 2
    else:
      self.model_path = analyst_model
      self.tracker_path = './trackers/tracker_5fps.yaml'
      self.device = '0'
      self.fps = 5
    
    # Static Hyperparameters
    self.conf_threshold = 0.2
    self.iou_association_threshold = 0.5
    self.imgsz = 640

  
  def _get_frame_skip(self, video_path):
    cap = cv2.VideoCapture(video_path)  
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = round(actual_fps / self.fps)
    return frame_skip

  
  def save(self, results, output_path='./output'):
    sharktrack_results = yolo2sharktrack(results, self.fps)

    # Construct Detections Folder
    max_conf_detections = extract_track_max_conf_detection(sharktrack_results)
    build_detection_folder(max_conf_detections, self.videos_folder, output_path, self.fps)

    # Save results to csv
    output_csv = os.path.join(output_path, 'output.csv')
    print(f'Saving results to {output_csv}...')
    sharktrack_results.to_csv(output_csv, index=False)

  
  def track(self, video_path):
    print(f'Processing video: {video_path}...')
    model = YOLO(self.model_path)

    results = model.track(
      video_path,
      conf=self.conf_threshold,
      iou=self.iou_association_threshold,
      imgsz=self.imgsz,
      tracker=self.tracker_path,
      vid_stride=self._get_frame_skip(video_path),
      device=self.device,
      verbose=False,
    )

    return results


  def run(self, videos_folder, stereo=False, save_results=True):
    all_results = {}
    self.videos_folder = videos_folder

    for video in os.listdir(videos_folder):
      video_path = os.path.join(videos_folder, video)
      if os.path.isdir(video_path):
        for chapter in os.listdir(video_path):
          stereo_filter = not stereo or 'LGX' in chapter # pick only left camera
          custom_filter = 'easy1' in chapter #TODO: remove
          if chapter.endswith('.mp4') and stereo_filter and custom_filter:
            chapter_id = os.path.join(video, chapter)
            chapter_path = os.path.join(videos_folder, chapter_id)
            chapter_results = self.track(chapter_path)
            all_results[chapter_id] = chapter_results

    self.results = all_results
    if save_results:
      self.save(all_results)

    if len(all_results) == 0:
      print('No chapters found in the given folder')
      print('Please ensure the folder structure resembles the following:')
      print('videos_folder')
      print('├── video1')
      print('│   ├── chapter1.mp4')
      print('│   ├── chapter2.mp4')
      print('└── video2')
      print('    ├── chapter1.mp4')
      print('    ├── chapter2.mp4')

    return all_results

  def get_results(self):
    # 2. From the results construct VIAME
    return self.results

#%% Cell3
def main():
  video_path = '/vol/biomedic3/bglocker/ugproj2324/fv220/datasets/frame_extraction_raw/val1'
  model = Model()
  results = model.run(video_path)
  
  # 1. Run tracker with configs
  # 2. From the results construct VIAME


if __name__ == '__main__':
  main()

