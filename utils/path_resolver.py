import os
import re
from pathlib import Path

default_output = "outputs"

def generate_output_path(user_output_path, input_path, annotation_folder="", resume=False,):
  if user_output_path:
    output_path = user_output_path
    if os.path.exists(output_path):
      if resume:
        print("Resuming SharkTrack run...")
      else:
        print("Using existing output directory...")
      return os.path.join(user_output_path, annotation_folder)
  else:
    # automatically create output
    output_name = os.path.basename(input_path).split(".")[0] + "_processed"
    original_output_path = os.path.join(default_output, output_name)
    output_path = original_output_path

    i = 0
    while os.path.exists(output_path):
      i += 1
      output_path = original_output_path + f"v{i}"

  if annotation_folder:
    output_path = os.path.join(output_path, annotation_folder)

  return output_path

def convert_to_abs_path(path):
  if path and not os.path.isabs(path):
    path = os.path.abspath(path)
  return path


def remove_input_prefix_from_video_path(video_path, input):
  frames_output = video_path.split(input)[1]
  if frames_output.startswith(os.path.sep):
    frames_output = frames_output[1:]
  return frames_output
   
def compute_frames_output_path(video_path, input, output_path, chapters=False) -> Path:
  if input:
    video_path = remove_input_prefix_from_video_path(video_path, input)
  extract_name = os.path.splitext(video_path)[0]
  if chapters:
    extract_name = os.path.dirname(extract_name)

  return Path(output_path) / extract_name

def sort_files(files):
  def extract_num(f):
    res = int(next(iter(re.findall(r"\d+", f)), 0))
    return res

  return sorted(files, key=extract_num)

