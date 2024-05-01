import os

default_output = "outputs"

def generate_output_path(user_output_path, input_path):
  if user_output_path:
    output_path = user_output_path
    if os.path.exists(output_path):
      print("Output path already exists. Provide a new path or remove the --output <path> argument to automatically output in the SharkTrack folder")
      return None
  else:
    # automatically create output
    output_name = os.path.basename(input_path).split(".")[0] + "_processed"
    original_output_path = os.path.join(default_output, output_name)
    output_path = original_output_path

    i = 0
    while os.path.exists(output_path):
      i += 1
      output_path = original_output_path + f"v{i}"

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
   
def compute_frames_output_path(video_path, input, output_path):
  frames_output = remove_input_prefix_from_video_path(video_path, input)
  extract_name = os.path.splitext(frames_output)[0]
  return os.path.join(output_path, extract_name)

