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
    output_path = os.path.join(default_output, output_name)

    i = 0
    while os.path.exists(output_path):
      if i > 0:
        # remove previous version
        output_path = output_path[:-2]
      i += 1
      output_path += f"v{i}"

  return output_path

