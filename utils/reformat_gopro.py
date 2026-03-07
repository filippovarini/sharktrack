#%%
from argparse import ArgumentParser
import os

#%%
def valid_video(video_path):
    video_name = os.path.basename(video_path).lower()
    right_suffix = video_name.endswith(".mp4") or video_name.endswith(".avi")
    right_prefix = not video_name.startswith(".")
    return right_suffix and right_prefix

def main(videos_root, output_root, stereo_prefix):
    processed_videos = []

    print('Reformatting GoPro videos')
    os.makedirs(output_root, exist_ok=True)

    for (root, _, files) in os.walk(videos_root):
        # Skip processing the output directory to avoid infinite recursion
        if root.startswith(output_root):
            continue
        output_dir = root.replace(videos_root, output_root)
        for file in files:
            stereo_filter = stereo_prefix is None or file.startswith(stereo_prefix)
            if valid_video(file) and stereo_filter:
                os.makedirs(output_dir, exist_ok=True)
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, file)
                if os.path.exists(output_path):
                    print(f"Skipping {input_path} as it already exists")
                    continue
                os.system(f'ffmpeg -i "{input_path}" -y -map 0:v -c copy "{output_path}"')
                processed_videos.append(input_path)

    if len(processed_videos) == 0:
      print("No new chapters found in the given folder")
      print("Please ensure the folder structure resembles the following:")
      print("video_path")
      print("├── video1")
      print("│   ├── chapter1.mp4")
      print("│   ├── chapter2.mp4")
      print("└── video2")
      print("    ├── chapter1.mp4")
      print("    ├── chapter2.mp4")

#%%

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the root directory of your video file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output root directory of your video file")
    parser.add_argument("--stereo_prefix", type=str, help="Prefix of the stereo video file")
    args = parser.parse_args()
    main(args.input, args.output, args.stereo_prefix)
