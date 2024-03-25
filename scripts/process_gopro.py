from argparse import ArgumentParser
import os

def valid_video(video_path):
    right_suffix = video_path.endswith(".mp4") or video_path.endswith(".MP4")
    right_prefix = not video_path.startswith(".")
    return right_suffix and right_prefix

def compute_output_path(output_root, chapter_id):
    output_path = os.path.join(output_root, chapter_id)
    output_path = output_path.replace(".MP4", ".mp4")
    return output_path

def process_gopros(videos_root, output_root, stereo):
    print('Processing GoPro videos')
    os.makedirs(output_root)
    processed_videos = []
    for video in os.listdir(videos_root):
        video_path = os.path.join(videos_root, video)
        if len(processed_videos) == 2: 
            break
        os.makedirs(os.path.join(output_root, video), exist_ok=True)
        if os.path.isdir(video_path):
            for chapter in os.listdir(video_path):
                if len(processed_videos) == 2: 
                    break
                stereo_filter = not stereo or "LGX" in chapter # pick only left camera
                if valid_video(chapter) and stereo_filter:
                    chapter_id = os.path.join(video, chapter)
                    input_path = os.path.join(videos_root, chapter_id)
                    output_path = compute_output_path(output_root, chapter_id)
                    os.system(f"ffmpeg -i {input_path} -y -map 0:v -c copy {output_path}")
                    processed_videos.append(input_path)

    if len(processed_videos) == 0:
      print("No chapters found in the given folder")
      print("Please ensure the folder structure resembles the following:")
      print("videos_root")
      print("├── video1")
      print("│   ├── chapter1.mp4")
      print("│   ├── chapter2.mp4")
      print("└── video2")
      print("    ├── chapter1.mp4")
      print("    ├── chapter2.mp4")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--videos_root", type=str, required=True, help="Path to the root directory of your video file")
    parser.add_argument("--output_root", type=str, required=True, help="Path to the output root directory of your video file")
    parser.add_argument("--stereo", action='store_true', help="Whether folder contains stereo BRUVS (LGX/RGX)")
    args = parser.parse_args()
    process_gopros(args.videos_root, args.output_root, args.stereo)
