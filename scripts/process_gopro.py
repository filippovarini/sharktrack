from argparse import ArgumentParser
import os

def valid_video(video_path):
    return video_path.endswith(".mp4") and not video_path.startswith(".")

def process_gopros(videos_root, output_root, stereo):
    print('Processing GoPro videos')
    os.makedirs(output_root, exist_ok=True)
    processed_videos = []
    for video in os.listdir(videos_root):
        os.makedirs(os.path.join(output_root, video), exist_ok=True)
        video_path = os.path.join(videos_root, video)
        if len(processed_videos) == 1: 
            break
        if os.path.isdir(video_path):
            for chapter in os.listdir(video_path):
                if len(processed_videos) == 1: 
                    break
                stereo_filter = not stereo or "LGX" in chapter # pick only left camera
                if valid_video(chapter) and stereo_filter:
                    chapter_id = os.path.join(video, chapter)
                    input_path = os.path.join(videos_root, chapter_id)
                    output_path = os.path.join(output_root, chapter_id) 
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
    print('hello')
    process_gopros(args.videos_root, args.output_root, args.stereo)
