import unittest
from utils.reformat_gopro import valid_video, main
from unittest.mock import patch, call
import os

class TestReformatGoPro(unittest.TestCase):
    def test_valid_video(self):
        self.assertTrue(valid_video("video.MP4"))
        self.assertTrue(valid_video("video.mp4"))
        self.assertTrue(valid_video("video.AVI"))
        self.assertTrue(valid_video("video.avi"))

        self.assertFalse(valid_video("video"))
        self.assertFalse(valid_video("video.mp4k"))
        self.assertFalse(valid_video("video.kmp4"))
        self.assertFalse(valid_video("kmp4"))

    @patch('os.system')
    @patch('os.makedirs')
    @patch('os.walk')
    def test_correct_reformat(self, mockedWalk, mockedMakedirs, mockedSystem):
        videos_root = "sample_root"
        output_root = "test/output_root"
        stereo_prefix = ""

        video_directories = ["test_videos1", "test_videos2", "test_videos1/test_videos3"]

        mock_video_directory = [
            (".", ["tests", "utils", "models", "trackers", "input_videos", "static", ".git", "outputs"], ["annotation-pipelines.md", ".DS_Store", "requirements.txt", "readme.md", "sharktrack-user-guide.md", ".gitignore", "app.py"]),
            ("./tests", ["__pycache__"], ["test_reformat_gopro.py", "test_compute_maxn.py", "__init__.py", "test_time_processor.py", "test_image_processor.py", "test_path_resolver.py"]),
            ("./utils", ["__pycache__"], ["species_classifier.py", "image_processor.py", "config.py", "sharktrack_annotations.py", "__init__.py", "compute_maxn.py", "video_iterators.py", "time_processor.py", "path_resolver.py", "reformat_gopro.py"]),
            ("./models", [], ["sharktrack.pt"]),
            *[(os.path.join(videos_root, video_dir), [], ["sample.mp4"]) for video_dir in video_directories]
        ]

        mockedWalk.return_value = (x for x in mock_video_directory)


        main(videos_root, output_root, stereo_prefix)

        makedirs_calls = [
            call(output_root, exist_ok=True),
            *[call(os.path.join(output_root, video_dir), exist_ok=True) for video_dir in video_directories]
        ]
        self.assertEqual(mockedMakedirs.call_count, 4)
        mockedMakedirs.assert_has_calls(makedirs_calls)

        system_calls = [
            call("ffmpeg -i {} -y -map 0:v -c copy {}".format(
                os.path.join(videos_root, video_dir, "sample.mp4"),
                os.path.join(output_root, video_dir, "sample.mp4")
                )) for video_dir in video_directories
        ]
        self.assertEqual(mockedSystem.call_count, 3)
        mockedSystem.assert_has_calls(system_calls)

    

        


