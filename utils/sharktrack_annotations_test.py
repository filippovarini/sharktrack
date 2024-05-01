import unittest
from sharktrack_annotations import compute_frames_output_path

class SaveCorrectOutput(unittest.TestCase):
    def test__compute_frames_output_path_correctly(self):
        self.assertEqual(compute_frames_output_path("input_videos/sample.mp4", "input_videos/sample.mp4", "outputs/sample"), "outputs/sample/")
        self.assertEqual(compute_frames_output_path("input_videos/sample.mp4", "input_videos/", "outputs/input_videos"), "outputs/input_videos/sample")
        self.assertEqual(compute_frames_output_path("input_videos/hello/sample.mp4", "input_videos/", "outputs/input_videos"), "outputs/input_videos/hello/sample")

    


if __name__ == '__main__':
    unittest.main()