import unittest
import os
from utils.path_resolver import generate_output_path, default_output, compute_frames_output_path, sort_files
import shutil

class TestGenerateCorrectOutput(unittest.TestCase):
    def test__manage_user_provided_output(self):
        self.assertFalse(generate_output_path("outputs", None))
        self.assertTrue(generate_output_path("outputs", None, annotation_folder="", resume=True))
        self.assertTrue(generate_output_path("new_output_directory", None))
        self.assertTrue(generate_output_path("new_output_directory/file.jpg", None))
        self.assertTrue(generate_output_path(None, "videos", "internal_results", resume=True))

    def test__generate_path_automatically(self):
        self.assertEqual(generate_output_path(None, "run1"), os.path.join(default_output, "run1_processed"))
        self.assertEqual(generate_output_path(None, "/abs/path/to/run1"), os.path.join(default_output, "run1_processed"))
        self.assertEqual(generate_output_path(None, "/abs/path/to/run1/video.mp4"), os.path.join(default_output, "video_processed"))

        self.assertEqual(generate_output_path(None, "run1", "expert_annotations"), os.path.join(default_output, "run1_processed", "expert_annotations"))
        self.assertEqual(generate_output_path(None, "/abs/path/to/run1", "expert_annotations"), os.path.join(default_output, "run1_processed", "expert_annotations"))
        self.assertEqual(generate_output_path(None, "/abs/path/to/run1/video.mp4", "expert_annotations"), os.path.join(default_output, "video_processed", "expert_annotations"))
        
        paths_to_create = [os.path.join(default_output, "run1_processed"), os.path.join(default_output, "run1_processedv1")]
        os.makedirs(paths_to_create[0], exist_ok=True)
        self.assertEqual(generate_output_path(None, "run1"), os.path.join(default_output, "run1_processedv1"))
        os.makedirs(paths_to_create[1], exist_ok=True)
        self.assertEqual(generate_output_path(None, "/abs/path/to/run1"), os.path.join(default_output, "run1_processedv2"))
        for path in paths_to_create:
            shutil.rmtree(path)

    def test__compute_frames_output_path_correctly(self):
        self.assertEqual(str(compute_frames_output_path("input_videos/sample.mp4", "input_videos/sample.mp4", "outputs/sample")), "outputs/sample")
        self.assertEqual(str(compute_frames_output_path("input_videos/sample.mp4", "input_videos/", "outputs/input_videos", chapters=True)), "outputs/input_videos")
        self.assertEqual(str(compute_frames_output_path("input_videos/sample.mp4", "input_videos/", "outputs/input_videos")), "outputs/input_videos/sample")
        self.assertEqual(str(compute_frames_output_path("input_videos/hello/sample.mp4", "input_videos/", "outputs/input_videos")), "outputs/input_videos/hello/sample")
        self.assertEqual(str(compute_frames_output_path("input_videos/hello/sample.mp4", "input_videos/", "outputs/input_videos", chapters=True)), "outputs/input_videos/hello")

    def test__sort_files_correctly(self):
        files_list_1 = [f"LG0{n}01.MP4" for n in range(12, 0, -1)]
        self.assertEqual(sort_files(files_list_1), files_list_1[::-1])
        files_list_2 = [f"test{'x' * n}" for n in range(10)]
        self.assertEqual(sort_files(files_list_2), files_list_2)
    
    


if __name__ == '__main__':
    unittest.main()