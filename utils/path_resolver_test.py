import unittest
import os
from path_resolver import generate_output_path, default_output
import shutil

class TestGenerateCorrectOutput(unittest.TestCase):
    def test__manage_user_provided_output(self):
        self.assertFalse(generate_output_path("outputs", None))
        self.assertTrue(generate_output_path("new_output_directory", None))
        self.assertTrue(generate_output_path("new_output_directory/file.jpg", None))

    def test__generate_path_automatically(self):
        self.assertEqual(generate_output_path(None, "run1"), os.path.join(default_output, "run1_processed"))
        self.assertEqual(generate_output_path(None, "/abs/path/to/run1"), os.path.join(default_output, "run1_processed"))
        self.assertEqual(generate_output_path(None, "/abs/path/to/run1/video.mp4"), os.path.join(default_output, "video_processed"))
        
        paths_to_create = [os.path.join(default_output, "run1_processed"), os.path.join(default_output, "run1_processedv1")]
        os.makedirs(paths_to_create[0], exist_ok=True)
        self.assertEqual(generate_output_path(None, "run1"), os.path.join(default_output, "run1_processedv1"))
        os.makedirs(paths_to_create[1], exist_ok=True)
        self.assertEqual(generate_output_path(None, "/abs/path/to/run1"), os.path.join(default_output, "run1_processedv2"))
        for path in paths_to_create:
            shutil.rmtree(path)
    
    


if __name__ == '__main__':
    unittest.main()