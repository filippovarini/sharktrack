import unittest
from utils.image_processor import construct_label_color_mapping, draw_bboxes, annotate_image, extract_frame_at_time
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

class TestImageProcessor(unittest.TestCase):
    def test_construct_label_color_mapping(self):
        # Test basic label mapping
        labels = ['car', 'person', 'bike']
        colors = [(255, 0, 0), (0, 255, 0)]
        mapping = construct_label_color_mapping(labels, colors)

        expected_mapping = {
            'car': colors[0],
            'person': colors[1],
            'bike': colors[0]  # Cycles back to the first color
        }
        self.assertEqual(mapping, expected_mapping)

        # Test empty labels
        self.assertIsNone(construct_label_color_mapping(None, colors))
    
    def test_draw_bboxes(self):
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        bboxes = [[10, 10, 50, 50], [60, 60, 90, 90]]
        labels = ['car', 'bike']
        track_ids = [1, 2]

        output_image = draw_bboxes(image, bboxes, labels, track_ids)

        # Ensure output image is of the same shape and not the same instance as the input image
        self.assertEqual(output_image.shape, image.shape)
        self.assertIsNot(output_image, image)

    def test_annotate_image(self):
        image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        text1, text2, text3 = "Label 1", "Label 2", "Label 3"

        annotated_image = annotate_image(image, text1, text2, text3)

        # Ensure the new image has the expected additional height
        self.assertEqual(annotated_image.shape[0], image.shape[0] + 100)
        self.assertEqual(annotated_image.shape[1], image.shape[1])

        # Check that the original part of the image is unchanged (white pixels)
        np.testing.assert_array_equal(annotated_image[:100, :], image)
    
    @patch('cv2.VideoCapture') # we mock cv2.VideoCapture for the duration of the test
    def test_extract_frame_at_time(self, mock_VideoCapture):
        mock_capture = MagicMock()
        mock_VideoCapture.return_value = mock_capture

        test_frame = np.ones((480, 640, 3), dtype=np.uint8) 
        mock_capture.read.return_value = (True, test_frame)

        video_path = "test_video.mp4"
        time_ms = 1000
        
        grabbed_frame = extract_frame_at_time(video_path, time_ms)

        mock_capture.set.assert_called_once_with(cv2.CAP_PROP_POS_MSEC, time_ms)
        self.assertIsInstance(grabbed_frame, np.ndarray)
        self.assertTrue(np.array_equal(grabbed_frame.shape, test_frame.shape))
