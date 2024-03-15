import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def draw_bboxes(image, bboxes, max_conf_bbox):
    """
    Draw bounding boxes on an image
    :param image: image to draw the bounding boxes on
    :param bboxes: list of bounding boxes
    :param color: color of the bounding boxes
    :param max_conf_bbox: maximum confidence bounding box which we want to highlight
    :return: image with bounding boxes drawn
    """
    img = image.copy()
    for bbox in bboxes:
        color= (0, 255, 255)
        thickness = 1
        if (bbox == max_conf_bbox).all():
            color = (0, 0, 255)
            thickness = 2
        bbox = np.array(bbox).astype(int)
        pt1, pt2 = (bbox[0], bbox[1]), (bbox[2], bbox[3])
        img = cv2.rectangle(img, pt1, pt2, color, thickness)
    return img

def annotate_image(img, video, chapter, time):
    padding_height = 100  # You can adjust the height as needed
    new_width = img.shape[1]
    new_height = img.shape[0] + padding_height

    new_image = 255 * np.ones((new_height, new_width, 3), np.uint8)
    new_image[:img.shape[0], :img.shape[1], :] = img

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 0)  # Black color
    line_type = 2

    # Position the text on the white padding
    cv2.putText(new_image, f"Video: {video}", (10, img.shape[0] + 30), font, font_scale, font_color, line_type)
    cv2.putText(new_image, f"Chapter: {chapter}", (10, img.shape[0] + 60), font, font_scale, font_color, line_type)
    cv2.putText(new_image, f"Time: {time}", (10, img.shape[0] + 90), font, font_scale, font_color, line_type)

    return new_image