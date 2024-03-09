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
