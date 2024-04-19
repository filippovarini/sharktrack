import cv2
import numpy as np


def draw_bbox(image, bbox, color=(0, 255, 0)):
    """
    """
    img = image.copy()
    thickness = 2
    bbox = np.array(bbox).astype(int)
    pt1, pt2 = (bbox[0], bbox[1]), (bbox[2], bbox[3])
    img = cv2.rectangle(img, pt1, pt2, color, thickness)
    return img

def annotate_image(img, chapter_path, time, conf):
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
    cv2.putText(new_image, f"Video: {chapter_path}", (10, img.shape[0] + 30), font, font_scale, font_color, line_type)
    if conf:
        cv2.putText(new_image, f"Confidence: {round(conf, 2)}", (10, img.shape[0] + 60), font, font_scale, font_color, line_type)
    cv2.putText(new_image, f"Time: {time}", (10, img.shape[0] + 90), font, font_scale, font_color, line_type)

    return new_image