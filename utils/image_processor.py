import cv2
import numpy as np


def construct_label_color_mapping(labels, colors):
    if labels is None:
        return None
    label_color_mapping = {}
    for l in labels:
        if l not in label_color_mapping:
            label_color_mapping[l] = colors[len(label_color_mapping) % len(colors)]
    
    return label_color_mapping


def draw_bboxes(image, bboxes, labels=None, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7, font_thickness=2):
    """
    """
    colors = [(242, 73, 248), (0, 255, 0), (144, 43, 147), (241, 70, 106), (255, 0, 0), (164, 102, 71)]
    color_mapping =  construct_label_color_mapping(labels, colors)
    if labels is not None:
        assert len(labels) == len(bboxes), f"There must be same number of labels {len(labels)} and boxes {len(bboxes)}"

    img = image.copy()
    thickness = 2
    
    for i in range(len(bboxes)):
        color = color_mapping[labels[i]] if labels is not None else colors[0]
        bbox = bboxes[i]
        bbox = np.array(bbox).astype(int)
        pt1, pt2 = (bbox[0], bbox[1]), (bbox[2], bbox[3])
        img = cv2.rectangle(img, pt1, pt2, color, thickness)

        if labels is not None:
            label = str(labels[i])
            label_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            label_pt = (pt1[0], pt1[1] - label_size[1] - 4 if pt1[1] - label_size[1] - 4 > 0 else pt1[1] + label_size[1] + 4) # draw above or below based on position of individual
            cv2.putText(img, label, label_pt, font, font_scale, color, font_thickness)

    return img

def annotate_image(img, text1, text2, text3):
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
    cv2.putText(new_image, text1, (10, img.shape[0] + 30), font, font_scale, font_color, line_type)
    cv2.putText(new_image, text2, (10, img.shape[0] + 60), font, font_scale, font_color, line_type)
    cv2.putText(new_image, text3, (10, img.shape[0] + 90), font, font_scale, font_color, line_type)

    return new_image

def extract_frame_at_time(video_path: str, time_ms: int):
    vidcap = cv2.VideoCapture(video_path)
    vidcap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
    ret, frame = vidcap.read()
    assert ret, f"Can't read {video_path} at time {time_ms} ms"
    return frame
