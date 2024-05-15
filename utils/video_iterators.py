import cv2

def stride_iterator(video_path, vid_stride):
    vidcap = cv2.VideoCapture(video_path)
    ret, frame = vidcap.read()
    n = 0

    while vidcap.isOpened():
        n += 1
        vidcap.grab()
        if n % vid_stride == 0:
            ret, frame = vidcap.retrieve()
            if ret:
                yield frame, vidcap.get(cv2.CAP_PROP_POS_MSEC), vidcap.get(cv2.CAP_PROP_POS_FRAMES)
            else: 
                break
            

        