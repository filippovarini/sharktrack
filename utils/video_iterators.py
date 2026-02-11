import cv2
import av
import av.datasets

def stride_iterator(video_path, vid_stride):
    """
    Iterates with cv2 each vid_stride frame
    """
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
            

def keyframe_iterator(video_path):
    """
    Iterates quickier with pyav each keyframe
    """
    content = av.datasets.curated(video_path)
    with av.open(content) as container:
      video_stream = container.streams.video[0]         # take only video stream
      video_stream.codec_context.skip_frame = 'NONKEY'  # and only keyframes (1fps)
      frame_idx = 0
      for frame in container.decode(video_stream):
        time_ms = round((frame.pts * video_stream.time_base) * 1000)
        yield frame.to_image(), time_ms, frame_idx
        frame_idx += 1