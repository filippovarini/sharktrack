import cv2
import os
import subprocess
import tempfile
import time

def _extract_frames_ffmpeg(video_path, output_video_path, vid_stride):
    """
    Uses FFmpeg to extract every vid_stride frame and save to output_video_path.
    Captures and displays FFmpeg output if show_log is True.
    """
    ffmpeg_cmd = [
        'ffmpeg', '-i', video_path,
        '-an',
        '-vf', f'select=not(mod(n\,{vid_stride}))',
        '-fps_mode', 'vfr',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        output_video_path
    ]
    ffmpeg_printed_cmd = ' \\\n    '.join(ffmpeg_cmd)
    print(f'Running ffmpeg command:\n$ {ffmpeg_printed_cmd}')
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def _stride_iterator(video_path, video_stride):
    """
    Iterates with cv2 each vid_stride frame from a .mkv file.
    """
    print(f'Started iterating over {video_path} with stride {video_stride}')
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_video_path = os.path.join(tmpdirname, 'output.mkv')

        start_time = time.time()
        _extract_frames_ffmpeg(video_path, output_video_path, video_stride)
        elapsed_time = time.time() - start_time
        print(f"FFmpeg processing time: {elapsed_time:.2f} seconds")

        vidcap = cv2.VideoCapture(output_video_path)
        original_frame_index = 0
        while vidcap.isOpened():
            ret, frame = vidcap.read()
            if not ret:
                break

            # TODO: this might be off by a few milliseconds. Need to improve it.
            original_frame_time = vidcap.get(cv2.CAP_PROP_POS_MSEC) * video_stride
            yield frame, original_frame_time, original_frame_index
            original_frame_index += video_stride
        vidcap.release()


# TODO: Remove this function once the new one is tested
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