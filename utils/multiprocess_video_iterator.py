import cv2
import time
from multiprocessing import Queue, Process

def frame_producer(video_path, vid_stride, start_frame, process_id, frame_queue, total_processes):
    vidcap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    while True:
        ret, frame = vidcap.read()
        if not ret:
            break
        frame_number = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
        if (frame_number - start_frame) % (vid_stride * total_processes) == 0:
            timestamp = vidcap.get(cv2.CAP_PROP_POS_MSEC)
            frame_queue.put((frame_number, frame, timestamp, process_id))
    vidcap.release()
    frame_queue.put(None)  # Signal that processing is done

def stride_iterator(video_path, vid_stride):
    num_processes = 16
    frame_queue = Queue(maxsize=20)
    processes = []

    for process_id in range(num_processes):
        start_frame = process_id * vid_stride
        process = Process(target=frame_producer, args=(video_path, vid_stride, start_frame, process_id, frame_queue, num_processes))
        processes.append(process)
        process.start()

    active_processes = num_processes
    frames_buffer = {}
    next_frame_to_yield = 0

    fetch_times = []
    yield_times = []
    global process_times  # Ensure process_times is referenced correctly

    start_time = time.time()
    one_minute_in_seconds = 60

    while active_processes > 0 and (time.time() - start_time) < one_minute_in_seconds:
        fetch_start = time.time()
        item = frame_queue.get()
        fetch_end = time.time()
        fetch_times.append(fetch_end - fetch_start)
        
        if item is None:
            active_processes -= 1
        else:
            frame_number, frame, timestamp, process_id = item
            frames_buffer[frame_number] = (frame, timestamp)
            while next_frame_to_yield in frames_buffer:
                yield_start = time.time()
                frame, timestamp = frames_buffer.pop(next_frame_to_yield)
                yield frame, timestamp, next_frame_to_yield
                yield_end = time.time()
                yield_times.append(yield_end - yield_start)
                next_frame_to_yield += vid_stride

    for process in processes:
        process.join()
    
    # Output profiling information
    print(f"Average fetch time: {sum(fetch_times) / len(fetch_times) * 1000:.2f} ms")
    print(f"Average yield time: {sum(yield_times) / len(yield_times) * 1000:.2f} ms")
    if process_times:
        print(f"Average processing time: {sum(process_times) / len(process_times) * 1000:.2f} ms")
