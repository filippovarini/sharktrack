#%%
import sys
sys.path.append("/vol/biomedic3/bglocker/ugproj2324/fv220/dev/sharktrack")
from app import main
import torch
import pandas as pd
from datetime import datetime

video_paths = [
    "/vol/biomedic3/bglocker/ugproj2324/fv220/datasets/videos_raw/mwitt/AXA_APR23_no_streams",
    "/vol/biomedic3/bglocker/ugproj2324/fv220/datasets/videos_raw/fra_redsea/RedSeaDeepBlue_BRUVs_processed",
    "/vol/biomedic3/bglocker/ugproj2324/fv220/datasets/downloads/danielle"
]
stereo_prefixes = [
    "LGX",
    "GH",
    None
]
output_paths = [
    "outputs/anguilla_speed_test",
    "outputs/redsea_speed_test",
    "outputs/maldives_speed_test",
]
device_overrides = [torch.device("cuda"), torch.device("cpu")]

max_video_cnt = 10
mobile = True
live=False

running_time = []

for i in range(len(video_paths)):
    for device_override in device_overrides:
        video_path = video_paths[i]
        output_path = output_paths[i] + str(device_override)
        stereo_prefix = stereo_prefixes[i]

        start_time = datetime.now()
        main(video_path, max_video_cnt, stereo_prefix, output_path, mobile, live, device_override)
        running_time.append({"test": output_path, "compute_time": datetime.now() - start_time})

pd.DataFrame(running_time).to_csv("./time_test.csv")



# %%
