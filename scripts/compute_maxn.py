from argparse import ArgumentParser
import pandas as pd
import os
from pathlib import Path

def get_labeled_detections(output_path: str):
    predefined_output_csv = "output.csv"
    output_csv_path = os.path.join(output_path, predefined_output_csv)
    assert os.path.exists(output_csv_path), f"To clean annotations locally you must have './detections' folder in the output path but {output_csv_path} doesn't exist!"

    valid_detections = [f.name for f in Path(output_path).rglob("*jpg")]
    assert len(valid_detections) == len(set(valid_detections)), "Detections don't have unique (track_id, species)"

    labeled_detections = {}
    for d in valid_detections:
        try:
            track_id = int(d.split("-")[0])
            label = d.split("-", maxsplit=1)[1].replace(".jpg", "")
            labeled_detections[track_id] = label
        except:
            raise Exception("All files in ./detections should be '{TRACK_ID}-{CLASS}.jpg' but there is failing file: " + d)
    return labeled_detections
    
def get_original_output(output_path):
    original_output_path = os.path.join(output_path, "output.csv")
    assert os.path.exists(original_output_path), f"No output.csv file found in {original_output_path}. output.csv represents the unclean output from sharktrack and is required to clean the annotations."
    return pd.read_csv(original_output_path)

def clean_annotations_locally(sharktrack_df, labeled_detections):
    filtered_sharktrack_df = sharktrack_df[sharktrack_df["track_id"].isin(labeled_detections.keys())]
    filtered_sharktrack_df.loc[:, "class"] = filtered_sharktrack_df["track_id"].apply(lambda k: labeled_detections[k])
    return filtered_sharktrack_df

def compute_species_max_n(original_output, labeled_detections):
    cleaned_annotations = clean_annotations_locally(original_output, labeled_detections)
    frame_box_cnt = cleaned_annotations.groupby(["video_path", "video_name", "frame", "class"], as_index=False).agg(time=("time", "first"), n=("track_id", "count"), tracks_in_maxn=("track_id", lambda x: list(x)))

    # for each chapter, species, get the max n and return video, species, max_n, chapter, time when that happens
    max_n = frame_box_cnt.sort_values("n", ascending=False).groupby(["video_path", "video_name", "class"], as_index=False).head(1)
    max_n = max_n.sort_values(["video_path", "n"], ascending=[True, False])
    max_n = max_n.reset_index(drop=True)

    return max_n


def main(output_path):
    if not os.path.exists(output_path):
        print(f"Output path {output_path} does not exist")
        return
    print(f"Computing MaxN from annotations cleaned locally...")
    original_output = get_original_output(output_path)
    labeled_detections = get_labeled_detections(output_path)
    max_n = compute_species_max_n(original_output, labeled_detections)

    max_n_path = os.path.join(output_path, "maxn.csv")
    max_n.to_csv(max_n_path, index=False)
    print(f"MaxN computed! Check in the folder {output_path}/maxn.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="./output", help="Path to the output folder of sharktrack")
    args = parser.parse_args()
    main(args.path)