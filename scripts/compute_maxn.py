from argparse import ArgumentParser
import pandas as pd
import os

def clean_annotations_locally(output_path):
    predefined_detection_folder = "detections"
    detections_path = os.path.join(output_path, predefined_detection_folder)
    assert os.path.exists(detections_path), f"To clean annotations locally you must have './detections' folder in the output path but {detections_path} doesn't exist!"

    valid_detections = [f for f in os.listdir(detections_path) if f.endswith(".jpg")]
    labeled_detections = {}
    for d in valid_detections:
        try:
            track_id = int(d.split("-")[0])
            label = d.split("-", maxsplit=1)[1].replace(".jpg", "")
            labeled_detections[track_id] = label
        except:
            raise Exception("All files in ./detections should be '{TRACK_ID}-{CLASS}.jpg' but there is failing file: " + d)
    
    original_output_path = os.path.join(output_path, "output.csv")
    assert os.path.exists(original_output_path), f"No output.csv file found in {original_output_path}. output.csv represents the unclean output from sharktrack and is required to clean the annotations."
    sharktrack_df = pd.read_csv(original_output_path)

    sharktrack_df = sharktrack_df[sharktrack_df["track_id"].isin(labeled_detections.keys())]
    sharktrack_df["class"] = sharktrack_df["track_id"].apply(lambda k: labeled_detections[k])

    return sharktrack_df

def compute_species_max_n(cleaned_annotations):
    frame_box_cnt = cleaned_annotations.groupby(["chapter_path", "time", "class"], as_index=False)["track_id"].count()
    frame_box_cnt.rename({"track_id": "n", "time": "time_sample"}, axis=1, inplace=True)

    # for each chapter, species, get the max n and return video, species, max_n, chapter, time when that happens
    max_n = frame_box_cnt.sort_values("n", ascending=False).groupby(["chapter_path", "class"], as_index=False).head(1)
    max_n = max_n.sort_values(["chapter_path", "n"], ascending=[True, False])
    max_n = max_n.reset_index(drop=True)

    return max_n


def main(output_path):
    if not os.path.exists(output_path):
        print(f"Output path {output_path} does not exist")
        return
    print(f"Computing MaxN from annotations cleaned locally...")
    cleaned_annotations = clean_annotations_locally(output_path)
    max_n = compute_species_max_n(cleaned_annotations)

    max_n_path = os.path.join(output_path, "maxn.csv")
    max_n.to_csv(max_n_path, index=False)
    print(f"MaxN computed! Check in the folder {output_path}/maxn.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="./output", help="Path to the output folder of sharktrack")
    args = parser.parse_args()
    main(args.path)