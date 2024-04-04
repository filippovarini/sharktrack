from argparse import ArgumentParser
import pandas as pd
import os

def clean_annotations_viame(output_path, viame_output_path):
    original_output_path = os.path.join(output_path, "output.csv")
    assert os.path.exists(original_output_path), f"No output.csv file found in {original_output_path}. output.csv represents the unclean output from sharktrack and is required to clean the annotations."
    sharktrack_df = pd.read_csv(original_output_path)

    viame_df = pd.read_csv(viame_output_path, skiprows=lambda x: x in [1])

    # Viame output contains only the tracks that are True Positive
    sharktrack_df = sharktrack_df[sharktrack_df["track_id"].isin(viame_df["# 1: Detection or Track-id"])]
    # Viame output contains the correct species
    class_mapping = viame_df.set_index("# 1: Detection or Track-id")["10-11+: Repeated Species"]
    sharktrack_df["class"] = sharktrack_df["track_id"].map(class_mapping)

    return sharktrack_df

def clean_annotations_locally(output_path):
    predefined_detection_folder = "detections"
    detections_path = os.path.join(output_path, predefined_detection_folder)
    assert os.path.exists(detections_path), f"To clean annotations locally you must have './detections' folder in the output path but {detections_path} doesn't exist!"

    valid_detections = [f for f in os.listdir(detections_path) if f.endswith(".jpg")]
    labeled_detections = {}
    for d in valid_detections:
        try:
            track_id = int(d.split("-")[0])
            label = d.split("-")[1].replace(".jpg", "")
            labeled_detections[track_id] = label
        except:
            raise Exception("All files in ./detections should be '{TRACK_ID}-{CLASS}.jpg' but there is failing file: " + d)
    
    original_output_path = os.path.join(output_path, "output.csv")
    assert os.path.exists(original_output_path), f"No output.csv file found in {original_output_path}. output.csv represents the unclean output from sharktrack and is required to clean the annotations."
    sharktrack_df = pd.read_csv(original_output_path)

    # Viame output contains only the tracks that are True Positive
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

def main(output_path, viame_output_path=None):
    if viame_output_path:
        print(f"Computing MaxN from annotations cleaned using VIAME...")
        cleaned_annotations = clean_annotations_viame(output_path, viame_output_path)
    else:
        print(f"Computing MaxN from annotations cleaned locally...")
        cleaned_annotations = clean_annotations_locally(output_path)
    max_n = compute_species_max_n(cleaned_annotations)

    max_n_path = './max_n.csv'
    max_n.to_csv(max_n_path, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--original_output", type=str, required=True, help="Path to the output folder of sharktrack")
    parser.add_argument("--viame_cleaned", type=str, help="Path to the output csv of viame")
    args = parser.parse_args()
    main(args.original_output, args.viame_cleaned)