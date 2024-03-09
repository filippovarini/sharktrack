import pandas as pd
from argparse import ArgumentParser
import os

def clean_annotations(output_path, viame_output_path):
    sharktrack_output_path = os.path.join(output_path, "output.csv")
    assert os.path.exists(sharktrack_output_path), f"No output.csv file found in {output_path}. output.csv represents the unclean output from sharktrack and is required to clean the annotations."
    sharktrack_df = pd.read_csv(sharktrack_output_path)

    viame_df = pd.read_csv(viame_output_path, skiprows=lambda x: x in [1])

    # Viame output contains only the tracks that are True Positive
    sharktrack_df = sharktrack_df[sharktrack_df["track_id"].isin(viame_df["# 1: Detection or Track-id"])]
    # Viame output contains the correct species
    class_mapping = viame_df.set_index("# 1: Detection or Track-id")["10-11+: Repeated Species"]
    sharktrack_df["class"] = sharktrack_df["track_id"].map(class_mapping)

    return sharktrack_df


def compute_species_max_n(cleaned_annotations):
    frame_box_cnt = cleaned_annotations.groupby(["video", "chapter", "time", "class"], as_index=False)["track_id"].count()
    frame_box_cnt.rename({"track_id": "n", "time": "time_sample"}, axis=1, inplace=True)

    # for each video, species, get the max n and return video, species, max_n, chapter, time when that happens
    max_n = frame_box_cnt.sort_values("n", ascending=False).groupby(["video", "class"], as_index=False).head(1)
    max_n = max_n.sort_values(["video", "n"], ascending=[True, False])
    max_n = max_n.reset_index(drop=True)

    return max_n

def main(output_path, viame_output_path):
    cleaned_annotations = clean_annotations(output_path, viame_output_path)
    max_n = compute_species_max_n(cleaned_annotations)

    max_n_path = os.path.join(output_path, "max_n.csv")
    max_n.to_csv(max_n_path, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", type=str, required=True, help="Path to the output folder of sharktrack")
    parser.add_argument("-v", type=str, required=True, help="Path to the output folder of viame")
    args = parser.parse_args()
    main(args.o, args.v)