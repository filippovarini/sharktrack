from argparse import ArgumentParser
import pandas as pd
import os
from pathlib import Path
import click
import sys
import cv2
import traceback
sys.path.append("utils")
from image_processor import extract_frame_at_time, draw_bboxes, annotate_image
from time_processor import string_to_ms
from path_resolver import compute_frames_output_path
from config import configs

def get_maxn_confidence(labeled_detections):
    completed_annotations = 0
    for k, v in labeled_detections.items():
        if v != configs["unclassifiable"]:
            completed_annotations += 1

    return completed_annotations / len(labeled_detections)

def get_labeled_detections(internal_results_path: str, output_csv_path: str):
    assert os.path.exists(output_csv_path), f"{output_csv_path} doesn't exist! Please provide the path to the top folder of the BRUVS study you want to analyise. It should contain a directory called 'internal_ "

    valid_detections = [f.name for f in Path(internal_results_path).rglob("*jpg")]
    assert len(valid_detections) == len(set(valid_detections)), "Detections don't have unique (track_id, species)"

    labeled_detections = {}
    for d in valid_detections:
        if d.split(".")[0].isnumeric():
            # not classified
            labeled_detections[int(d.split(".")[0])] = None
        else:
            try:
                track_id = int(d.split("-")[0])
                label = d.split("-", maxsplit=1)[1].replace(".jpg", "")
                labeled_detections[track_id] = label
            except:
                raise Exception("All files in ./detections should be '{TRACK_ID}-{CLASS}.jpg' but there is failing file: " + d)

    return labeled_detections
    
def get_original_output(original_output_path):
    assert os.path.exists(original_output_path), f"No output.csv file found in {original_output_path}. output.csv represents the unclean output from sharktrack and is required to clean the annotations."
    return pd.read_csv(original_output_path)

def clean_annotations_locally(sharktrack_df, labeled_detections):
    filtered_sharktrack_df = sharktrack_df[sharktrack_df["track_id"].isin(labeled_detections.keys())]
    if len(filtered_sharktrack_df) == 0:
        print("output csv empty!")
        return
    filtered_sharktrack_df.loc[:, "label"] = filtered_sharktrack_df.apply((lambda row: labeled_detections[row.track_id] or row.label), axis=1)
    return filtered_sharktrack_df

def compute_species_maxn(cleaned_annotations, chapter):
    groupby_columns = ["video_path", "video_name", "frame", "label"]
    directory_columns = [c for c in cleaned_annotations.columns if c.startswith("folder")]
    aggregations = {"time": ("time", "first"), "n":("track_id", "count"), "tracks_in_maxn":("track_id", lambda x: list(x))}
    frame_box_cnt = cleaned_annotations.groupby(groupby_columns + directory_columns, as_index=False, dropna=False).agg(**aggregations)

    # for each chapter, species, get the max n and return video, species, maxn, chapter, time when that happens
    if chapter: 
        maxn = frame_box_cnt.groupby(directory_columns + ["label"], as_index=False, dropna=False).apply(lambda grp: grp.nlargest(1, "n"))
    else:
        maxn = frame_box_cnt.groupby(["video_path", "video_name", "label"], as_index=False, dropna=False).apply(lambda grp: grp.nlargest(1, "n"))
    maxn = maxn.sort_values(["video_path", "n"], ascending=[True, False])
    maxn = maxn.reset_index(drop=True)

    return maxn

def save_maxn_frames(cleaned_output: pd.DataFrame, maxn: pd.DataFrame, videos_path: Path, analysis_output_path: Path, chapters: bool):
    # Remove previous MaxN Visualisation
    for image in analysis_output_path.rglob("*.jpg"):
        image.unlink()

    for idx, row in maxn.iterrows():
        video_relative_path = row["video_path"]
        label = row["label"]
        video_path = videos_path / video_relative_path if str(video_relative_path).strip() else videos_path
        try:
            time_ms = string_to_ms(row["time"])
            frame = extract_frame_at_time(str(video_path), time_ms)
            maxn_sightings = cleaned_output[(cleaned_output["time"] == row["time"]) & (cleaned_output["video_path"] == video_relative_path)]
            bboxes = maxn_sightings[["xmin", "ymin", "xmax", "ymax"]].values
            labels = maxn_sightings["label"].values
            track_ids = maxn_sightings["track_id"].values
            plot = draw_bboxes(frame, bboxes, labels, track_ids)
            plot = annotate_image(plot, f"Video: {video_relative_path}", f"Time: {row['time']}", f"MaxN: {row['n']}")
            frames_folder = compute_frames_output_path(video_relative_path, input=None, output_path=analysis_output_path, chapters=chapters)
            frames_folder.mkdir(exist_ok=True, parents=True)
            image_filename = frames_folder / (label + ".jpg")
            cv2.imwrite(str(image_filename), plot)
        except:
            traceback.print_exc()
            print(f"Failed reading video {video_path}. \n You provided video path {videos_path}, please make sure you provide only the root path that joins with relative path{video_relative_path}")
            # return

@click.command()
@click.option("--path", "-p", type=str, required=True, prompt="Provide path to original output", help="Path to the output folder of sharktrack")
@click.option("--videos", "-v", type=str, default="N/A", show_default=True, prompt="Path to original videos (to compute maxn screenshots)", help="Path to the folder that contains all videos, to extract MaxN")
@click.option("--chapters",  is_flag=True, default=False, show_default=True, prompt="Are your videos split in chapters?", help="Aggreagate chapter information into a single video")
def main(path, videos, chapters):
    final_analysis_folder = "analysed"
    internal_results_folder = "internal_results"

    maxn_filename = "maxn.csv"
    analysis_path = Path(path) / final_analysis_folder
    analysis_path.mkdir(exist_ok=True)

    internal_results_path = Path(path) / internal_results_folder
    original_output_path = internal_results_path / "output.csv"
    if not internal_results_path.exists():
        print(f"**Error** Please provide the path to the folder containing the {internal_results_path} subfolder")
        return

    print(f"Computing MaxN from annotations cleaned locally...")
    original_output = get_original_output(original_output_path)
    labeled_detections = get_labeled_detections(internal_results_path, original_output_path)
    maxn_confidence = get_maxn_confidence(labeled_detections)

    cleaned_annotations = clean_annotations_locally(original_output, labeled_detections)
    maxn = compute_species_maxn(cleaned_annotations, chapters)

    maxn_path = analysis_path / maxn_filename
    maxn.to_csv(str(maxn_path), index=False)
    print(f"MaxN computed! Check in the folder {maxn_path}")
    print(f"MaxN confidence achieved {int(maxn_confidence*100)}%")

    if videos == "N/A":
        # extract the frame from each maxn and annotate it with output.csv
        print("Provide the path to the original videos to compute MaxN screenshots")
    else:
        videos_path = Path(videos)
        save_maxn_frames(cleaned_annotations, maxn, videos_path, analysis_path, chapters)


if __name__ == "__main__":
    main()