#%%
import pandas as pd
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

# if __name__ == "__main__":
    # df = clean_annotations("output", "output/test123.csv")
# %%
