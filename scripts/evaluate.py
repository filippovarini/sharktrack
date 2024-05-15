"""
Code to evaluate predicted MaxN accuracy using the Mean Max N proposed in the SharkTrack paper:

Max N Accuracy in one video v for the species i:
MaxNAcc = True_MaxN / (True_MaxN + MaxN_Error) where MaxN_Error = |True_MaxN - MaxN|

Mean Max N Accuracy for all videos:
mMaxNAcc = sum_v(sum_i(MaxNAcc)) / num_videos*video_species
"""
from argparse import ArgumentParser
import pandas as pd
import os

def max():
    """
    """

def evaluate_maxn(pred, target):
    """
    Assumption: MaxN in SharkTrack format: chapter_path, label, n
    Convert target MaxN to that format before evaluating it
    """
    required_cols = ["chapter_path", "label", "n"]
    assert [c in pred.columns and c in target.columns for c in required_cols], f"Both pred and target MaxN file must have cols {required_cols}"

    evaluation = pd.merge(pred, target, on=["chapter_path", "label"], how="outer", suffixes=("_pred", "_target"), validate="one_to_one")
    evaluation.fillna(0, inplace=True)

    # assert no chapter_path, label duplicates
    assert evaluation[["chapter_path", "label"]].duplicated().sum() == 0, "chapter_path, label duplicates found in evaluation"
    
    evaluation["MaxN_Error"] = abs(evaluation["n_pred"] - evaluation["n_target"])
    evaluation["MaxNAcc"] = evaluation["n_target"] / (evaluation["n_target"] + evaluation["MaxN_Error"])
    
    mMaxNAcc = evaluation["MaxNAcc"].mean()
    return mMaxNAcc

def main(pred_path, target_path):
    if not os.path.exists(pred_path):
        f"Predicted MaxN path {pred_path} doesn't exist"
        return
    if not os.path.exists(target_path):
        f"Target MaxN path {target_path} doesn't exist"
        return

    pred = pd.read_csv(pred_path)
    target = pd.read_csv(target_path)
    mMaxNAcc = evaluate_maxn(pred, target)
    print(f"Mean MaxN Accuracy is {mMaxNAcc}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pred_maxn", type=str, required=True, help="Path to the PREDICTED MaxN file")
    parser.add_argument("--target_maxn", type=str, required=True, help="Path to the TRAGET MaxN file")
    args = parser.parse_args()
    main(args.pred_maxn, args.target_maxn)