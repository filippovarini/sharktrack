from argparse import ArgumentParser
import pandas as pd

def species_match():
    """
    """

def evaluate_maxn(pred, target):
    """
    Assumption: MaxN in SharkTrack format: chapter_path, class, n
    Convert target MaxN to that format before evaluating it
    """
    required_cols = ["chapter_path", "class", "n"]
    assert [c in pred.columns and c in target.columns for c in required_cols], f"Both pred and target MaxN file must have cols {required_cols}"

    # for each video
    # for each 

def main(pred_path, target_path):
    if not os.path.exists(pred_path):
        f"Predicted MaxN path {pred_path} doesn't exist"
        return
    if not os.path.exists(target_path):
        f"Target MaxN path {target_path} doesn't exist"
        return

    pred = pd.read_csv(pred_path)
    target = pd.read_csv(target_path)

    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pred_maxn", type=str, required=True, help="Path to the PREDICTED MaxN file")
    parser.add_argument("--target_maxn", type=str, required=True, help="Path to the TRAGET MaxN file")
    args = parser.parse_args()
    main(args.pred_maxn, args.target_maxn)