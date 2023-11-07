"""
This script loads in the contrasts made in "prep_for_decoding.py" and runs a decoding analysis on them. Each subject is decoded separately, and the results are saved.
"""
import pickle
from pathlib import Path
import numpy as np


def load_contrasts_dir(path:Path):
    """
    Loads in all contrasts from a directory and returns them in a list.
    """
    contrasts_pos = []
    contrasts_neg = []

    # loop through all files in directory
    for f in path.glob("*"):
        # load in contrast
        with open(f, "rb") as f:
            contrast = pickle.load(f)

        # append to list
        if "positive" in f.name:
            contrasts_pos.append(contrast)
        elif "negative" in f.name:
            contrasts_neg.append(contrast)

    return contrasts_pos, contrasts_neg

def prep_X_y(pos_contrasts:list, neg_contrasts:list):
    """
    Prepares contrasts for decoding analysis.
    """
    
    # concatenate all contrasts
    y = [1] * len(pos_contrasts) + [0] * len(neg_contrasts)

    # concatenate all contrasts and turn into numpy array
    X = pos_contrasts + neg_contrasts
    X = np.array(X)

    return X, y





if __name__ in "__main__":
    path = Path(__file__).parents[1]

    # path to contrasts
    contrasts_path = path / "data" / "decoding" / "contrasts"

    subjects = [f.name for f in contrasts_path.glob("*") if f.is_dir()]

    # path to save results
    results_path = path / "results"

    # ensure results path exists
    if not results_path.exists():
        results_path.mkdir()

    # loop through subjects
    for subject in subjects:
        print(f"Running decoding analysis for subject {subject}...")

        # load in all contrasts
        contrasts_pos, contrasts_neg = load_contrasts_dir(contrasts_path / subject)

        # prep contrasts for decoding
        X, y = prep_X_y(contrasts_pos, contrasts_neg)

        print(f"Number of positive contrasts: {len(contrasts_pos)}")
        print(f"Number of negative contrasts: {len(contrasts_neg)}")

        print(f"Shape of X: {X.shape}")
