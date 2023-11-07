"""
This script loads in the contrasts made in "prep_for_decoding.py" and runs a decoding analysis on them. Each subject is decoded separately, and the results are saved.
"""
import pickle
from pathlib import Path
import numpy as np
import nibabel as nib
from nilearn.decoding import SearchLight
from sklearn.svm import LinearSVC


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
        
        contrast = contrast.get_fdata()

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
    contrasts_path = path / "data" / "decoding" 

    subjects = ["0116"]#, "0117", "0118", "0119", "0120", "0121", "0122", "0123"]

    # path to save results
    results_path = path / "results"

    # ensure results path exists
    if not results_path.exists():
        results_path.mkdir()

    # loop through subjects
    for subject in subjects:
        print(f"Running decoding analysis for subject {subject}...")

        # load in all contrasts
        print(contrasts_path / f"sub-{subject}")
        contrasts_pos, contrasts_neg = load_contrasts_dir(contrasts_path / f"sub-{subject}")

        # prep contrasts for decoding
        X, y = prep_X_y(contrasts_pos, contrasts_neg)

        # brain mask 
        mask_wb_filename = Path('/work/82777/BIDS/derivatives/sub-0096/anat/sub-{subject}_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
        mask_wb = nib.load(mask_wb_filename).get_fdata()

        searchlight = SearchLight(
            mask_wb,
            estimator=LinearSVC(penalty='l2'),
            radius=5, 
            n_jobs=-1,
            verbose=10, 
            cv=10
            )
        
        searchlight.fit(X, y)

        # save results
        with open(results_path / f"sub-{subject}.pkl", "wb") as f:
            pickle.dump(searchlight, f)

