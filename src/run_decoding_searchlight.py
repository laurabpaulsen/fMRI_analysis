"""
This script loads in the contrasts made in "prep_for_decoding.py" and runs a decoding analysis on them. Each subject is decoded separately, and the results are saved.
"""
import pickle
from pathlib import Path
from nilearn.image import load_img, concat_imgs
from nilearn.decoding import SearchLight
from sklearn.svm import LinearSVC
import numpy as np
from nilearn.maskers import NiftiMasker
from nilearn.image import new_img_like
from nilearn import plotting
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import train_test_split


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

    # create list of indices for cross validation
    indices = np.arange(len(y))

    # split into train and test set (50/50)
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify = y)

    # create train and test set
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]

    # concatenate images
    X_train = concat_imgs(X_train)
    X_test = concat_imgs(X_test)


    return X_train, y_train, X_test, y_test




if __name__ in "__main__":
    path = Path(__file__).parents[1]

    # path to contrasts
    contrasts_path = path / "data" / "decoding" 

    subjects = ["0116", "0117", "0118", "0119", "0120", "0121", "0122", "0123"]

    # path to save results
    results_path = path / "data" / "searchlight"
    output_path = path / "data" / "classification"

    # ensure output path exists
    if not output_path.exists():
        output_path.mkdir(exist_ok = True)

    # ensure results path exists
    if not results_path.exists():
        results_path.mkdir(exist_ok = True)

    # loop through subjects
    for subject in subjects:
        print(f"Running decoding analysis for subject {subject}...")

        # load in all contrasts
        contrasts_pos, contrasts_neg = load_contrasts_dir(contrasts_path / f"sub-{subject}")

        # prep contrasts for decoding
        X_sl, y_sl, X_cl, y_cl = prep_X_y(contrasts_pos, contrasts_neg)

        # brain mask 
        mask_wb_filename = Path('/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-0116/anat/sub-0116_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
        mask_img = load_img(mask_wb_filename)

        searchlight = SearchLight(
            mask_img,
            estimator=LinearSVC(penalty='l2'),
            radius=5, 
            n_jobs=-1,
            verbose=10, 
            cv=3,
            )
        
        searchlight.fit(X_sl, y_sl)

        # save results
        with open(results_path / f"sub-{subject}.pkl", "wb") as f:
            pickle.dump(searchlight, f)

        # use most explanatory voxels for classification
        mask_wb_filename=f'/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-{subject}/anat/sub-{subject}_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
        anat_filename=f'/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-{subject}/anat/sub-{subject}_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'
        
        searchlight_img = new_img_like(anat_filename, searchlight.scores_)
        
        # get the 500 most predictive voxels
        perc = 100 * (1-500.0/searchlight.scores_.size)

        # cutoff
        cutoff = np.percentile(searchlight.scores_, perc)

        # load whole brain mask
        mask_wb = load_img(mask_wb_filename)

        process_mask2 = mask_wb.get_fdata().astype(int)
        process_mask2[searchlight.scores_<=cutoff] = 0
        process_mask2_img = new_img_like(mask_wb, process_mask2)

        masker = NiftiMasker(mask_img=process_mask2_img, standardize=False)

        # plot voxels
        #plotting.plot_glass_brain effects
        fig = plotting.plot_glass_brain(searchlight_img, threshold = cutoff)
        fig.savefig(path / "fig" / f"InSpe_neg_vs_but_searchlightNB_glass_500_{subject}.png", dpi=300)

        # load contrast images
        contrasts_path = path / "data" / "decoding" / f"sub-{subject}"
        contrasts_pos, contrasts_neg = load_contrasts_dir(contrasts_path)

        # mask the images
        masker = NiftiMasker(mask_img=process_mask2_img, standardize = False)
        fmri_masked = masker.fit_transform(X_cl)
        print("Now running classifier")
        score_cv_test, scores_perm, pvalue= permutation_test_score(
            GaussianNB(), fmri_masked, y_cl, cv=10, n_permutations=1000, 
            n_jobs=-1, random_state=0, verbose=0, scoring=None)
        
        # write results to file
        with open(output_path / f"sub-{subject}_classification.txt", "w") as f:
            f.write(f"Classification Accuracy: {score_cv_test} (pvalue : {pvalue})")

        print(f"Classification Accuracy: {score_cv_test} (pvalue : {pvalue})")


