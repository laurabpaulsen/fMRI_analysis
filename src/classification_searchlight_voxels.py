"""

"""
from pathlib import Path
import pickle
from nilearn.image import new_img_like, load_img
import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiMasker
from run_decoding_searchlight import load_contrasts_dir, prep_X_y
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import permutation_test_score


if __name__ in "__main__":
    path = Path(__file__).parents[1]

    # path to searchlight results
    searchlight_path = path / "data" / "searchlight"

    subjects = ["0116", "0117", "0118", "0119", "0120", "0121", "0122", "0123"]

    for subject in subjects:
        # load results (for now just one subject)
        subject_path = searchlight_path / f"sub-{subject}.pkl"
        
        with open(subject_path, "rb") as f:
            searchlight = pickle.load(f)

        mask_wb_filename=f'/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-{subject}/anat/sub-{subject}_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
        anat_filename=f'/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-{subject}/anat/sub-{subject}_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'
        

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

        # load contrast images
        contrasts_path = path / "data" / "decoding" / f"sub-{subject}"
        contrasts_pos, contrasts_neg = load_contrasts_dir(contrasts_path)

        # prep contrasts for decoding
        X, y = prep_X_y(contrasts_pos, contrasts_neg)

        # mask the images
        masker = NiftiMasker(mask_img=process_mask2_img, standardize = False)
        fmri_masked = masker.fit_transform(X)
        print("now running classifier")
        score_cv_test, scores_perm, pvalue= permutation_test_score(
            GaussianNB(), fmri_masked, y, cv=10, n_permutations=1000, 
            n_jobs=-1, random_state=0, verbose=0, scoring=None)
        
        print("Classification Accuracy: %s (pvalue : %s)" % (score_cv_test, pvalue))



        
