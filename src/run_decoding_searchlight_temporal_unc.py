"""
This script loads in the contrasts made in "prep_for_decoding.py" and runs a decoding analysis on them. Each subject is decoded separately, and the results are saved.
"""
import pickle
from pathlib import Path
import numpy as np
import nibabel as nib
from nilearn.image import load_img, new_img_like, concat_imgs
from nilearn.decoding import SearchLight
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold





if __name__ in "__main__":
    path = Path(__file__).parents[1]

    # path to contrasts
    contrasts_path = path / "data" / "decoding" 

    subjects = ["0116", "0117", "0118", "0119", "0120", "0121", "0122", "0123"]

    # path to save results
    results_path = path / "data" / "searchlight"

    # ensure results path exists
    if not results_path.exists():
        results_path.mkdir(exist_ok = True)

    runs = ["1"]
    # loop through subjects
    for subject in subjects:
        print(f"Running decoding analysis for subject {subject}...")

        # load data for subject
        beta_maps = []
        ys = []
        for run in runs:
            path_beta_maps = contrasts_path / f"sub-{subject}" / f"beta_maps_temporal_unc_run_{run}.pkl"
            beta_maps = pickle.load(open(path_beta_maps, "rb"))

            # load trial types
            path_ys = contrasts_path / f"sub-{subject}" / f"conditions_run_{run}.pkl"
            y = pickle.load(open(path_ys, "rb"))
            
            index = [i for i, y in enumerate(ys) if "positive" in y or "negative" in y]
            y = np.array(ys)[index]

            beta_maps.append(beta_maps[index])
            ys.append(y)


        
        # stack the beta maps
        beta_maps = np.vstack(beta_maps)
        # stack the ys
        ys = np.hstack(ys)

        # get the positive and negative trials


    




        # prep contrasts for decoding
        #X, y = prep_X_y(contrasts_pos, contrasts_neg)

        # brain mask 
        mask_wb_filename = Path('/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-0116/anat/sub-0116_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
        mask_img = load_img(mask_wb_filename)

        cv = KFold(n_splits=3, shuffle=True, random_state=42)

        searchlight = SearchLight(
            mask_img,
            estimator=LinearSVC(penalty='l2'),
            #process_mask_img=process_mask_img,
            radius=5, 
            n_jobs=-1,
            verbose=10, 
            cv=cv
            )
        

        searchlight.fit(X, y)

        # save results
        with open(results_path / f"sub-{subject}.pkl", "wb") as f:
            pickle.dump(searchlight, f)


