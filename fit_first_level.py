import os
import matplotlib.pyplot as plt
from nilearn import image as nimg
from nilearn import plotting as nplot
from bids import BIDSLayout
from pathlib import Path
import numpy as np
import pandas as pd
from nilearn import masking
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel

#for inline visualization in jupyter notebook
%matplotlib inline 


def fit_first_level_subject(subject, bids_dir, runs = [1], space = "MNI152NLin2009cAsym"):
    """

    Parameters
    ----------
    subject : str
        Subject identifier e.g. "0102".

    bids_dir : Path
        Path to the root of the bids directory.

    runs : list of int
        List of runs to load.

    space : str
        Name of the space of the data to load.
    

    """
    
    bids_func_dir  = bids_dir / f"sub-{subject}" / "func"
    fprep_func_dir  = bids_dir / "derivatives" / f"sub-{subject}" / "func"
    

    # paths to functional preprocessed data for all runs
    fprep_func_paths = [fprep_func_dir / f"sub-{subject}_task-boldinnerspeech_run-{run}_echo-1_space-{space}_desc-preproc_bold.nii.gz" for run in runs]
    
    # paths to raw functional data for all runs
    raw_func_paths = [bids_func_dir / f"sub-{subject}_task-boldinnerspeech_run-{run}_echo-1_space-{space}_desc-preproc_bold.nii.gz" for run in runs]
    
    events, confounds, masks = [], [], [] 
    
    for f_prep_path, f_raw_path in zip(fprep_func_paths, raw_func_paths):
        # get the corresponding events
        event_path = str(f_raw_path).replace(f'_echo-1_space-{space}_desc-preproc_bold.nii.gz', '_events.tsv')
        event_df = pd.read_csv(event_path, sep='\t')

        # get the data corresponding to the events and only keep the needed columns
        event_df = event_df.loc[:, ["onset", "duration", "trial_type"]]

        # only keep trial types "IMG_POS" and "IMG_NEG"
        event_df = event_df[event_df['trial_type'].isin(['IMG_PO', 'IMG_NO', 'IMG_PS', 'IMG_NS'])]
        events.append(event_df)


        # get the corresponding confounds
        confounds_path = str(f_prep_path).replace(f'_echo-1_space-{space}_desc-preproc_bold.nii.gz', '_desc-confounds_timeseries.tsv')
        confounds_df = pd.read_csv(confounds_path, sep='\t').loc[:, ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']]
        confounds.append(confounds_df)

        # get the corresponding masks
        mask_path = str(f_prep_path).replace(f'_echo-1_space-{space}_desc-preproc_bold.nii.gz', f'_space-{space}_desc-brain_mask.nii.gz')
        mask = nib.load(mask_path)

        masks.append(mask)


    # merge the masks
    mask_img = masking.intersect_masks(masks, threshold=0.8)

    first_level_model = FirstLevelModel(t_r=nib.load(func).header['pixdim'][4], slice_time_ref=0.5, mask_img=mask_img)
    first_level_model.fit(fprep_func_paths, events, confounds, verbose = 1)
    
    return first_level_model


if __name__ in "__main__":
    bids_dir = "/projects/MINDLAB2022_MR-semantics-of-depression/scratch/bachelor_scratch/BIDS"


    flm = fit_firstlevel(subject, bids_dir, drift_model='cosine', high_pass=0.01) 

    file_name = "insert here"
    pickle.dump(flm, open(file_name, 'wb'))
        
