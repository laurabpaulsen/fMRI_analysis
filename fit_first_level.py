"""
BEAWARE OF WARNING!

/work/LauraBockPaulsen#1941/fMRI_analysis/env/lib/python3.10/site-packages/nilearn/glm/first_level/first_level.py:76: UserWarning: Mean values of 0 observed.The data have probably been centered.Scaling might not work as expected

Try and solve <3
"""

import os
from nilearn import image as nimg
from nilearn import plotting as nplot
from bids import BIDSLayout
from pathlib import Path
import numpy as np
import pandas as pd
from nilearn import masking
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel
import pickle

def add_button_presses(event_df, trial_type_col = "trial_type", response_col = "RT"):
    """
    Takes an event dataframe and adds button presses to it by looking at the "IMG_BI" events and the corresponding "response_time".

    Parameters
    ----------
    event_df : pd.DataFrame
        Dataframe containing the events.
    
    trial_type_col : str
        Name of the column containing the trial types.

    response_col : str
        Name of the column containing the response times.
    
    Returns
    -------
    event_df : pd.DataFrame
        Dataframe containing the events with button presses added.
    """

    # get the indices of the button presses
    button_img_indices = event_df.index[event_df[trial_type_col] == "IMG_BI"].tolist()
    #print(button_img_indices)

    for index in button_img_indices:

        # get the response time
        response_time = event_df.loc[index, response_col]

        # get the onset
        onset = response_time + event_df.loc[index, "onset"]
        
        # new row to add to the dataframe
        new_row = pd.DataFrame({"onset": onset, "duration": 0, "trial_type": "button_press"})

        # concatenate the new row to the dataframe
        event_df = pd.concat([event_df, new_row], ignore_index=True)


        print(event_df.tail(5))

    # sort the dataframe by onset
    event_df = event_df.sort_values(by=["onset"])
    #print(event_df.head(10), "\n\n")

    return event_df





def update_trial_type(trial_type):
    if trial_type in ['IMG_PO', 'IMG_PS']:
        return "positive"
    elif trial_type in ['IMG_NO', 'IMG_NS']:
        return "negative"
    elif trial_type == "IMG_BI":
        return "IMG_button"
    else:
        return trial_type


def fit_first_level_subject(subject, bids_dir, runs = [1, 2, 3, 4, 5, 6], space = "MNI152NLin2009cAsym"):
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

        # add button presses to the event dataframe
        event_df = add_button_presses(event_df)

        # get the data corresponding to the events and only keep the needed columns
        event_df = event_df.loc[:, ["onset", "duration", "trial_type"]]

        event_df["trial_type"] = event_df["trial_type"].apply(update_trial_type)
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

    first_level_model = FirstLevelModel(t_r=int(nib.load(f_prep_path).header['pixdim'][4]), verbose = 1)
    first_level_model.fit(fprep_func_paths, events, confounds)
    
    return first_level_model


if __name__ in "__main__":
    path = Path(__file__).parent
    print(path)

    output_path = path / "flms"

    # make sure that output path exists
    if not output_path.exists():
        output_path.mkdir()

    bids_dir = Path("/work/816119/InSpePosNegData/BIDS_2023E")
    subjects = ["0116", "0117", "0118", "0119", "0120", "0121", "0122", "0123"]

    for subject in subjects:
        flm = fit_first_level_subject(subject, bids_dir) 
        file_name = f"flm_{subject}.pkl"
        pickle.dump(flm, open(output_path / file_name, 'wb'))
            