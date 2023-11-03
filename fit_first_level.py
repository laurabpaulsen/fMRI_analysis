"""
This script loops over all the participants and fits a first level model to their data

Author: Laura Bock Paulsen
Modified from code by Emma Olsen and Sirid Wihlborg (https://github.com/emmarisgaardolsen/BSc_project_fMRI/blob/main/fmri_analysis_scripts/first_level_fit_function.py)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from nilearn import masking
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel
import pickle

def load_prep_events(path): 
    """
    Loads the event tsv and modifies it to contain the events we want

    Parameters
    ----------
    path : Path
        Path to tsv file containing the events
    
    Returns
    -------
    event_df : pd.DataFrame
        Pandas dataframe containing the events
    """
    # load the data
    event_df = pd.read_csv(path, sep='\t')

    # add button presses to the event dataframe
    event_df = add_button_presses(event_df)

    # get the data corresponding to the events and only keep the needed columns
    event_df = event_df.loc[:, ["onset", "duration", "trial_type"]]

    event_df["trial_type"] = event_df["trial_type"].apply(update_trial_type)

    return event_df

def load_prep_confounds(path, confound_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']):
    """
    Loads the confound tsv and modifies it to contain the events we want

    Parameters
    ----------
    path : Path
        Path to tsv file containing the events
    confound_cols : list of strings
        List of the column names that should be included in the confounds_df
    
    Returns
    -------
    confounds_df : pd.DataFrame
        Pandas dataframe containing the confounds
    """
    # load the confounds
    confounds_df = pd.read_csv(path, sep='\t')

    # choose specific columns
    confounds_df = confounds_df. loc[:, confound_cols]

    return confounds_df

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

    for index in button_img_indices:

        # get the response time
        response_time = event_df.loc[index, response_col]

        # get the onset
        onset = response_time + event_df.loc[index, "onset"]
        
        if np.isnan(onset) == False: # not including missed button presses where RT is NaN
            # new row to add to the dataframe
            # NOTE: NOT SURE WHAT THE APPROPRIATE DURATION IS HERE?
            new_row = pd.DataFrame({"onset": [onset], "duration": [0.5], "trial_type": ["button_press"]})

            # concatenate the new row to the dataframe
            event_df = pd.concat([event_df, new_row], ignore_index=True)

    # sort the dataframe by onset
    event_df = event_df.sort_values(by=["onset"])

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
    
    Returns
    -------
    first_level_model : FirstLevelModel
        First level model fitted for one subject

    """
    
    bids_func_dir  = bids_dir / f"sub-{subject}" / "func"
    fprep_func_dir  = bids_dir / "derivatives" / f"sub-{subject}" / "func"
    

    # paths to functional preprocessed data for all runs
    fprep_func_paths = [fprep_func_dir / f"sub-{subject}_task-boldinnerspeech_run-{run}_echo-1_space-{space}_desc-preproc_bold.nii.gz" for run in runs]
    
    # paths to raw functional data for all runs
    raw_func_paths = [bids_func_dir / f"sub-{subject}_task-boldinnerspeech_run-{run}_echo-1_space-{space}_desc-preproc_bold.nii.gz" for run in runs]

    # prepare event files
    event_paths = [bids_func_dir / f"sub-{subject}_task-boldinnerspeech_run-{run}_events.tsv" for run in runs]
    events = [load_prep_events(path) for path in event_paths]

    # paths to confounds
    confounds_paths = [fprep_func_dir / f"sub-{subject}_task-boldinnerspeech_run-{run}_desc-confounds_timeseries.tsv" for run in runs]
    confounds = [load_prep_confounds(path) for path in confounds_paths]
    
    # prep masks
    mask_paths = [fprep_func_dir / f"sub-{subject}_task-boldinnerspeech_run-{run}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz" for run in runs]
    masks = [nib.load(path) for path in mask_paths]

    # merge the masks
    mask_img = masking.intersect_masks(masks, threshold=0.8)


    # fit first level model
    first_level_model = FirstLevelModel(t_r=int(nib.load(fprep_func_paths[0]).header['pixdim'][4]), mask_img = mask_img, verbose = 1)
    first_level_model.fit(fprep_func_paths, events, confounds)
    
    return first_level_model


if __name__ in "__main__":
    path = Path(__file__).parent
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
            
