"""
This script prepares the data for decoding. 
- This is done by fitting first level models with design matrices that include one regressor per trial.
- A contrast is calculated for each regressor, and the resulting maps are saved.
"""
from pathlib import Path
from fit_first_level import load_prep_events, load_prep_confounds
from pathlib import Path
import numpy as np
import pandas as pd
from nilearn import masking
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel
import pickle

def modify_events(event_df):
    """
    Renames the events so we have one regressor per trial

    parameters
    ----------
    event_df : pd.DataFrame
        Pandas dataframe containing the events
    """

    # unique trial types
    trial_types = event_df["trial_type"].unique()

    # rename the events
    for trial_type in trial_types:
        event_df.loc[event_df["trial_type"] == trial_type, "trial_type"] = [f"{trial_type}_{i}" for i in range(sum(event_df["trial_type"] == trial_type))]
    
    return event_df


def fit_first_level_subject_per_trial(subject, bids_dir, runs = [1, 2, 3, 4, 5, 6], space = "MNI152NLin2009cAsym"):
    """
    Fits a first level model for one subject, with one regressor per trial

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

    # prepare event files
    event_paths = [bids_func_dir / f"sub-{subject}_task-boldinnerspeech_run-{run}_events.tsv" for run in runs]
    events = [load_prep_events(path) for path in event_paths]

    # rename the events
    events = [modify_events(event) for event in events]
    print(events[0].head(10))

    # paths to confounds
    confounds_paths = [fprep_func_dir / f"sub-{subject}_task-boldinnerspeech_run-{run}_desc-confounds_timeseries.tsv" for run in runs]
    confounds = [load_prep_confounds(path) for path in confounds_paths]
    
    # prep masks
    mask_paths = [fprep_func_dir / f"sub-{subject}_task-boldinnerspeech_run-{run}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz" for run in runs]
    masks = [nib.load(path) for path in mask_paths]

    # merge the masks
    mask_img = masking.intersect_masks(masks, threshold=0.8)

    # fit first level model
    first_level_model = FirstLevelModel(
        t_r=int(nib.load(fprep_func_paths[0]).header['pixdim'][4]), 
        mask_img = mask_img, 
        slice_time_ref = 0.5,
        hrf_model = "glover",
        verbose = 1
        )

    first_level_model.fit(fprep_func_paths, events, confounds)
    
    return first_level_model



if __name__ in "__main__":
    path  = Path(__file__).parents[1]

    outpath = path / "data" / "decoding" 

    bids_dir = Path("/work/816119/InSpePosNegData/BIDS_2023E")
    subjects = ["0116", "0117", "0118", "0119", "0120", "0121", "0122", "0123"]
    
    for subject in subjects:
        outpath_subject = outpath / f"sub-{subject}"
        
        # ensure outpath exists 
        if not outpath_subject.exists():
            outpath.mkdir(parents=True)
        
        flm = fit_first_level_subject_per_trial(subject, bids_dir)

        #pickle.dump(flm, open(output_path / / file_name, 'wb'))
            
    
