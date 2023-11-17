from pathlib import Path
from pathlib import Path
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel
import pickle
import pandas as pd
import numpy as np

# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils import load_first_level_models, plot_contrast_all_subjects


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

    # exclude button images
    # event_df = event_df[event_df["trial_type"] != "IMG_BI"]

    # get the data corresponding to the events and only keep the needed columns
    event_df = event_df.loc[:, ["onset", "duration", "trial_type"]]

    event_df["trial_type"] = event_df["trial_type"].apply(update_trial_type)

    event_df = event_df.reset_index(drop = True)

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
            # NOTE: duration set to 0 as discussed with Mikkel
            new_row = pd.DataFrame({"onset": [onset], "duration": [0], "trial_type": ["button_press"]})

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


def fit_first_level_subject_per_trial(subject, bids_dir, mask, runs = [1, 2, 3, 4, 5, 6], space = "MNI152NLin2009cAsym"):
    """
    Fits a first level model for one subject, with one regressor per trial

    Parameters
    ----------
    subject : str
        Subject identifier e.g. "0102".

    bids_dir : Path
        Path to the root of the bids directory.
    
    mask : nibabel.nifti1.Nifti1Image
        Mask to use for the first level model.

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
    event_paths = [bids_func_dir / f"sub-{subject}_task-boldinnerspeech_run-{run}_echo-1_events.tsv" for run in runs]
    events = [load_prep_events(path) for path in event_paths]

    # rename the events
    events = [modify_events(event) for event in events]

    # paths to confounds
    confounds_paths = [fprep_func_dir / f"sub-{subject}_task-boldinnerspeech_run-{run}_desc-confounds_timeseries.tsv" for run in runs]
    confounds = [load_prep_confounds(path) for path in confounds_paths]

    # fit first level model for each run
    flms = []

    for i in range(len(runs)):
        print(f"Fitting model for run {i} for subject {subject}")
            
        # fit first level model
        first_level_model = FirstLevelModel(
            t_r=int(nib.load(fprep_func_paths[0]).header['pixdim'][4]), 
            mask_img = mask, 
            slice_time_ref = 0.5,
            hrf_model = "glover",
            )
        
        # fit the model
        flm = first_level_model.fit(fprep_func_paths[i], events[i], confounds[i])
        
        flms.append(flm)
    
    return flms

def get_contrast(regressor, flm, output_type = "effect_size"):
    """
    Calculates the contrast of a given trial type
    """
    contrast_map  = flm.compute_contrast(regressor, output_type = output_type)

    return contrast_map


if __name__ in "__main__":
    path  = Path(__file__).parents[1]

    bids_dir = Path("/work/816119/InSpePosNegData/BIDS_2023E")
    subject = "0116"
    outpath = path / "data" / "decoding_buttonpress" / subject

    # ensure output path exists
    if not outpath.exists():
        outpath.mkdir(exist_ok = True, parents=True)

    mask = nib.load(f"/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-{subject}/func/sub-{subject}_task-boldinnerspeech_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")
       
    flms = fit_first_level_subject_per_trial(subject, bids_dir, mask = mask)


    for i, flm in enumerate(flms):
        # get the names of the regressors
        regressor_names = flm.design_matrices_[0].columns

        # only keep the ones with positive or negative
        regressor_names = [name for name in regressor_names if "button_press" in name or "negative" in name]

        # get the contrasts
        for reg in regressor_names:
            contrast = flm.compute_contrast(reg, output_type = "effect_size")
                
            # save to pickle
            file_name = f"contrast_{reg}_run_{i}.pkl"

            pickle.dump(contrast, open(outpath / file_name, 'wb'))
            
    
