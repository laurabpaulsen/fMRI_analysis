"""
This script prepares the data for decoding. 
- This is done by fitting first level models with design matrices that include one regressor per trial.
- A contrast is calculated for each regressor, and the resulting maps are saved.
"""
from pathlib import Path
from fit_first_level import load_prep_events, load_prep_confounds
from pathlib import Path
import numpy as np
from scipy.linalg import sqrtm
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel
import pickle

# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils import load_first_level_models, plot_contrast_all_subjects


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


def fit_first_level_subject_per_trial(subject, bids_dir, mask, runs = [1], space = "MNI152NLin2009cAsym"):
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

def get_contrast(regressor, flm, output_type = "z_score"):
    """
    Calculates the contrast of a given trial type
    """
    contrast_map  = flm.compute_contrast(regressor, output_type = output_type)

    return contrast_map


if __name__ in "__main__":
    path  = Path(__file__).parents[1]

    outpath = path / "data" / "decoding" 

    bids_dir = Path("/work/816119/InSpePosNegData/BIDS_2023E")
    subjects = ["0116"]

    
    for subject in subjects:
        outpath_subject = outpath / f"sub-{subject}"
        
        # ensure outpath exists 
        if not outpath_subject.exists():
            outpath_subject.mkdir(exist_ok = True, parents = True)

        mask = nib.load(f"/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-{subject}/anat/sub-{subject}_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")
        
        flms = fit_first_level_subject_per_trial(subject, bids_dir, mask = mask)


        for i, flm in enumerate(flms):
            # get the names of the regressors
            regressor_names = flm.design_matrices_[0].columns

            # only keep the ones with positive or negative
            regressor_names = [name for name in regressor_names if "positive" in name or "negative" in name]
            
            beta_maps = [] 
            # get the contrasts
            for reg in regressor_names:
                contrast = flm.compute_contrast(reg, output_type = "effect_size")

                beta_maps.append(contrast)
                
                # save to pickle
                #file_name = f"contrast_{reg}_run_{i}.pkl"
                # pickle.dump(contrast, open(outpath_subject / file_name, 'wb'))
            
            # vstack the beta maps
            beta_maps = np.vstack(beta_maps)

            # get design matrix
            design_matrix = flm._design_matrices[0]

            # get temporally uncorrerlated beta maps
            beta_maps_temporal_unc = sqrtm(np.linalg.inv(design_matrix.T @ design_matrix)) @ beta_maps

            # save to pickle
            file_name = f"beta_maps_temporal_unc_run_{i}.pkl"
            pickle.dump(beta_maps_temporal_unc, open(outpath_subject / file_name, 'wb'))

            # save the conditions to a pickle
            file_name = f"conditions_run_{i}.pkl"
            pickle.dump(regressor_names, open(outpath_subject / file_name, 'wb'))



            
    
