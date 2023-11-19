"""
This script prepares the data for decoding. 
- This is done by fitting first level models with design matrices that include one regressor per trial.
- A contrast is calculated for each regressor, and the resulting maps are saved.
"""
from pathlib import Path
from fit_first_level import load_prep_events, load_prep_confounds
from pathlib import Path
from scipy.linalg import sqrtm
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel
import pickle
import numpy as np
import matplotlib.pyplot as plt
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.masking import unmask
from nilearn.image import index_img
from nilearn.masking import apply_mask

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

    event_df = event_df.loc[~event_df['trial_type'].str.contains('IMG')]
    
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

    # make a design matrix for each run
    lsa_dms = []

    for i in range(len(runs)):
        print(f"Fitting model for run {i} for subject {subject}")
            
        # fit first level model
        first_level_model = FirstLevelModel(
            t_r=int(nib.load(fprep_func_paths[i]).header['pixdim'][4]), 
            mask_img = mask, 
            slice_time_ref = 0.5,
            hrf_model = "glover",
            smoothing_fwhm=1
            )
        
        # fit the model
        flm = first_level_model.fit(fprep_func_paths[i], events[i], confounds[i])

        # Load the NIfTI file
        img = nib.load(fprep_func_paths[i])

        # Get the number of volumes
        num_volumes = img.shape[-1]

        # Get the TR
        TR = img.header['pixdim'][4]

        # Get the time series data
        ts = img.get_fdata()

        frame_times = np.arange(0, num_volumes) * TR

        # Let's make a standard LSA design matrix
        lsa_dm = make_first_level_design_matrix(
            frame_times=frame_times,  # we defined this earlier for interpolation!
            events=events[i],
            hrf_model='glover',
            drift_model=None  # assume data is already high-pass filtered
        )



        lsa_dm = lsa_dm.iloc[:, :90]

        lsa_dm = lsa_dm[events[i]['trial_type'].values]

        flms.append(flm)

        lsa_dms.append(lsa_dm)
    
    return flms, lsa_dms, events

if __name__ in "__main__":
    path  = Path(__file__).parents[1]

    outpath = path / "data" / "decoding" 

    bids_dir = Path("/work/816119/InSpePosNegData/BIDS_2023E")
    subjects = ["0116", "0117", "0118", "0119", "0120", "0121", "0122", "0123"]
    
    for subject in subjects:
        outpath_subject = outpath / f"sub-{subject}"
        
        # ensure outpath exists 
        if not outpath_subject.exists():
            outpath_subject.mkdir(exist_ok = True, parents = True)

        mask = nib.load(f"/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-{subject}/func/sub-{subject}_task-boldinnerspeech_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")
       
        flms, lsa_dms, events = fit_first_level_subject_per_trial(subject, bids_dir, mask = mask)

        for i, flm in enumerate(flms):
            R_face = []

            for col in events[i]['trial_type']:
                contrast = flm.compute_contrast(col, output_type = "effect_size")
                R_face.append(apply_mask(contrast, mask))
            
            R_face = np.vstack(R_face)

            # Do temporal uncorrelation
            X = lsa_dms[i].to_numpy()
            R_unc = sqrtm(X.T @ X) @ R_face

            # #There seems to be some outliers in the data that we need to filter out.
            R_unc[R_unc>5]=5
            R_unc[R_unc<-5]=-5

            # Unmask the 4D data
            contrasts_run = unmask(R_unc, mask)

            # Assuming you have a list named 'trial_types' with trial descriptions
            positive_indices = [i for i, trial_type in enumerate(events[i]['trial_type']) if 'positive' in trial_type]
            negative_indices = [i for i, trial_type in enumerate(events[i]['trial_type']) if 'negative' in trial_type]

            positive_contrasts_img = index_img(contrasts_run, positive_indices)
            negative_contrasts_img = index_img(contrasts_run, negative_indices)

            # save the conditions to a pickle
            file_name_pos = f"pos_contrasts_run_{i}.pkl"
            file_name_neg = f"neg_contrasts_run_{i}.pkl"

            pickle.dump(positive_contrasts_img, open(outpath_subject / file_name_pos, 'wb'))
            pickle.dump(negative_contrasts_img, open(outpath_subject / file_name_neg, 'wb'))
    
    print(f"{subject} done")