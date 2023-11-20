import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from nilearn.plotting import plot_glass_brain
from nilearn.image import new_img_like
from nilearn.image import threshold_img

# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils import chance_level

def plot_searchlight_subject(searchlight_img, ax, threshold = False, title = None, **kwargs):
    """
    Calculates and plots the contrast for the given first level model.

    Parameters
    ----------
    searchlight_img : 
        The results of the searchlight analysis
    threshold : int
        The chance level threshold
    ax : matplotlib.axes
        Axis to plot on. Defaults to None. If None, a new figure is created.
    title : str
        Title of the ax. Defaults to None.
    **kwargs : dict
        Additional arguments to pass to the plotting function.
    

    Returns
    -------
    None.

    """

    plot_glass_brain(
        searchlight_img,
        colorbar=True,
        cmap='jet',
        threshold = threshold,
        axes=ax,
        **kwargs)
    
    if title:
        ax.set_title(title)


def plot_searchlight_all_subjects(search_light_imgs, subject_ids, threshold = None, save_path = None, **kwargs):
    """
    Plots a given contrast for all subjects in the given list of first level models.

    Parameters
    ----------
    search_light_imgs : list
        List of searchlight images
    subject_ids : list
        List of subject IDs.
    threshold : int
        The chance level accuracy
    save_path : str, optional
        Path to save the figure to. The default is None.
    
    Returns
    -------
    None.
    """
    fig, axes = plt.subplots(int(len(search_light_imgs)/2), 2, figsize=(10, 12))
    
    for i, (searchlight_img, subject_id) in enumerate(zip(search_light_imgs, subject_ids)):
        ax = axes.flatten()[i]
        plot_searchlight_subject(searchlight_img, ax, threshold = threshold, title = f"Subject {subject_id}", **kwargs)

    # add super title in bold
    fig.suptitle(f"Searchlight", fontweight="bold", fontsize=20)

    if save_path:
        plt.savefig(save_path, dpi=300)


if __name__ in "__main__":
    path = Path(__file__).parents[1]
    results_path = path / "data" / "searchlight"
    fig_path = path / "fig" / "searchlight"

    # ensure results path exists
    if not fig_path.exists():
        fig_path.mkdir(exist_ok = True)

    chance = chance_level(n = 60*6, alpha = 0.001, p = 0.5)


    subjects = ["0116", "0117", "0118", "0119", "0120", "0121", "0122", "0123"]
    search_lights = []

    for subject in subjects:
        # load results (for now just one subject)
        subject_path = results_path / f"sub-{subject}.pkl"
        
        with open(subject_path, "rb") as f:
            searchlight = pickle.load(f)

        mask_wb_filename=f'/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-{subject}/anat/sub-{subject}_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
        anat_filename=f'/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-{subject}/anat/sub-{subject}_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'
        n_voxels = searchlight.scores_.shape[0]
        
        #Create an image of the searchlight scores
        searchlight_img = new_img_like(anat_filename, searchlight.scores_)

        search_lights.append(searchlight_img)

    plot_searchlight_all_subjects(
        search_lights, subjects, 
        threshold = chance, 
        save_path = fig_path / "search_light_all_subjects_glass_brain.png", 
        plot_abs = False
        )
