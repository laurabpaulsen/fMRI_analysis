from pathlib import Path
import pickle
from nilearn import plotting
import matplotlib.pyplot as plt
from nilearn.glm import threshold_stats_img
from scipy.stats import binom

def chance_level(n, alpha = 0.001, p = 0.5):
    """
    THIS FUNCTION WAS COPIED DIRECTLY FROM https://github.com/laurabpaulsen/DecodingMagneto/blob/main/utils/analysis/tools.py
    Calculates the chance level for a given number of trials and alpha level

    Parameters
    ----------
    n : int
        The number of trials.
    alpha : float
        The alpha level.
    p : float
        The probability of a correct response.

    Returns
    -------
    chance_level : float
        The chance level.
    """
    k = binom.ppf(1-alpha, n, p)
    chance_level = k/n
    
    return chance_level

def load_first_level_models(path: Path, return_subject_ids = False):
    """
    Loads the first level models from the given path. Assumes that the models are pickled and no other files are present in the directory.

    Parameters
    ----------
    path : Path
        Path to the directory containing the first level models.
    
    Returns
    -------
    flms : list
        List of first level models.
    """
    flm_files = [f for f in path.iterdir() if f.is_file() and f.suffix == ".pkl"]
    flm_files.sort()

    flms = []

    for model in flm_files: # looping over the flms and loading them
        flm_model = pickle.load(open(model,'rb')) 
        flms.append(flm_model)
    
    if return_subject_ids:
        subject_ids = [path.stem[-4:] for path in flm_files]
        return flms, subject_ids
    else:
        return flms



# ----------------- PLOTTING FUNCTIONS ----------------- #


def sanitise_contrast(contrast:str):
    """
    Sanitises the contrast name for plotting
    """
    contrast = contrast.replace("_", " ")
    contrast = contrast.title()

    return contrast


def plot_contrast_subject_level(flm, subject_id, ax, threshold = False, contrast = "button_press", output_type = "z_score"):
    """
    Calculates and plots the contrast for the given first level model.

    Parameters
    ----------
    flm : FirstLevelModel
        First level model.
    subject_id : stre
        Subject ID.
    threshold : bool, optional
        if True, a bonferroni corrected threshold is applied. The default is False.
    ax : matplotlib.axes
        Axis to plot on. Defaults to None. If None, a new figure is created.
    contrast : str, optional
        Contrast to calculate and plot. The default is "button_press".
    
    Returns
    -------
    None.

    """

    contrast_map  = flm.compute_contrast(contrast, output_type = output_type)

    if threshold:
        contrast_map, threshold = threshold_stats_img(
            contrast_map, 
            alpha=0.05, 
            height_control='bonferroni')

    plotting.plot_glass_brain(
        contrast_map,
        colorbar=True,
        plot_abs=False, 
        cmap='RdBu',
        axes=ax)
    
    ax.set_title(f"Subject {subject_id}")

def plot_contrast_all_subjects(flms, subject_ids, threshold = False, save_path = None, contrast = "button_press", output_type = "z_score"):
    """
    Plots a given contrast for all subjects in the given list of first level models.

    Parameters
    ----------
    flms : list
        List of first level models.
    subject_ids : list
        List of subject IDs.
    threshold : bool
        if True, a bonferroni corrected threshold is applied. The default is False.
    save_path : str, optional
        Path to save the figure to. The default is None.
    contrast : str, optional
        Contrast to calculate and plot. The default is "button_press".
    
    Returns
    -------
    None.
    """
    fig, axes = plt.subplots(int(len(flms)/2), 2, figsize=(10, 12))
    
    for i, (flm, subject_id) in enumerate(zip(flms, subject_ids)):
        ax = axes.flatten()[i]
        plot_contrast_subject_level(flm, subject_id, ax, threshold, contrast = contrast, output_type = output_type)

    # add super title in bold
    fig.suptitle(f"{sanitise_contrast(contrast)}", fontweight="bold", fontsize=20)

    if save_path:
        plt.savefig(save_path, dpi=300)