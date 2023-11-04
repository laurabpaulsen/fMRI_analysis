"""
This file holds the code for generating plots used for sanity checks.
"""
from pathlib import Path
from utils import load_first_level_models
from nilearn import plotting
import matplotlib.pyplot as plt

def plot_contrast_subject_level(flm, subject_id, ax = None, contrast = "button_press", output_type = "z_score"):
    """
    Calculates and plots the contrast for the given first level model.

    Parameters
    ----------
    flm : FirstLevelModel
        First level model.
    subject_id : stre
        Subject ID.
    ax : matplotlib.axes
        Axis to plot on. Defaults to None. If None, a new figure is created.
    contrast : str, optional
        Contrast to calculate and plot. The default is "button_press".
    
    Returns
    -------
    """

    contrast_map  = flm.compute_contrast(contrast, output_type = output_type)

    plotting.plot_glass_brain(
        contrast_map, 
        colorbar=True,
        title=f"Button press ({output_type}), Subject {subject_id}",
        plot_abs=False, 
        cmap='RdBu',
        axes=ax)

def plot_all_subjects(flms, subject_ids, save_path = None, contrast = "button_press", output_type = "z_score"):
    """
    Plots the contrast for all subjects in the given list of first level models.

    Parameters
    ----------
    flms : list
        List of first level models.
    subject_ids : list
        List of subject IDs.
    save_path : str, optional
        Path to save the figure to. The default is None.
    contrast : str, optional
        Contrast to calculate and plot. The default is "button_press".
    
    Returns
    -------
    """
    fig, axes = plt.subplots(2, int(len(flms)/2), figsize=(15, 5))
    
    for i, (flm, subject_id) in enumerate(zip(flms, subject_ids)):
        plot_contrast_subject_level(flm, subject_id, ax = axes.flatten()[i], contrast = contrast, output_type = output_type)

    if save_path:
        plt.savefig(save_path, dpi=300)


if __name__ in "__main__":
    flm_path = Path("/work/LauraBockPaulsen#1941/fMRI_analysis/flms")
    flms, subject_ids = load_first_level_models(flm_path, return_subject_ids = True)

    # plot button press contrast for all subjects
    plot_all_subjects(flms, subject_ids, save_path = "plots/button_press_contrast.png", contrast = "button_press", output_type = "z_score")
