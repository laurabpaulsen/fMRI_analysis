"""
Code inspired by Emma Olsen and Sirid Wihlborg (https://github.com/emmarisgaardolsen/BSc_project_fMRI/blob/main/fmri_analysis_scripts/second_level.ipynb)
"""

from pathlib import Path
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
from nilearn import plotting
import matplotlib.pyplot as plt

# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils import load_first_level_models, sanitise_contrast



if __name__ in "__main__":
    path = Path(__file__).parents[1]

    path_flms = path / "flms"
    flms, subject_ids = load_first_level_models(path_flms, return_subject_ids = True)

    slm = SecondLevelModel(smoothing_fwhm=8.0)
    slm.fit(flms)
    
    for contrast in ['positive-negative', 'button_press']:

        zmap = slm.compute_contrast(first_level_contrast=contrast, output_type='z_score')

        thresholded_map, threshold = threshold_stats_img(
            zmap, 
            alpha=.05, 
            height_control='bonferroni'
            ) 

        fig, ax = plt.subplots(1,1, figsize = (10, 12), dpi = 300)

        plotting.plot_glass_brain(
            thresholded_map, 
            colorbar=True,
            plot_abs=False, 
            ax = ax, 
            cmap='RdBu')

        ax.set_title(f"{sanitise_contrast(contrast)} (Bonferroni, alpha < 0.05)")

        plt.savefig(path / fig / f"{contrast}_across_subjects.png")
