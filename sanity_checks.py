"""
This file holds the code for generating plots used for sanity checks.
"""
from pathlib import Path
from utils import load_first_level_models, plot_contrast_all_subjects



if __name__ in "__main__":
    path = Path(__file__).parent
    flm_path = Path("/work/LauraBockPaulsen#1941/fMRI_analysis/flms")
    flms, subject_ids = load_first_level_models(flm_path, return_subject_ids = True)
    output_path = path / "fig" / "sanity_checks"

    if not output_path.exists():
        output_path.mkdir(parents = True)

    # plot button press contrast for all subjects
    plot_contrast_all_subjects(
        flms, subject_ids, 
        threshold=True,
        save_path = output_path / "button_press_contrast.png", 
        contrast = "button_press", 
        output_type = "z_score"
        )
