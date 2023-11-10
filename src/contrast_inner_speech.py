"""
This file holds the code for contrasting positive and negative inner speech
"""
from pathlib import Path

# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils import load_first_level_models, plot_contrast_all_subjects

if __name__ in "__main__":
    path = Path(__file__).parents[1]
    flm_path = flm_path = path / "data" / "flms"
    flms, subject_ids = load_first_level_models(flm_path, return_subject_ids = True)
    output_path = path / "fig" / "contrasts"

    if not output_path.exists():
        output_path.mkdir(parents = True)

    # plot button press contrast for all subjects
    plot_contrast_all_subjects(
        flms, subject_ids, 
        threshold=True,
        save_path = output_path / "pos_neg_contrast.png", 
        contrast = "positive - negative", 
        output_type = "z_score"
        )
