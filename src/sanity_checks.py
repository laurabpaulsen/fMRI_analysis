"""
This file holds the code for generating plots used for sanity checks.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils import load_first_level_models, plot_contrast_all_subjects

if __name__ in "__main__":
    path = Path(__file__).parents[1]
    flm_path = path / "data" / "flms"
    flms, subject_ids = load_first_level_models(flm_path, return_subject_ids = True)
    output_path = path / "fig" / "sanity_checks"

    if not output_path.exists():
        output_path.mkdir(parents = True)
    """
    # plot button press contrast for all subjects
    plot_contrast_all_subjects(
        flms, subject_ids, 
        threshold=True,
        save_path = output_path / "button_press_contrast.png", 
        contrast = "button_press", 
        output_type = "z_score"
        )
    """
    subjects = ["0116", "0117", "0118", "0119", "0120", "0121", "0122", "0123"]

    bids_dir = Path("/work/816119/InSpePosNegData/BIDS_2023E")


    fig, axes = plt.subplots(2,4, figsize = (20, 10))

    for i, subject in enumerate(subjects):
        ax = axes.flatten()[i]
        bids_func_dir  = bids_dir / f"sub-{subject}" / "func"
        # load tsv
        tsvs = [bids_func_dir / f"sub-{subject}_task-boldinnerspeech_run-{run}_echo-1_events.tsv" for run in [1, 2, 3, 4, 5, 6]]
        
        dfs = []
        for tsv in tsvs:
            dfs.append(pd.read_csv(tsv, sep = "\t"))

        # exclude rows with no RT
        dfs = [df[df["RT"] > 0] for df in dfs]
        
        # make a bar plot with the number of IMG_BI events per run
        img_bi = [df[df["trial_type"] == "IMG_BI"].shape[0] for df in dfs]

        # plot the bar plot
        ax.bar(range(1,7), img_bi)

    
    plt.savefig(output_path / "img_bi_per_run.png")
        



