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

    # plot button press contrast for all subjects
    plot_contrast_all_subjects(
        flms, subject_ids, 
        threshold=True,
        save_path = output_path / "button_press_contrast.png", 
        contrast = "button_press", 
        output_type = "z_score"
        )

    subject = "0119"
    # plot button presses for subject 0119
    # load tsv
    tsvs = [path / "events" / f"sub-{subject}_task-boldinnerspeech_run-{run}_echo-1_events.tsv" for run in [1, 2, 3, 4, 5, 6]]

    dfs = [pd.read_csv(tsv, sep = "\t") for tsv in tsvs]

    dfs = [df[df["trial_type"] == "IMG_BI"] for df in dfs]

    # plot button presses
    fig, axes = plt.subplots(2, len(dfs) // 2, figsize = (20, 10))

    for i, ax in enumerate(axes.flatten()):
        df = dfs[i]
        ax.scatter(df["onset"], df["duration"])
        ax.set_title(f"Run {i + 1}")


    plt.savefig(output_path / f"button_presses_{subject}.png")
