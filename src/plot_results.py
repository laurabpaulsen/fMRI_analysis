import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from nilearn.plotting import plot_glass_brain, plot_stat_map
from nilearn.image import new_img_like, load_img


if __name__ in "__main__":
    path = Path(__file__).parents[1]

    results_path = path / "results"
    for subject in ["0116"]:
        # load results (for now just one subject)
        subject_path = results_path / f"sub-{subject}.pkl"
        
        with open(subject_path, "rb") as f:
            searchlight = pickle.load(f)

        mask_wb_filename=f'/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-{subject}/anat/sub-{subject}_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
        anat_filename=f'/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-{subject}/anat/sub-{subject}_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'

        #Create an image of the searchlight scores
        searchlight_img = new_img_like(anat_filename, searchlight.scores_)


        plot_glass_brain(searchlight_img, cmap='jet',colorbar=True, threshold=0.5,
                          title='Image pos vs Image neg (unthresholded)',
                          plot_abs=False)
        plt.savefig("hej1.png")

        plot_glass_brain(searchlight_img,cmap='prism',colorbar=True,threshold=0.60,title='pos vs neg (Acc>0.6')
        plt.savefig("hej2.png")

        plot_stat_map(searchlight_img, cmap='jet',threshold=0.6, cut_coords=[-30,-20,-10,0,10,20,30],
                    display_mode='z',  black_bg=False,
                    title='pos vs neg (Acc>0.6)')

        plt.savefig("hej3.png")

