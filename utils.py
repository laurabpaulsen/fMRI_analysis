from pathlib import Path
import pickle

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