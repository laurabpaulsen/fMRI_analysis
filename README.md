# fMRI_analysis
This repository holds the code for the fMRI analysis portfolio assignment for the Advanced Cognitive Neuroscience course (F2023). 



## Pipeline
To reproduce the results, run the following commands from the root of the repository:

Create a virtual environment with the required packages:
```
bash setup_env.sh
```

Activate the virtual environment:
```
source env/bin/activate
```

Fit the first level models
```
python src/fit_first_level.py
```

Generate plots for sanity checks
```
python src/sanity_checks.py
```

Calculate and plot the contrast between positive and negative inner speech
```
python src/contrast_inner_speech.py # NOT IMPLEMENTED YET
```

Run decoding analysis
```
python src/decoding.py # NOT IMPLEMENTED YET
```

## Project organisation

## Collaborators
The project was done by study group 8 consisting of:
- [Pernille](https://github.com/PernilleBrams)
- [Luke](https://github.com/zeyus)
- [Aleksandrs](https://github.com/sashapustota)
- [Christoffer](https://github.com/clandberger)
- [Laura](https://github.com/laurabpaulsen)