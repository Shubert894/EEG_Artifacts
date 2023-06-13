# Artifact Detection and Removal

This repository contains the code for the paper: 
Artifact Detection and Removal in Single Channel EEG
(by Nichita Motoc)

# Structure
- `recordings` contains some EEG recordings collected from Curvex EEG headset (Fp1) for the experiments.
- `models` contains the models for the support vector classifier used in detection
- `detection.py` contains the code used to detect EEG artifacts (SWT + Statistical features + SVC)
- `correction.py` contains the code used to correct EEG artifacts (Wavelet Thresholding, Wavelet Quantile Normalization)
- `helpers.py` contains utility functions and classes
- `main.ipynb` combines the detection and correction