# EEG_Classification  
A deep-learning pipeline for classifying EEG signals (especially SSVEP) using PyTorch Lightning.  

## Project Description  
This repository implements a complete workflow for EEG-based classification, including:  
- Data segmentation (EEG windows), filtering, optionally transforming via FFT/Mel/STFT  
- Custom PyTorch `Dataset` classes to load and preprocess EEG windows  
- Multiple deep-learning model architectures (1D CNN/TCN for raw EEG, 2D CNN for spectrograms)  
- A training and evaluation pipeline using PyTorch Lightning (with checkpointing, logging, mixed precision)  
- Test-time evaluation including confusion matrix, per-class accuracy, and visualization  

The goal is to provide efficient models suitable both for offline experimentation and potential embedded/real-time BCI scenarios.  

## Performance

| Scenario                                           | Test Accuracy |
|----------------------------------------------------|--------------|
| 10 SSVEP frequencies                                | 0.801        |
| 4 SSVEP frequencies + “no-signal” class             | 0.817        |
| 4 SSVEP frequencies only                            | 0.970        |

## Repository Structure  

```bash
EEG_Classification/
│
├── Result_presentation.ipynb # to present results and performance of differnet ML models
│
├── Main.ipynb # Main Notebook to train, test and visualise data
│
├── Base_Model.py # includes Base Model for all other Models
├── Models_1D.py # 1D models (TCN, CNN, EEGNet variants)
├── Models_2D.py # 2D models for spectrogram inputs
│
├── Dataset_torch # Dataset class for EEG Data, inclused notch filtering and normalisation
│
├── Model_Trainer.py # Main training/validation/testing script/pipeline
│
├── Utils.py/
│ ├── plot_training_metrics.py # Utility to plot loss, accuracy, confusion matrix
│ ├── get_eeg_data_segmented.py # Windowing functions for EEG data
│ └── load_and_concat_ssvep_datasets.py # Load and combine multiple SSVEP EEG datasets from CSV files
│
├── Preprocessing.ipynb # convert raw data to csv, analyse, process and save data
│
├── datasets/
│ └── numpy/ # Pre-segmented .npz EEG datasets (X, y) (not on github due to data privacy)
│
├── logs/ # Trainer output: versioned logs and checkpoint folders
│
└── README.md # This file
```

## Dependencies


Install required packages:
```bash
pip install torch torchaudio pytorch-lightning numpy scipy matplotlib
```

## Use Case for BCI & Drone Control

While the core pipeline focuses on EEG classification, the models and architecture are designed for low-latency, lightweight deployment (e.g. embedded systems or drones).
For example, you can integrate the trained model output with a real-time BCI controlling a drone (e.g. using Crazyflie 2.1+) where each detected frequency class maps to a flight command.