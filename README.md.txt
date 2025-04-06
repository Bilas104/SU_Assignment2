# Speech Understanding - Assignment 2

## Overview
This repository contains solutions for Assignment 2, which involves tasks on speech enhancement, speaker verification, and MFCC feature extraction for Indian languages. The implementation uses Python and PyTorch for modeling and analysis.

## Files
- `SU2_Q1_PartI.ipynb`: Implements the first part of Question 1, including dataset preparation, speaker verification using pre-trained models, and fine-tuning with LoRA and ArcFace loss.
- `SU2_Q1_PartII.ipynb`: Continues Question 1 by creating multi-speaker scenarios, performing speaker separation using SepFormer, and designing a novel pipeline for speech enhancement and speaker identification.
- `su2_q2 (1).ipynb`: Implements Question 2, focusing on MFCC feature extraction and classification of Indian languages. It includes:
  - MFCC extraction from audio samples.
  - Visualization of MFCC spectrograms for selected languages.
  - Statistical analysis of MFCC features across languages.
  - A classifier to predict the language of an audio sample using extracted MFCC features.

## Requirements
- Python 3.x
- Libraries: numpy, pandas, matplotlib, seaborn, librosa, scikit-learn, torch, torchaudio

## Running the Code
1. Install the required libraries using `pip install -r requirements.txt`.
2. Execute the notebooks in order:
   - For Question 1: Run `SU2_Q1_PartI.ipynb` followed by `SU2_Q1_PartII.ipynb`.
   - For Question 2: Run `su2_q2 (1).ipynb`.

## Results
### Question 1
- Speaker verification performance metrics such as EER (Equal Error Rate) and TAR@1%FAR are reported for both pre-trained and fine-tuned models.

### Question 2
- MFCC spectrograms are visualized for three Indian languages (e.g., Hindi, Bengali, Urdu).
- Statistical differences in MFCC features across languages are analyzed.
- A language classifier achieves high accuracy in predicting the language of an audio sample.

This README provides an overview of the files and instructions to run the code for Assignment 2.
