# TensorFlow Model for Breast Cancer Detection with Data Leakage Analysis

This repository contains a TensorFlow implementation for detecting breast fibroadenoma from ultrasound images, showcasing the journey through data leakage identification to achieving a 43% accuracy rate on the corrected dataset. For those interested in the improved model using PyTorch with an 84% accuracy rate, please visit the [PyTorch Model Repository](https://github.com/ErikaMolnar/ultrasound_classification).

## Overview

### Dataset
The model utilizes a dataset comprising ultrasound images of breast cancer, categorized into benign, malignant, and normal cases:
- **Benign Images:** 437
- **Malignant Images:** 210
- **Normal Images:** 133

### Approach
- **Transfer Learning:** Applied to leverage knowledge from pre-trained models.
- **Data Augmentation:** Included techniques like random affine transformations and color jittering to mitigate the effects of a small and imbalanced dataset.

### Data Leakage Analysis
A critical phase in this project involved identifying and rectifying data leakage, which initially led to an inflated accuracy of 89%. The analysis notebook details the steps taken to uncover and correct this issue, providing valuable insights into ensuring the integrity of machine learning models.

## TensorFlow Model
After addressing the data leakage, the TensorFlow model's performance settled at 43% accuracy. This section of the repo includes:
- The model implementation code.
- The data preparation and augmentation scripts.
- Hyperparameter tuning and regularization techniques were applied in an attempt to improve performance.

## Why 43% Accuracy?
The detailed exploration into the causes of the reduced accuracy post-data leakage correction is documented, along with the limitations encountered during the TensorFlow model's development.

## Transition to PyTorch
Encouraged by the need for improved performance, a subsequent model was developed using PyTorch, achieving a remarkable 84% accuracy. For complete details on the PyTorch model, please refer to the [PyTorch Model Repository](https://github.com/ErikaMolnar/ultrasound_classification).

## Conclusion
This repository not only documents the technical journey of developing a machine learning model for a critical healthcare application but also emphasizes the importance of vigilance against data leakage, illustrating the stark differences in outcomes between TensorFlow and PyTorch implementations.

