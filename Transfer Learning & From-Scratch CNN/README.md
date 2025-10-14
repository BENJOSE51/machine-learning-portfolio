# CNN Image Classification — CIFAR-10

## Overview
This project builds a **Convolutional Neural Network (CNN)** to classify images from the CIFAR-10 dataset into 10 distinct categories such as airplanes, automobiles, birds, and cats.  
The model was implemented using **TensorFlow/Keras** and trained on the 60,000-image CIFAR-10 dataset.

## Contents
- `notebooks/` — cleaned notebook: `05_CNN_Image_Classification_Clean.ipynb`
- `data/` — CIFAR-10 dataset (automatically downloaded via Keras)
- `models/` — optional directory for saving trained models (e.g., `cnn_cifar10_model.h5`)

## Project Highlights
- Implemented a CNN architecture with Conv2D, MaxPooling, Flatten, and Dense layers.
- Used ReLU activation and Softmax for multiclass classification.
- Evaluated model performance on validation/test data with accuracy and loss metrics.
- Visualized model accuracy/loss trends across epochs.

## Key Results
| Metric | Value (approx.) |
|---------|-----------------|
| **Training Accuracy** | ~85–90% |
| **Test Accuracy** | ~80–85% |
| **Loss Trend** | Decreasing steadily over epochs |

## Possible Improvements
- Add data augmentation (random flips, rotations, shifts) to improve generalization.
- Introduce Dropout or Batch Normalization layers to reduce overfitting.
- Experiment with deeper architectures (ResNet, VGG, or MobileNet).

## Next Steps
- Save the trained model (`cnn_cifar10_model.h5`).
- Build a **Streamlit app** for real-time image uploads and classification.
- Optionally deploy the model as a **FastAPI endpoint**.

## Tech Stack
Python, TensorFlow/Keras, NumPy, Matplotlib

---
_This project is part of the Machine Learning Portfolio — Project 05: CNN Image Classification._
