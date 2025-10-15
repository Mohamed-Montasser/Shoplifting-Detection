# Shoplifting Detection using CNN-LSTM

A deep learning-based video classification system for detecting shoplifting behavior in surveillance footage using a custom CNN-LSTM architecture built from scratch in PyTorch.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [License](#license)

## 🎯 Overview

This project implements a video classification model that can automatically detect shoplifting behavior from surveillance camera footage. The model combines Convolutional Neural Networks (CNN) for spatial feature extraction and Long Short-Term Memory (LSTM) networks for temporal pattern recognition.

## ✨ Features

- **Custom CNN-LSTM Architecture**: Built from scratch without pretrained models
- **Temporal Analysis**: Processes 16 uniformly sampled frames per video
- **Class Imbalance Handling**: Implements class weighting and balanced loss functions
- **Data Augmentation**: Random flips, rotations, and color jittering for better generalization
- **GPU Acceleration**: Full CUDA support for faster training
- **Deployment Ready**: Compatible with Streamlit and Flask for easy deployment

## 🏗️ Architecture

The model consists of three main components:

### 1. CNN Feature Extractor
- 4 convolutional blocks with batch normalization
- Progressive channel expansion (64 → 128 → 256 → 512)
- Max pooling for spatial dimension reduction
- Adaptive average pooling for flexible input sizes

### 2. Bidirectional LSTM
- 2-layer bidirectional LSTM
- Hidden size: 256
- Captures temporal dependencies across video frames

### 3. Classifier Head
- Fully connected layers with dropout (0.5)
- ReLU activation
- Binary classification output (Shoplifting vs Normal)

**Model Parameters**: ~15-20 million trainable parameters

## 📊 Dataset

- **Classes**: 2 (Non-Shoplifter, Shoplifter)
- **Train/Val/Test Split**: 80/10/10
- **Class Distribution**: 
  - Non-Shoplifter: ~62%
  - Shoplifter: ~38%
- **Video Processing**: 16 frames per video, resized to 112×112 pixels
- **Stratified Splitting**: Maintains class balance across all splits

## 🔧 Requirements

```
torch>=1.9.0
torchvision>=0.10.0
opencv-python-headless>=4.5.0
numpy>=1.19.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
tqdm>=4.60.0
```

## 📈 Model Performance

### Confusion Matrix
![Confusion Matrix](assets/confusion_matrix.png)

### Training Curves
![Training Curves](assets/training_curves.png)


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
