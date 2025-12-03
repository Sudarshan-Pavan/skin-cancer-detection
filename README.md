Skin Lesion Classification using Deep Learning (HAM10000 Dataset)

A complete deep-learning pipeline for multi-class skin-lesion classification using the HAM10000 dataset, featuring extensive model benchmarking and a high-performance custom CNN that significantly outperforms several state-of-the-art pretrained architectures.

📌 Project Overview

This project focuses on automatic classification of skin lesions into seven diagnostic categories using deep learning. Multiple transfer-learning models were evaluated to determine their performance on the HAM10000 dataset, followed by the development of a custom convolutional neural network that achieved superior accuracy.

The goal is to build an efficient, lightweight, high-accuracy model suitable for real-world medical screening applications.

📊 Dataset Information

Dataset: HAM10000 (Human Against Machine with 10,000 training images)

Original size: 10,015 dermoscopic images

Classes: 7 (e.g., MEL, NV, BKL, BCC, AKIEC, DF, VASC)

Balanced dataset: Expanded to ~45,000+ images through augmentation

Image resolution used: 28×28×3 for efficiency-optimized model training

🧠 Models Implemented
Transfer Learning Architectures

The following pretrained models (ImageNet weights) were trained and evaluated:

XceptionNet

ShuffleNet

ResNet-50

MobileNetV2

EfficientNet-B0

DenseNet-121

Each model was fine-tuned with custom classification heads and evaluated across three data splits: 80–20, 70–30, 60–40.

Custom CNN Architecture

A purpose-built deep convolutional network optimized for:

Small-resolution images

Balanced depth and parameter count

Efficient feature extraction

High accuracy on multi-class classification

Architecture Summary:

4 Convolutional blocks (16 → 32 → 64 → 128 filters)

MaxPooling after each block

Flatten → Dense(64) → Dense(32)

Final Dense(7) with Softmax

~2.1M total parameters

Optimizer: Adam with tunable learning rate

Loss: Sparse Categorical Crossentropy

📈 Model Performance
Test Accuracy Across Models
Model	80–20 Split	70–30 Split	60–40 Split
XceptionNet	46.66%	46.70%	45.82%
ShuffleNet	58.24%	55.44%	57.98%
ResNet-50	25.31%	24.82%	24.69%
MobileNetV2	49.37%	47.59%	46.41%
EfficientNet-B0	14.29%	14.29%	14.29%
DenseNet-121	52.14%	51.67%	50.82%
➡️ Custom CNN	99.2%	98.78%	98.42%
Key Achievement

The custom CNN outperformed all pretrained architectures by a large margin, achieving ~99% accuracy, demonstrating that a specialized architecture can exceed transfer learning on small, highly augmented datasets.

🛠️ Technologies & Tools

Languages:

Python 3.x

Frameworks / Libraries:

TensorFlow / Keras

NumPy

OpenCV

Matplotlib / Seaborn

Scikit-learn

Environment:

Jupyter Notebook / Google Colab

🚀 Features of This Project

Full deep-learning pipeline (preprocessing → training → evaluation)

Dataset balancing and heavy augmentation

Benchmarking of multiple state-of-the-art architectures

Custom model creation and hyperparameter tuning

Confusion matrices, metrics, and split-wise stability checks

Structure suitable for reproducible research

Ready for deployment upgrades (TFLite / ONNX)

📌 Future Enhancements

Continual Learning integration for real-time adaptation

Deploying lightweight model for mobile/edge devices

Improved augmentation tailored to dermoscopic patterns

Ensemble methods for robust clinical prediction

Grad-CAM visualization for explainability
