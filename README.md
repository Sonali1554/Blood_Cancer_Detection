Blood Cancer Detection & Classification (Explainable AI)
Project Overview
This repository contains a comprehensive deep learning pipeline designed to detect and classify various types of blood cancer from microscopic cell imagery. Beyond just raw classification, this project integrates Explainable AI (XAI) techniques to provide visual justifications for model predictions, making it a viable tool for clinical decision support.

Key Features
Multi-Model Benchmarking: Implemented and compared several SOTA architectures including VGG16, VGG19, ResNet50, and DenseNet-121[cite: 26, 29].
High Precision: Achieved a peak testing accuracy of 96% using a fine-tuned VGG16 backbone[cite: 29].
Visual Interpretability: Integrated Grad-CAM (Gradient-weighted Class Activation Mapping) to highlight the specific regions of a cell that influenced the model's classification[cite: 26, 30].
Robust Evaluation: Benchmarked performance using precision, recall, and F1-score to ensure reliability in medical contexts[cite: 30].

Tech Stack
Deep Learning: PyTorch
Architectures: VGG16, VGG19, ResNet50, DenseNet-121.
Explainability: Grad-CAM
Machine Learning: XGBoost
Languages: Python.

The models were trained on a massive dataset of 25,000 images. Through transfer learning, the following results were achieved:

Model	Accuracy	Status
VGG16	96%	Peak Performance
ResNet50	Benchmarked	Supported
DenseNet-121	Benchmarked	Supported

🔬 Explainable AI (XAI) Implementation
In medical diagnostics, "Black Box" models are difficult to trust. This project solves this by using Grad-CAM to generate heatmaps on the original input images, providing visual explainability of CNN predictions

Repository Structure
├── data/               # Dataset directory (25,000 images)
├── notebooks/          # Training and Grad-CAM visualization notebooks
├── models/             # Saved model weights (.pth)
├── src/
│   ├── preprocess.py   # Image augmentation and normalization
│   ├── train.py        # Model training and transfer learning script
│   └── explain.py      # Grad-CAM heatmap generation
└── requirements.txt    # Project dependencies

