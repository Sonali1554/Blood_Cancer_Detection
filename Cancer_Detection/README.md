# Computer Vision Project
Detection and Classification of Blood Cancer Using CNNs, Aided by Explainable AI

1. Project Overview
This project presents an AI-based system for detecting and classifying PML-RARA blood cancer using Convolutional Neural Networks (CNNs) and Explainable AI. A dataset of 25,000 microscopic blood cell images categorized into Infected and Uninfected classes is used to train and compare five models: VGG16, VGG19, ResNet50, DenseNet121, and XGBoost.
Transfer learning is applied for CNN models, and Explainable AI using GradCAM highlights the image regions influencing predictions. The highest accuracy model is VGG16 (96%), demonstrating excellent performance for medical diagnostics and early detection.

2. Repository Structure
├── data/
│   ├── infected/
│   └── uninfected/
├── models/
│   ├── vgg16.pth
│   ├── resnet50.pth
│   ├── densenet121.pth
│   └── xgboost_cancer_model.json
├── src/
│   ├── preprocess.py
│   ├── train_vgg16.py
│   ├── train_resnet50.py
│   ├── train_densenet.py
│   ├── train_xgboost.py
│   ├── predict.py
│   ├── gradcam.py
│   └── utils.py
├── results/
│   ├── confusion_matrices/
│   ├── roc_curves/
│   └── gradcam_visuals/
├── README.md
└── requirements.txt

4. Installation
Install Dependencies
pip install -r requirements.txt

Dataset Structure
data/
 ├── infected/
 └── uninfected/

4. Data Preprocessing
Resize images to 224 × 224
Normalize pixel values
Convert to tensors
80% training, 20% validation split
Additional 200-image test set

Torch transform:
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

5. Models Implemented
VGG16 (Top performer)

Pretrained ImageNet model
Modified final classifier
Achieved 96% accuracy

Other Models
VGG19 – deeper, but lower accuracy due to overfitting
ResNet50 – skip connections, 90% accuracy
DenseNet121 – dense connectivity, 90% accuracy
XGBoost – uses ResNet features, 87% accuracy

6. Training the Models
Train VGG16
python src/train_vgg16.py
Train ResNet50
python src/train_resnet50.py
Train DenseNet121
python src/train_densenet.py
Train XGBoost
python src/train_xgboost.py

7. Running Predictions
python src/predict.py --image sample.jpg --model models/vgg16.pth


Outputs:

Class label
Confidence
Optional GradCAM heatmap

8. Explainable AI (GradCAM)
Generate heatmap:
python src/gradcam.py --image sample.jpg --model models/vgg16.pth
Visualizations saved in:
results/gradcam_visuals/
GradCAM shows which image regions influence classification — essential for medical trust and interpretability.

9. Performance Evaluation
Final Metrics Summary
Model	Accuracy	Precision	Recall	F1-Score
VGG16	96%	96%	96%	96%
VGG19	83%	83.5%	83%	83%
ResNet50	90%	90%	88%	90%
DenseNet121	90%	91%	90%	90%
XGBoost	87%	87%	87%	87%
Validation Accuracy

VGG16: 96.8%
DenseNet121: 84.8%
ResNet50: 82.1%
XGBoost: 80.3%

10. Visual Outputs

Outputs are stored in the results folder:
Confusion Matrices
results/confusion_matrices/
ROC Curves
results/roc_curves/
GradCAM Heatmaps
results/gradcam_visuals/

11. Real-World Applications
Early blood cancer screening
Clinical diagnostic support
Telemedicine
Pathology image analysis
Medical student training
Research in Explainable AI

12. Limitations
No user interface
CPU-only training → slower
Dataset covers only two classes
Not deployed in real-time clinical systems

13. Future Enhancements
Web UI with Streamlit/Flask
GPU-based cloud training
Detection of multiple leukemia subtypes
Add SHAP explainability
Mobile/edge deployment

Real-time hospital integration via API



