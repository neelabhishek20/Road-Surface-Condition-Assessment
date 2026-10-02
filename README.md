# AI-Based Road Surface Condition Assessment Using Computer Vision

A computer vision system that analyzes road images and classifies road surface damage into three categories: **Crack, Pothole, and Surface Erosion**.

The project uses **ResNet18 transfer learning and fine-tuning** to perform road surface condition classification. It also provides prediction confidence and **Grad-CAM explainability**.

---

## Problem Statement

Road damage such as cracks, potholes, and surface erosion can affect road safety, driving comfort, and maintenance planning. This project explores an automated image-based approach for identifying different types of road surface damage.

---

## Project Objectives

- Classify road surface damage into three categories.
- Build a deep learning image classification model using ResNet18.
- Improve the baseline model using fine-tuning.
- Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
- Predict the condition of new road images.
- Use Grad-CAM to visualize regions influencing the model's prediction.

---

## Dataset

The dataset contains **1,530 RGB road images** belonging to three classes.

| Class | Images |
|---|---:|
| Crack | 478 |
| Pothole | 401 |
| Surface Erosion | 651 |
| **Total** | **1530** |

### Dataset Split

| Split | Images |
|---|---:|
| Training | 1069 |
| Validation | 230 |
| Testing | 231 |

The original dataset was kept unchanged.

---

## Methodology

```text
Road Image
     ↓
Image Preprocessing
     ↓
Data Augmentation
     ↓
ResNet18
     ↓
Transfer Learning
     ↓
Fine-Tuning
     ↓
Road Condition Classification
     ↓
Confidence Score
     ↓
Grad-CAM Explainability
Model Development
V1 - Baseline Model

A pretrained ResNet18 model was used as the baseline. The backbone was initially frozen and the final classification layer was trained for the three road-condition classes.

V1 Test Accuracy: 70.13%

V2 - Fine-Tuned Model

For V2, the deeper layer4 of ResNet18 and the final fully connected classification layer were fine-tuned for the road damage dataset.

V2 Test Accuracy: 85.71%

V1 vs V2
Metric	V1	V2
Test Accuracy	70.13%	85.71%
Macro F1 Score	0.6902	0.8561
V2 Results
Class	Precision	Recall	F1-Score
Crack	0.9130	0.8750	0.8936
Pothole	0.8868	0.7705	0.8246
Surface Erosion	0.8073	0.8980	0.8502
Confusion Matrix

Sample Prediction

The trained V2 model was tested on a road image.

Predicted Condition: Crack

Confidence: 76.60%

Class Probabilities
Class	Probability
Crack	76.60%
Pothole	11.99%
Surface Erosion	11.41%
Input Image

Grad-CAM Explainability

Grad-CAM was implemented to visualize the regions of the image that contributed to the model's prediction.

The Grad-CAM output provides a visual explanation alongside the model prediction.

Technologies Used
Python
PyTorch
Torchvision
ResNet18
NumPy
Matplotlib
Scikit-learn
Pillow