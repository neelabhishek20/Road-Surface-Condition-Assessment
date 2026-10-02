from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = PROJECT_ROOT / "data" / "test"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "road_condition_resnet18_v2.pth"
)

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Test Transform
# ============================================================

test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Load Test Dataset
# ============================================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transforms
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

class_names = test_dataset.classes
num_classes = len(class_names)


# ============================================================
# Header
# ============================================================

print("=" * 70)
print("ROAD SURFACE CONDITION - V2 TEST EVALUATION")
print("=" * 70)

print(f"Device       : {DEVICE}")
print(f"Test images  : {len(test_dataset)}")

print("\nClasses:")

for index, name in enumerate(class_names):
    print(f"{index}: {name}")


# ============================================================
# Check Model File
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nV2 model not found:\n{MODEL_PATH}\n\n"
        "Make sure Model V2 training completed successfully."
    )


# ============================================================
# Build ResNet18
# ============================================================

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)


# ============================================================
# Load V2 Model
# ============================================================

print("\nLoading V2 model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model = model.to(DEVICE)

model.eval()

print("V2 model loaded successfully.")


# ============================================================
# Test Prediction
# ============================================================

all_predictions = []
all_labels = []

correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# ============================================================
# Accuracy
# ============================================================

test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)


print("\n" + "=" * 70)
print("V2 TEST RESULTS")
print("=" * 70)

print(
    f"\nTest Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# Classification Report
# ============================================================

print("\n" + "=" * 70)
print("V2 CLASSIFICATION REPORT")
print("=" * 70)


report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


print("\n" + "=" * 70)
print("V2 CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ============================================================
# Plot Confusion Matrix
# ============================================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(
    figsize=(8, 7)
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

plt.title(
    "Road Surface Condition - V2 Confusion Matrix"
)

plt.tight_layout()


cm_path = (
    RESULTS_DIR
    / "v2_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Save Metrics
# ============================================================

metrics_path = (
    RESULTS_DIR
    / "v2_evaluation_metrics.txt"
)


with open(
    metrics_path,
    "w"
) as f:

    f.write(
        "ROAD SURFACE CONDITION - MODEL V2 EVALUATION\n"
    )

    f.write(
        "=" * 55 + "\n\n"
    )

    f.write(
        f"Test Images: {len(test_dataset)}\n"
    )

    f.write(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%\n\n"
    )

    f.write(
        "Classification Report\n"
    )

    f.write(
        "-" * 55 + "\n"
    )

    f.write(report)

    f.write(
        "\n\nConfusion Matrix\n"
    )

    f.write(
        "-" * 55 + "\n"
    )

    f.write(
        str(cm)
    )


# ============================================================
# Final Output
# ============================================================

print("\n" + "=" * 70)
print("V2 EVALUATION COMPLETE")
print("=" * 70)

print("\nConfusion matrix saved to:")
print(cm_path)

print("\nMetrics saved to:")
print(metrics_path)