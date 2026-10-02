from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_ROOT / "data" / "train"
VAL_DIR = PROJECT_ROOT / "data" / "val"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 12

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# ============================================================
# DATA TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD DATASET
# ============================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

print("\nClasses:", train_dataset.classes)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

train_labels = train_dataset.targets

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)

print("\nClass weights:")

for cls, weight in zip(
    train_dataset.classes,
    class_weights
):
    print(f"{cls}: {weight.item():.4f}")


# ============================================================
# CREATE RESNET18
# ============================================================

print("\nLoading pretrained ResNet18...")

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    len(train_dataset.classes)
)


# ============================================================
# LOAD V1 CHECKPOINT
# ============================================================

v1_model_path = (
    MODEL_DIR /
    "road_condition_resnet18_v1.pth"
)

if not v1_model_path.exists():

    raise FileNotFoundError(
        f"\nV1 model not found:\n"
        f"{v1_model_path}\n\n"
        f"Make sure the V1 model was copied correctly."
    )


print("\nLoading V1 model...")

checkpoint = torch.load(
    v1_model_path,
    map_location=DEVICE
)


# ------------------------------------------------------------
# The V1 checkpoint contains:
# model_state_dict
# class_names
# image_size
# ------------------------------------------------------------

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    print("Checkpoint format detected: full checkpoint")

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    print("Checkpoint format detected: state_dict")

    model.load_state_dict(
        checkpoint
    )


print("V1 model loaded successfully.")


# ============================================================
# FREEZE ENTIRE MODEL
# ============================================================

for param in model.parameters():

    param.requires_grad = False


# ============================================================
# UNFREEZE RESNET18 LAYER4
# ============================================================

for param in model.layer4.parameters():

    param.requires_grad = True


# ============================================================
# UNFREEZE FINAL CLASSIFIER
# ============================================================

for param in model.fc.parameters():

    param.requires_grad = True


# Move model to device
model = model.to(DEVICE)


# ============================================================
# SHOW TRAINABLE LAYERS
# ============================================================

print("\nTrainable layers:")

for name, param in model.named_parameters():

    if param.requires_grad:

        print(name)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    [
        {
            "params": model.layer4.parameters(),
            "lr": 0.00001
        },

        {
            "params": model.fc.parameters(),
            "lr": 0.0001
        }
    ],

    weight_decay=0.0001
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# TRAINING HISTORY
# ============================================================

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

best_val_accuracy = 0.0


# ============================================================
# TRAINING
# ============================================================

print("\n==========================================")
print("Starting Model V2 training...")
print("==========================================\n")


for epoch in range(EPOCHS):

    # ========================================================
    # TRAINING PHASE
    # ========================================================

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0


    for images, labels in train_loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)


        # Clear gradients

        optimizer.zero_grad()


        # Forward pass

        outputs = model(images)


        # Calculate loss

        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation

        loss.backward()


        # Update weights

        optimizer.step()


        # Statistics

        running_loss += (
            loss.item() *
            images.size(0)
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_loss /
        len(train_dataset)
    )

    train_accuracy = (
        100 *
        correct /
        total
    )


    # ========================================================
    # VALIDATION PHASE
    # ========================================================

    model.eval()

    val_running_loss = 0.0

    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            val_running_loss += (
                loss.item() *
                images.size(0)
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_loss = (
        val_running_loss /
        len(val_dataset)
    )

    val_accuracy = (
        100 *
        val_correct /
        val_total
    )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler.step(val_accuracy)


    # ========================================================
    # STORE HISTORY
    # ========================================================

    train_losses.append(train_loss)

    val_losses.append(val_loss)

    train_accuracies.append(
        train_accuracy
    )

    val_accuracies.append(
        val_accuracy
    )


    # ========================================================
    # PRINT EPOCH RESULTS
    # ========================================================

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2f}%"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        v2_model_path = (
            MODEL_DIR /
            "road_condition_resnet18_v2.pth"
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "class_names":
                    train_dataset.classes,

                "image_size":
                    IMAGE_SIZE
            },

            v2_model_path
        )


        print(
            f"  -> New best V2 model saved "
            f"({best_val_accuracy:.2f}%)"
        )


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n==========================================")
print("Model V2 training complete!")
print("==========================================")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    "\nModel saved at:"
)

print(
    MODEL_DIR /
    "road_condition_resnet18_v2.pth"
)


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, EPOCHS + 1),
    train_accuracies,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    range(1, EPOCHS + 1),
    val_accuracies,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.title(
    "Model V2 Training and Validation Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "v2_training_accuracy.png",
    dpi=300
)

plt.show()


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, EPOCHS + 1),
    train_losses,
    marker="o",
    label="Training Loss"
)

plt.plot(
    range(1, EPOCHS + 1),
    val_losses,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Model V2 Training and Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "v2_training_loss.png",
    dpi=300
)

plt.show()