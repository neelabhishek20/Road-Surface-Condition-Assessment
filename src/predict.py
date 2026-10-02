from pathlib import Path
import sys

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "road_condition_resnet18_v2.pth"
)

TEST_IMAGES_DIR = (
    PROJECT_ROOT
    / "test_images"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Crack",
    "Pothole",
    "Surface Erosion"
]


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )


# ============================================================
# BUILD MODEL
# ============================================================

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES)
)


# ============================================================
# LOAD MODEL
# ============================================================

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


# ============================================================
# GET IMAGE PATH
# ============================================================

if len(sys.argv) > 1:

    image_path = Path(sys.argv[1])

else:

    print("\nNo image path provided.")

    print(
        "\nYou can place an image inside:"
    )

    print(TEST_IMAGES_DIR)

    image_name = input(
        "\nEnter image filename: "
    ).strip()

    image_path = (
        TEST_IMAGES_DIR
        / image_name
    )


# ============================================================
# CHECK IMAGE
# ============================================================

if not image_path.exists():

    raise FileNotFoundError(
        f"\nImage not found:\n{image_path}"
    )


# ============================================================
# LOAD IMAGE
# ============================================================

try:

    image = Image.open(
        image_path
    ).convert("RGB")

except Exception as e:

    raise RuntimeError(
        f"\nCould not open image:\n{e}"
    )


# ============================================================
# PREPROCESS IMAGE
# ============================================================

input_tensor = transform(
    image
).unsqueeze(0)

input_tensor = input_tensor.to(DEVICE)


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    outputs = model(
        input_tensor
    )

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence, predicted_class = torch.max(
        probabilities,
        dim=1
    )


predicted_index = predicted_class.item()

predicted_label = (
    CLASS_NAMES[predicted_index]
)

confidence_value = (
    confidence.item() * 100
)


# ============================================================
# TOP 3 PREDICTIONS
# ============================================================

top_probabilities, top_indices = torch.topk(
    probabilities[0],
    k=len(CLASS_NAMES)
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("ROAD SURFACE CONDITION PREDICTION")
print("=" * 60)

print(
    f"\nImage       : {image_path.name}"
)

print(
    f"Device      : {DEVICE}"
)

print(
    f"\nPredicted Condition: "
    f"{predicted_label.upper()}"
)

print(
    f"Confidence          : "
    f"{confidence_value:.2f}%"
)


print("\n" + "-" * 60)
print("CLASS PROBABILITIES")
print("-" * 60)

for probability, index in zip(
    top_probabilities,
    top_indices
):

    print(
        f"{CLASS_NAMES[index.item()]:20s} : "
        f"{probability.item() * 100:.2f}%"
    )


print("\n" + "=" * 60)
print("PREDICTION COMPLETE")
print("=" * 60)