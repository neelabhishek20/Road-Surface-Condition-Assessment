from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, models

from PIL import Image
import matplotlib.pyplot as plt


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

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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
# GET IMAGE PATH
# ============================================================

if len(sys.argv) > 1:

    image_path = Path(sys.argv[1])

else:

    image_name = input(
        "\nEnter image filename from test_images: "
    ).strip()

    image_path = (
        TEST_IMAGES_DIR
        / image_name
    )


if not image_path.exists():

    raise FileNotFoundError(
        f"\nImage not found:\n{image_path}"
    )


# ============================================================
# LOAD IMAGE
# ============================================================

try:

    original_image = Image.open(
        image_path
    ).convert("RGB")

except Exception as e:

    raise RuntimeError(
        f"\nCould not open image:\n{e}"
    )


# ============================================================
# BUILD RESNET18
# ============================================================

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES)
)


# ============================================================
# LOAD V2 CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

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
# GRAD-CAM VARIABLES
# ============================================================

activations = None
gradients = None


# ============================================================
# HOOK FUNCTIONS
# ============================================================

def forward_hook(module, input, output):
    global activations
    activations = output


def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


# ============================================================
# TARGET LAYER
# ============================================================

target_layer = model.layer4[-1].conv2


forward_handle = target_layer.register_forward_hook(
    forward_hook
)

backward_handle = target_layer.register_full_backward_hook(
    backward_hook
)


# ============================================================
# PREPARE IMAGE
# ============================================================

input_tensor = transform(
    original_image
).unsqueeze(0)

input_tensor = input_tensor.to(DEVICE)

input_tensor.requires_grad_(True)


# ============================================================
# FORWARD PASS
# ============================================================

output = model(
    input_tensor
)

probabilities = torch.softmax(
    output,
    dim=1
)

predicted_index = torch.argmax(
    probabilities,
    dim=1
).item()

predicted_class = (
    CLASS_NAMES[predicted_index]
)

confidence = (
    probabilities[0, predicted_index].item()
    * 100
)


# ============================================================
# BACKWARD PASS
# ============================================================

model.zero_grad()

target_score = output[
    0,
    predicted_index
]

target_score.backward()


# ============================================================
# REMOVE HOOKS
# ============================================================

forward_handle.remove()
backward_handle.remove()


# ============================================================
# PREPARE GRAD-CAM
# ============================================================

activation = activations[0]

gradient = gradients[0]


# Global average pooling of gradients

weights = gradient.mean(
    dim=(1, 2)
)


# Weighted sum of feature maps

cam = torch.zeros(
    activation.shape[1:],
    dtype=torch.float32,
    device=DEVICE
)


for channel, weight in enumerate(weights):

    cam += (
        weight *
        activation[channel]
    )


# ReLU

cam = F.relu(cam)


# Normalize

cam -= cam.min()

if cam.max() > 0:

    cam /= cam.max()


# ============================================================
# RESIZE HEATMAP TO ORIGINAL IMAGE SIZE
# ============================================================

cam = cam.unsqueeze(0).unsqueeze(0)

cam = F.interpolate(
    cam,
    size=(
        original_image.height,
        original_image.width
    ),
    mode="bilinear",
    align_corners=False
)

cam = cam.squeeze().detach().cpu().numpy()


# ============================================================
# ORIGINAL IMAGE ARRAY
# ============================================================

original_array = np.array(
    original_image
).astype(np.float32) / 255.0


# ============================================================
# CREATE OUTPUT FIGURE
# ============================================================

plt.figure(
    figsize=(15, 5)
)


# ============================================================
# ORIGINAL IMAGE
# ============================================================

plt.subplot(
    1,
    3,
    1
)

plt.imshow(
    original_image
)

plt.title(
    "Original Image"
)

plt.axis("off")


# ============================================================
# GRAD-CAM HEATMAP
# ============================================================

plt.subplot(
    1,
    3,
    2
)

plt.imshow(
    original_image
)

plt.imshow(
    cam,
    cmap="jet",
    alpha=0.5
)

plt.title(
    "Grad-CAM Heatmap"
)

plt.axis("off")


# ============================================================
# FINAL OVERLAY
# ============================================================

plt.subplot(
    1,
    3,
    3
)

plt.imshow(
    original_image
)

plt.imshow(
    cam,
    cmap="jet",
    alpha=0.45
)

plt.title(
    f"Prediction: {predicted_class}\n"
    f"Confidence: {confidence:.2f}%"
)

plt.axis("off")


plt.tight_layout()


# ============================================================
# SAVE RESULT
# ============================================================

output_path = (
    RESULTS_DIR
    / "gradcam_road_test.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PRINT RESULT
# ============================================================

print("\n" + "=" * 65)
print("GRAD-CAM EXPLAINABILITY")
print("=" * 65)

print(
    f"\nImage      : {image_path.name}"
)

print(
    f"Prediction : {predicted_class}"
)

print(
    f"Confidence : {confidence:.2f}%"
)

print(
    "\nGrad-CAM result saved to:"
)

print(output_path)

print("\n" + "=" * 65)
print("COMPLETE")
print("=" * 65)