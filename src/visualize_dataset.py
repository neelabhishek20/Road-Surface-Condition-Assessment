from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "data" / "train"

CLASSES = [
    "Crack",
    "Pothole",
    "Surface Erosion"
]

IMAGES_PER_CLASS = 3

random.seed(42)


# --------------------------------------------------
# Load Sample Images
# --------------------------------------------------

fig, axes = plt.subplots(
    len(CLASSES),
    IMAGES_PER_CLASS,
    figsize=(12, 9)
)

fig.suptitle(
    "Road Surface Condition - Training Samples",
    fontsize=16
)

for row, class_name in enumerate(CLASSES):

    class_dir = TRAIN_DIR / class_name

    image_files = [
        p for p in class_dir.iterdir()
        if p.is_file()
    ]

    selected_images = random.sample(
        image_files,
        min(IMAGES_PER_CLASS, len(image_files))
    )

    for col in range(IMAGES_PER_CLASS):

        ax = axes[row, col]

        if col < len(selected_images):

            image_path = selected_images[col]

            image = Image.open(image_path).convert("RGB")

            ax.imshow(image)

            ax.set_title(class_name)

        ax.axis("off")


plt.tight_layout()

# Save the figure
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

output_path = RESULTS_DIR / "training_samples.png"

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

print(f"\nSaved visualization to:")
print(output_path)

plt.show()