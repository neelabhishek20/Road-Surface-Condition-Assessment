from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


# --------------------------------------------------
# Dataset Configuration
# --------------------------------------------------

# Find the project root first, then access dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

CLASSES = [
    "Crack",
    "Pothole",
    "Surface Erosion"
]


# --------------------------------------------------
# Check Images
# --------------------------------------------------

print("=" * 60)
print("ROAD SURFACE CONDITION DATASET CHECK")
print("=" * 60)

print(f"\nDataset location:")
print(DATASET_DIR)

total_images = 0
class_counts = {}
image_sizes = {}

for class_name in CLASSES:

    class_dir = DATASET_DIR / class_name

    valid_images = 0
    invalid_images = 0
    sizes = []

    if not class_dir.exists():
        print(f"\nWARNING: Folder not found -> {class_dir}")
        continue

    for image_path in class_dir.rglob("*"):

        if not image_path.is_file():
            continue

        try:
            # Verify image
            with Image.open(image_path) as img:
                img.verify()

            # Read image dimensions
            with Image.open(image_path) as img:
                sizes.append(img.size)

            valid_images += 1

        except Exception:
            invalid_images += 1

    class_counts[class_name] = valid_images
    total_images += valid_images
    image_sizes[class_name] = sizes

    print(f"\nClass: {class_name}")
    print(f"Valid images   : {valid_images}")
    print(f"Invalid images : {invalid_images}")


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

for class_name in CLASSES:
    print(f"{class_name:<20} : {class_counts.get(class_name, 0)}")

print("-" * 60)
print(f"{'Total images':<20} : {total_images}")


# --------------------------------------------------
# Class Distribution Plot
# --------------------------------------------------

if class_counts:

    plt.figure(figsize=(8, 5))

    names = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.bar(names, counts)

    plt.title("Road Surface Dataset - Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")

    plt.xticks(rotation=15)

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Image Size Summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("IMAGE SIZE SUMMARY")
print("=" * 60)

for class_name, sizes in image_sizes.items():

    if not sizes:
        continue

    unique_sizes = set(sizes)

    print(f"\n{class_name}")

    if len(unique_sizes) <= 10:

        for size in sorted(unique_sizes):
            print(f"  {size}")

    else:
        print(f"  {len(unique_sizes)} different image sizes")


print("\nDataset check completed.")