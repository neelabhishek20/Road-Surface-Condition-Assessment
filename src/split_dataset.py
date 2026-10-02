from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "data"

CLASSES = [
    "Crack",
    "Pothole",
    "Surface Erosion"
]

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42


# --------------------------------------------------
# Create output folders
# --------------------------------------------------

for split in ["train", "val", "test"]:
    for class_name in CLASSES:
        (OUTPUT_DIR / split / class_name).mkdir(
            parents=True,
            exist_ok=True
        )


# --------------------------------------------------
# Split Dataset
# --------------------------------------------------

print("=" * 60)
print("ROAD SURFACE DATASET SPLITTING")
print("=" * 60)

total_train = 0
total_val = 0
total_test = 0

for class_name in CLASSES:

    source_class_dir = SOURCE_DIR / class_name

    image_files = [
        p for p in source_class_dir.rglob("*")
        if p.is_file()
    ]

    if not image_files:
        print(f"\nNo images found for {class_name}")
        continue

    # First: 70% train, 30% temporary
    train_files, temp_files = train_test_split(
        image_files,
        test_size=(1 - TRAIN_RATIO),
        random_state=RANDOM_STATE
    )

    # Second: split temporary into 15% validation and 15% test
    val_files, test_files = train_test_split(
        temp_files,
        test_size=0.50,
        random_state=RANDOM_STATE
    )

    # Copy files
    for image_path in train_files:
        destination = OUTPUT_DIR / "train" / class_name / image_path.name
        shutil.copy2(image_path, destination)

    for image_path in val_files:
        destination = OUTPUT_DIR / "val" / class_name / image_path.name
        shutil.copy2(image_path, destination)

    for image_path in test_files:
        destination = OUTPUT_DIR / "test" / class_name / image_path.name
        shutil.copy2(image_path, destination)

    total_train += len(train_files)
    total_val += len(val_files)
    total_test += len(test_files)

    print(f"\n{class_name}")
    print(f"  Total      : {len(image_files)}")
    print(f"  Train      : {len(train_files)}")
    print(f"  Validation : {len(val_files)}")
    print(f"  Test       : {len(test_files)}")


# --------------------------------------------------
# Final Summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL SPLIT")
print("=" * 60)

print(f"Training   : {total_train}")
print(f"Validation : {total_val}")
print(f"Testing    : {total_test}")
print(f"Total      : {total_train + total_val + total_test}")

print("\nDataset splitting completed.")