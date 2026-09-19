"""
Verify the project-local TrashCan-Instance dataset.

This script does NOT modify the dataset.

It verifies:
    1. Required directories exist.
    2. Train/validation image counts.
    3. COCO annotation files exist.
    4. JSON image counts match filesystem images.
    5. Every JSON image filename exists on disk.
    6. No unexpected images exist.
    7. Expected number of classes.
    8. Category IDs and names.
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "trashcan-1-0"
)

EXPECTED_TRAIN_IMAGES = 6065
EXPECTED_VAL_IMAGES = 1147
EXPECTED_CLASSES = 22


TRAIN_JSON = DATASET_ROOT / "instances_train_trashcan.json"
VAL_JSON = DATASET_ROOT / "instances_val_trashcan.json"

TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_image_files(directory: Path) -> set[str]:
    """Return all JPG filenames in a directory."""
    return {
        path.name
        for path in directory.glob("*.jpg")
        if path.is_file()
    }


def verify_split(
    split_name: str,
    image_directory: Path,
    annotation_file: Path,
    expected_count: int,
) -> bool:
    """
    Verify one dataset split.

    Returns:
        True if all checks pass, otherwise False.
    """
    print(f"\n{'=' * 70}")
    print(f"VERIFYING {split_name.upper()} SPLIT")
    print(f"{'=' * 70}")

    passed = True

    # ---------------------------------------------------------
    # Check annotation file
    # ---------------------------------------------------------
    if not annotation_file.exists():
        print(f"[FAIL] Missing annotation file: {annotation_file}")
        return False

    print(f"[PASS] Annotation file exists")

    # ---------------------------------------------------------
    # Check image directory
    # ---------------------------------------------------------
    if not image_directory.exists():
        print(f"[FAIL] Missing image directory: {image_directory}")
        return False

    print(f"[PASS] Image directory exists")

    # ---------------------------------------------------------
    # Load COCO annotation data
    # ---------------------------------------------------------
    data = load_json(annotation_file)

    json_images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    json_filenames = {
        image["file_name"]
        for image in json_images
    }

    actual_filenames = get_image_files(
        image_directory
    )

    # ---------------------------------------------------------
    # Image count check
    # ---------------------------------------------------------
    print(f"\nExpected images : {expected_count}")
    print(f"JSON images     : {len(json_images)}")
    print(f"Actual JPG files: {len(actual_filenames)}")

    if len(json_images) == expected_count:
        print("[PASS] JSON image count")
    else:
        print("[FAIL] JSON image count")
        passed = False

    if len(actual_filenames) == expected_count:
        print("[PASS] Filesystem image count")
    else:
        print("[FAIL] Filesystem image count")
        passed = False

    # ---------------------------------------------------------
    # Filename consistency
    # ---------------------------------------------------------
    missing_images = json_filenames - actual_filenames
    extra_images = actual_filenames - json_filenames

    print(f"\nMissing images: {len(missing_images)}")
    print(f"Extra images  : {len(extra_images)}")

    if not missing_images:
        print("[PASS] No JSON-referenced images are missing")
    else:
        print("[FAIL] Some JSON-referenced images are missing")
        passed = False

        for filename in sorted(missing_images)[:10]:
            print(f"    Missing: {filename}")

    if not extra_images:
        print("[PASS] No extra images found")
    else:
        print("[FAIL] Extra images found")
        passed = False

        for filename in sorted(extra_images)[:10]:
            print(f"    Extra: {filename}")

    # ---------------------------------------------------------
    # Annotation statistics
    # ---------------------------------------------------------
    print(f"\nAnnotations: {len(annotations)}")
    print(f"Categories  : {len(categories)}")

    if len(categories) == EXPECTED_CLASSES:
        print("[PASS] Category count")
    else:
        print(
            f"[FAIL] Expected {EXPECTED_CLASSES} categories"
        )
        passed = False

    return passed


def verify_categories(annotation_file: Path) -> bool:
    """Verify the category list and IDs."""
    print(f"\n{'=' * 70}")
    print("VERIFYING CATEGORIES")
    print(f"{'=' * 70}")

    data = load_json(annotation_file)

    categories = sorted(
        data["categories"],
        key=lambda category: category["id"],
    )

    passed = True

    print(f"\nNumber of categories: {len(categories)}")
    print()

    for category in categories:
        print(
            f"ID {category['id']:>2} "
            f"→ {category['name']}"
        )

    category_ids = [
        category["id"]
        for category in categories
    ]

    if len(category_ids) != len(set(category_ids)):
        print("\n[FAIL] Duplicate category IDs found")
        passed = False
    else:
        print("\n[PASS] Category IDs are unique")

    return passed


def main() -> None:
    """Run all dataset verification checks."""
    print("=" * 70)
    print("TrashCan-Instance Dataset Verification")
    print("=" * 70)

    print(f"\nDataset root:")
    print(DATASET_ROOT)

    # ---------------------------------------------------------
    # Dataset root
    # ---------------------------------------------------------
    if not DATASET_ROOT.exists():
        print("\n[FAIL] Dataset directory does not exist.")
        print(
            "\nExpected:"
            f"\n{DATASET_ROOT}"
        )
        return

    print("\n[PASS] Dataset root exists")

    # ---------------------------------------------------------
    # Train verification
    # ---------------------------------------------------------
    train_passed = verify_split(
        split_name="train",
        image_directory=TRAIN_DIR,
        annotation_file=TRAIN_JSON,
        expected_count=EXPECTED_TRAIN_IMAGES,
    )

    # ---------------------------------------------------------
    # Validation verification
    # ---------------------------------------------------------
    val_passed = verify_split(
        split_name="validation",
        image_directory=VAL_DIR,
        annotation_file=VAL_JSON,
        expected_count=EXPECTED_VAL_IMAGES,
    )

    # ---------------------------------------------------------
    # Category verification
    # ---------------------------------------------------------
    categories_passed = verify_categories(
        TRAIN_JSON
    )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("FINAL DATASET VERIFICATION")
    print(f"{'=' * 70}")

    all_passed = (
        train_passed
        and val_passed
        and categories_passed
    )

    if all_passed:
        print("\n✓ DATASET VERIFICATION PASSED")
        print("\nTrashCan-Instance is ready for conversion.")
    else:
        print("\n✗ DATASET VERIFICATION FAILED")
        print("\nFix the reported problems before conversion.")


if __name__ == "__main__":
    main()