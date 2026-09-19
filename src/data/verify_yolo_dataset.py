"""
Verify the converted TrashCan-Instance YOLO detection dataset.

This script does NOT modify the dataset.

Checks:
    1. Required directories exist.
    2. Train/validation image counts.
    3. Train/validation label counts.
    4. Every image has a corresponding label file.
    5. Every label has valid YOLO formatting.
    6. Class IDs are within the expected range.
    7. Bounding-box values are normalized to [0, 1].
    8. data.yaml exists and contains the expected configuration.
"""

from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trashcan_instance"
)

DATA_YAML = DATASET_ROOT / "data.yaml"

EXPECTED_TRAIN_IMAGES = 6065
EXPECTED_VAL_IMAGES = 1147
EXPECTED_CLASSES = 22

MIN_CLASS_ID = 0
MAX_CLASS_ID = EXPECTED_CLASSES - 1


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_files(directory: Path, extension: str) -> set[str]:
    """Return filenames with the specified extension."""
    return {
        file.name
        for file in directory.glob(f"*{extension}")
        if file.is_file()
    }


def verify_split(
    split_name: str,
    expected_images: int,
) -> bool:
    """Verify one YOLO dataset split."""
    print(f"\n{'=' * 70}")
    print(f"VERIFYING {split_name.upper()} YOLO SPLIT")
    print(f"{'=' * 70}")

    passed = True

    image_dir = (
        DATASET_ROOT
        / "images"
        / split_name
    )

    label_dir = (
        DATASET_ROOT
        / "labels"
        / split_name
    )

    # -----------------------------------------------------------------------
    # Directory checks
    # -----------------------------------------------------------------------

    if image_dir.exists():
        print("[PASS] Image directory exists")
    else:
        print(f"[FAIL] Missing image directory: {image_dir}")
        return False

    if label_dir.exists():
        print("[PASS] Label directory exists")
    else:
        print(f"[FAIL] Missing label directory: {label_dir}")
        return False

    # -----------------------------------------------------------------------
    # Count images and labels
    # -----------------------------------------------------------------------

    image_files = get_files(image_dir, ".jpg")
    label_files = get_files(label_dir, ".txt")

    print(f"\nExpected images : {expected_images}")
    print(f"Actual JPG files: {len(image_files)}")
    print(f"Label files     : {len(label_files)}")

    if len(image_files) == expected_images:
        print("[PASS] Image count")
    else:
        print("[FAIL] Image count")
        passed = False

    # -----------------------------------------------------------------------
    # Every image should have a label file.
    # -----------------------------------------------------------------------

    expected_labels = {
        Path(filename).stem + ".txt"
        for filename in image_files
    }

    missing_labels = expected_labels - label_files
    extra_labels = label_files - expected_labels

    print(f"\nMissing labels: {len(missing_labels)}")
    print(f"Extra labels  : {len(extra_labels)}")

    if not missing_labels:
        print("[PASS] Every image has a label file")
    else:
        print("[FAIL] Some images are missing labels")
        passed = False

        for filename in sorted(missing_labels)[:10]:
            print(f"    Missing: {filename}")

    if not extra_labels:
        print("[PASS] No extra label files")
    else:
        print("[FAIL] Extra label files found")
        passed = False

        for filename in sorted(extra_labels)[:10]:
            print(f"    Extra: {filename}")

    # -----------------------------------------------------------------------
    # Validate every YOLO label.
    # -----------------------------------------------------------------------

    invalid_files = []
    invalid_lines = 0
    total_annotations = 0

    for label_filename in sorted(label_files):
        label_path = label_dir / label_filename

        try:
            lines = label_path.read_text(
                encoding="utf-8"
            ).splitlines()

            for line_number, line in enumerate(
                lines,
                start=1,
            ):
                if not line.strip():
                    continue

                values = line.split()

                # YOLO detection format contains exactly
                # class_id + 4 bounding-box values.
                if len(values) != 5:
                    invalid_files.append(
                        (
                            label_filename,
                            line_number,
                            "Expected 5 values",
                        )
                    )
                    invalid_lines += 1
                    continue

                try:
                    class_id = int(values[0])

                    x_center = float(values[1])
                    y_center = float(values[2])
                    width = float(values[3])
                    height = float(values[4])

                except ValueError:
                    invalid_files.append(
                        (
                            label_filename,
                            line_number,
                            "Non-numeric value",
                        )
                    )
                    invalid_lines += 1
                    continue

                # -----------------------------------------------------------
                # Class ID validation
                # -----------------------------------------------------------

                if not (
                    MIN_CLASS_ID
                    <= class_id
                    <= MAX_CLASS_ID
                ):
                    invalid_files.append(
                        (
                            label_filename,
                            line_number,
                            f"Invalid class ID: {class_id}",
                        )
                    )
                    invalid_lines += 1
                    continue

                # -----------------------------------------------------------
                # Bounding-box validation
                # -----------------------------------------------------------

                bbox_values = (
                    x_center,
                    y_center,
                    width,
                    height,
                )

                if not all(
                    0.0 <= value <= 1.0
                    for value in bbox_values
                ):
                    invalid_files.append(
                        (
                            label_filename,
                            line_number,
                            "Bounding box outside [0, 1]",
                        )
                    )
                    invalid_lines += 1
                    continue

                if width <= 0.0 or height <= 0.0:
                    invalid_files.append(
                        (
                            label_filename,
                            line_number,
                            "Zero or negative box size",
                        )
                    )
                    invalid_lines += 1
                    continue

                total_annotations += 1

        except OSError as error:
            invalid_files.append(
                (
                    label_filename,
                    0,
                    f"Could not read file: {error}",
                )
            )
            invalid_lines += 1

    print(f"\nValid annotations: {total_annotations}")
    print(f"Invalid annotations: {invalid_lines}")

    if invalid_lines == 0:
        print("[PASS] All YOLO annotations are valid")
    else:
        print("[FAIL] Invalid YOLO annotations found")
        passed = False

        for filename, line_number, reason in invalid_files[:10]:
            print(
                f"    {filename}"
                f":{line_number} → {reason}"
            )

    return passed


def verify_data_yaml() -> bool:
    """Verify the Ultralytics data.yaml configuration."""
    print(f"\n{'=' * 70}")
    print("VERIFYING DATA.YAML")
    print(f"{'=' * 70}")

    if not DATA_YAML.exists():
        print(f"[FAIL] Missing data.yaml: {DATA_YAML}")
        return False

    print("[PASS] data.yaml exists")

    try:
        with DATA_YAML.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as error:
        print(f"[FAIL] Invalid YAML: {error}")
        return False

    passed = True

    # -----------------------------------------------------------------------
    # Class count
    # -----------------------------------------------------------------------

    names = data.get("names", {})
    nc = data.get("nc")

    print(f"\nNumber of classes: {nc}")

    if nc == EXPECTED_CLASSES:
        print("[PASS] Class count")
    else:
        print(
            f"[FAIL] Expected {EXPECTED_CLASSES} classes"
        )
        passed = False

    if len(names) == EXPECTED_CLASSES:
        print("[PASS] Class names")
    else:
        print("[FAIL] Incorrect class-name count")
        passed = False

    # -----------------------------------------------------------------------
    # Train/validation paths
    # -----------------------------------------------------------------------

    train_path = data.get("train")
    val_path = data.get("val")

    print(f"\nTrain path: {train_path}")
    print(f"Val path  : {val_path}")

    if train_path == "images/train":
        print("[PASS] Train path")
    else:
        print("[FAIL] Unexpected train path")
        passed = False

    if val_path == "images/val":
        print("[PASS] Validation path")
    else:
        print("[FAIL] Unexpected validation path")
        passed = False

    # -----------------------------------------------------------------------
    # Print class mapping.
    # -----------------------------------------------------------------------

    print("\nClass mapping:")

    for class_id in range(EXPECTED_CLASSES):
        if isinstance(names, dict):
            class_name = names.get(class_id)

            # YAML may load numeric keys as integers.
            if class_name is None:
                class_name = names.get(str(class_id))
        else:
            class_name = names[class_id]

        print(
            f"  {class_id:>2} → {class_name}"
        )

    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete YOLO dataset verification."""
    print("=" * 70)
    print("TrashCan-Instance YOLO Dataset Verification")
    print("=" * 70)

    print("\nDataset root:")
    print(DATASET_ROOT)

    if not DATASET_ROOT.exists():
        print("\n[FAIL] Dataset root does not exist.")
        return

    train_passed = verify_split(
        split_name="train",
        expected_images=EXPECTED_TRAIN_IMAGES,
    )

    val_passed = verify_split(
        split_name="val",
        expected_images=EXPECTED_VAL_IMAGES,
    )

    yaml_passed = verify_data_yaml()

    # -----------------------------------------------------------------------
    # Final result
    # -----------------------------------------------------------------------

    print(f"\n{'=' * 70}")
    print("FINAL YOLO DATASET VERIFICATION")
    print(f"{'=' * 70}")

    all_passed = (
        train_passed
        and val_passed
        and yaml_passed
    )

    if all_passed:
        print("\n✓ YOLO DATASET VERIFICATION PASSED")
        print("\nThe dataset is ready for baseline training.")
    else:
        print("\n✗ YOLO DATASET VERIFICATION FAILED")
        print("\nFix the reported problems before training.")


if __name__ == "__main__":
    main()