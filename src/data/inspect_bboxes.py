"""
Inspect COCO bounding boxes that fall outside image boundaries.

This script does NOT modify the dataset.

It reports:
    - how many boxes extend outside the image
    - how far they extend outside
    - examples from train and validation
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

SPLITS = {
    "train": DATASET_ROOT / "instances_train_trashcan.json",
    "val": DATASET_ROOT / "instances_val_trashcan.json",
}


def inspect_split(split_name: str, annotation_file: Path) -> None:
    """Inspect bounding boxes for one split."""

    print(f"\n{'=' * 70}")
    print(f"INSPECTING {split_name.upper()} BOUNDING BOXES")
    print(f"{'=' * 70}")

    with annotation_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    images = {
        image["id"]: image
        for image in data["images"]
    }

    problematic = []

    for annotation in data["annotations"]:
        image = images[annotation["image_id"]]

        image_width = image["width"]
        image_height = image["height"]

        x, y, width, height = annotation["bbox"]

        x2 = x + width
        y2 = y + height

        outside_left = x < 0
        outside_top = y < 0
        outside_right = x2 > image_width
        outside_bottom = y2 > image_height

        if (
            outside_left
            or outside_top
            or outside_right
            or outside_bottom
        ):
            problematic.append(
                {
                    "annotation_id": annotation["id"],
                    "image_id": annotation["image_id"],
                    "filename": image["file_name"],
                    "image_width": image_width,
                    "image_height": image_height,
                    "bbox": annotation["bbox"],
                    "x2": x2,
                    "y2": y2,
                    "left": outside_left,
                    "top": outside_top,
                    "right": outside_right,
                    "bottom": outside_bottom,
                }
            )

    print(f"\nTotal annotations : {len(data['annotations'])}")
    print(f"Problematic boxes : {len(problematic)}")

    # -----------------------------------------------------------------------
    # Count the type of boundary violation.
    # -----------------------------------------------------------------------

    left = sum(item["left"] for item in problematic)
    top = sum(item["top"] for item in problematic)
    right = sum(item["right"] for item in problematic)
    bottom = sum(item["bottom"] for item in problematic)

    print("\nBoundary violations:")
    print(f"  Left   : {left}")
    print(f"  Top    : {top}")
    print(f"  Right  : {right}")
    print(f"  Bottom : {bottom}")

    # -----------------------------------------------------------------------
    # Print examples.
    # -----------------------------------------------------------------------

    print("\nFirst 20 problematic annotations:")
    print("-" * 70)

    for item in problematic[:20]:
        print(
            f"\nAnnotation ID : {item['annotation_id']}"
            f"\nImage         : {item['filename']}"
            f"\nImage size    : "
            f"{item['image_width']} x {item['image_height']}"
            f"\nOriginal bbox : {item['bbox']}"
            f"\nBottom-right  : "
            f"({item['x2']:.2f}, {item['y2']:.2f})"
            f"\nOutside       : "
            f"left={item['left']}, "
            f"top={item['top']}, "
            f"right={item['right']}, "
            f"bottom={item['bottom']}"
        )

    # -----------------------------------------------------------------------
    # Measure maximum excursion outside image boundaries.
    # -----------------------------------------------------------------------

    max_left = 0.0
    max_top = 0.0
    max_right = 0.0
    max_bottom = 0.0

    for item in problematic:
        x, y, width, height = item["bbox"]

        x2 = x + width
        y2 = y + height

        max_left = max(max_left, -x)
        max_top = max(max_top, -y)
        max_right = max(
            max_right,
            x2 - item["image_width"],
        )
        max_bottom = max(
            max_bottom,
            y2 - item["image_height"],
        )

    print("\nMaximum boundary excursion:")
    print(f"  Left   : {max_left:.2f} pixels")
    print(f"  Top    : {max_top:.2f} pixels")
    print(f"  Right  : {max_right:.2f} pixels")
    print(f"  Bottom : {max_bottom:.2f} pixels")


def main() -> None:
    """Inspect all dataset splits."""

    print("=" * 70)
    print("TrashCan-Instance Bounding Box Diagnostic")
    print("=" * 70)

    for split_name, annotation_file in SPLITS.items():
        inspect_split(
            split_name,
            annotation_file,
        )


if __name__ == "__main__":
    main()