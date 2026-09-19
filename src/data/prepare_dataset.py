"""
Convert TrashCan-Instance from COCO format to YOLO detection format.

Source:
    data/raw/trashcan-1-0/

Output:
    data/processed/trashcan_instance/

The source dataset is NOT modified.

COCO bounding boxes:
    [x_min, y_min, width, height]

YOLO bounding boxes:
    class_id x_center y_center width height

All YOLO coordinates are normalized to [0, 1].

Important:
    Some original TrashCan-Instance COCO bounding boxes extend
    slightly beyond the image boundary. Before conversion, each
    bounding box is clipped to the valid image region.

    This ensures that the resulting YOLO bounding boxes are
    geometrically valid and remain inside the image.
"""

import json
import shutil
from pathlib import Path


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "trashcan-1-0"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trashcan_instance"
)


# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

SPLITS = {
    "train": {
        "image_dir": SOURCE_ROOT / "train",
        "annotation_file": (
            SOURCE_ROOT / "instances_train_trashcan.json"
        ),
    },
    "val": {
        "image_dir": SOURCE_ROOT / "val",
        "annotation_file": (
            SOURCE_ROOT / "instances_val_trashcan.json"
        ),
    },
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_coco_json(annotation_file: Path) -> dict:
    """
    Load a COCO annotation JSON file.

    Args:
        annotation_file: Path to the COCO JSON annotation file.

    Returns:
        Parsed COCO annotation dictionary.
    """
    with annotation_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def create_output_directories() -> None:
    """
    Create the YOLO dataset directory structure.

    Structure:
        data/processed/trashcan_instance/
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
    """
    for split in SPLITS:
        image_output_dir = (
            OUTPUT_ROOT
            / "images"
            / split
        )

        label_output_dir = (
            OUTPUT_ROOT
            / "labels"
            / split
        )

        image_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        label_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


def build_category_mapping(
    categories: list[dict],
) -> dict[int, int]:
    """
    Convert COCO category IDs to zero-based YOLO class IDs.

    The mapping is based on ascending COCO category IDs.

    Example:
        COCO ID 1  -> YOLO ID 0
        COCO ID 2  -> YOLO ID 1
        ...
        COCO ID 22 -> YOLO ID 21

    Args:
        categories: COCO category definitions.

    Returns:
        Dictionary mapping COCO IDs to YOLO IDs.
    """
    sorted_categories = sorted(
        categories,
        key=lambda category: category["id"],
    )

    return {
        category["id"]: index
        for index, category in enumerate(
            sorted_categories
        )
    }


def convert_bbox_to_yolo(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """
    Clip a COCO bounding box to the image boundary and convert it
    to normalized YOLO format.

    COCO format:
        [x_min, y_min, width, height]

    YOLO format:
        [x_center, y_center, width, height]

    Example:

        Image:
            width  = 480
            height = 270

        Original box:
            x_min = 0
            y_min = 217
            width = 481
            height = 54

        Original bottom-right:
            x_max = 481
            y_max = 271

        Clipped bottom-right:
            x_max = 480
            y_max = 270

    This prevents the resulting normalized bounding box from
    extending outside [0, 1].

    Args:
        bbox:
            COCO bounding box
            [x_min, y_min, width, height].

        image_width:
            Image width in pixels.

        image_height:
            Image height in pixels.

    Returns:
        Tuple containing:
            x_center,
            y_center,
            normalized_width,
            normalized_height

        Returns None if the clipped bounding box has zero or
        negative area.
    """

    x_min, y_min, bbox_width, bbox_height = bbox

    # ------------------------------------------------------------------------
    # Convert COCO [x, y, width, height] representation into
    # [x_min, y_min, x_max, y_max].
    # ------------------------------------------------------------------------

    x_max = x_min + bbox_width
    y_max = y_min + bbox_height

    # ------------------------------------------------------------------------
    # Clip the bounding box to the image boundaries.
    #
    # Valid image coordinate range:
    #
    #     0 <= x <= image_width
    #     0 <= y <= image_height
    # ------------------------------------------------------------------------

    clipped_x_min = max(
        0.0,
        min(
            x_min,
            float(image_width),
        ),
    )

    clipped_y_min = max(
        0.0,
        min(
            y_min,
            float(image_height),
        ),
    )

    clipped_x_max = max(
        0.0,
        min(
            x_max,
            float(image_width),
        ),
    )

    clipped_y_max = max(
        0.0,
        min(
            y_max,
            float(image_height),
        ),
    )

    # ------------------------------------------------------------------------
    # Recalculate width and height after clipping.
    # ------------------------------------------------------------------------

    clipped_width = (
        clipped_x_max - clipped_x_min
    )

    clipped_height = (
        clipped_y_max - clipped_y_min
    )

    # ------------------------------------------------------------------------
    # Reject boxes that have no visible area after clipping.
    # ------------------------------------------------------------------------

    if (
        clipped_width <= 0.0
        or clipped_height <= 0.0
    ):
        return None

    # ------------------------------------------------------------------------
    # Calculate the center coordinates of the clipped box.
    # ------------------------------------------------------------------------

    x_center = (
        clipped_x_min
        + clipped_width / 2.0
    )

    y_center = (
        clipped_y_min
        + clipped_height / 2.0
    )

    # ------------------------------------------------------------------------
    # Normalize coordinates to [0, 1].
    # ------------------------------------------------------------------------

    normalized_x_center = (
        x_center / image_width
    )

    normalized_y_center = (
        y_center / image_height
    )

    normalized_width = (
        clipped_width / image_width
    )

    normalized_height = (
        clipped_height / image_height
    )

    return (
        normalized_x_center,
        normalized_y_center,
        normalized_width,
        normalized_height,
    )


def write_yolo_label(
    label_file: Path,
    annotations: list[dict],
    category_mapping: dict[int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    """
    Convert and write all annotations for one image.

    Args:
        label_file:
            Destination YOLO label file.

        annotations:
            COCO annotations belonging to the image.

        category_mapping:
            COCO-to-YOLO category mapping.

        image_width:
            Image width in pixels.

        image_height:
            Image height in pixels.

    Returns:
        Tuple containing:
            valid_annotations_written,
            skipped_annotations
    """

    valid_annotations = []
    skipped_annotations = 0

    for annotation in annotations:

        # --------------------------------------------------------------------
        # Validate category.
        # --------------------------------------------------------------------

        category_id = annotation["category_id"]

        if category_id not in category_mapping:
            print(
                "[WARNING] Unknown category ID: "
                f"{category_id}"
            )

            skipped_annotations += 1
            continue

        # --------------------------------------------------------------------
        # Retrieve and validate bounding box.
        # --------------------------------------------------------------------

        bbox = annotation.get("bbox")

        if not bbox or len(bbox) != 4:
            print(
                "[WARNING] Invalid bounding box: "
                f"{bbox}"
            )

            skipped_annotations += 1
            continue

        _, _, bbox_width, bbox_height = bbox

        # --------------------------------------------------------------------
        # Reject invalid source boxes with zero or negative dimensions.
        # --------------------------------------------------------------------

        if (
            bbox_width <= 0
            or bbox_height <= 0
        ):
            print(
                "[WARNING] Skipping invalid bbox: "
                f"{bbox}"
            )

            skipped_annotations += 1
            continue

        # --------------------------------------------------------------------
        # Convert COCO bounding box to clipped YOLO format.
        # --------------------------------------------------------------------

        yolo_bbox = convert_bbox_to_yolo(
            bbox=bbox,
            image_width=image_width,
            image_height=image_height,
        )

        # --------------------------------------------------------------------
        # If clipping produces a zero-area box, skip it.
        # --------------------------------------------------------------------

        if yolo_bbox is None:
            print(
                "[WARNING] Bounding box has no visible area "
                f"after clipping: {bbox}"
            )

            skipped_annotations += 1
            continue

        (
            x_center,
            y_center,
            normalized_width,
            normalized_height,
        ) = yolo_bbox

        class_id = category_mapping[
            category_id
        ]

        valid_annotations.append(
            (
                class_id,
                x_center,
                y_center,
                normalized_width,
                normalized_height,
            )
        )

    # ------------------------------------------------------------------------
    # Write YOLO label file.
    #
    # Format:
    #
    # class_id x_center y_center width height
    #
    # Example:
    #
    # 10 0.523400 0.612300 0.231200 0.184500
    # ------------------------------------------------------------------------

    with label_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        for annotation in valid_annotations:

            (
                class_id,
                x_center,
                y_center,
                bbox_width,
                bbox_height,
            ) = annotation

            file.write(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{bbox_width:.6f} "
                f"{bbox_height:.6f}\n"
            )

    return (
        len(valid_annotations),
        skipped_annotations,
    )


def convert_split(
    split_name: str,
    split_config: dict,
    category_mapping: dict[int, int],
) -> tuple[int, int, int]:
    """
    Convert one COCO dataset split into YOLO format.

    Args:
        split_name:
            Dataset split name, e.g. "train" or "val".

        split_config:
            Image directory and annotation configuration.

        category_mapping:
            COCO-to-YOLO category mapping.

    Returns:
        Tuple containing:
            images_processed,
            annotations_converted,
            annotations_skipped
    """

    print(f"\n{'=' * 70}")
    print(
        f"CONVERTING {split_name.upper()} SPLIT"
    )
    print(f"{'=' * 70}")

    image_dir = split_config["image_dir"]
    annotation_file = split_config[
        "annotation_file"
    ]

    # ------------------------------------------------------------------------
    # Load COCO annotations.
    # ------------------------------------------------------------------------

    coco_data = load_coco_json(
        annotation_file
    )

    # ------------------------------------------------------------------------
    # Map image ID -> image metadata.
    # ------------------------------------------------------------------------

    images = {
        image["id"]: image
        for image in coco_data["images"]
    }

    # ------------------------------------------------------------------------
    # Group annotations by image ID.
    # ------------------------------------------------------------------------

    annotations_by_image: dict[
        int,
        list[dict],
    ] = {}

    for annotation in coco_data["annotations"]:

        image_id = annotation["image_id"]

        annotations_by_image.setdefault(
            image_id,
            [],
        ).append(annotation)

    # ------------------------------------------------------------------------
    # Create output directories.
    # ------------------------------------------------------------------------

    output_image_dir = (
        OUTPUT_ROOT
        / "images"
        / split_name
    )

    output_label_dir = (
        OUTPUT_ROOT
        / "labels"
        / split_name
    )

    output_image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_label_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # Conversion counters.
    # ------------------------------------------------------------------------

    total_images = 0
    total_annotations = 0
    total_skipped = 0

    # ------------------------------------------------------------------------
    # Process every image in the COCO annotation file.
    # ------------------------------------------------------------------------

    for image_id, image_info in images.items():

        filename = image_info["file_name"]

        image_width = image_info["width"]
        image_height = image_info["height"]

        source_image = (
            image_dir / filename
        )

        output_image = (
            output_image_dir / filename
        )

        # --------------------------------------------------------------------
        # Verify that the source image exists.
        # --------------------------------------------------------------------

        if not source_image.exists():

            print(
                "[WARNING] Missing image: "
                f"{filename}"
            )

            continue

        # --------------------------------------------------------------------
        # Copy original image without modification.
        # --------------------------------------------------------------------

        shutil.copy2(
            source_image,
            output_image,
        )

        # --------------------------------------------------------------------
        # Create corresponding YOLO label filename.
        # --------------------------------------------------------------------

        label_filename = (
            Path(filename).stem + ".txt"
        )

        label_file = (
            output_label_dir
            / label_filename
        )

        # --------------------------------------------------------------------
        # Retrieve annotations for this image.
        # --------------------------------------------------------------------

        annotations = (
            annotations_by_image.get(
                image_id,
                [],
            )
        )

        # --------------------------------------------------------------------
        # Convert annotations.
        # --------------------------------------------------------------------

        (
            annotation_count,
            skipped_count,
        ) = write_yolo_label(
            label_file=label_file,
            annotations=annotations,
            category_mapping=category_mapping,
            image_width=image_width,
            image_height=image_height,
        )

        total_images += 1
        total_annotations += (
            annotation_count
        )
        total_skipped += skipped_count

    # ------------------------------------------------------------------------
    # Split report.
    # ------------------------------------------------------------------------

    print(
        f"\nImages converted          : "
        f"{total_images}"
    )

    print(
        f"Annotations converted     : "
        f"{total_annotations}"
    )

    print(
        f"Annotations skipped       : "
        f"{total_skipped}"
    )

    return (
        total_images,
        total_annotations,
        total_skipped,
    )


def write_data_yaml(
    categories: list[dict],
) -> None:
    """
    Create the Ultralytics data.yaml file.

    Category ordering follows ascending COCO category IDs.
    """

    sorted_categories = sorted(
        categories,
        key=lambda category: category["id"],
    )

    class_names = [
        category["name"]
        for category in sorted_categories
    ]

    yaml_file = (
        OUTPUT_ROOT / "data.yaml"
    )

    lines = [
        f"path: {OUTPUT_ROOT.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]

    for index, class_name in enumerate(
        class_names
    ):
        lines.append(
            f"  {index}: {class_name}"
        )

    yaml_file.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(
        f"\n[PASS] Created: {yaml_file}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """
    Run the complete COCO-to-YOLO conversion.
    """

    print("=" * 70)
    print(
        "TrashCan-Instance COCO → YOLO Conversion"
    )
    print("=" * 70)

    print("\nSource:")
    print(SOURCE_ROOT)

    print("\nOutput:")
    print(OUTPUT_ROOT)

    # ------------------------------------------------------------------------
    # Verify source dataset exists.
    # ------------------------------------------------------------------------

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(
            "Source dataset does not exist:\n"
            f"{SOURCE_ROOT}"
        )

    # ------------------------------------------------------------------------
    # Verify required source files exist.
    # ------------------------------------------------------------------------

    for split_name, split_config in SPLITS.items():

        image_dir = split_config[
            "image_dir"
        ]

        annotation_file = split_config[
            "annotation_file"
        ]

        if not image_dir.exists():
            raise FileNotFoundError(
                f"Missing {split_name} image directory:\n"
                f"{image_dir}"
            )

        if not annotation_file.exists():
            raise FileNotFoundError(
                f"Missing {split_name} annotation file:\n"
                f"{annotation_file}"
            )

    # ------------------------------------------------------------------------
    # Read categories from training annotations.
    # ------------------------------------------------------------------------

    train_json = load_coco_json(
        SPLITS["train"]["annotation_file"]
    )

    categories = train_json[
        "categories"
    ]

    # ------------------------------------------------------------------------
    # TrashCan-Instance should contain 22 categories.
    # ------------------------------------------------------------------------

    if len(categories) != 22:
        raise ValueError(
            "Expected 22 categories, "
            f"found {len(categories)}."
        )

    # ------------------------------------------------------------------------
    # Build COCO -> YOLO category mapping.
    # ------------------------------------------------------------------------

    category_mapping = (
        build_category_mapping(
            categories
        )
    )

    print("\nCategory mapping:")
    print("-" * 50)

    for category in sorted(
        categories,
        key=lambda item: item["id"],
    ):

        coco_id = category["id"]

        yolo_id = (
            category_mapping[coco_id]
        )

        name = category["name"]

        print(
            f"COCO {coco_id:>2} "
            f"→ YOLO {yolo_id:>2} "
            f"→ {name}"
        )

    # ------------------------------------------------------------------------
    # Create output directories.
    # ------------------------------------------------------------------------

    create_output_directories()

    # ------------------------------------------------------------------------
    # Convert train and validation splits.
    # ------------------------------------------------------------------------

    total_images = 0
    total_annotations = 0
    total_skipped = 0

    for split_name, split_config in SPLITS.items():

        (
            images,
            annotations,
            skipped,
        ) = convert_split(
            split_name=split_name,
            split_config=split_config,
            category_mapping=category_mapping,
        )

        total_images += images
        total_annotations += (
            annotations
        )
        total_skipped += skipped

    # ------------------------------------------------------------------------
    # Write Ultralytics dataset YAML.
    # ------------------------------------------------------------------------

    write_data_yaml(
        categories
    )

    # ------------------------------------------------------------------------
    # Final report.
    # ------------------------------------------------------------------------

    print(f"\n{'=' * 70}")
    print("CONVERSION COMPLETE")
    print(f"{'=' * 70}")

    print(
        f"\nTotal images              : "
        f"{total_images}"
    )

    print(
        f"Total annotations         : "
        f"{total_annotations}"
    )

    print(
        f"Total annotations skipped : "
        f"{total_skipped}"
    )

    print("\nOutput dataset:")
    print(OUTPUT_ROOT)

    print("\nNext step:")
    print(
        "Run the YOLO dataset verification "
        "script before training any model."
    )


if __name__ == "__main__":
    main()
