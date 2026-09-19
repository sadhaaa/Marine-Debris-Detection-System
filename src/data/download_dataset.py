"""
Download the TrashCan 1.0 dataset from Kaggle.

This script only downloads the dataset.
Dataset structure and annotation conversion are handled separately.
"""

from pathlib import Path

import kagglehub


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "trashcan-1-0"


def main() -> None:
    """Download TrashCan 1.0 and copy/reference it in the project."""
    print("=" * 60)
    print("TrashCan 1.0 Dataset Download")
    print("=" * 60)

    downloaded_path = kagglehub.dataset_download(
        "mexwell/trashcan-1-0"
    )

    print("\nKaggleHub download completed.")
    print(f"Downloaded dataset location:\n{downloaded_path}")

    print("\nProject raw-data directory:")
    print(RAW_DATA_DIR)

    print(
        "\nIMPORTANT:"
        "\nKaggleHub may store the downloaded dataset in its own cache."
        "\nWe will inspect the downloaded structure before copying or"
        "\nconverting anything."
    )


if __name__ == "__main__":
    main()