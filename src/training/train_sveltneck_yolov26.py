"""
Train SvelteNeck-YOLO26 on TrashCan-Instance.

Experimental protocol:
    - Same dataset as frozen Phase 1 baselines
    - 640x640 input
    - Batch size 8
    - 30 epochs
    - Seed 42
    - Deterministic training
    - Same optimizer selection policy
    - AMP enabled
    - No dataset caching

Model:
    YOLO26n backbone
    + SvelteNeck (e=0.50)
    + native YOLO26 Detect head
"""

from pathlib import Path

from ultralytics import YOLO


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(
    r"C:\Users\HP\OneDrive\Desktop\Computer_Science\DL_project"
)

DATA_YAML = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trashcan_instance"
    / "data.yaml"
)

MODEL_YAML = (
    PROJECT_ROOT
    / "configs"
    / "sveltneck_yolov26.yaml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "sveltneck_yolo26"
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

IMG_SIZE = 640
BATCH_SIZE = 8
EPOCHS = 30
SEED = 42
WORKERS = 2


def main() -> None:
    """Train the SvelteNeck-YOLO26 model."""

    # Verify required files before starting a long training run.
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found:\n{DATA_YAML}"
        )

    if not MODEL_YAML.exists():
        raise FileNotFoundError(
            f"Model YAML not found:\n{MODEL_YAML}"
        )

    print("=" * 70)
    print("SVELTENECK-YOLO26 TRAINING")
    print("=" * 70)

    print(f"Dataset : {DATA_YAML}")
    print(f"Model   : {MODEL_YAML}")
    print(f"Image   : {IMG_SIZE}")
    print(f"Batch   : {BATCH_SIZE}")
    print(f"Epochs  : {EPOCHS}")
    print(f"Seed    : {SEED}")
    print(f"Workers : {WORKERS}")
    print("=" * 70)

    # Build the proposed architecture from YAML.
    #
    # The model contains:
    #
    #   YOLO26n backbone
    #          +
    #   SvelteNeck (e=0.50)
    #          +
    #   native YOLO26 Detect head
    #
    # No pretrained checkpoint is loaded here because our custom
    # architecture does not have a compatible pretrained checkpoint.
    model = YOLO(str(MODEL_YAML))

    # Print the model summary before committing to training.
    print("\nMODEL SUMMARY")
    model.info()

    print("\nStarting training...\n")

    results = model.train(
        data=str(DATA_YAML),
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        epochs=EPOCHS,
        seed=SEED,
        deterministic=True,
        device=0,
        workers=WORKERS,
        pretrained=False,
        amp=True,
        cache=False,
        val=True,
        save=True,
        plots=True,
        project=str(OUTPUT_DIR),
        name="sveltneck_yolov26_e050_seed42",
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(f"Results directory: {OUTPUT_DIR}")
    print(f"Training result object: {results}")


if __name__ == "__main__":
    main()