"""
Train the native YOLO26n baseline on the TrashCan-Instance dataset.

Model 1:
    Native YOLO26n

Controlled baseline:
    - TrashCan-Instance
    - 6,065 train images
    - 1,147 validation images
    - 640x640 input
    - No custom modules
    - No custom preprocessing
    - Seed = 42
"""

from pathlib import Path

import torch
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_YAML = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trashcan_instance"
    / "data.yaml"
)

EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "baseline_yolo26n"
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

MODEL_NAME = "yolo26n.pt"

IMAGE_SIZE = 640
BATCH_SIZE = 2
EPOCHS = 30

SEED = 42
WORKERS = 0

DEVICE = 0

RUN_NAME = "yolo26n_baseline_seed42"


# ============================================================
# ENVIRONMENT CHECK
# ============================================================

def verify_environment():
    """Verify CUDA and GPU availability."""

    print("=" * 70)
    print("SvelteNeck-YOLO26 | YOLO26n BASELINE")
    print("=" * 70)

    print("\n[ENVIRONMENT]")
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA build      : {torch.version.cuda}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Training has been stopped."
        )

    print(
        f"GPU             : "
        f"{torch.cuda.get_device_name(DEVICE)}"
    )

    gpu_memory = (
        torch.cuda.get_device_properties(DEVICE).total_memory
        / (1024 ** 3)
    )

    print(f"GPU VRAM        : {gpu_memory:.2f} GB")


# ============================================================
# DATASET CHECK
# ============================================================

def verify_dataset():
    """Verify that the YOLO dataset configuration exists."""

    print("\n[DATASET]")
    print(f"Dataset YAML    : {DATASET_YAML}")

    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found:\n{DATASET_YAML}"
        )

    print("Dataset YAML    : FOUND")


# ============================================================
# TRAINING
# ============================================================

def main():
    """Train the native YOLO26n baseline."""

    verify_environment()
    verify_dataset()

    print("\n[EXPERIMENT CONFIGURATION]")
    print(f"Model           : {MODEL_NAME}")
    print(f"Image size      : {IMAGE_SIZE}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Epochs          : {EPOCHS}")
    print(f"Seed            : {SEED}")
    print(f"Workers         : {WORKERS}")
    print(f"Device          : CUDA:{DEVICE}")
    print(f"Run name        : {RUN_NAME}")
    print(f"Output folder   : {EXPERIMENT_DIR}")

    print("\n" + "=" * 70)
    print("Loading native YOLO26n...")
    print("=" * 70)

    model = YOLO(MODEL_NAME)

    print("\nYOLO26n loaded successfully.")

    print("\n" + "=" * 70)
    print("STARTING YOLO26n BASELINE TRAINING")
    print("=" * 70)

    model.train(
        # Dataset
        data=str(DATASET_YAML),

        # Training resolution
        imgsz=IMAGE_SIZE,

        # Training schedule
        epochs=EPOCHS,
        batch=BATCH_SIZE,

        # Reproducibility
        seed=SEED,
        deterministic=True,

        # Hardware
        device=DEVICE,
        workers=WORKERS,

        # GPU memory optimization
        amp=True,
        cache="disk",

        # Validation
        val=True,

        # Save checkpoints
        save=True,

        # Experiment output
        project=str(EXPERIMENT_DIR),
        name=RUN_NAME,
        exist_ok=False,

        # Generate result plots
        plots=True,
    )

    print("\n" + "=" * 70)
    print("YOLO26n BASELINE TRAINING COMPLETE")
    print("=" * 70)

    print("\nResults saved to:")
    print(EXPERIMENT_DIR / RUN_NAME)

    print("\nBest model:")
    print(EXPERIMENT_DIR / RUN_NAME / "weights" / "best.pt")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()