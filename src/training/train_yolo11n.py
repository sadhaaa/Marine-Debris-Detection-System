"""
Train YOLO11n baseline on TrashCan-Instance dataset.

Model 1B:
    YOLO11n — comparison baseline against YOLO26n.

Protocol (same as YOLO26n):
    - 50 epochs
    - batch = 16
    - imgsz = 640
    - AdamW, lr0 = 0.001
    - Seed = 42
    - AMP = True
"""

from pathlib import Path

import torch
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_YAML = (
    PROJECT_ROOT / "configs" / "yolov11n_trashcan.yaml"
)

EXPERIMENT_DIR = (
    PROJECT_ROOT / "experiments" / "02_yolo11n"
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

MODEL_NAME    = "yolo11n.pt"
IMAGE_SIZE    = 640
BATCH_SIZE    = 16
EPOCHS        = 50
SEED          = 42
WORKERS       = 4
DEVICE        = 0
LR0           = 0.001
RUN_NAME      = "yolo11n_baseline_seed42"


# ============================================================
# HELPERS
# ============================================================

def verify_environment():
    print("=" * 70)
    print("SvelteNeck-YOLO26 | YOLO11n BASELINE")
    print("=" * 70)
    print(f"\nPyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.version.cuda}")
    print(f"GPU     : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Stopping.")
    print(f"Device  : {torch.cuda.get_device_name(DEVICE)}")


def main():
    verify_environment()

    print(f"\nDataset YAML : {DATASET_YAML}")
    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"Missing: {DATASET_YAML}")

    model = YOLO(MODEL_NAME)

    model.train(
        data=str(DATASET_YAML),
        imgsz=IMAGE_SIZE,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        optimizer="AdamW",
        lr0=LR0,
        seed=SEED,
        deterministic=True,
        device=DEVICE,
        workers=WORKERS,
        amp=True,
        cache=False,
        val=True,
        save=True,
        plots=True,
        project=str(EXPERIMENT_DIR),
        name=RUN_NAME,
        exist_ok=False,
        # Augmentation (matched to YOLO26n baseline)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        fliplr=0.5,
    )

    best = EXPERIMENT_DIR / RUN_NAME / "weights" / "best.pt"
    print(f"\nDone. Best model: {best}")


if __name__ == "__main__":
    main()
