"""
Model validation and evaluation script on the TrashCan-Instance benchmark.
"""

from pathlib import Path
import torch
from ultralytics import YOLO


def main():
    import sys
    ROOT = Path(__file__).resolve().parent
    enhanced_yaml = ROOT / "data" / "processed" / "trashcan_instance_uw_enhanced" / "data.yaml"
    raw_yaml = ROOT / "data" / "processed" / "trashcan_instance" / "data.yaml"
    DATA_YAML = enhanced_yaml if enhanced_yaml.exists() else raw_yaml
    
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        CKPT = Path(sys.argv[1])
    else:
        CKPT = ROOT / "models" / "final" / "sveltef3m_yolo26_best.pt"
        if not CKPT.exists():
            CKPT = ROOT / "app" / "best.pt"


    print(f"Loading checkpoint: {CKPT}")
    model = YOLO(str(CKPT))

    print("\n" + "=" * 70)
    print("MODEL VALIDATION EVALUATION (TrashCan-Instance Benchmark)")
    print("=" * 70)

    # Evaluate model on validation split
    results = model.val(
        data=str(DATA_YAML),
        imgsz=640,
        batch=16,
        workers=0,
        verbose=True,
        device=0 if torch.cuda.is_available() else "cpu",
    )

    print("\n" + "=" * 70)
    print("VALIDATION PERFORMANCE SUMMARY")
    print("=" * 70)
    total_params = sum(x.numel() for x in model.model.parameters()) / 1e6
    print(f"Total Parameters : {total_params:.3f} M")
    print(f"Precision        : {results.box.mp:.4f} ({results.box.mp * 100:.2f}%)")
    print(f"Recall           : {results.box.mr:.4f} ({results.box.mr * 100:.2f}%)")
    print(f"mAP@0.50         : {results.box.map50:.4f} ({results.box.map50 * 100:.2f}%)")
    print(f"mAP@0.50:0.95    : {results.box.map:.4f} ({results.box.map * 100:.2f}%)")
    print(f"Fitness Score    : {results.fitness:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
