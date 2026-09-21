"""
train_sveltef3m_yolo26.py
Primary training pipeline for SvelteF3M-YOLO26 on TrashCan-Instance.

Architecture:
  - Backbone: YOLO26n pre-trained feature extractor
  - Neck: SvelteNeck (EHConv + EHSCSP blocks, e=0.50)
  - Attention: F3M (Feature-level Frequency & Spatial-Channel Attention Gates)
  - Preprocessing: Beer-Lambert Optical Compensation + CLAHE
  - Head: NMS-free End-to-End Detect Head
"""

import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules

# Import custom architecture blocks
from src.models.ehconv import EHConv
from src.models.ehs_bottleneck import EHSBottleneck
from src.models.ehs_csp import EHSCSP
from src.models.sveltneck import SvelteNeck
from src.models.f3m import F3M

# Register modules dynamically
for m in [EHConv, EHSBottleneck, EHSCSP, SvelteNeck, F3M]:
    setattr(modules, m.__name__, m)
    setattr(tasks, m.__name__, m)

DATA_YAML = PROJECT_ROOT / "data" / "processed" / "trashcan_instance" / "data.yaml"
CFG_YAML  = PROJECT_ROOT / "configs" / "sveltneck_f3m_yolo26.yaml"
EXP_DIR   = PROJECT_ROOT / "experiments" / "04_sveltef3m_yolo26"
BASE_PT   = PROJECT_ROOT / "experiments" / "01_yolo26n" / "weights" / "best.pt"

def transfer_backbone_weights(target_model, base_weights_path):
    if not Path(base_weights_path).exists():
        print(f"Base checkpoint not found at {base_weights_path}. Training from scratch.")
        return target_model
    ck = torch.load(base_weights_path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck).state_dict() if hasattr(ck.get("model", ck), "state_dict") else ck.get("model", ck)
    dd = target_model.model.state_dict()
    transferred = 0
    for k, v in sd.items():
        if k in dd and dd[k].shape == v.shape:
            dd[k] = v
            transferred += 1
    target_model.model.load_state_dict(dd, strict=False)
    print(f"Transferred {transferred} backbone weights from baseline {base_weights_path.name}.")
    return target_model

def main():
    print("=" * 70)
    print("SvelteF3M-YOLO26 TRAINING PIPELINE")
    print("=" * 70)
    print(f"Dataset Config : {DATA_YAML}")
    print(f"Model Config   : {CFG_YAML}")
    print(f"Output Path    : {EXP_DIR}")
    print("=" * 70)

    model = YOLO(str(CFG_YAML))
    model = transfer_backbone_weights(model, BASE_PT)

    train_args = dict(
        data=str(DATA_YAML),
        epochs=30,
        imgsz=640,
        batch=8,
        workers=0,
        device=0 if torch.cuda.is_available() else "cpu",
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        warmup_epochs=3,
        mosaic=1.0,
        flipud=0.2,
        fliplr=0.5,
        scale=0.5,
        project=str(EXP_DIR.parent),
        name=EXP_DIR.name,
        exist_ok=True,
        verbose=True,
    )

    results = model.train(**train_args)
    print("\nTraining completed successfully!")
    print(f"Best weights saved to: {EXP_DIR / 'weights' / 'best.pt'}")

if __name__ == "__main__":
    main()
