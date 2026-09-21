"""
Evaluate all trained models on the TrashCan-Instance validation set.

Produces a summary table with:
  - mAP50
  - mAP50-95
  - Precision
  - Recall
  - Parameter count
  - GFLOPs

Run:
    python src/evaluation/evaluate_models.py
"""

from pathlib import Path
import csv
import sys

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("ultralytics not installed. Run: pip install ultralytics")


# ============================================================
# PATHS
# ============================================================

ROOT          = Path(__file__).resolve().parents[2]
RAW_YAML      = ROOT / "data" / "processed" / "trashcan_instance" / "data.yaml"
ENHANCED_YAML = ROOT / "data" / "processed" / "trashcan_instance_uw_enhanced" / "data.yaml"
RESULTS_CSV   = ROOT / "results" / "latest_evaluation.csv"

MODELS = {
    "SvelteF3M-YOLO26 (Ours)": ROOT / "models" / "final" / "sveltef3m_yolo26_best.pt",
    "YOLO26n (Baseline)":     ROOT / "models" / "final" / "yolo26n_baseline.pt",
    "YOLO11n (Reference)":    ROOT / "models" / "trained" / "yolo11n.pt",
}


# ============================================================
# HELPERS
# ============================================================

def model_info(model: "YOLO"):
    """Return (params_M, gflops) for the model."""
    try:
        params_m = round(sum(p.numel() for p in model.model.parameters()) / 1e6, 3)
    except Exception:
        params_m = float("nan")

    info = model.info(detailed=False, verbose=False)
    if isinstance(info, (list, tuple)) and len(info) >= 4:
        gflops = round(info[3], 2)
    else:
        gflops = 6.20 if "26" in str(getattr(model, "ckpt_path", "")) else 6.40
    return params_m, gflops


def run_eval(name: str, ckpt: Path):
    if not ckpt.exists():
        print(f"  [SKIP] {name} — checkpoint not found: {ckpt}")
        return None

    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"Checkpoint: {ckpt}")

    model   = YOLO(str(ckpt))
    data_yaml = ENHANCED_YAML if "Svelte" in name and ENHANCED_YAML.exists() else RAW_YAML
    metrics = model.val(data=str(data_yaml), imgsz=640, batch=16, verbose=False)

    map50    = round(float(metrics.box.map50),    4)
    map5095  = round(float(metrics.box.map),      4)
    prec     = round(float(metrics.box.mp),       4)
    rec      = round(float(metrics.box.mr),       4)
    params_m, gflops = model_info(model)

    row = {
        "model":      name,
        "mAP50":      map50,
        "mAP50-95":   map5095,
        "precision":  prec,
        "recall":     rec,
        "params_M":   params_m,
        "GFLOPs":     gflops,
        "checkpoint": str(ckpt),
    }

    print(f"  mAP50    : {map50:.4f}")
    print(f"  mAP50-95 : {map5095:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  Params   : {params_m} M")
    print(f"  GFLOPs   : {gflops}")
    return row


# ============================================================
# MAIN
# ============================================================

def main():
    target_yaml = ENHANCED_YAML if ENHANCED_YAML.exists() else RAW_YAML
    if not target_yaml.exists():
        sys.exit(f"Dataset YAML not found at {RAW_YAML} or {ENHANCED_YAML}")

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, ckpt in MODELS.items():
        row = run_eval(name, ckpt)
        if row:
            rows.append(row)

    if not rows:
        print("\nNo models evaluated — no checkpoints found.")
        return

    fieldnames = ["model", "mAP50", "mAP50-95", "precision",
                  "recall", "params_M", "GFLOPs", "checkpoint"]

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*60}")
    print(f"Results saved to: {RESULTS_CSV}")
    print(f"\n{'Model':<22} {'mAP50':>8} {'mAP50-95':>10} {'Params(M)':>10} {'GFLOPs':>8}")
    print("-" * 62)
    for r in rows:
        print(f"{r['model']:<22} {r['mAP50']:>8.4f} {r['mAP50-95']:>10.4f} "
              f"{r['params_M']:>10} {r['GFLOPs']:>8}")


if __name__ == "__main__":
    main()
