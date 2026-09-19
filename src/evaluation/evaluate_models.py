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
DATA_YAML     = ROOT / "data" / "processed" / "trashcan_instance" / "data.yaml"
RESULTS_CSV   = ROOT / "results" / "master_results.csv"

# Map model name → checkpoint path
MODELS = {
    "YOLO26n":           ROOT / "experiments" / "01_yolo26n"       / "yolo26n_baseline_seed42"  / "weights" / "best.pt",
    "YOLO11n":           ROOT / "experiments" / "02_yolo11n"       / "yolo11n_baseline_seed42"  / "weights" / "best.pt",
    "SvelteNeck-e050":   ROOT / "experiments" / "03_sveltneck_e050"/ "sveltneck_e050_seed42"    / "weights" / "best.pt",
    "SvelteNeck-e075":   ROOT / "experiments" / "03_sveltneck_e075"/ "sveltneck_e075_seed42"    / "weights" / "best.pt",
    "YOLO26n+F3M":       ROOT / "experiments" / "04_yolo26n_f3m"   / "yolo26n_f3m_seed42"       / "weights" / "best.pt",
    "SvelteNeck+F3M":    ROOT / "experiments" / "05_sveltneck_f3m" / "sveltneck_f3m_seed42"     / "weights" / "best.pt",
}


# ============================================================
# HELPERS
# ============================================================

def model_info(model: "YOLO"):
    """Return (params_M, gflops) for the model."""
    info = model.info(detailed=False, verbose=False)
    # info returns (layers, params, gradients, flops)
    if isinstance(info, (list, tuple)) and len(info) >= 4:
        params_m = info[1] / 1e6
        gflops   = info[3]
    else:
        params_m = gflops = float("nan")
    return round(params_m, 3), round(gflops, 2)


def run_eval(name: str, ckpt: Path):
    if not ckpt.exists():
        print(f"  [SKIP] {name} — checkpoint not found: {ckpt}")
        return None

    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"Checkpoint: {ckpt}")

    model   = YOLO(str(ckpt))
    metrics = model.val(data=str(DATA_YAML), imgsz=640, batch=16, verbose=False)

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
    if not DATA_YAML.exists():
        sys.exit(f"data.yaml not found: {DATA_YAML}")

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
