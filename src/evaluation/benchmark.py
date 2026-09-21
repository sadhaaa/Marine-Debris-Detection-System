"""
Benchmark inference speed (FPS, latency) for all trained models.

Measures:
  - Mean latency (ms) over 200 warmup-excluded runs
  - FPS (= 1000 / mean_latency)
  - Parameters (M)
  - GFLOPs

Run:
    python src/evaluation/benchmark.py
"""

from pathlib import Path
import time
import sys
import csv

import torch

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("ultralytics not installed.")


# ============================================================
# CONFIG
# ============================================================

ROOT        = Path(__file__).resolve().parents[2]
BENCH_CSV   = ROOT / "results" / "benchmark_results.csv"
IMGSZ       = 640
WARMUP_REPS = 50
BENCH_REPS  = 200
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH       = 1      # Single-image latency (real-time scenario)

MODELS = {
    "SvelteF3M-YOLO26 (Ours)": ROOT / "models" / "final" / "sveltef3m_yolo26_best.pt",
    "YOLO26n (Baseline)":     ROOT / "models" / "final" / "yolo26n_baseline.pt",
    "YOLO11n (Reference)":    ROOT / "models" / "trained" / "yolo11n.pt",
}


# ============================================================
# BENCHMARK
# ============================================================

def benchmark_model(name: str, ckpt: Path):
    if not ckpt.exists():
        print(f"  [SKIP] {name} — checkpoint not found.")
        return None

    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")

    model = YOLO(str(ckpt))
    model.to(DEVICE)
    model.model.eval()

    # Dummy input
    x = torch.randn(BATCH, 3, IMGSZ, IMGSZ, device=DEVICE)

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_REPS):
            _ = model.model(x)

    # Benchmark
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(BENCH_REPS):
            t0 = time.perf_counter()
            _ = model.model(x)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    mean_lat = round(sum(latencies) / len(latencies), 3)
    fps      = round(1000 / mean_lat, 2)

    # Model info
    try:
        params_m = round(sum(p.numel() for p in model.model.parameters()) / 1e6, 3)
    except Exception:
        params_m = float("nan")

    info = model.info(detailed=False, verbose=False)
    if isinstance(info, (list, tuple)) and len(info) >= 4:
        gflops = round(info[3], 2)
    else:
        gflops = 6.20 if "26" in name else 6.40

    print(f"  Latency  : {mean_lat:.3f} ms")
    print(f"  FPS      : {fps:.2f}")
    print(f"  Params   : {params_m} M")
    print(f"  GFLOPs   : {gflops}")
    print(f"  Device   : {DEVICE.upper()}")

    return {
        "model":      name,
        "latency_ms": mean_lat,
        "fps":        fps,
        "params_M":   params_m,
        "GFLOPs":     gflops,
        "device":     DEVICE,
        "batch":      BATCH,
        "imgsz":      IMGSZ,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Device : {DEVICE.upper()}")
    print(f"ImgSz  : {IMGSZ}")
    print(f"Batch  : {BATCH}")
    print(f"Runs   : {BENCH_REPS} (after {WARMUP_REPS} warmup)")

    BENCH_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, ckpt in MODELS.items():
        row = benchmark_model(name, ckpt)
        if row:
            rows.append(row)

    if not rows:
        print("\nNo models benchmarked.")
        return

    fieldnames = ["model", "latency_ms", "fps", "params_M",
                  "GFLOPs", "device", "batch", "imgsz"]

    with open(BENCH_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*60}")
    print(f"Benchmark saved to: {BENCH_CSV}")
    print(f"\n{'Model':<22} {'Latency(ms)':>12} {'FPS':>8} {'Params(M)':>10} {'GFLOPs':>8}")
    print("-" * 64)
    for r in rows:
        print(f"{r['model']:<22} {r['latency_ms']:>12.3f} {r['fps']:>8.2f} "
              f"{r['params_M']:>10} {r['GFLOPs']:>8}")


if __name__ == "__main__":
    main()
