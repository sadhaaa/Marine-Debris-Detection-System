"""
Print Master Metrics and Benchmark Summary for DL Project
=========================================================
Displays all empirical metrics, comparisons across models, parameter counts,
and hardware latency benchmarks in clean, formatted tables.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

def main():
    print("=" * 85)
    print("      SVELTEF3M-YOLO26 UNDERWATER OBJECT DETECTION BENCHMARK METRICS")
    print("=" * 85)

    master_csv = ROOT / "results" / "master_results.csv"
    if master_csv.exists():
        df_master = pd.read_csv(master_csv).dropna(subset=["model"])
        print("\n[1] MASTER PERFORMANCE & ARCHITECTURE COMPARISON TABLE")
        print("-" * 85)
        # Select key columns for clean terminal display
        cols = ["model", "mAP50", "mAP50-95", "precision", "recall", "params_M", "GFLOPs", "latency_ms", "FPS"]
        cols_available = [c for c in cols if c in df_master.columns]
        formatted_df = df_master[cols_available].copy()
        
        # Rename for presentation
        rename_map = {
            "model": "Model Architecture",
            "mAP50": "mAP@0.50",
            "mAP50-95": "mAP@.5:.95",
            "precision": "Precision",
            "recall": "Recall",
            "params_M": "Params (M)",
            "GFLOPs": "GFLOPs",
            "latency_ms": "Latency (ms)",
            "FPS": "FPS"
        }
        formatted_df.rename(columns=rename_map, inplace=True)
        print(formatted_df.to_string(index=False))
        print("-" * 85)
    else:
        print("Master results file not found at results/master_results.csv")

    bench_csv = ROOT / "results" / "benchmark_results.csv"
    if bench_csv.exists():
        print("\n[2] HARDWARE INFERENCE LATENCY & THROUGHPUT (CUDA / Local MX450 GPU)")
        print("-" * 85)
        df_bench = pd.read_csv(bench_csv).dropna(subset=["model"])
        print(df_bench.to_string(index=False))
        print("-" * 85)

    training_csv = ROOT / "results" / "results.csv"
    if training_csv.exists():
        df_train = pd.read_csv(training_csv)
        best_epoch = df_train.loc[df_train["metrics/mAP50(B)"].idxmax()]
        print("\n[3] SVELTEF3M-YOLO26 TRAINING PROGRESSION & CONVERGENCE")
        print("-" * 85)
        print(f"Total Epochs Completed : {len(df_train)}")
        print(f"Peak Epoch             : Epoch {int(best_epoch['epoch'])}")
        print(f"Peak Validation mAP@50 : {best_epoch['metrics/mAP50(B)'] * 100:.2f}%")
        print(f"Peak mAP@0.50:0.95     : {best_epoch['metrics/mAP50-95(B)'] * 100:.2f}%")
        print(f"Peak Precision         : {best_epoch['metrics/precision(B)'] * 100:.2f}%")
        print(f"Peak Recall            : {best_epoch['metrics/recall(B)'] * 100:.2f}%")
        print(f"Final Train Box Loss   : {df_train.iloc[-1]['train/box_loss']:.4f}")
        print(f"Final Train Cls Loss   : {df_train.iloc[-1]['train/cls_loss']:.4f}")
        print("-" * 85)

    print("\n[4] VERIFIED CHECKPOINTS AVAILABLE LOCALLY")
    print("-" * 85)
    checkpoints = [
        ("SvelteF3M-YOLO26 (Ours)", ROOT / "models" / "final" / "sveltef3m_yolo26_best.pt"),
        ("YOLO26n Baseline",       ROOT / "models" / "final" / "yolo26n_baseline.pt"),
        ("YOLO11n Reference",      ROOT / "models" / "trained" / "yolo11n.pt"),
    ]
    for name, path in checkpoints:
        status = f"Exists ({path.stat().st_size / (1024*1024):.2f} MB)" if path.exists() else "Missing"
        print(f"  * {name:<26} : {status} -> {path.relative_to(ROOT) if path.exists() else path}")
    print("-" * 85)

    print("\nNOTE: To run a live frame-by-frame validation over all 1,147 test images")
    print("      and display individual per-class mAP scores for all 22 categories, run:")
    print("      python validate_live.py\n")

if __name__ == "__main__":
    main()
