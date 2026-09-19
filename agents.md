# Project Map & Agent Guide (`agents.md`)

This file provides a clear overview of key directory locations, training scripts, datasets, model architectures, and notebooks within this repository.

---

## 1. Datasets (`data/`)

The project uses the **TrashCan** underwater instance detection / segmentation dataset formatted for YOLO.

- **Raw Data**:
  - `data/raw/trashcan-1-0/`: Unprocessed/original dataset split into `train/` and `val/`.
- **Processed YOLO Dataset**:
  - `data/processed/trashcan_instance/`: Cleaned and structured YOLO format dataset.
    - `images/train/` & `images/val/`: Underwater image files.
    - `labels/train/` & `labels/val/`: YOLO-format bounding box annotations (`.txt`).

---

## 2. Training Scripts (`src/training/` & `notebooks/`)

### Core Python Training Scripts
Located under `src/training/`:
- **`src/training/train_yolo26n.py`**: Baseline training pipeline for YOLO26n on TrashCan.
- **`src/training/train_yolo11n.py`**: Baseline training pipeline for YOLO11n comparison.
- **`src/training/train_sveltneck_yolov26.py`**: Training script for the customized model incorporating SveltNeck.

### Jupyter Training & Experiment Notebooks
Located under `notebooks/` (and archived under `archive/notebooks/`):
- **`notebooks/01_setup.ipynb`**: Environment setup and dataset verification.
- **`notebooks/02_yolo26n.ipynb`**: Interactive notebook for YOLO26n training & validation.
- **`notebooks/03_yolo11n.ipynb`**: Interactive notebook for YOLO11n training & evaluation.
- **`notebooks/04_sveltneck_yolo26.ipynb`**: Interactive notebook for SveltNeck YOLO26 training.
- **`notebooks/05_final_results.ipynb`**: Aggregated performance comparison, metrics, and plots.
- *Root exploratory notebooks*: `Untitled0.ipynb`, `notebook4d3e8da6b2 (3).ipynb`.

---

## 3. Configuration Files (`configs/`)

YOLO dataset configurations and model hyperparameter definitions:
- **`configs/yolov26n_trashcan.yaml`**: Dataset paths and class mappings for YOLO26n.
- **`configs/yolov11n_trashcan.yaml`**: Dataset paths and class mappings for YOLO11n.
- **`configs/sveltneck_yolov26.yaml`**: Custom network configuration specifying SveltNeck architecture layers.

---

## 4. Source Code Architecture (`src/`)

- **Data Processing (`src/data/`)**:
  - `download_dataset.py`: Script to fetch the raw TrashCan dataset.
  - `prepare_dataset.py`: Prepares YOLO-compatible splits and directory layouts.
  - `inspect_bboxes.py`: Visualizes and inspects bounding boxes.
  - `verify_dataset.py` & `verify_yolo_dataset.py`: Data integrity, label verification, and sanity checks.
- **Model Components (`src/models/`)**:
  - `sveltneck.py`: Implementation of the lightweight SveltNeck module.
  - `ehconv.py`: Enhanced Hybrid Convolution modules.
  - `ehs_bottleneck.py` & `ehs_csp.py`: EHS bottleneck & Cross-Stage Partial network blocks.
  - `test_*.py` & `analyze_ehscsp_budget.py`: Unit tests and parameter/FLOPs budget analysis for model blocks.
- **Evaluation & Benchmarking (`src/evaluation/`)**:
  - `evaluate_models.py`: Runs multi-model metric calculations (mAP50, mAP50-95, precision, recall).
  - `benchmark.py`: Measures inference latency, throughput (FPS), and FLOPs.
  - `audit_sveltneck.py`: Deep diagnostic audit of SveltNeck activations and gradient flow.
- **Utilities (`src/utils/`)**:
  - `system_info.py`: Hardware, GPU/CUDA, and environment diagnostic utilities.

---

## 5. Other Key Directories

- **`experiments/`**: Run logs, trained weights, validation plots, and prediction samples:
  - `experiments/01_yolo26n/`
  - `experiments/02_yolo11n/`
  - `experiments/03_sveltneck_yolo26/`
  - `experiments/baseline_yolo26n/` & `experiments/sveltneck_yolo26/`
- **`models/`**: Exported checkpoints (`final/`, `pretrained/`, `trained/`).
- **`weights/`**: Checkpoint storage (e.g. `yolo26n.pt`).
- **`app/`**: Interactive UI/inference demonstration application (`app/app.py`).
- **`tools/`**: Helper utilities for packaging runs (e.g., `create_kaggle_zip.py`, `verify_kaggle_zip.py`).
- **`reports/`**: Summarized documentation, experiment results, and methodology write-ups.
