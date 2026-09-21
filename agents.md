# Project Map & Agent Guide (`AGENTS.md`)

This guide provides a comprehensive overview of key directory layouts, training pipelines, datasets, models, evaluation scripts, and notebooks within this repository.

---

## 1. Datasets (`data/`)

The project benchmarks object detection models on the **TrashCan-Instance 1.0** dataset (derived from deep-sea ROV imagery curated by JAMSTEC), with 22 granular classes (13 marine debris categories, 8 marine fauna categories, and 1 ROV category).

- **Raw Data**:
  - `data/raw/trashcan-1-0/`: Unprocessed/original dataset partition.
- **Processed YOLO Dataset**:
  - `data/processed/trashcan_instance/`: Cleaned and structured YOLO format dataset.
    - `images/train/` & `images/val/`: Underwater image files (6,065 train, 1,147 val).
    - `labels/train/` & `labels/val/`: YOLO-format bounding box annotations (`.txt`).
    - `data.yaml`: Dataset configuration and class mapping.

---

## 2. Core Python Training Scripts (`src/training/`)

- **`src/training/train_sveltef3m_yolo26.py`**: Primary training and validation pipeline for the novel SvelteF3M-YOLO26 architecture combining SvelteNeck, F3M attention, and Beer–Lambert optical compensation.
- **`src/training/train_yolo26n.py`**: Baseline training pipeline for native YOLO26n.
- **`src/training/train_yolo11n.py`**: Baseline training pipeline for reference YOLO11n comparison.
- **`src/training/train_sveltneck_yolov26.py`**: Training pipeline for the SveltNeck-compressed ablation architecture.

---

## 3. Jupyter Notebooks (`notebooks/`)

- **`notebooks/sveltef3m_yolo26_training.ipynb`**: End-to-end multi-stage training pipeline with Beer-Lambert optical compensation, SvelteNeck, F3M attention gates, and long-tail class balancing. Synchronized with Kaggle kernel `yukthasa/sveltef3m-yolo26-underwater-debris-training`.
- **`notebooks/ablation_kaggle.ipynb`**: Complete ablation and multi-stage training pipeline (Stage 1A, 1B, 2A, 2B, 3B). Directly synchronized with Kaggle kernel `yukthasa/sveltef3m-stages-2-and-3-only`.

- **`notebooks/01_setup.ipynb`**: Environment setup and dataset verification.
- **`notebooks/02_yolo26n.ipynb`**: Interactive notebook for YOLO26n training & validation.
- **`notebooks/03_yolo11n.ipynb`**: Interactive notebook for YOLO11n training & evaluation.
- **`notebooks/04_sveltneck_yolo26.ipynb`**: Interactive notebook for SveltNeck YOLO26 training.
- **`notebooks/05_final_results.ipynb`**: Aggregated performance comparison, metrics, and plots.

---

## 4. Source Code Architecture (`src/`)

- **Data Processing (`src/data/`)**:
  - `underwater_augment.py`: Physics-grounded optical compensation module (Beer–Lambert red channel restoration and CLAHE contrast enhancement).
  - `prepare_uw_dataset.py`: Batch dataset processor applying optical compensation.
  - `download_dataset.py`, `prepare_dataset.py`, `verify_dataset.py`: Data ingestion, splitting, and verification.
- **Model Architectures (`src/models/`)**:
  - `sveltneck.py`: Lightweight multi-scale feature-fusion neck replacing standard PANet with EHConv blocks.
  - `f3m.py`: Feature-level Frequency Fusion Module (dual channel + spatial attention gates).
  - `ehconv.py`, `ehs_bottleneck.py`, `ehs_csp.py`: Enhanced Hybrid Convolution and CSP blocks.
- **Evaluation & Benchmarking (`src/evaluation/`)**:
  - `plot_figures.py`: Generates all academic figures, loss/mAP epoch convergence analysis, and speed/accuracy graphs in `results/figures/`.
  - `benchmark.py`: Measures inference latency, throughput (FPS), parameters, and GFLOPs across CUDA/CPU.
  - `evaluate_models.py`: Runs multi-model metric calculations (mAP@0.5, mAP@0.5:0.95, Precision, Recall).
- **Interactive Application (`app/`)**:
  - `app/app.py`: Streamlit-based real-time marine debris detector featuring live optical compensation toggling and categorical breakdown.

---

## 5. Experiment Artifacts & Checkpoints

- **`models/final/`**:
  - `sveltef3m_yolo26_best.pt`: Primary novel architecture checkpoint (SvelteNeck + F3M attention + optical compensation).
  - `yolo26n_baseline.pt`: Native baseline checkpoint (66.68% mAP@0.5, 2.38M parameters).
- **`models/trained/`**:
  - `yolo26n.pt`, `yolo11n.pt`: Exported trained weights from verified experiments.
- **`models/pretrained/`**:
  - `yolo26n_coco.pt`: Official base weights.
- **`results/`**:
  - `master_results.csv`: Standardized master metrics table.
  - `benchmark_results.csv`: FPS and latency benchmarking output.
  - `figures/`: Generated high-resolution plots (`optimal_epochs_analysis.png`, `benchmark_map_comparison.png`, `benchmark_speed_fps.png`, `benchmark_precision_recall.png`, `benchmark_efficiency_frontier.png`).
  - `predictions/`: Annotated sample visual detection outputs (`pred_*.jpg`).
- **`reports/`**:
  - `comprehensive_project_report.md`: Complete 10-page equivalent academic project report with literature review table, methodology diagram, and results analysis.
  - `final_project_report.md`: Condensed project summary.

---

## 6. GitHub Preparation & Synchronization

All files within `DL_project/` (including all local Python modules, configs, results, figures, reports, and notebooks) are organized for immediate tracking and export to GitHub. Large weights (`.pt`) over 50MB should be managed with Git LFS if needed (all active nano checkpoints are under 6 MB and can be committed directly).
