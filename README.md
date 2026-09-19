# SvelteNeck-YOLO26: Lightweight Underwater Marine Debris Detection

A deep learning project engineered for efficient underwater marine debris and marine life detection on Autonomous Underwater Vehicles (AUVs) and Remotely Operated Vehicles (ROVs).

---

## 1. Project Overview

Autonomous underwater operations require lightweight, fast, and robust object detection models capable of running under resource-constrained embedded systems with low latency and low power consumption. This repository provides:
- **Baseline Models**: YOLO26n and YOLO11n architectures trained on marine debris benchmarks.
- **Custom Lightweight Architecture**: SvelteNeck-YOLO26 featuring Enhanced Hybrid Convolutions (`EHConv`) and Cross-Stage Partial bottleneck blocks (`EHSCSP`) for compact feature fusion.
- **Interactive UI**: A ready-to-use Streamlit web interface for uploading images or inspecting sample underwater frames with real-time detection, bounding box localization, inference timing, and FPS metrics.

---

## 2. Dataset (`TrashCan 1.0 Instance`)

The project is trained and benchmarked on the **TrashCan 1.0 Instance** dataset formatted for YOLO:
- **Total Images**: 7,212 underwater frames (6,065 train, 1,147 validation).
- **Classes (22 categories)**:
  - **Marine Debris**: `trash_clothing`, `trash_pipe`, `trash_bottle`, `trash_bag`, `trash_snack_wrapper`, `trash_can`, `trash_cup`, `trash_container`, `trash_unknown_instance`, `trash_branch`, `trash_wreckage`, `trash_tarp`, `trash_rope`, `trash_net`.
  - **Marine Life / Objects**: `rov`, `plant`, `animal_fish`, `animal_starfish`, `animal_shells`, `animal_crab`, `animal_eel`, `animal_etc`.
- **Kaggle Dataset Native Path**: `mexwell/trashcan-1-0` (`/kaggle/input/trashcan-1-0/`).

---

## 3. Historical Benchmark Results

| Model | mAP50 (%) | mAP50-95 (%) | Precision (%) | Recall (%) | Parameters | GFLOPs | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO26n** | **65.2** | **47.5** | **74.9** | **61.4** | 2.379 M | 5.3 | 2.6 |
| **YOLO11n** | **65.1** | **46.7** | **72.7** | **58.4** | 2.586 M | 6.4 | 3.1 |

*Note: Historical benchmark results recorded under standardized test protocols.*

---

## 4. Repository Structure

```text
DL_project/
├── app/
│   ├── app.py              # Streamlit interactive application
│   └── best.pt             # Best trained detector checkpoint (TrashCan 22 classes)
├── demo_images/            # Curated underwater validation sample images
├── data/
│   ├── raw/                # Original TrashCan dataset
│   └── processed/          # Processed YOLO formatted images and labels
├── configs/                # YAML model and dataset configuration files
├── src/
│   ├── data/               # Dataset preparation, download, and verification scripts
│   ├── models/             # SvelteNeck, EHConv, and EHSCSP neural network components
│   ├── training/           # Training scripts for YOLO26n, YOLO11n, and SvelteNeck
│   ├── evaluation/         # Evaluation, benchmark, and audit tools
│   └── utils/              # Hardware and diagnostic utilities
├── experiments/            # Experiment checkpoints, training plots, and logs
├── models/                 # Model checkpoints directory
├── notebooks/              # Jupyter walkthroughs and experiment notebooks
├── results/                # Consolidated metrics and comparisons
├── reports/                # Project reports and audits
├── requirements.txt        # Environment dependencies
└── README.md               # Documentation and execution guide
```

---

## 5. Installation & Setup

### Clone and Environment Setup
```bash
git clone <repo-url>
cd DL_project

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 6. Running the Streamlit Application

Launch the presentation-ready user interface:

```bash
streamlit run app/app.py
```

### App Features:
1. **Interactive Image Upload**: Upload any `.jpg`, `.jpeg`, or `.png` underwater image.
2. **Demo Sample Selector**: Test immediately by choosing sample underwater images directly from the sidebar dropdown.
3. **Confidence Threshold Slider**: Adjust detection sensitivity (0.05 to 0.95) on the fly.
4. **Live Metrics**: Displays detected object counts, inference latency in milliseconds, and real-time approximate FPS.
5. **Detection Table**: Complete breakdown of detected classes, confidence percentages, and bounding box coordinates.

---

## 7. Training Protocol

When training or reproducing baseline models:
- **Epochs**: 30
- **Image Size (`imgsz`)**: 640
- **Batch Size**: 8
- **Seed**: 42 (Deterministic: True)
- **Workers**: 2
- **Pretrained**: True
- **AMP**: True
- **Cache**: False
- **Validation**: True
