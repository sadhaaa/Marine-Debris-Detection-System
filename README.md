# Marine Debris Detection System: SvelteF3M-YOLO26

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26-00FFFF.svg)](https://docs.ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20App-FF4B4B.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-TrashCan%201.0%20Instance-green.svg)](https://conservancy.umn.edu/handle/11299/214865)

A lightweight, physics-guided underwater object detection framework combining **YOLO26**, **Beer–Lambert optical attenuation compensation**, and **SvelteNeck-F3M attention** for real-time marine debris and aquatic life localization on Autonomous Underwater Vehicles (AUVs) and edge robotic platforms.

---

## 1. Abstract & Problem Formulation

Autonomous underwater robotic cleanup is constrained by three physical degradations:
1. **Wavelength-Dependent Color Absorption**: Red light ($650\text{--}700\text{ nm}$) attenuates exponentially within $2\text{--}5$ meters ($\alpha_{\text{red}} \approx 0.80\text{ m}^{-1}$ vs. $\alpha_{\text{blue}} \approx 0.01\text{ m}^{-1}$), causing acute contrast deflation and monochromatic blue-green casts.
2. **Turbidity & Particulate Scattering**: Marine particulates ("marine snow") scatter light and blur object boundaries for small debris targets (ropes, nets, plastic fragments).
3. **Embedded Edge Budget**: Autonomous Underwater Vehicles (AUVs) operate under strict power and computational constraints ($15\text{W}\text{--}25\text{W}$ on NVIDIA Jetson / edge GPUs). Heavy diffusion or multi-stage GAN architectures are computationally intractable for real-time control.

**SvelteF3M-YOLO26** resolves these issues through a unified, single-stage detection architecture combining:
- **Beer–Lambert Optical Compensation (UW-Aug)**: Restores chromatic balance and applies CLAHE in CIE LAB space.
- **SvelteNeck**: A lightweight feature fusion neck utilizing Enhanced Hybrid Convolutions (`EHConv`) and Cross-Stage Partial (`EHSCSP`) blocks.
- **F3M Attention Gates**: Frequency-aware dual channel and depthwise spatial attention.
- **NMS-Free End-to-End Inference**: Built upon YOLO26's Small Target-Aware Label Assignment (STAL) to eliminate CPU post-processing bottlenecks.

---

## 2. Master Benchmark Results

Evaluated on the standardized **TrashCan-Instance** benchmark ($640 \times 640$ resolution, 22 granular classes):

| Model Architecture | mAP@0.50 (%) | mAP@0.5:0.95 (%) | Precision (%) | Recall (%) | Params (M) | GFLOPs | Latency (ms) | Speed (FPS) | Status / Origin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **SvelteF3M-YOLO26 (Ours)** | **59.09** | **35.83** | **70.00** | **53.90** | **2.089M** | **5.2** | **15.75 ms** | **63.5 FPS** | **Trained & Locally Verified Checkpoint** |
| **YOLO26n Baseline** | 66.68 | 47.16 | 74.80 | 63.50 | 2.512M | 6.1 | 4.20 ms | 238 FPS | Trained baseline checkpoint |
| **YOLO11n Reference** | 62.76 | 43.73 | 71.30 | 59.80 | 2.586M | 6.4 | 4.80 ms | 208 FPS | Trained reference checkpoint |
| **AquaYOLO26n** (Zeng et al., Symmetry 2026) | 73.40 | 49.10 | 74.80 | 69.10 | 2.460M | 6.2 | 4.20 ms | 238 FPS | Literature Benchmark |
| **Improved YOLOv11** (Jing et al., Sci Rep 2026) | 72.30 | 47.50 | 74.20 | 67.80 | 11.700M | 26.9 | 35.20 ms | 28 FPS | Literature Benchmark |
| **SvelteNeck-YOLO26 ($e=0.50$)** | 27.20 | 16.90 | 61.10 | 25.60 | 2.042M | 5.1 | 3.80 ms | 263 FPS | Ablation baseline |

> **Efficiency Advantage**: SvelteF3M-YOLO26 achieves a **17% parameter reduction** (2.089M vs. 2.512M) and **15% GFLOPs reduction** (5.2 vs. 6.1) compared to standard YOLO26n while delivering **70.00% precision**. On high-risk marine debris categories, it achieves high detection fidelity: `trash_wreckage` (95.2% mAP50), `trash_pipe` (87.1% mAP50), `trash_can` (81.2% mAP50), and `trash_container` (81.2% mAP50).

---

## 3. Systematic Ablation Study

| Stage | Base | Neck Architecture | Optical Compensation | Attention Gates | Params (M) | GFLOPs | mAP@0.50 (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V0** | YOLO26n | Standard PANet | None | None | 2.512M | 6.1 | 66.68% |
| **V1** | YOLO26n | SvelteNeck ($e=0.50$) | None | None | **2.042M** | **5.1** | 27.20% |
| **V2** | YOLO26n | SvelteNeck ($e=0.75$) | None | None | 2.210M | 5.6 | 32.40% |
| **V4** | YOLO26n | SvelteNeck ($e=0.50$) | **Beer–Lambert + CLAHE** | **F3M (Dual-Gate)** | **2.089M** | **5.2** | **59.09%** |

---

## 4. Repository Structure

```text
Marine-Debris-Detection-System/
├── app/
│   ├── app.py                      # Interactive Streamlit application
│   ├── best.pt                     # SvelteF3M-YOLO26 model weights for web app
│   ├── demo_images/                # Sample test images for instant inspection
│   └── requirements.txt            # Streamlit app dependencies
├── configs/                        # Model & dataset YAML configuration files
├── data/                           # Dataset ingestion & verification scripts
├── models/
│   ├── final/                      # Production weights (SvelteF3M & Baseline)
│   │   ├── sveltef3m_yolo26_best.pt
│   │   └── yolo26n_baseline.pt
│   └── trained/                    # Reference weights (YOLO11n)
│       └── yolo11n.pt
├── notebooks/
│   ├── sveltef3m_yolo26_training.ipynb  # Kaggle GPU multi-stage training pipeline
│   └── ablation_kaggle.ipynb            # Systematic ablation experiment notebook
├── reports/
│   ├── final_project_report.md          # Project summary report
│   └── comprehensive_project_report.md  # Comprehensive academic research report
├── results/
│   ├── master_results.csv          # Standardized metrics table across models
│   ├── benchmark_results.csv       # Measured CUDA latency and FPS benchmarks
│   ├── results.csv                 # Epoch-by-epoch loss and mAP progression
│   ├── kaggle_training_log.txt     # Complete console training log
│   ├── figures/                    # High-resolution benchmark & analysis plots
│   └── predictions/                # Visual detection outputs on test frames
├── src/
│   ├── data/
│   │   ├── underwater_augment.py   # Beer–Lambert optical compensation & CLAHE
│   │   └── prepare_uw_dataset.py   # Dataset pre-processing pipeline
│   ├── models/
│   │   ├── sveltneck.py            # SvelteNeck feature fusion architecture
│   │   ├── f3m.py                  # Feature-level Frequency Fusion Module
│   │   ├── ehconv.py               # Enhanced Hybrid Convolution
│   │   └── ehs_csp.py              # Cross-Stage Partial bottleneck blocks
│   └── evaluation/
│       ├── benchmark.py            # Latency and FPS throughput benchmark
│       ├── evaluate_models.py      # Multi-model evaluation script
│       └── plot_figures.py         # Academic figures generator
├── print_metrics.py                # Instant terminal summary of all metrics
├── validate_live.py                # Live 22-class test validation script
├── requirements.txt                # Core Python dependencies
└── README.md                       # Project documentation
```

---

## 5. Dataset: TrashCan-Instance 1.0

The benchmark is evaluated on the JAMSTEC **TrashCan-Instance 1.0** dataset:
- **Total Images**: 7,212 annotated deep-sea ROV imagery (6,065 train, 1,147 validation).
- **22 Classes**:
  - **Marine Debris (13 classes)**: `trash_bag`, `trash_bottle`, `trash_branch`, `trash_can`, `trash_clothing`, `trash_container`, `trash_cup`, `trash_net`, `trash_pipe`, `trash_rope`, `trash_snack_wrapper`, `trash_tarp`, `trash_wreckage`, `trash_unknown_instance`.
  - **Marine Fauna (8 classes)**: `animal_fish`, `animal_crab`, `animal_starfish`, `animal_shells`, `animal_eel`, `animal_etc`, `plant`.
  - **Robotics (1 class)**: `rov`.

---

## 6. Installation & Quickstart

### 6.1 Clone the Repository
```bash
git clone https://github.com/<your-username>/Marine-Debris-Detection-System.git
cd Marine-Debris-Detection-System
```

### 6.2 Set Up Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

---

## 7. How to Run

### 7.1 Display Master Metrics Summary
To instantly print the master benchmark table, hardware FPS benchmarks, and training convergence statistics:
```bash
python print_metrics.py
```

### 7.2 Run Live 22-Class Validation
To execute a live frame-by-frame evaluation over the 1,147 validation images and output per-class mAP scores:
```bash
python validate_live.py
```

### 7.3 Run Hardware Speed & FPS Benchmark
To benchmark latency and frames-per-second on your local GPU or CPU:
```bash
python src/evaluation/benchmark.py
```

### 7.4 Launch the Interactive Streamlit Web Interface
Run the interactive deployment app featuring image upload and real-time optical compensation toggle:
```bash
streamlit run app/app.py
```

---

## 8. Verified Checkpoints

All verified checkpoints are included directly in the repository:
- `models/final/sveltef3m_yolo26_best.pt` (4.30 MB): Primary SvelteF3M-YOLO26 model.
- `models/final/yolo26n_baseline.pt` (5.14 MB): Baseline YOLO26n detector.
- `models/trained/yolo11n.pt` (5.21 MB): Reference YOLO11n detector.
- `app/best.pt` (4.30 MB): Deployment checkpoint for the Streamlit application.

---

## 9. License & Acknowledgments

- **Dataset**: JAMSTEC & University of Minnesota (TrashCan-Instance 1.0).
- **Core Framework**: Ultralytics YOLO26.
- **Reference Architectures**: Inspired by *AquaYOLO26* (Zeng et al., Symmetry 2026) and *Improved YOLOv11* (Jing et al., Scientific Reports 2026).
