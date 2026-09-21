"""
Marine Debris Detection System
Streamlit app using trained YOLO26n on TrashCan-Instance dataset (22 classes)
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# ── Ensure project root is in sys.path ─────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Marine Debris Detector",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_CANDIDATES = [
    ROOT / "models" / "final" / "sveltef3m_yolo26_best.pt",
    ROOT / "experiments" / "04_sveltef3m_yolo26" / "weights" / "best.pt",
    ROOT / "experiments" / "01_yolo26n" / "weights" / "best.pt",
    APP_DIR / "best.pt",
    ROOT / "weights" / "yolo26n.pt",
]

CLASS_NAMES = [
    "rov", "plant", "animal_fish", "animal_starfish", "animal_shells",
    "animal_crab", "animal_eel", "animal_etc", "trash_clothing", "trash_pipe",
    "trash_bottle", "trash_bag", "trash_snack_wrapper", "trash_can", "trash_cup",
    "trash_container", "trash_unknown_instance", "trash_branch", "trash_wreckage",
    "trash_tarp", "trash_rope", "trash_net",
]

# Category groupings for summary
TRASH_CLASSES = {n for n in CLASS_NAMES if n.startswith("trash_")}
ANIMAL_CLASSES = {n for n in CLASS_NAMES if n.startswith("animal_")}


# ── Inlined Optical Compensation to guarantee standalone robustness ──────────
def underwater_degradation_compensation(image_bgr: np.ndarray) -> np.ndarray:
    """
    Applies software-level channel-asymmetric color attenuation recovery
    and contrast enhancement without modifying neural network layers:
      1. Red channel boost (Beer-Lambert attenuation prior compensation).
      2. CLAHE on Luminance channel in LAB color space (turbidity reduction).
    """
    img = image_bgr.astype(np.float32)
    b_mean = np.mean(img[:, :, 0]) + 1e-5
    g_mean = np.mean(img[:, :, 1]) + 1e-5
    r_mean = np.mean(img[:, :, 2]) + 1e-5
    compensation_factor = np.clip((g_mean + b_mean) / (2.0 * r_mean), 1.0, 2.5)
    img[:, :, 2] = np.clip(img[:, :, 2] * compensation_factor, 0, 255)
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    return enhanced


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading detection model…")
def load_model():
    from ultralytics import YOLO
    import ultralytics.nn.modules as um
    import ultralytics.nn.tasks as ut
    from src.models.sveltneck import SvelteNeck
    from src.models.f3m import F3M
    from src.models.ehconv import EHConv
    from src.models.ehs_bottleneck import EHSBottleneck
    from src.models.ehs_csp import EHSCSP
    for cls in [SvelteNeck, F3M, EHConv, EHSBottleneck, EHSCSP]:
        setattr(um, cls.__name__, cls)
        setattr(ut, cls.__name__, cls)

    for path in MODEL_CANDIDATES:
        if path.exists():
            try:
                model = YOLO(str(path))
                return model, path
            except Exception:
                continue
    return None, None



# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; }
.trash-badge  { background:#ff4b4b; color:white; padding:2px 8px; border-radius:12px; font-size:0.8rem; }
.animal-badge { background:#21c354; color:white; padding:2px 8px; border-radius:12px; font-size:0.8rem; }
.other-badge  { background:#4b6bfb; color:white; padding:2px 8px; border-radius:12px; font-size:0.8rem; }
h1 { color: #0e4d92; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 Marine Debris\nDetector")
    st.markdown("---")

    model, model_path = load_model()

    if model is None:
        st.error("❌ No trained model found.\nCheck that weights exist in `experiments/01_yolo26n/weights/best.pt`")
        st.stop()

    st.success("✅ Model loaded")
    model_name = model_path.parent.parent.name if model_path else "unknown"
    st.caption(f"**Model:** `{model_path.name}`")
    st.caption(f"**Run:** `{model_name}`")

    try:
        info = model.info(verbose=False)
        if isinstance(info, (list, tuple)) and len(info) >= 4:
            st.metric("Parameters", f"{info[1]/1e6:.2f} M")
            st.metric("GFLOPs",     f"{info[3]:.1f}")
    except Exception:
        pass

    st.markdown("---")
    st.markdown("### ⚙️ Detection Settings")
    conf_thresh = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    iou_thresh  = st.slider("NMS IoU threshold",    0.10, 0.90, 0.45, 0.05)
    img_size    = st.select_slider("Image size", [320, 480, 640], value=640)
    enable_uw_comp = st.checkbox(
        "🌊 Enable Optical Compensation (Beer-Lambert + CLAHE)",
        value=True,
        help="Applies Beer-Lambert red channel restoration and CLAHE contrast enhancement"
    )

    st.markdown("---")
    st.markdown("### 📊 Benchmark Comparison")
    st.markdown("| Model | mAP50 | Params | GFLOPs |")
    st.markdown("|:---|:---:|:---:|:---:|")
    st.markdown("| **SvelteF3M-YOLO26 (Ours)** | **59.09%** | **2.09M** | **5.2** |")

    st.markdown("| YOLO26n Baseline (Ours) | 66.68% | 2.51M | 6.1 |")
    st.markdown("| YOLO11n Reference (Ours) | 62.76% | 2.58M | 6.4 |")
    st.markdown("| AquaYOLO26n (Zeng et al.) | 73.40% | 2.46M | 6.2 |")
    st.markdown("| Improved YOLOv11 (2026) | 72.30% | 11.70M | 26.9 |")
    st.markdown("| YOLOTrashCan (2023) | 65.01% | 214.0M | — |")

    st.markdown("---")
    st.markdown("### 🏷️ 22 Classes")
    for cls in CLASS_NAMES:
        badge = "🗑️" if cls in TRASH_CLASSES else ("🐟" if cls in ANIMAL_CLASSES else "🤖")
        st.caption(f"{badge} {cls}")


# ── Inference helper ───────────────────────────────────────────────────────────
def run_inference(image: Image.Image):
    img_array = np.array(image)
    if enable_uw_comp:
        img_bgr = img_array[:, :, ::-1]
        enhanced_bgr = underwater_degradation_compensation(img_bgr)
        infer_input = enhanced_bgr[:, :, ::-1]
    else:
        infer_input = img_array

    t0 = time.perf_counter()
    results = model.predict(
        source=infer_input,
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=img_size,
        verbose=False,
        save=False,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    result = results[0]

    annotated = Image.fromarray(result.plot())

    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            w_box, h_box = x2 - x1, y2 - y1
            detections.append({
                "Class":      name,
                "Category":   "🗑️ Trash" if name in TRASH_CLASSES else ("🐟 Animal" if name in ANIMAL_CLASSES else "🤖 Other"),
                "Confidence": f"{conf:.1%}",
                "Conf_val":   conf,
                "BBox (px)":  f"[{x1},{y1}] {w_box}×{h_box}",
            })
    detections.sort(key=lambda d: d["Conf_val"], reverse=True)
    return annotated, detections, latency_ms


# ── Main ───────────────────────────────────────────────────────────────────────
st.title("🌊 Underwater Marine Debris Detection")
st.caption(
    "**YOLO26n** · TrashCan-Instance · 22 classes · "
    "mAP50 = 66.68% · mAP50-95 = 47.16%"
)

tab_upload, tab_demo, tab_about = st.tabs(["📤 Upload Image", "🖼️ Demo", "ℹ️ About"])

# ── Upload tab ─────────────────────────────────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader(
        "Upload an underwater image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Upload any image — works best with underwater/marine imagery",
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            st.subheader("Input")
            st.image(image, width="stretch")
            st.caption(f"Size: {image.width}×{image.height} px")

        with col2:
            st.subheader("Detections")
            with st.spinner("Running inference…"):
                annotated, detections, latency_ms = run_inference(image)
            st.image(annotated, width="stretch")

        n_trash  = sum(1 for d in detections if "Trash" in d["Category"])
        n_animal = sum(1 for d in detections if "Animal" in d["Category"])
        n_other  = len(detections) - n_trash - n_animal

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total", len(detections))
        m2.metric("🗑️ Trash",  n_trash)
        m3.metric("🐟 Animal", n_animal)
        m4.metric("🤖 Other",  n_other)
        m5.metric("Latency",   f"{latency_ms:.0f} ms")

        if detections:
            st.markdown("#### Detection Details")
            df = pd.DataFrame(detections)[["Class", "Category", "Confidence", "BBox (px)"]]
            st.dataframe(df, width="stretch", hide_index=True)

            st.markdown("#### Category Summary")
            cats = {}
            for d in detections:
                cats[d["Category"]] = cats.get(d["Category"], 0) + 1
            c1, c2, c3 = st.columns(3)
            for i, (cat, cnt) in enumerate(sorted(cats.items())):
                [c1, c2, c3][i % 3].metric(cat, cnt)
        else:
            st.info(f"No objects detected above {conf_thresh:.0%} confidence. Try lowering the threshold.")

# ── Demo tab ───────────────────────────────────────────────────────────────────
with tab_demo:
    DEMO_DIR = APP_DIR / "demo_images"
    demo_images = sorted(DEMO_DIR.glob("*.jpg")) + sorted(DEMO_DIR.glob("*.png")) if DEMO_DIR.exists() else []

    if not demo_images:
        st.info(
            "📁 No demo images found. Add some `.jpg` or `.png` files to `app/demo_images/` "
            "and they'll appear here automatically.\n\n"
            "You can also try the **Upload Image** tab with your own images."
        )
    else:
        selected = st.selectbox("Select a demo image", [f.name for f in demo_images])
        image = Image.open(DEMO_DIR / selected).convert("RGB")

        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.subheader("Input")
            st.image(image, width="stretch")

        with col2:
            st.subheader("Detections")
            with st.spinner("Running inference…"):
                annotated, detections, latency_ms = run_inference(image)
            st.image(annotated, width="stretch")

        m1, m2, m3 = st.columns(3)
        m1.metric("Detections", len(detections))
        m2.metric("Latency", f"{latency_ms:.0f} ms")
        m3.metric("FPS", f"{1000/latency_ms:.1f}" if latency_ms > 0 else "—")

        if detections:
            df = pd.DataFrame(detections)[["Class", "Category", "Confidence"]]
            st.dataframe(df, width="stretch", hide_index=True)

# ── About tab ──────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
## 🌊 Marine Debris Detection System

This application detects underwater marine debris and animals using a YOLO-based
object detection model trained on the **TrashCan-Instance** dataset.

### 🏗️ Architecture
| Component | Details |
|-----------|---------|
| **Backbone** | YOLO26n (scale n, width=0.25) |
| **Dataset** | TrashCan-Instance — 6065 train / 1147 val images |
| **Classes** | 22 (13 trash types + 8 animal types + ROV) |
| **Framework** | Ultralytics YOLO |

### 📊 Training Results
| Model | mAP50 | mAP50-95 | Params |
|-------|-------|----------|--------|
| YOLO26n (baseline) | **66.68%** | 47.16% | ~2.2 M |
| YOLO11n (reference) | 62.76% | 43.73% | ~2.6 M |

### 🗑️ Trash Classes (13)
`trash_bag`, `trash_bottle`, `trash_branch`, `trash_can`, `trash_clothing`,
`trash_container`, `trash_cup`, `trash_net`, `trash_pipe`, `trash_rope`,
`trash_snack_wrapper`, `trash_tarp`, `trash_unknown_instance`, `trash_wreckage`

### 🐟 Animal Classes (8)
`animal_crab`, `animal_eel`, `animal_etc`, `animal_fish`,
`animal_shells`, `animal_starfish`

### 🤖 Other Classes
`rov` (Remote Operated Vehicle), `plant`

### 🚀 Usage Tips
- Toggle **Enable Optical Compensation** to dynamically correct underwater color distortion & backscatter.
- Lower the **Confidence threshold** to detect more objects (may increase false positives).
- Use **640px** image size for best accuracy.
""")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🌊 Marine Debris Detector · YOLO26n · TrashCan-Instance (22 classes) · "
    "Adjust the confidence slider in the sidebar if detections seem off."
)
