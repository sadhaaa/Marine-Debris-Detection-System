"""
SvelteNeck-YOLO26 — Underwater Marine Debris Detection
Streamlit inference app.
"""

from pathlib import Path
import time

import streamlit as st
from PIL import Image
import numpy as np

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Marine Debris Detector",
    page_icon="🌊",
    layout="wide",
)

# ── Paths ─────────────────────────────────────────────────────
APP_DIR    = Path(__file__).resolve().parent
ROOT       = APP_DIR.parent
MODEL_DIR  = ROOT / "app"
DEMO_DIR   = APP_DIR / "demo_images"

# Best model preference order
MODEL_CANDIDATES = [
    MODEL_DIR / "best.pt",
    ROOT / "experiments" / "05_sveltneck_f3m"  / "sveltneck_f3m_seed42"   / "weights" / "best.pt",
    ROOT / "experiments" / "03_sveltneck_e050" / "sveltneck_e050_seed42"  / "weights" / "best.pt",
    ROOT / "experiments" / "01_yolo26n"        / "yolo26n_baseline_seed42" / "weights" / "best.pt",
    ROOT / "weights" / "yolo26n.pt",
]

CLASS_NAMES = [
    "can", "carton", "metal", "misc-debris", "paper", "pipe",
    "plastic-bag", "plastic-bottle", "plastic-container", "plastic-gloves",
    "plastic-net", "rope", "rov", "rubber-gloves", "scarf", "shoe",
    "snack-wrapper", "textile", "tire", "towel", "wood", "fishing-gear",
]

# Colour palette (one per class)
PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
    "#ffffff", "#000000",
]


# ── Model loading ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    try:
        from ultralytics import YOLO
    except ImportError:
        st.error("ultralytics not installed. Run: pip install ultralytics")
        return None

    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            try:
                model = YOLO(str(candidate))
                return model, str(candidate)
            except Exception as e:
                continue

    return None, None


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/dolphin.png", width=80)
    st.title("🌊 Marine Debris\nDetector")
    st.markdown("---")

    conf_thresh = st.slider("Confidence threshold", 0.01, 0.95, 0.25, 0.01)
    iou_thresh  = st.slider("NMS IoU threshold",    0.10, 0.95, 0.45, 0.05)

    st.markdown("---")
    st.subheader("Model Info")

    model_result = load_model()
    if model_result and model_result[0] is not None:
        model, model_path = model_result
        st.success("Model loaded ✓")
        st.caption(f"`{Path(model_path).name}`")
        try:
            info = model.info(verbose=False)
            if isinstance(info, (list, tuple)) and len(info) >= 4:
                st.metric("Parameters", f"{info[1]/1e6:.3f} M")
                st.metric("GFLOPs", f"{info[3]:.2f}")
        except Exception:
            pass
    else:
        st.error("No model found. Train first!")
        st.stop()

    st.markdown("---")
    st.markdown("**Target metrics:**")
    st.markdown("- mAP50 > 89.8%")
    st.markdown("- mAP50-95 > 69.5%")
    st.markdown("- Params ≤ 2.243 M")
    st.markdown("- GFLOPs ≤ 5.8")


# ── Main area ─────────────────────────────────────────────────
st.title("🌊 Underwater Marine Debris Detection")
st.caption("SvelteNeck-YOLO26 · TrashCan-Instance · 22 classes")

tab_upload, tab_demo = st.tabs(["📤 Upload Image", "🖼️ Demo Images"])

# ── Helper: run inference ─────────────────────────────────────
def run_inference(image: Image.Image):
    img_array = np.array(image)

    t0      = time.perf_counter()
    results = model.predict(
        source=img_array,
        conf=conf_thresh,
        iou=iou_thresh,
        verbose=False,
        save=False,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    fps        = 1000 / latency_ms if latency_ms > 0 else 0

    result = results[0]

    # Annotated image (ultralytics renders bboxes)
    annotated = Image.fromarray(result.plot())

    # Detection table
    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            detections.append({
                "Class":      name,
                "Confidence": f"{conf:.2%}",
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })

    return annotated, detections, latency_ms, fps


# ── Upload tab ────────────────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader(
        "Upload an underwater image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Detections")
            with st.spinner("Running inference…"):
                annotated, detections, latency_ms, fps = run_inference(image)
            st.image(annotated, use_container_width=True)

        # Metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("Detections", len(detections))
        m2.metric("Latency", f"{latency_ms:.1f} ms")
        m3.metric("Throughput", f"{fps:.1f} FPS")

        if detections:
            import pandas as pd
            df = pd.DataFrame(detections)
            st.dataframe(df[["Class", "Confidence"]], use_container_width=True)
        else:
            st.info("No objects detected above confidence threshold.")

# ── Demo tab ──────────────────────────────────────────────────
with tab_demo:
    demo_images = sorted(DEMO_DIR.glob("*.jpg")) if DEMO_DIR.exists() else []

    if not demo_images:
        st.warning("No demo images found in `app/demo_images/`. Add some .jpg files.")
    else:
        demo_names = [f.name for f in demo_images]
        selected   = st.selectbox("Select a demo image", demo_names)
        demo_path  = DEMO_DIR / selected

        image = Image.open(demo_path).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Detections")
            with st.spinner("Running inference…"):
                annotated, detections, latency_ms, fps = run_inference(image)
            st.image(annotated, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Detections", len(detections))
        m2.metric("Latency", f"{latency_ms:.1f} ms")
        m3.metric("Throughput", f"{fps:.1f} FPS")

        if detections:
            import pandas as pd
            df = pd.DataFrame(detections)
            st.dataframe(df[["Class", "Confidence"]], use_container_width=True)
        else:
            st.info("No objects detected above the threshold.")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "SvelteNeck-YOLO26 · TrashCan-Instance (22 classes) · "
    "Increase confidence threshold if you see too many false positives."
)
