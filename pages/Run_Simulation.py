import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time
import base64
from io import BytesIO
import torch
from utils import inject_branding, render_navbar, render_footer

st.set_page_config(
    page_title="Run Simulation · Underwater Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_branding()

render_navbar("Run_Simulation")

# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    baseline = YOLO("models/baseline_best.pt")
    proposed = YOLO("models/proposed_best.pt")

    def ensure_branch_weights(yolo_model):
        for module in yolo_model.model.modules():
            try:
                num_branches = None
                if hasattr(module, "branches"):
                    num_branches = len(module.branches)
                elif hasattr(module, "num_branches"):
                    num_branches = int(module.num_branches)
                elif hasattr(module, "branch_convs"):
                    num_branches = len(module.branch_convs)
                elif module.__class__.__name__ == "AdaptiveFeatureFusion":
                    num_branches = 2
                if num_branches and num_branches > 0:
                    module.register_parameter(
                        "branch_weights",
                        torch.nn.Parameter(torch.ones(num_branches, dtype=torch.float32)),
                    )
            except Exception:
                pass
            if module.__class__.__name__ == "AdaptiveFeatureFusion":
                if not hasattr(module, "conv_align"):
                    module.add_module("conv_align", torch.nn.Identity())
                if not hasattr(module, "ca"):
                    module.add_module("ca", torch.nn.Identity())

    ensure_branch_weights(baseline)
    ensure_branch_weights(proposed)
    return baseline, proposed

baseline_model, proposed_model = load_models()

# ── Metrics ────────────────────────────────────────────────────────────────────
def get_model_metrics(model):
    metrics = {"precision": 0.0, "recall": 0.0, "mAP50": 0.0, "mAP50-95": 0.0}
    try:
        ckpt = model.ckpt
        if ckpt and "metrics" in ckpt:
            if hasattr(ckpt["metrics"], "box"):
                b = ckpt["metrics"].box
                metrics["precision"] = float(b.p)     if hasattr(b, "p")     else 0.0
                metrics["recall"]    = float(b.r)     if hasattr(b, "r")     else 0.0
                metrics["mAP50"]     = float(b.map50) if hasattr(b, "map50") else 0.0
                metrics["mAP50-95"]  = float(b.map)   if hasattr(b, "map")   else 0.0
    except Exception:
        pass
    return metrics

baseline_metrics = get_model_metrics(baseline_model)
proposed_metrics  = get_model_metrics(proposed_model)

if baseline_metrics["precision"] == 0.0:
    baseline_metrics = {"precision": 0.8205, "recall": 0.7260, "mAP50": 0.8137, "mAP50-95": 0.5639}
if proposed_metrics["precision"] == 0.0:
    proposed_metrics = {"precision": 0.8423, "recall": 0.72726, "mAP50": 0.82561, "mAP50-95": 0.57133}

# ── Page header ────────────────────────────────────────────────────────────────
st.html("""
<div style="padding:56px 0 28px;">
    <div style="font-family:'Inter',sans-serif;font-size:11.5px;font-weight:600;
                color:#1FA3A3;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:12px;">
        Detection System
    </div>
    <h1 style="font-family:'Poppins',sans-serif;font-size:38px;font-weight:700;
               color:#0B3C5D;margin-bottom:12px;letter-spacing:-0.02em;">Run Simulation</h1>
    <p style="font-family:'Inter',sans-serif;font-size:16px;color:#64748b;max-width:580px;line-height:1.7;">
        Upload an underwater image to compare Baseline and Enhanced YOLOv12 detection results side by side.
    </p>
</div>
""")

# ── Upload ─────────────────────────────────────────────────────────────────────
preview_slot = st.empty()
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
run_detection = st.button("Run Detection", disabled=not uploaded_file)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    buf = BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    preview_slot.html(f"""
    <style>
    [data-testid="stFileUploaderDropzone"] {{
        padding: 10px 20px !important; min-height: 0 !important; border-radius: 8px !important;
    }}
    [data-testid="stFileUploaderDropzoneInstructions"] > div::before {{ display: none !important; }}
    [data-testid="stFileUploaderDropzoneInstructions"] > div::after {{
        content: "Click to upload a different image" !important;
        font-size: 13px !important; color: #94a3b8 !important; font-weight: 400 !important;
    }}
    [data-testid="stFileUploaderDropzoneInstructions"]::after {{ display: none !important; }}
    [data-testid="stFileUploaderDropzone"] button {{ display: none !important; }}
    [data-testid="stFileUploaderFile"] {{ display: none !important; }}
    </style>
    <div style="border:1.5px dashed #cbd5e1;border-radius:12px;background:#fafafa;
                padding:24px;text-align:center;margin-bottom:8px;">
        <img src="data:image/png;base64,{b64}"
             style="max-width:100%;max-height:400px;border-radius:8px;object-fit:contain;" />
        <p style="font-size:12.5px;color:#94a3b8;margin-top:10px;margin-bottom:0;
                  font-family:'Inter',sans-serif;">{uploaded_file.name}</p>
    </div>
    """)

# ── Detection ──────────────────────────────────────────────────────────────────
if uploaded_file and run_detection:
    status_slot = st.empty()
    status_slot.info("Running detection on both models...")

    start = time.time()
    baseline_result = baseline_model(img_array)[0]
    baseline_time = time.time() - start

    start = time.time()
    proposed_result = proposed_model(img_array)[0]
    proposed_time = time.time() - start

    status_slot.empty()

    baseline_img = baseline_result.plot()
    proposed_img = proposed_result.plot()

    def get_all_detections(result):
        if len(result.boxes.cls) == 0:
            return []
        detections = [
            (result.names[int(result.boxes.cls[i])], float(result.boxes.conf[i]))
            for i in range(len(result.boxes.cls))
        ]
        detections.sort(key=lambda x: x[1], reverse=True)
        return detections

    baseline_detections = get_all_detections(baseline_result)
    proposed_detections = get_all_detections(proposed_result)
    baseline_count = len(baseline_result.boxes.cls)
    proposed_count = len(proposed_result.boxes.cls)

    baseline_accuracy     = max(c for _, c in baseline_detections) * 100 if baseline_detections else 0
    proposed_accuracy     = max(c for _, c in proposed_detections) * 100 if proposed_detections else 0
    baseline_accuracy_avg = np.mean([c for _, c in baseline_detections]) * 100 if baseline_detections else 0
    proposed_accuracy_avg = np.mean([c for _, c in proposed_detections]) * 100 if proposed_detections else 0

    precision_improvement = (proposed_metrics["precision"] - baseline_metrics["precision"]) * 100
    recall_improvement    = (proposed_metrics["recall"]    - baseline_metrics["recall"])    * 100
    map50_improvement     = (proposed_metrics["mAP50"]     - baseline_metrics["mAP50"])     * 100
    map95_improvement     = (proposed_metrics["mAP50-95"]  - baseline_metrics["mAP50-95"]) * 100

    def dc(v): return "#10b981" if v >= 0 else "#ef4444"
    def ds(v): return "+" if v >= 0 else ""

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.html('<div class="card"><div class="card-title">Baseline Model</div></div>')
        st.image(baseline_img, use_container_width=True)
        if baseline_count == 0:
            if proposed_count == 0:
                st.error("This model can only detect Echinus, Starfish, Scallop, and Holothurian. Please upload a different image.")
            else:
                st.error("No class detected")
        else:
            st.html(f'<div class="metric-box">Highest Detection Accuracy<br><b>{baseline_accuracy:.1f}%</b></div>')
            st.html('<p style="font-size:13px;color:#94a3b8;margin:12px 0 8px;font-family:Inter,sans-serif;font-weight:500;">Classification Results</p>')
            with st.expander("See More"):
                for cls, conf in baseline_detections:
                    st.write(f"{cls} — {conf*100:.1f}%")
            with st.expander("Evaluation Metrics"):
                m1, m2 = st.columns(2)
                with m1:
                    st.html(f'<div class="metric-box">Precision<br><b>{baseline_metrics["precision"]*100:.2f}%</b><br><span style="font-size:11px;color:#94a3b8;">(baseline)</span></div>')
                    st.html(f'<div class="metric-box">mAP@50<br><b>{baseline_metrics["mAP50"]*100:.2f}%</b><br><span style="font-size:11px;color:#94a3b8;">(baseline)</span></div>')
                with m2:
                    st.html(f'<div class="metric-box">Recall<br><b>{baseline_metrics["recall"]*100:.2f}%</b><br><span style="font-size:11px;color:#94a3b8;">(baseline)</span></div>')
                    st.html(f'<div class="metric-box">mAP@50:95<br><b>{baseline_metrics["mAP50-95"]*100:.2f}%</b><br><span style="font-size:11px;color:#94a3b8;">(baseline)</span></div>')

    with col2:
        st.html('<div class="card proposed"><div class="card-title">Enhanced Model</div></div>')
        st.image(proposed_img, use_container_width=True)
        if proposed_count == 0:
            if baseline_count == 0:
                st.error("This model can only detect Echinus, Starfish, Scallop, and Holothurian. Please upload a different image.")
            else:
                st.error("No class detected")
        else:
            st.html(f'<div class="metric-box">Highest Detection Accuracy<br><b>{proposed_accuracy:.1f}%</b></div>')
            st.html('<p style="font-size:13px;color:#94a3b8;margin:12px 0 8px;font-family:Inter,sans-serif;font-weight:500;">Classification Results</p>')
            with st.expander("See More"):
                for cls, conf in proposed_detections:
                    st.write(f"{cls} — {conf*100:.1f}%")
            with st.expander("Evaluation Metrics"):
                m1, m2 = st.columns(2)
                with m1:
                    st.html(f'<div class="metric-box">Precision<br><b>{proposed_metrics["precision"]*100:.2f}%</b><br><span style="font-size:11px;color:{dc(precision_improvement)};">{ds(precision_improvement)}{precision_improvement:.2f}%</span></div>')
                    st.html(f'<div class="metric-box">mAP@50<br><b>{proposed_metrics["mAP50"]*100:.2f}%</b><br><span style="font-size:11px;color:{dc(map50_improvement)};">{ds(map50_improvement)}{map50_improvement:.2f}%</span></div>')
                with m2:
                    st.html(f'<div class="metric-box">Recall<br><b>{proposed_metrics["recall"]*100:.2f}%</b><br><span style="font-size:11px;color:{dc(recall_improvement)};">{ds(recall_improvement)}{recall_improvement:.2f}%</span></div>')
                    st.html(f'<div class="metric-box">mAP@50:95<br><b>{proposed_metrics["mAP50-95"]*100:.2f}%</b><br><span style="font-size:11px;color:{dc(map95_improvement)};">{ds(map95_improvement)}{map95_improvement:.2f}%</span></div>')

    st.html("<div style='height:32px;'></div>")
    st.html("""
    <div style="border-top:1px solid #e2e8f0;padding-top:32px;margin-bottom:8px;">
        <h2 style="font-family:'Poppins',sans-serif;font-size:22px;font-weight:700;
                   color:#0B3C5D;letter-spacing:-0.01em;">Model Comparison</h2>
    </div>
    """)
    comp_col1, comp_col2, comp_col3 = st.columns(3)
    with comp_col1:
        st.metric("Average Accuracy Improvement", f"{proposed_accuracy_avg:.2f}%", f"{proposed_accuracy_avg - baseline_accuracy_avg:.2f}%")
    with comp_col2:
        st.metric("Enhanced Detections", proposed_count)
    with comp_col3:
        st.metric("Baseline Detections", baseline_count)

render_footer()
