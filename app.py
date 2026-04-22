import streamlit as st
from utils import inject_branding, render_navbar, render_footer

st.set_page_config(
    page_title="Underwater Object Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_branding()
render_navbar("Home")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 80px 0 60px; text-align: center;">
    <p style="font-family:'Inter',sans-serif;font-size:13px;font-weight:600;
              color:#1FA3A3;letter-spacing:2px;text-transform:uppercase;margin-bottom:16px;">
        University of Mindanao · Thesis Research
    </p>
    <h1 style="font-family:'Poppins',sans-serif;font-size:44px;font-weight:700;
               color:#0B3C5D;line-height:1.25;margin-bottom:20px;">
        An Enhanced YOLOv12 Architecture<br>with Dual-Branch Network for<br>Underwater Object Detection
    </h1>
    <p style="font-family:'Inter',sans-serif;font-size:17px;color:#4b6a7d;
              max-width:720px;margin:0 auto 40px;line-height:1.75;">
        Improving underwater object detection by enhancing YOLOv12 with a Dual-Branch Input Stem
        that preserves visual features while reducing the effects of underwater noise and image degradation.
    </p>
    <div style="display:flex;gap:16px;justify-content:center;flex-wrap:wrap;">
        <a href="/Run_Simulation" style="
            display:inline-block;background:#0B3C5D;color:#ffffff;
            text-decoration:none;font-family:'Inter',sans-serif;
            font-weight:600;font-size:15px;padding:14px 32px;border-radius:8px;">
            Run Simulation
        </a>
        <a href="/Instructions" style="
            display:inline-block;background:transparent;color:#0B3C5D;
            border:2px solid #0B3C5D;text-decoration:none;
            font-family:'Inter',sans-serif;font-weight:600;
            font-size:15px;padding:12px 30px;border-radius:8px;">
            View Instructions
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Class badges ──────────────────────────────────────────────────────────────
classes = ["Echinus", "Holothurian", "Scallop", "Starfish"]
badges = "".join(
    f'<span style="background:#E8F1F2;color:#0B3C5D;border:1px solid #CFE3E6;'
    f'border-radius:20px;padding:6px 18px;font-family:Inter,sans-serif;'
    f'font-size:14px;font-weight:500;">{c}</span>'
    for c in classes
)
st.markdown(
    f'<div style="display:flex;justify-content:center;gap:12px;flex-wrap:wrap;margin-bottom:64px;">{badges}</div>',
    unsafe_allow_html=True,
)

st.divider()

# ── About the Study ───────────────────────────────────────────────────────────
col_txt, col_card = st.columns([3, 2], gap="large")
with col_txt:
    st.markdown("""
    <h2 style="font-family:'Poppins',sans-serif;font-size:30px;font-weight:700;
               color:#0B3C5D;margin-bottom:16px;">About the Study</h2>
    <p style="font-family:'Inter',sans-serif;font-size:16px;color:#374151;line-height:1.75;margin-bottom:16px;">
        Underwater images often suffer from blur, color distortion, poor visibility, and environmental noise.
        These challenges make object detection difficult and reduce model performance.
    </p>
    <p style="font-family:'Inter',sans-serif;font-size:16px;color:#374151;line-height:1.75;margin-bottom:24px;">
        This study addresses that problem by improving YOLOv12 through a dual-branch architecture composed of
        a <strong>standard branch</strong> for structural feature extraction, a <strong>denoising branch</strong>
        for noise suppression, and <strong>adaptive feature fusion</strong> to combine both outputs effectively.
    </p>
    <a href="#" style="
        display:inline-flex;align-items:center;gap:8px;
        background:#1FA3A3;color:#ffffff;text-decoration:none;
        font-family:'Inter',sans-serif;font-weight:600;font-size:14px;
        padding:12px 24px;border-radius:8px;">
        📄 View Thesis Document
    </a>
    """, unsafe_allow_html=True)
with col_card:
    st.markdown("""
    <div style="background:#0B3C5D;border-radius:16px;padding:40px 32px;color:#ffffff;
                display:flex;flex-direction:column;justify-content:center;min-height:300px;">
        <div style="font-family:'Poppins',sans-serif;font-size:12px;font-weight:600;
                    color:#1FA3A3;letter-spacing:2px;margin-bottom:20px;">DUAL-BRANCH ARCHITECTURE</div>
        <div style="display:flex;flex-direction:column;gap:14px;">
            <div style="background:rgba(31,163,163,0.15);border-left:3px solid #1FA3A3;padding:12px 16px;border-radius:4px;">
                <div style="font-family:'Poppins',sans-serif;font-weight:600;font-size:14px;margin-bottom:4px;">Standard Branch</div>
                <div style="font-family:'Inter',sans-serif;font-size:13px;color:#94b4c5;">Structural feature extraction</div>
            </div>
            <div style="background:rgba(31,163,163,0.15);border-left:3px solid #1FA3A3;padding:12px 16px;border-radius:4px;">
                <div style="font-family:'Poppins',sans-serif;font-weight:600;font-size:14px;margin-bottom:4px;">Denoising Branch</div>
                <div style="font-family:'Inter',sans-serif;font-size:13px;color:#94b4c5;">Noise suppression & correction</div>
            </div>
            <div style="background:rgba(31,163,163,0.25);border-left:3px solid #ffffff;padding:12px 16px;border-radius:4px;">
                <div style="font-family:'Poppins',sans-serif;font-weight:600;font-size:14px;margin-bottom:4px;">Adaptive Feature Fusion</div>
                <div style="font-family:'Inter',sans-serif;font-size:13px;color:#94b4c5;">Combined output for final detection</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ── Objectives ────────────────────────────────────────────────────────────────
st.markdown("""
<h2 style="font-family:'Poppins',sans-serif;font-size:30px;font-weight:700;
           color:#0B3C5D;margin-bottom:8px;text-align:center;">Objectives</h2>
<p style="font-family:'Inter',sans-serif;font-size:16px;color:#6b7280;
          text-align:center;margin-bottom:36px;">What this research aims to achieve</p>
""", unsafe_allow_html=True)

obj_col1, obj_col2 = st.columns(2, gap="large")
with obj_col1:
    st.markdown("""
    <div class="card" style="height:100%;">
        <div style="font-family:'Poppins',sans-serif;font-size:12px;font-weight:600;
                    color:#1FA3A3;letter-spacing:1.5px;margin-bottom:12px;">GENERAL OBJECTIVE</div>
        <p style="font-family:'Inter',sans-serif;font-size:15px;color:#374151;line-height:1.7;margin:0;">
            To enhance YOLOv12's performance as a detection model for underwater objects by developing an
            improved architecture that can better handle noise in underwater environments.
        </p>
    </div>
    """, unsafe_allow_html=True)
with obj_col2:
    specific = [
        "Preprocess underwater datasets for training and evaluation",
        "Develop a new model by integrating a dual-branch input stem into the baseline YOLOv12",
        "Compare the baseline and proposed model under varying noise levels",
        "Evaluate both models using Precision, Recall, mAP@50, and mAP@50:95",
    ]
    items = "".join(
        f'<li style="font-family:Inter,sans-serif;font-size:15px;color:#374151;'
        f'line-height:1.7;margin-bottom:8px;">{s}</li>'
        for s in specific
    )
    st.markdown(f"""
    <div class="card" style="height:100%;">
        <div style="font-family:'Poppins',sans-serif;font-size:12px;font-weight:600;
                    color:#1FA3A3;letter-spacing:1.5px;margin-bottom:12px;">SPECIFIC OBJECTIVES</div>
        <ol style="padding-left:20px;margin:0;">{items}</ol>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ── Key Features ──────────────────────────────────────────────────────────────
st.markdown("""
<h2 style="font-family:'Poppins',sans-serif;font-size:30px;font-weight:700;
           color:#0B3C5D;margin-bottom:8px;text-align:center;">Key Features</h2>
<p style="font-family:'Inter',sans-serif;font-size:16px;color:#6b7280;
          text-align:center;margin-bottom:36px;">What makes this system stand out</p>
""", unsafe_allow_html=True)

features = [
    ("🔬", "Model Comparison", "Side-by-side comparison of Baseline YOLOv12 and Enhanced YOLOv12 on the same image."),
    ("🌊", "Underwater Classes", "Detection of Echinus, Holothurian, Scallop, and Starfish in underwater imagery."),
    ("🛡️", "Noise Robustness", "Improved robustness in noisy underwater conditions via the denoising branch."),
    ("📊", "Standard Metrics", "Evaluation using Precision, Recall, mAP@50, and mAP@50:95."),
]
feat_cols = st.columns(4, gap="medium")
for col, (icon, title, desc) in zip(feat_cols, features):
    col.markdown(f"""
    <div class="card" style="text-align:center;height:100%;">
        <div style="font-size:32px;margin-bottom:12px;">{icon}</div>
        <div style="font-family:'Poppins',sans-serif;font-weight:600;font-size:16px;
                    color:#0B3C5D;margin-bottom:8px;">{title}</div>
        <p style="font-family:'Inter',sans-serif;font-size:14px;color:#6b7280;
                  line-height:1.6;margin:0;">{desc}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ── Study Highlights ──────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#0B3C5D;border-radius:16px;padding:48px;text-align:center;margin-bottom:48px;">
    <div style="font-family:'Poppins',sans-serif;font-size:12px;font-weight:600;
                color:#1FA3A3;letter-spacing:2px;margin-bottom:16px;">STUDY HIGHLIGHTS</div>
    <p style="font-family:'Poppins',sans-serif;font-size:22px;font-weight:600;
              color:#ffffff;line-height:1.6;max-width:780px;margin:0 auto 32px;">
        The Enhanced YOLOv12 model consistently outperformed the baseline in both clean and noisy scenarios.
        Improvements became more noticeable under degraded conditions, especially in Recall, mAP@50, and mAP@50:95.
    </p>
    <div style="display:flex;justify-content:center;gap:48px;flex-wrap:wrap;">
        <div>
            <div style="font-family:'Poppins',sans-serif;font-size:36px;font-weight:700;color:#1FA3A3;">84.2%</div>
            <div style="font-family:'Inter',sans-serif;font-size:13px;color:#94b4c5;margin-top:4px;">Enhanced Precision</div>
        </div>
        <div>
            <div style="font-family:'Poppins',sans-serif;font-size:36px;font-weight:700;color:#1FA3A3;">82.6%</div>
            <div style="font-family:'Inter',sans-serif;font-size:13px;color:#94b4c5;margin-top:4px;">Enhanced mAP@50</div>
        </div>
        <div>
            <div style="font-family:'Poppins',sans-serif;font-size:36px;font-weight:700;color:#1FA3A3;">57.1%</div>
            <div style="font-family:'Inter',sans-serif;font-size:13px;color:#94b4c5;margin-top:4px;">Enhanced mAP@50:95</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

render_footer()
