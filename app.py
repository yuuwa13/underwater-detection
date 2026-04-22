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
st.html("""
<div style="padding:88px 0 72px;text-align:center;border-bottom:1px solid #e2e8f0;">
    <div style="display:inline-flex;align-items:center;gap:8px;
                background:#f0fafa;border:1px solid #c7eaea;
                border-radius:100px;padding:5px 14px;margin-bottom:28px;">
        <span style="display:inline-block;width:7px;height:7px;
                     background:#1FA3A3;border-radius:50%;"></span>
        <span style="font-family:'Inter',sans-serif;font-size:12.5px;font-weight:500;
                     color:#0B7A7A;letter-spacing:0.04em;">
            University of Mindanao &nbsp;·&nbsp; Thesis Research
        </span>
    </div>
    <h1 style="font-family:'Poppins',sans-serif;font-size:48px;font-weight:700;
               color:#0B3C5D;line-height:1.18;margin-bottom:22px;letter-spacing:-0.02em;">
        An Enhanced YOLOv12 Architecture<br>
        <span style="color:#1FA3A3;">with Dual-Branch Network</span><br>
        for Underwater Object Detection
    </h1>
    <p style="font-family:'Inter',sans-serif;font-size:17px;color:#64748b;
              max-width:640px;margin:0 auto 40px;line-height:1.75;font-weight:400;">
        A research prototype comparing Baseline and Enhanced YOLOv12 models
        for detecting marine objects under noisy underwater conditions.
    </p>
    <div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap;">
        <a href="/Run_Simulation" style="
            display:inline-block;background:#0B3C5D;color:#ffffff;
            text-decoration:none;font-family:'Inter',sans-serif;
            font-weight:600;font-size:14px;padding:13px 28px;
            border-radius:8px;letter-spacing:0.01em;
            box-shadow:0 1px 3px rgba(11,60,93,0.25);">
            Run Simulation →
        </a>
        <a href="/Instructions" style="
            display:inline-block;background:#ffffff;color:#0B3C5D;
            border:1.5px solid #cbd5e1;text-decoration:none;
            font-family:'Inter',sans-serif;font-weight:500;
            font-size:14px;padding:12px 24px;border-radius:8px;">
            View Instructions
        </a>
    </div>
</div>
""")

# ── Class badges ──────────────────────────────────────────────────────────────
classes = [
    ("🦔", "Echinus"),
    ("🐙", "Holothurian"),
    ("🐚", "Scallop"),
    ("⭐", "Starfish"),
]
badges = "".join(
    f'<div style="display:flex;align-items:center;gap:8px;'
    f'background:#f8fafc;border:1px solid #e2e8f0;'
    f'border-radius:10px;padding:10px 20px;">'
    f'<span style="font-size:18px;">{icon}</span>'
    f'<span style="font-family:Inter,sans-serif;font-size:13.5px;'
    f'font-weight:500;color:#334155;">{name}</span>'
    f'</div>'
    for icon, name in classes
)
st.html(
    f'<div style="display:flex;justify-content:center;gap:10px;flex-wrap:wrap;'
    f'padding:32px 0 48px;">'
    f'<span style="font-family:Inter,sans-serif;font-size:12px;font-weight:500;'
    f'color:#94a3b8;letter-spacing:0.06em;align-self:center;margin-right:8px;">DETECTABLE CLASSES</span>'
    f'{badges}</div>'
)

# ── About the Study ───────────────────────────────────────────────────────────
col_txt, col_card = st.columns([3, 2], gap="large")
with col_txt:
    st.html("""
    <div style="padding:16px 0;">
        <div style="font-family:'Inter',sans-serif;font-size:11.5px;font-weight:600;
                    color:#1FA3A3;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:14px;">
            About the Study
        </div>
        <h2 style="font-family:'Poppins',sans-serif;font-size:28px;font-weight:700;
                   color:#0B3C5D;margin-bottom:18px;line-height:1.3;letter-spacing:-0.01em;">
            Addressing the Challenges of Underwater Vision
        </h2>
        <p style="font-family:'Inter',sans-serif;font-size:15.5px;color:#475569;
                  line-height:1.8;margin-bottom:16px;">
            Underwater images often suffer from blur, color distortion, poor visibility, and
            environmental noise — making object detection significantly harder than in terrestrial settings.
        </p>
        <p style="font-family:'Inter',sans-serif;font-size:15.5px;color:#475569;
                  line-height:1.8;margin-bottom:28px;">
            This study introduces a <strong style="color:#0B3C5D;">Dual-Branch Input Stem</strong>
            into the YOLOv12 architecture — combining a <em>standard feature branch</em>
            and a <em>denoising branch</em> through adaptive feature fusion to improve
            detection accuracy under degraded conditions.
        </p>
        <a href="#" style="
            display:inline-flex;align-items:center;gap:8px;
            color:#1FA3A3;text-decoration:none;
            font-family:'Inter',sans-serif;font-weight:600;font-size:14px;
            border-bottom:1.5px solid #1FA3A3;padding-bottom:2px;">
            📄 View Thesis Document
        </a>
    </div>
    """)
with col_card:
    st.html("""
    <div style="background:#0B3C5D;border-radius:16px;padding:36px;color:#ffffff;
                height:100%;min-height:320px;display:flex;flex-direction:column;justify-content:center;">
        <div style="font-family:'Inter',sans-serif;font-size:11px;font-weight:600;
                    color:#1FA3A3;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:24px;">
            Architecture Overview
        </div>
        <div style="display:flex;flex-direction:column;gap:12px;">
            <div style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);
                        border-left:3px solid #1FA3A3;padding:14px 16px;border-radius:8px;">
                <div style="font-family:'Poppins',sans-serif;font-weight:600;font-size:13.5px;
                            margin-bottom:4px;">Standard Branch</div>
                <div style="font-family:'Inter',sans-serif;font-size:12.5px;
                            color:rgba(255,255,255,0.5);">Structural feature extraction</div>
            </div>
            <div style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);
                        border-left:3px solid #1FA3A3;padding:14px 16px;border-radius:8px;">
                <div style="font-family:'Poppins',sans-serif;font-weight:600;font-size:13.5px;
                            margin-bottom:4px;">Denoising Branch</div>
                <div style="font-family:'Inter',sans-serif;font-size:12.5px;
                            color:rgba(255,255,255,0.5);">Noise suppression & correction</div>
            </div>
            <div style="background:rgba(31,163,163,0.15);border:1px solid rgba(31,163,163,0.3);
                        border-left:3px solid #ffffff;padding:14px 16px;border-radius:8px;">
                <div style="font-family:'Poppins',sans-serif;font-weight:600;font-size:13.5px;
                            margin-bottom:4px;">Adaptive Feature Fusion</div>
                <div style="font-family:'Inter',sans-serif;font-size:12.5px;
                            color:rgba(255,255,255,0.5);">Combined output for final detection</div>
            </div>
        </div>
    </div>
    """)

st.html("<div style='height:56px;'></div>")

# ── Objectives ────────────────────────────────────────────────────────────────
st.html("""
<div style="border-top:1px solid #e2e8f0;padding-top:56px;margin-bottom:40px;">
    <div style="font-family:'Inter',sans-serif;font-size:11.5px;font-weight:600;
                color:#1FA3A3;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:10px;">
        Objectives
    </div>
    <h2 style="font-family:'Poppins',sans-serif;font-size:28px;font-weight:700;
               color:#0B3C5D;letter-spacing:-0.01em;">What this research aims to achieve</h2>
</div>
""")

obj_col1, obj_col2 = st.columns(2, gap="large")
with obj_col1:
    st.html("""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
                padding:28px;height:100%;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
        <div style="font-family:'Inter',sans-serif;font-size:11px;font-weight:600;
                    color:#1FA3A3;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:14px;">
            General Objective
        </div>
        <p style="font-family:'Inter',sans-serif;font-size:15px;color:#475569;line-height:1.75;margin:0;">
            To enhance YOLOv12's performance as a detection model for underwater objects by developing
            an improved architecture that can better handle noise in underwater environments.
        </p>
    </div>
    """)
with obj_col2:
    specific = [
        "Preprocess underwater datasets for training and evaluation",
        "Develop a new model by integrating a dual-branch input stem into the baseline YOLOv12",
        "Compare the baseline and proposed model under varying noise levels",
        "Evaluate both models using Precision, Recall, mAP@50, and mAP@50:95",
    ]
    items = "".join(
        f'<li style="font-family:Inter,sans-serif;font-size:15px;color:#475569;'
        f'line-height:1.75;margin-bottom:10px;padding-left:4px;">{s}</li>'
        for s in specific
    )
    st.html(f"""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
                padding:28px;height:100%;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
        <div style="font-family:'Inter',sans-serif;font-size:11px;font-weight:600;
                    color:#1FA3A3;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:14px;">
            Specific Objectives
        </div>
        <ol style="padding-left:18px;margin:0;">{items}</ol>
    </div>
    """)

st.html("<div style='height:56px;'></div>")

# ── Key Features ──────────────────────────────────────────────────────────────
st.html("""
<div style="border-top:1px solid #e2e8f0;padding-top:56px;margin-bottom:40px;">
    <div style="font-family:'Inter',sans-serif;font-size:11.5px;font-weight:600;
                color:#1FA3A3;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:10px;">
        Key Features
    </div>
    <h2 style="font-family:'Poppins',sans-serif;font-size:28px;font-weight:700;
               color:#0B3C5D;letter-spacing:-0.01em;">What makes this system stand out</h2>
</div>
""")

features = [
    ("🔬", "Model Comparison", "Side-by-side comparison of Baseline and Enhanced YOLOv12 on the same input image."),
    ("🌊", "Underwater Classes", "Trained to detect Echinus, Holothurian, Scallop, and Starfish."),
    ("🛡️", "Noise Robustness", "Dual-branch design targets noise suppression in degraded underwater imagery."),
    ("📊", "Standard Metrics", "Evaluated with Precision, Recall, mAP@50, and mAP@50:95."),
]
feat_cols = st.columns(4, gap="medium")
for col, (icon, title, desc) in zip(feat_cols, features):
    col.html(f"""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
                padding:24px;text-align:left;height:100%;
                box-shadow:0 1px 3px rgba(0,0,0,0.04);">
        <div style="font-size:28px;margin-bottom:14px;">{icon}</div>
        <div style="font-family:'Poppins',sans-serif;font-weight:600;font-size:15px;
                    color:#0B3C5D;margin-bottom:8px;">{title}</div>
        <p style="font-family:'Inter',sans-serif;font-size:13.5px;color:#64748b;
                  line-height:1.65;margin:0;">{desc}</p>
    </div>
    """)

st.html("<div style='height:56px;'></div>")

# ── Study Highlights ──────────────────────────────────────────────────────────
st.html("""
<div style="background:#0B3C5D;border-radius:16px;padding:56px 64px;
            margin-bottom:0;border:1px solid rgba(255,255,255,0.05);">
    <div style="font-family:'Inter',sans-serif;font-size:11px;font-weight:600;
                color:#1FA3A3;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:16px;">
        Study Highlights
    </div>
    <p style="font-family:'Poppins',sans-serif;font-size:20px;font-weight:500;
              color:rgba(255,255,255,0.85);line-height:1.65;max-width:700px;margin:0 0 44px;">
        The Enhanced YOLOv12 model consistently outperformed the baseline in both clean and noisy
        scenarios — with the most notable gains in Recall, mAP@50, and mAP@50:95 under degraded conditions.
    </p>
    <div style="display:flex;gap:56px;flex-wrap:wrap;border-top:1px solid rgba(255,255,255,0.1);
                padding-top:36px;">
        <div>
            <div style="font-family:'Poppins',sans-serif;font-size:38px;font-weight:700;
                        color:#1FA3A3;letter-spacing:-0.02em;">84.2%</div>
            <div style="font-family:'Inter',sans-serif;font-size:13px;
                        color:rgba(255,255,255,0.45);margin-top:6px;">Enhanced Precision</div>
        </div>
        <div>
            <div style="font-family:'Poppins',sans-serif;font-size:38px;font-weight:700;
                        color:#1FA3A3;letter-spacing:-0.02em;">82.6%</div>
            <div style="font-family:'Inter',sans-serif;font-size:13px;
                        color:rgba(255,255,255,0.45);margin-top:6px;">Enhanced mAP@50</div>
        </div>
        <div>
            <div style="font-family:'Poppins',sans-serif;font-size:38px;font-weight:700;
                        color:#1FA3A3;letter-spacing:-0.02em;">57.1%</div>
            <div style="font-family:'Inter',sans-serif;font-size:13px;
                        color:rgba(255,255,255,0.45);margin-top:6px;">Enhanced mAP@50:95</div>
        </div>
    </div>
</div>
""")

render_footer()
