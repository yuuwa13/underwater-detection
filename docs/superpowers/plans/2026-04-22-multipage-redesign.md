# Multi-Page Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the single-page Streamlit detection app into a branded, multi-page web app (Home, Instructions, Contact Us, Run Simulation) matching the color palette and typography spec.

**Architecture:** Extract shared CSS/navbar/footer into `utils.py`; convert `app.py` to the homepage; move detection logic to `pages/Run_Simulation.py`; add `pages/Instructions.py` and `pages/Contact_Us.py`. Streamlit's built-in `pages/` directory handles routing.

**Tech Stack:** Streamlit ≥1.28, Python, Google Fonts (Poppins + Inter via CDN), custom HTML/CSS via `st.markdown(unsafe_allow_html=True)`

---

### Task 1: Create shared utils.py (branding, navbar, footer)

**Files:**
- Create: `utils.py`

- [ ] **Step 1: Create utils.py**

```python
# utils.py
import streamlit as st

BRAND_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; }
body, .stApp { font-family: 'Inter', sans-serif !important; }

[data-testid="stHeader"]         { display: none !important; }
[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

.stApp { background-color: #E8F1F2 !important; }

[data-testid="stAppViewContainer"] > section:first-child { padding-top: 64px !important; }
[data-testid="stMainBlockContainer"],
.main .block-container,
section.main > div.block-container {
    padding-left: 5rem !important;
    padding-right: 5rem !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}

.navbar {
    position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
    background: #0B3C5D;
    height: 64px;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 calc((100vw - min(1400px, 100vw)) / 2 + 5rem);
}
.navbar-brand {
    font-family: 'Poppins', sans-serif;
    font-size: 15px; font-weight: 600;
    color: #ffffff; text-decoration: none; white-space: nowrap;
}
.navbar-links { display: flex; gap: 2rem; align-items: center; }
.nav-link {
    font-family: 'Inter', sans-serif;
    font-size: 14px; font-weight: 500;
    color: #94b4c5; text-decoration: none;
    padding-bottom: 3px;
    border-bottom: 2px solid transparent;
    transition: color 0.2s, border-color 0.2s;
}
.nav-link:hover { color: #ffffff; border-color: #1FA3A3; }
.nav-link.active { color: #ffffff; border-color: #1FA3A3; }

.card {
    background: #ffffff;
    border-radius: 14px; padding: 28px;
    border: 1px solid #CFE3E6;
    box-shadow: 0 2px 8px rgba(11,60,93,0.07);
    margin-bottom: 20px;
}
.card-title {
    font-family: 'Poppins', sans-serif;
    font-size: 20px; font-weight: 600;
    color: #0B3C5D; margin-bottom: 12px;
}
.proposed { border: 2px solid #1FA3A3; }

.metric-box {
    background: #E8F1F2;
    border-radius: 10px; padding: 14px;
    border: 1px solid #CFE3E6;
    margin-bottom: 10px; color: #0B3C5D;
}
.metric-box b { color: #0B3C5D; }

h1, h2, h3 { font-family: 'Poppins', sans-serif !important; color: #0B3C5D !important; }
.stMarkdown, .stMarkdown p, .stMarkdown div { color: #1f2937; }

.stButton > button {
    background: #0B3C5D !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}
.stButton > button:hover:not(:disabled) { background: #1FA3A3 !important; }
.stButton > button:disabled {
    background: #CFE3E6 !important;
    color: #9ca3af !important;
    cursor: not-allowed !important;
}

[data-testid="stExpander"] {
    border: 1px solid #CFE3E6 !important;
    border-radius: 10px !important;
    background: #f0f7f8 !important;
    margin-bottom: 8px !important;
}
[data-testid="stExpander"] summary {
    font-size: 14px !important; font-weight: 600 !important;
    color: #0B3C5D !important; padding: 10px 14px !important;
    background: #f0f7f8 !important; border-radius: 10px !important;
}
[data-testid="stExpander"] summary:hover { background: #E8F1F2 !important; }
[data-testid="stExpander"] summary svg { color: #0B3C5D !important; fill: #0B3C5D !important; }
[data-testid="stExpander"][open] summary { border-radius: 10px 10px 0 0 !important; }
[data-testid="stExpander"] > div[data-testid="stExpanderDetails"] {
    border-top: 1px solid #CFE3E6 !important;
    padding: 12px 14px !important; color: #0B3C5D !important;
    background: #f0f7f8 !important; border-radius: 0 0 10px 10px !important;
}
[data-testid="stExpander"] * { background-color: transparent !important; }
[data-testid="stExpander"] .metric-box { background: #ffffff !important; }

[data-testid="stMetric"] label,
[data-testid="stMetric"] [data-testid="stMetricValue"],
[data-testid="stMetric"] [data-testid="stMetricDelta"] { color: #0B3C5D !important; }
[data-testid="stMetricValue"] > div { color: #0B3C5D !important; }

[data-testid="stFileUploader"] { width: 100%; }
[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #1FA3A3 !important;
    border-radius: 16px !important;
    padding: 180px 40px 190px 40px !important;
    text-align: center !important;
    background: white !important;
    cursor: pointer !important;
    transition: border-color 0.2s, background 0.2s !important;
    min-height: 220px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #0B3C5D !important;
    background: #f0f7f8 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: flex !important; flex-direction: column !important;
    align-items: center !important; justify-content: center !important;
    width: 100% !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div {
    display: flex !important; flex-direction: column !important;
    align-items: center !important; width: 100% !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] svg { display: none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small { display: none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] > div::before {
    content: "";
    display: block; width: 60px; height: 60px;
    background-color: #cef0f0;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%231FA3A3' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/%3E%3Cpolyline points='17 8 12 3 7 8'/%3E%3Cline x1='12' y1='3' x2='12' y2='15'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: center;
    background-size: 28px 28px; border-radius: 50%;
    margin: 0 auto 16px auto;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div::after {
    content: "Click to upload image";
    display: block; font-size: 15px; font-weight: 500;
    color: #0B3C5D; margin-bottom: 4px;
}
[data-testid="stFileUploaderDropzoneInstructions"]::after {
    content: "PNG, JPG up to 10MB";
    display: block; font-size: 13px; color: #9ca3af;
    margin-top: 4px; text-align: center;
}
[data-testid="stFileUploaderDropzone"] button { display: none !important; }
[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="uploadedFileName"],
[data-testid="stFileUploader"] .uploadedFileName { color: #0B3C5D !important; }
[data-testid="stFileUploader"] > div > div > div { color: #0B3C5D !important; }

@media (max-width: 1200px) {
    .navbar { padding: 0 2rem; }
    [data-testid="stMainBlockContainer"],
    .main .block-container,
    section.main > div.block-container {
        padding-left: 2rem !important; padding-right: 2rem !important; max-width: 100% !important;
    }
}
@media (max-width: 768px) {
    .navbar { padding: 0 1rem; height: auto; padding-top: 12px; padding-bottom: 12px; flex-wrap: wrap; }
    .navbar-links { gap: 1rem; flex-wrap: wrap; }
    [data-testid="stMainBlockContainer"],
    .main .block-container,
    section.main > div.block-container {
        padding-left: 1rem !important; padding-right: 1rem !important;
    }
}
</style>
"""


def inject_branding():
    st.markdown(BRAND_CSS, unsafe_allow_html=True)


def render_navbar(active: str = "Home"):
    pages = [
        ("Home",           "/",               "Home"),
        ("Instructions",   "/Instructions",   "Instructions"),
        ("Contact_Us",     "/Contact_Us",     "Contact Us"),
        ("Run_Simulation", "/Run_Simulation", "Run Simulation"),
    ]
    links_html = "".join(
        f'<a href="{href}" class="nav-link{" active" if key == active else ""}">{label}</a>'
        for key, href, label in pages
    )
    st.markdown(
        f'<div class="navbar">'
        f'<a href="/" class="navbar-brand">Underwater Object Detection</a>'
        f'<div class="navbar-links">{links_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown("""
    <div style="
        background: #0B3C5D; color: #94b4c5;
        text-align: center; padding: 40px 5rem;
        font-family: 'Inter', sans-serif; font-size: 14px;
        width: 100vw; margin-left: calc(-50vw + 50%);
        margin-top: 4rem;
    ">
        <p style="font-family:'Poppins',sans-serif;font-weight:600;color:#ffffff;font-size:15px;margin:0 0 8px 0;">
            An Enhanced YOLOv12 Architecture with Dual-Branch Network for Underwater Object Detection
        </p>
        <p style="margin: 4px 0;">University of Mindanao</p>
        <p style="margin: 4px 0;">© 2026 All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)
```

- [ ] **Step 2: Commit**

```bash
git add utils.py
git commit -m "feat: add shared branding utils (navbar, footer, CSS)"
```

---

### Task 2: Rewrite app.py as the Homepage

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Replace app.py with homepage**

```python
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
        "Compare the baseline and enhanced model under varying noise levels",
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
```

- [ ] **Step 2: Commit**

```bash
git add app.py
git commit -m "feat: implement homepage with hero, about, objectives, features, highlights"
```

---

### Task 3: Create Instructions page

**Files:**
- Create: `pages/Instructions.py`

- [ ] **Step 1: Create pages/Instructions.py**

```python
# pages/Instructions.py
import streamlit as st
from utils import inject_branding, render_navbar, render_footer

st.set_page_config(
    page_title="Instructions · Underwater Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_branding()
render_navbar("Instructions")

st.markdown("""
<div style="padding: 48px 0 32px;">
    <p style="font-family:'Inter',sans-serif;font-size:13px;font-weight:600;
              color:#1FA3A3;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">
        Getting Started
    </p>
    <h1 style="font-family:'Poppins',sans-serif;font-size:38px;font-weight:700;
               color:#0B3C5D;margin-bottom:12px;">How to Use</h1>
    <p style="font-family:'Inter',sans-serif;font-size:16px;color:#6b7280;max-width:600px;">
        Follow these steps to run the underwater object detection simulation and interpret the results.
    </p>
</div>
""", unsafe_allow_html=True)

steps = [
    ("01", "Go to Run Simulation",
     "Navigate to the <strong>Run Simulation</strong> page using the top navigation bar."),
    ("02", "Upload an Underwater Image",
     "Click the upload area or drag and drop an image file (PNG or JPG, up to 10 MB). "
     "For best results, use actual underwater photographs of marine objects."),
    ("03", "Get Sample Images",
     'Need test images? <a href="#" style="color:#1FA3A3;font-weight:600;">Browse the dataset on Google Drive</a> '
     "for sample underwater images of Echinus, Starfish, Scallop, and Holothurian."),
    ("04", "Click Run Detection",
     "Press the <strong>Run Detection</strong> button. Both models will process the image simultaneously."),
    ("05", "Wait for Processing",
     "The system will run inference through both the Baseline and Enhanced YOLOv12 models. "
     "This typically takes a few seconds depending on server load."),
    ("06", "Compare Results",
     "Review the side-by-side results for both models, including detection overlays, "
     "classification confidence scores, and evaluation metrics."),
]

for num, title, desc in steps:
    st.markdown(f"""
    <div style="display:flex;gap:20px;align-items:flex-start;
                background:#ffffff;border:1px solid #CFE3E6;border-radius:14px;
                padding:24px;margin-bottom:16px;">
        <div style="background:#0B3C5D;color:#ffffff;font-family:'Poppins',sans-serif;
                    font-weight:700;font-size:18px;border-radius:10px;
                    min-width:52px;height:52px;display:flex;align-items:center;
                    justify-content:center;flex-shrink:0;">
            {num}
        </div>
        <div>
            <div style="font-family:'Poppins',sans-serif;font-size:17px;font-weight:600;
                        color:#0B3C5D;margin-bottom:6px;">{title}</div>
            <p style="font-family:'Inter',sans-serif;font-size:15px;color:#374151;
                      line-height:1.65;margin:0;">{desc}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<h2 style="font-family:'Poppins',sans-serif;font-size:26px;font-weight:700;
           color:#0B3C5D;margin-bottom:20px;">Understanding the Results</h2>
""", unsafe_allow_html=True)

metric_col1, metric_col2 = st.columns(2, gap="large")
with metric_col1:
    for name, desc in [
        ("Highest Detection Accuracy", "The confidence score of the top bounding box detection, expressed as a percentage."),
        ("Classification Results", "A ranked list of all detected objects and their individual confidence scores."),
    ]:
        st.markdown(f"""
        <div class="metric-box" style="margin-bottom:12px;">
            <b style="font-family:'Poppins',sans-serif;font-size:15px;">{name}</b>
            <p style="font-family:'Inter',sans-serif;font-size:14px;color:#6b7280;margin:6px 0 0;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)
with metric_col2:
    for name, desc in [
        ("Precision", "Fraction of detections that are correct. High precision = fewer false positives."),
        ("Recall", "Fraction of actual objects that were detected. High recall = fewer missed detections."),
        ("mAP@50", "Mean Average Precision at IoU threshold 0.50. Standard detection benchmark."),
        ("mAP@50:95", "Mean Average Precision averaged across IoU 0.50–0.95. Stricter localization metric."),
    ]:
        st.markdown(f"""
        <div class="metric-box" style="margin-bottom:12px;">
            <b style="font-family:'Poppins',sans-serif;font-size:15px;">{name}</b>
            <p style="font-family:'Inter',sans-serif;font-size:14px;color:#6b7280;margin:6px 0 0;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<h2 style="font-family:'Poppins',sans-serif;font-size:26px;font-weight:700;
           color:#0B3C5D;margin-bottom:16px;">Notes</h2>
""", unsafe_allow_html=True)

notes = [
    "The system is intended for <strong>underwater object images</strong> only.",
    "Detection results may vary depending on <strong>noise level and image quality</strong>.",
    "The enhanced model is expected to perform better under <strong>noisy underwater conditions</strong>.",
    "Detectable classes are limited to: <strong>Echinus, Starfish, Scallop, and Holothurian</strong>.",
]
notes_html = "".join(
    f'<li style="font-family:Inter,sans-serif;font-size:15px;color:#374151;'
    f'line-height:1.7;margin-bottom:10px;">{n}</li>'
    for n in notes
)
st.markdown(f'<div class="card"><ul style="padding-left:20px;margin:0;">{notes_html}</ul></div>', unsafe_allow_html=True)

render_footer()
```

- [ ] **Step 2: Commit**

```bash
git add pages/Instructions.py
git commit -m "feat: add Instructions page"
```

---

### Task 4: Create Contact Us page

**Files:**
- Create: `pages/Contact_Us.py`

- [ ] **Step 1: Create pages/Contact_Us.py**

```python
# pages/Contact_Us.py
import streamlit as st
from utils import inject_branding, render_navbar, render_footer

st.set_page_config(
    page_title="Contact Us · Underwater Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_branding()
render_navbar("Contact_Us")

st.markdown("""
<div style="padding: 48px 0 32px;">
    <p style="font-family:'Inter',sans-serif;font-size:13px;font-weight:600;
              color:#1FA3A3;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">
        Get in Touch
    </p>
    <h1 style="font-family:'Poppins',sans-serif;font-size:38px;font-weight:700;
               color:#0B3C5D;margin-bottom:12px;">Contact Us</h1>
    <p style="font-family:'Inter',sans-serif;font-size:16px;color:#6b7280;max-width:600px;">
        For inquiries about this research, the dataset, or the detection system, feel free to reach out.
    </p>
</div>
""", unsafe_allow_html=True)

contact_col, info_col = st.columns([3, 2], gap="large")

with contact_col:
    st.markdown("""
    <h2 style="font-family:'Poppins',sans-serif;font-size:22px;font-weight:700;
               color:#0B3C5D;margin-bottom:20px;">Research Team</h2>
    """, unsafe_allow_html=True)
    for role, institution, email in [
        ("Researcher", "University of Mindanao", "researcher@umindanao.edu.ph"),
        ("Thesis Adviser", "University of Mindanao", "adviser@umindanao.edu.ph"),
    ]:
        st.markdown(f"""
        <div style="background:#ffffff;border:1px solid #CFE3E6;border-radius:14px;
                    padding:24px;margin-bottom:16px;display:flex;gap:20px;align-items:flex-start;">
            <div style="background:#E8F1F2;border-radius:50%;width:48px;height:48px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:20px;flex-shrink:0;">👤</div>
            <div>
                <div style="font-family:'Poppins',sans-serif;font-weight:600;font-size:16px;
                            color:#0B3C5D;margin-bottom:4px;">{role}</div>
                <div style="font-family:'Inter',sans-serif;font-size:14px;color:#6b7280;
                            margin-bottom:4px;">{institution}</div>
                <a href="mailto:{email}" style="font-family:'Inter',sans-serif;font-size:14px;
                           color:#1FA3A3;font-weight:500;text-decoration:none;">{email}</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

with info_col:
    st.markdown("""
    <div style="background:#0B3C5D;border-radius:16px;padding:32px;color:#ffffff;">
        <div style="font-family:'Poppins',sans-serif;font-size:12px;font-weight:600;
                    color:#1FA3A3;letter-spacing:2px;margin-bottom:20px;">INSTITUTION</div>
        <div style="font-family:'Poppins',sans-serif;font-size:18px;font-weight:600;
                    margin-bottom:8px;">University of Mindanao</div>
        <div style="font-family:'Inter',sans-serif;font-size:14px;color:#94b4c5;
                    line-height:1.6;margin-bottom:24px;">Bolton St., Davao City, Philippines</div>
        <hr style="border-color:rgba(255,255,255,0.1);margin-bottom:24px;">
        <div style="font-family:'Poppins',sans-serif;font-size:12px;font-weight:600;
                    color:#1FA3A3;letter-spacing:2px;margin-bottom:16px;">ABOUT THIS SYSTEM</div>
        <p style="font-family:'Inter',sans-serif;font-size:14px;color:#94b4c5;line-height:1.65;margin:0;">
            This simulation was developed as part of a thesis study on enhanced underwater object detection
            using a dual-branch YOLOv12 architecture. It is intended for academic and research purposes.
        </p>
    </div>
    """, unsafe_allow_html=True)

render_footer()
```

- [ ] **Step 2: Commit**

```bash
git add pages/Contact_Us.py
git commit -m "feat: add Contact Us page"
```

---

### Task 5: Create Run Simulation page (migrate detection logic)

**Files:**
- Create: `pages/Run_Simulation.py`

- [ ] **Step 1: Create pages/Run_Simulation.py**

```python
# pages/Run_Simulation.py
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

# Pin Run Detection button to top-right of navbar on this page only
st.markdown("""
<style>
.stButton > button {
    position: fixed !important;
    top: 12px !important;
    right: calc((100vw - min(1400px,100vw)) / 2 + 5rem) !important;
    z-index: 10000 !important;
    width: auto !important;
    min-width: 0 !important;
}
@media (max-width: 1200px) { .stButton > button { right: 2rem !important; } }
@media (max-width: 768px)  { .stButton > button { right: 1rem !important; } }
</style>
""", unsafe_allow_html=True)

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
st.markdown("""
<div style="padding: 48px 0 24px;">
    <p style="font-family:'Inter',sans-serif;font-size:13px;font-weight:600;
              color:#1FA3A3;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">
        Detection System
    </p>
    <h1 style="font-family:'Poppins',sans-serif;font-size:38px;font-weight:700;
               color:#0B3C5D;margin-bottom:12px;">Run Simulation</h1>
    <p style="font-family:'Inter',sans-serif;font-size:16px;color:#6b7280;max-width:600px;">
        Upload an underwater image to compare detection results between the Baseline and Enhanced YOLOv12 models.
    </p>
</div>
""", unsafe_allow_html=True)

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

    preview_slot.markdown(f"""
    <style>
    [data-testid="stFileUploaderDropzone"] {{
        padding: 10px 20px !important; min-height: 0 !important; border-radius: 10px !important;
    }}
    [data-testid="stFileUploaderDropzoneInstructions"] > div::before {{ display: none !important; }}
    [data-testid="stFileUploaderDropzoneInstructions"] > div::after {{
        content: "Click to upload a different image" !important;
        font-size: 13px !important; color: #6b7280 !important; font-weight: 400 !important;
    }}
    [data-testid="stFileUploaderDropzoneInstructions"]::after {{ display: none !important; }}
    [data-testid="stFileUploaderDropzone"] button {{ display: none !important; }}
    [data-testid="stFileUploaderFile"] {{ display: none !important; }}
    </style>
    <div style="border:2px dashed #1FA3A3;border-radius:16px;background:white;
                padding:24px;text-align:center;margin-bottom:8px;">
        <img src="data:image/png;base64,{b64}"
             style="max-width:100%;max-height:420px;border-radius:10px;object-fit:contain;" />
        <p style="font-size:13px;color:#6b7280;margin-top:10px;margin-bottom:0;">{uploaded_file.name}</p>
    </div>
    """, unsafe_allow_html=True)

# ── Detection ──────────────────────────────────────────────────────────────────
if uploaded_file and run_detection:
    status_slot = st.empty()
    status_slot.info("Running detection...")

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
        st.markdown('<div class="card"><div class="card-title">Baseline Model</div></div>', unsafe_allow_html=True)
        st.image(baseline_img, use_container_width=True)
        if baseline_count == 0:
            if proposed_count == 0:
                st.error("This model can only detect Echinus, Starfish, Scallop, and Holothurian. Please upload a different image.")
            else:
                st.error("No class detected. This model can only detect Echinus, Starfish, Scallop, and Holothurian. Please upload a different image.")
        else:
            st.markdown(f'<div class="metric-box">Highest Detection Accuracy<br><b>{baseline_accuracy:.1f}%</b></div>', unsafe_allow_html=True)
            st.markdown('<p style="font-size:14px;color:#6b7280;margin-top:16px;margin-bottom:8px;">Classification Results</p>', unsafe_allow_html=True)
            with st.expander("See More"):
                for cls, conf in baseline_detections:
                    st.write(f"{cls} — {conf*100:.1f}%")
            with st.expander("Evaluation Metrics"):
                m1, m2 = st.columns(2)
                with m1:
                    st.markdown(f'<div class="metric-box">Precision<br><b>{baseline_metrics["precision"]*100:.2f}%</b><br><span style="font-size:12px;color:#9ca3af">(baseline)</span></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-box">mAP@50<br><b>{baseline_metrics["mAP50"]*100:.2f}%</b><br><span style="font-size:12px;color:#9ca3af">(baseline)</span></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="metric-box">Recall<br><b>{baseline_metrics["recall"]*100:.2f}%</b><br><span style="font-size:12px;color:#9ca3af">(baseline)</span></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-box">mAP@50:95<br><b>{baseline_metrics["mAP50-95"]*100:.2f}%</b><br><span style="font-size:12px;color:#9ca3af">(baseline)</span></div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card proposed"><div class="card-title">Enhanced Model</div></div>', unsafe_allow_html=True)
        st.image(proposed_img, use_container_width=True)
        if proposed_count == 0:
            if baseline_count == 0:
                st.error("This model can only detect Echinus, Starfish, Scallop, and Holothurian. Please upload a different image.")
            else:
                st.error("No class detected. This model can only detect Echinus, Starfish, Scallop, and Holothurian. Please upload a different image.")
        else:
            st.markdown(f'<div class="metric-box">Highest Detection Accuracy<br><b>{proposed_accuracy:.1f}%</b></div>', unsafe_allow_html=True)
            st.markdown('<p style="font-size:14px;color:#6b7280;margin-top:16px;margin-bottom:8px;">Classification Results</p>', unsafe_allow_html=True)
            with st.expander("See More"):
                for cls, conf in proposed_detections:
                    st.write(f"{cls} — {conf*100:.1f}%")
            with st.expander("Evaluation Metrics"):
                m1, m2 = st.columns(2)
                with m1:
                    st.markdown(f'<div class="metric-box">Precision<br><b>{proposed_metrics["precision"]*100:.2f}%</b><br><span style="font-size:12px;color:{dc(precision_improvement)}">{ds(precision_improvement)}{precision_improvement:.2f}%</span></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-box">mAP@50<br><b>{proposed_metrics["mAP50"]*100:.2f}%</b><br><span style="font-size:12px;color:{dc(map50_improvement)}">{ds(map50_improvement)}{map50_improvement:.2f}%</span></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="metric-box">Recall<br><b>{proposed_metrics["recall"]*100:.2f}%</b><br><span style="font-size:12px;color:{dc(recall_improvement)}">{ds(recall_improvement)}{recall_improvement:.2f}%</span></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-box">mAP@50:95<br><b>{proposed_metrics["mAP50-95"]*100:.2f}%</b><br><span style="font-size:12px;color:{dc(map95_improvement)}">{ds(map95_improvement)}{map95_improvement:.2f}%</span></div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Model Comparison")
    comp_col1, comp_col2, comp_col3 = st.columns(3)
    with comp_col1:
        st.metric("Average Accuracy Improvement", f"{proposed_accuracy_avg:.2f}%", f"{proposed_accuracy_avg - baseline_accuracy_avg:.2f}%")
    with comp_col2:
        st.metric("Enhanced Detections", proposed_count)
    with comp_col3:
        st.metric("Baseline Detections", baseline_count)

render_footer()
```

- [ ] **Step 2: Commit**

```bash
git add pages/Run_Simulation.py
git commit -m "feat: add Run Simulation page with detection logic"
```

---

### Task 6: Push to remote

- [ ] **Step 1: Push all commits**

```bash
git push origin main
```

---

**To run locally:** `streamlit run app.py`
