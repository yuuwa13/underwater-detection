import streamlit as st

from utils import inject_branding, render_footer, render_navbar

st.set_page_config(
    page_title="Instructions · Underwater Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_branding()
render_navbar("Instructions")

st.html("""
<div style="padding:16px 0 36px;">
    <div style="font-family:'Inter',sans-serif;font-size:13px;font-weight:600;
                color:#1FA3A3;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:14px;">
        Getting Started
    </div>
    <h1 style="font-family:'Poppins',sans-serif;font-size:42px;font-weight:700;
               color:#0B3C5D;margin-bottom:14px;letter-spacing:-0.02em;">How to Use</h1>
    <p style="font-family:'Inter',sans-serif;font-size:17px;color:#64748b;max-width:600px;line-height:1.75;">
        Follow these steps to run the underwater object detection simulation and interpret the results.
    </p>
</div>
""")

steps = [
    (
        "01",
        "Go to Run Simulation",
        "Navigate to the <strong style='color:#0B3C5D;'>Run Simulation</strong> page using the top navigation bar.",
    ),
    (
        "02",
        "Upload an Underwater Image",
        "Click the upload area or drag and drop a PNG or JPG file (up to 10 MB). "
        "For best results, use actual underwater photographs of marine objects.",
    ),
    (
        "03",
        "Get Sample Images",
        'Need test images? <a href="https://drive.google.com/drive/folders/1G_tf8SJ7tz_vPVahpuAigtlq59ys3rvo?usp=sharing" target="_blank" style="color:#1FA3A3;font-weight:600;text-decoration:none;border-bottom:1px solid #1FA3A3;">Browse the dataset on Google Drive</a> '
        "for sample underwater images of Echinus, Starfish, Scallop, and Holothurian.",
    ),
    (
        "04",
        "Click Run Detection",
        "Press the <strong style='color:#0B3C5D;'>Run Detection</strong> button. Both models will process the image simultaneously.",
    ),
    (
        "05",
        "Wait for Processing",
        "The system runs inference through both the Baseline and Enhanced YOLOv12 models. "
        "This typically takes a few seconds depending on server load.",
    ),
    (
        "06",
        "Compare Results",
        "Review the side-by-side results — detection overlays, confidence scores, and evaluation metrics for both models.",
    ),
]

for num, title, desc in steps:
    st.html(f"""
    <div style="display:flex;gap:24px;align-items:flex-start;
                background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;
                padding:32px;margin-bottom:14px;
                box-shadow:0 1px 3px rgba(0,0,0,0.04);">
        <div style="background:#0B3C5D;color:#ffffff;font-family:'Poppins',sans-serif;
                    font-weight:700;font-size:16px;border-radius:12px;
                    min-width:56px;height:56px;display:flex;align-items:center;
                    justify-content:center;flex-shrink:0;letter-spacing:0.02em;">
            {num}
        </div>
        <div style="padding-top:4px;">
            <div style="font-family:'Poppins',sans-serif;font-size:18px;font-weight:600;
                        color:#0B3C5D;margin-bottom:8px;">{title}</div>
            <p style="font-family:'Inter',sans-serif;font-size:16px;color:#475569;
                      line-height:1.75;margin:0;">{desc}</p>
        </div>
    </div>
    """)

st.html("<div style='height:64px;'></div>")

# ── Understanding the Results ─────────────────────────────────────────────────
st.html("""
<div style="border-top:1px solid #e2e8f0;padding-top:64px;margin-bottom:48px;">
    <div style="font-family:'Inter',sans-serif;font-size:13px;font-weight:600;
                color:#1FA3A3;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:14px;">
        Results Guide
    </div>
    <h2 style="font-family:'Poppins',sans-serif;font-size:36px;font-weight:700;
               color:#0B3C5D;letter-spacing:-0.02em;">Understanding the Results</h2>
</div>
""")

# Output indicators row
out_col1, out_col2 = st.columns(2, gap="large")
for col, (name, desc) in zip(
    [out_col1, out_col2],
    [
        ("Highest Detection Accuracy", "The confidence score of the top bounding-box detection, expressed as a percentage."),
        ("Classification Results", "A ranked list of all detected objects and their individual confidence scores."),
    ],
):
    col.html(f"""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;
                padding:32px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
        <div style="font-family:'Poppins',sans-serif;font-size:17px;font-weight:600;
                    color:#0B3C5D;margin-bottom:10px;">{name}</div>
        <p style="font-family:'Inter',sans-serif;font-size:15px;color:#64748b;margin:0;line-height:1.7;">{desc}</p>
    </div>
    """)

# Evaluation metrics — 4 in a 2x2 grid
st.html("""
<div style="font-family:'Inter',sans-serif;font-size:13px;font-weight:600;
            color:#1FA3A3;letter-spacing:0.1em;text-transform:uppercase;
            margin-top:8px;margin-bottom:20px;">
    Evaluation Metrics
</div>
""")

metrics = [
    ("Precision", "Fraction of detections that are correct. High precision = fewer false positives."),
    ("Recall", "Fraction of actual objects detected. High recall = fewer missed detections."),
    ("mAP@50", "Mean Average Precision at IoU threshold 0.50. Standard detection benchmark."),
    ("mAP@50:95", "Mean AP averaged across IoU 0.50–0.95. Stricter localization metric."),
]
m_cols = st.columns(2, gap="large")
for i, (name, desc) in enumerate(metrics):
    m_cols[i % 2].html(f"""
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;
                padding:28px;margin-bottom:16px;">
        <div style="font-family:'Poppins',sans-serif;font-size:17px;font-weight:600;
                    color:#0B3C5D;margin-bottom:10px;">{name}</div>
        <p style="font-family:'Inter',sans-serif;font-size:15px;color:#64748b;margin:0;line-height:1.7;">{desc}</p>
    </div>
    """)

st.html("<div style='height:64px;'></div>")

# ── Notes ─────────────────────────────────────────────────────────────────────
st.html("""
<div style="border-top:1px solid #e2e8f0;padding-top:64px;margin-bottom:32px;">
    <div style="font-family:'Inter',sans-serif;font-size:13px;font-weight:600;
                color:#1FA3A3;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:14px;">
        Important
    </div>
    <h2 style="font-family:'Poppins',sans-serif;font-size:36px;font-weight:700;
               color:#0B3C5D;letter-spacing:-0.02em;">Notes</h2>
</div>
""")

notes = [
    ("🖼️", "The system is intended for <strong style='color:#0B3C5D;'>underwater object images</strong> only."),
    ("📉", "Detection results may vary depending on <strong style='color:#0B3C5D;'>noise level and image quality</strong>."),
    ("🛡️", "The enhanced model is expected to perform better under <strong style='color:#0B3C5D;'>noisy underwater conditions</strong>."),
    ("🐚", "Detectable classes are limited to: <strong style='color:#0B3C5D;'>Echinus, Starfish, Scallop, and Holothurian</strong>."),
]
notes_html = "".join(
    f"""<div style="display:flex;align-items:flex-start;gap:20px;
                    padding:24px 0;border-bottom:1px solid #f1f5f9;">
        <span style="font-size:24px;flex-shrink:0;margin-top:2px;">{icon}</span>
        <p style="font-family:'Inter',sans-serif;font-size:16px;color:#475569;
                  line-height:1.75;margin:0;">{text}</p>
    </div>"""
    for icon, text in notes
)
st.html(f"""
<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;padding:8px 32px;
            box-shadow:0 1px 3px rgba(0,0,0,0.04);">
    {notes_html}
</div>
""")

render_footer()
