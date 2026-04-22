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
