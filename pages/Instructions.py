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

st.html("""
<div style="padding:56px 0 36px;">
    <div style="font-family:'Inter',sans-serif;font-size:11.5px;font-weight:600;
                color:#1FA3A3;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:12px;">
        Getting Started
    </div>
    <h1 style="font-family:'Poppins',sans-serif;font-size:38px;font-weight:700;
               color:#0B3C5D;margin-bottom:12px;letter-spacing:-0.02em;">How to Use</h1>
    <p style="font-family:'Inter',sans-serif;font-size:16px;color:#64748b;max-width:580px;line-height:1.7;">
        Follow these steps to run the underwater object detection simulation and interpret the results.
    </p>
</div>
""")

steps = [
    ("01", "Go to Run Simulation",
     "Navigate to the <strong style='color:#0B3C5D;'>Run Simulation</strong> page using the top navigation bar."),
    ("02", "Upload an Underwater Image",
     "Click the upload area or drag and drop a PNG or JPG file (up to 10 MB). "
     "For best results, use actual underwater photographs of marine objects."),
    ("03", "Get Sample Images",
     'Need test images? <a href="#" style="color:#1FA3A3;font-weight:600;text-decoration:none;border-bottom:1px solid #1FA3A3;">Browse the dataset on Google Drive</a> '
     "for sample underwater images of Echinus, Starfish, Scallop, and Holothurian."),
    ("04", "Click Run Detection",
     "Press the <strong style='color:#0B3C5D;'>Run Detection</strong> button in the top-right corner. Both models will process the image simultaneously."),
    ("05", "Wait for Processing",
     "The system runs inference through both the Baseline and Enhanced YOLOv12 models. "
     "This typically takes a few seconds depending on server load."),
    ("06", "Compare Results",
     "Review the side-by-side results — detection overlays, confidence scores, and evaluation metrics for both models."),
]

for num, title, desc in steps:
    st.html(f"""
    <div style="display:flex;gap:20px;align-items:flex-start;
                background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
                padding:24px;margin-bottom:12px;
                box-shadow:0 1px 3px rgba(0,0,0,0.04);">
        <div style="background:#0B3C5D;color:#ffffff;font-family:'Poppins',sans-serif;
                    font-weight:700;font-size:15px;border-radius:10px;
                    min-width:48px;height:48px;display:flex;align-items:center;
                    justify-content:center;flex-shrink:0;letter-spacing:0.02em;">
            {num}
        </div>
        <div>
            <div style="font-family:'Poppins',sans-serif;font-size:16px;font-weight:600;
                        color:#0B3C5D;margin-bottom:6px;">{title}</div>
            <p style="font-family:'Inter',sans-serif;font-size:14.5px;color:#475569;
                      line-height:1.7;margin:0;">{desc}</p>
        </div>
    </div>
    """)

st.html("<div style='height:48px;'></div>")

st.html("""
<div style="border-top:1px solid #e2e8f0;padding-top:48px;margin-bottom:28px;">
    <h2 style="font-family:'Poppins',sans-serif;font-size:24px;font-weight:700;
               color:#0B3C5D;letter-spacing:-0.01em;">Understanding the Results</h2>
</div>
""")

metric_col1, metric_col2 = st.columns(2, gap="large")
with metric_col1:
    for name, desc in [
        ("Highest Detection Accuracy", "The confidence score of the top bounding-box detection, expressed as a percentage."),
        ("Classification Results", "A ranked list of all detected objects and their individual confidence scores."),
    ]:
        st.html(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
                    padding:16px 18px;margin-bottom:12px;">
            <div style="font-family:'Poppins',sans-serif;font-size:14px;font-weight:600;
                        color:#0B3C5D;margin-bottom:6px;">{name}</div>
            <p style="font-family:'Inter',sans-serif;font-size:13.5px;color:#64748b;margin:0;line-height:1.65;">{desc}</p>
        </div>
        """)
with metric_col2:
    for name, desc in [
        ("Precision", "Fraction of detections that are correct. High precision = fewer false positives."),
        ("Recall", "Fraction of actual objects detected. High recall = fewer missed detections."),
        ("mAP@50", "Mean Average Precision at IoU threshold 0.50. Standard detection benchmark."),
        ("mAP@50:95", "Mean AP averaged across IoU 0.50–0.95. Stricter localization metric."),
    ]:
        st.html(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
                    padding:16px 18px;margin-bottom:12px;">
            <div style="font-family:'Poppins',sans-serif;font-size:14px;font-weight:600;
                        color:#0B3C5D;margin-bottom:6px;">{name}</div>
            <p style="font-family:'Inter',sans-serif;font-size:13.5px;color:#64748b;margin:0;line-height:1.65;">{desc}</p>
        </div>
        """)

st.html("<div style='height:48px;'></div>")

st.html("""
<div style="border-top:1px solid #e2e8f0;padding-top:48px;margin-bottom:20px;">
    <h2 style="font-family:'Poppins',sans-serif;font-size:24px;font-weight:700;
               color:#0B3C5D;letter-spacing:-0.01em;">Notes</h2>
</div>
""")

notes = [
    "The system is intended for <strong style='color:#0B3C5D;'>underwater object images</strong> only.",
    "Detection results may vary depending on <strong style='color:#0B3C5D;'>noise level and image quality</strong>.",
    "The enhanced model is expected to perform better under <strong style='color:#0B3C5D;'>noisy underwater conditions</strong>.",
    "Detectable classes are limited to: <strong style='color:#0B3C5D;'>Echinus, Starfish, Scallop, and Holothurian</strong>.",
]
notes_html = "".join(
    f'<li style="font-family:Inter,sans-serif;font-size:15px;color:#475569;'
    f'line-height:1.75;margin-bottom:10px;padding-left:4px;">{n}</li>'
    for n in notes
)
st.html(f"""
<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:28px;
            box-shadow:0 1px 3px rgba(0,0,0,0.04);">
    <ul style="padding-left:20px;margin:0;">{notes_html}</ul>
</div>
""")

render_footer()
