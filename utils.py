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
    """Render the fixed top navbar. active is one of: Home, Instructions, Contact_Us, Run_Simulation."""
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
