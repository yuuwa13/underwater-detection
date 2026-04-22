import streamlit as st

_BRAND_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #eef2f7 !important;
    color: #0f172a;
}

/* ── Hide Streamlit chrome ── */
[data-testid="stHeader"]         { display: none !important; }
[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stDecoration"]     { display: none !important; }
#MainMenu                        { display: none !important; }
footer                           { display: none !important; }

/* ── Content offset below floating navbar ── */
[data-testid="stAppViewContainer"] > section:first-child {
    padding-top: 88px !important;
}
[data-testid="stMainBlockContainer"],
.main .block-container,
section.main > div.block-container {
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding-bottom: 0 !important;
}

/* ── Floating pill navbar ── */
.navbar {
    position: fixed; top: 16px;
    left: 50%; transform: translateX(-50%);
    z-index: 9999;
    background: rgba(11, 60, 93, 0.82);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    width: min(1200px, calc(100vw - 32px));
    height: 56px;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 28px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 8px 32px rgba(11,60,93,0.25), 0 1px 2px rgba(0,0,0,0.12);
    white-space: nowrap;
}
.navbar-brand {
    font-family: 'Poppins', sans-serif;
    font-size: 13.5px; font-weight: 600;
    color: #ffffff; text-decoration: none;
    letter-spacing: 0.01em;
}
.navbar-links { display: flex; gap: 2rem; align-items: center; }
.nav-link {
    font-family: 'Inter', sans-serif;
    font-size: 13px; font-weight: 500;
    color: rgba(255,255,255,0.55); text-decoration: none;
    transition: color 0.15s;
    position: relative;
}
.nav-link::after {
    content: '';
    position: absolute; bottom: -4px; left: 0; right: 0;
    height: 2px; background: #1FA3A3;
    transform: scaleX(0); transition: transform 0.15s;
}
.nav-link:hover { color: #ffffff; }
.nav-link:hover::after { transform: scaleX(1); }
.nav-link.active { color: #ffffff; }
.nav-link.active::after { transform: scaleX(1); }

.nav-cta {
    font-family: 'Inter', sans-serif;
    font-size: 13px; font-weight: 600;
    color: #ffffff; text-decoration: none;
    background: #1FA3A3;
    padding: 8px 18px; border-radius: 9px;
    letter-spacing: 0.01em;
    transition: background 0.15s, box-shadow 0.15s;
    box-shadow: 0 1px 4px rgba(31,163,163,0.4);
}
.nav-cta:hover { background: #17888a; box-shadow: 0 4px 12px rgba(31,163,163,0.45); }
.nav-cta-active { background: #17888a; }

/* ── Cards ── */
.card {
    background: #ffffff;
    border-radius: 12px; padding: 28px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
    margin-bottom: 16px;
}
.card-title {
    font-family: 'Poppins', sans-serif;
    font-size: 18px; font-weight: 600;
    color: #0B3C5D; margin-bottom: 12px;
}
.proposed { border: 1.5px solid #1FA3A3; box-shadow: 0 0 0 3px rgba(31,163,163,0.08); }

/* ── Metric boxes ── */
.metric-box {
    background: #f8fafc;
    border-radius: 10px; padding: 14px 16px;
    border: 1px solid #e2e8f0;
    margin-bottom: 10px; color: #0f172a;
    font-family: 'Inter', sans-serif; font-size: 14px;
}
.metric-box b { color: #0B3C5D; font-size: 18px; font-weight: 600; }

/* ── Headings ── */
h1, h2, h3 { font-family: 'Poppins', sans-serif !important; color: #0B3C5D !important; }

/* ── Streamlit text ── */
.stMarkdown, .stMarkdown p { color: #374151; }

/* ── Buttons ── */
.stButton > button {
    background: #0B3C5D !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 22px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    letter-spacing: 0.01em !important;
    transition: background 0.15s, box-shadow 0.15s !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
}
.stButton > button:hover:not(:disabled) {
    background: #1FA3A3 !important;
    box-shadow: 0 4px 12px rgba(31,163,163,0.25) !important;
}
.stButton > button:disabled {
    background: #e2e8f0 !important;
    color: #94a3b8 !important;
    cursor: not-allowed !important;
    box-shadow: none !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    background: #f8fafc !important;
    margin-bottom: 8px !important;
    box-shadow: none !important;
}
[data-testid="stExpander"] summary {
    font-size: 13.5px !important; font-weight: 600 !important;
    color: #0B3C5D !important; padding: 10px 14px !important;
    background: #f8fafc !important; border-radius: 10px !important;
}
[data-testid="stExpander"] summary:hover { background: #f1f5f9 !important; }
[data-testid="stExpander"] summary svg { color: #0B3C5D !important; fill: #0B3C5D !important; }
[data-testid="stExpander"][open] summary { border-radius: 10px 10px 0 0 !important; }
[data-testid="stExpander"] > div[data-testid="stExpanderDetails"] {
    border-top: 1px solid #e2e8f0 !important;
    padding: 12px 14px !important;
    color: #374151 !important;
    background: #f8fafc !important;
    border-radius: 0 0 10px 10px !important;
}
[data-testid="stExpander"] * { background-color: transparent !important; }
[data-testid="stExpander"] .metric-box { background: #ffffff !important; }

/* ── Metrics ── */
[data-testid="stMetric"] label { color: #64748b !important; font-size: 13px !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #0B3C5D !important; font-family: 'Poppins', sans-serif !important; }
[data-testid="stMetricValue"] > div { color: #0B3C5D !important; }

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid #e2e8f0 !important; margin: 40px 0 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] { width: 100%; }
[data-testid="stFileUploaderDropzone"] {
    border: 1.5px dashed #cbd5e1 !important;
    border-radius: 12px !important;
    padding: 160px 40px !important;
    text-align: center !important;
    background: #fafafa !important;
    cursor: pointer !important;
    transition: border-color 0.2s, background 0.2s !important;
    min-height: 200px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #1FA3A3 !important;
    background: #f0fafa !important;
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
    display: block; width: 52px; height: 52px;
    background-color: #e0f4f4;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%231FA3A3' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/%3E%3Cpolyline points='17 8 12 3 7 8'/%3E%3Cline x1='12' y1='3' x2='12' y2='15'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: center;
    background-size: 24px 24px; border-radius: 50%;
    margin: 0 auto 14px auto;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div::after {
    content: "Click to upload image";
    display: block; font-size: 14px; font-weight: 500;
    color: #0B3C5D; margin-bottom: 4px;
}
[data-testid="stFileUploaderDropzoneInstructions"]::after {
    content: "PNG, JPG up to 10MB";
    display: block; font-size: 12px; color: #94a3b8;
    margin-top: 4px; text-align: center;
}
[data-testid="stFileUploaderDropzone"] button { display: none !important; }
[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="uploadedFileName"],
[data-testid="stFileUploader"] .uploadedFileName { color: #0B3C5D !important; }
[data-testid="stFileUploader"] > div > div > div { color: #0B3C5D !important; }

/* ── Alert ── */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ── Responsive ── */
/* Tablet */
@media (max-width: 768px) {
    .navbar {
        width: calc(100vw - 24px);
        padding: 0 16px;
        height: 50px;
        border-radius: 12px;
    }
    .navbar-brand { font-size: 12.5px; }
    .navbar-links { gap: 1rem; }
    .nav-link { font-size: 12px; }
    .nav-cta { font-size: 12px; padding: 6px 12px; }
    [data-testid="stAppViewContainer"] > section:first-child { padding-top: 78px !important; }
    [data-testid="stMainBlockContainer"],
    .main .block-container,
    section.main > div.block-container {
        padding-left: 1.25rem !important; padding-right: 1.25rem !important;
    }
}
/* Mobile */
@media (max-width: 480px) {
    .navbar {
        width: calc(100vw - 16px);
        padding: 0 12px;
        border-radius: 10px;
    }
    .navbar-brand { display: none; }
    .navbar-links { gap: 0.75rem; width: 100%; justify-content: space-between; }
    .nav-link { font-size: 11.5px; }
    .nav-cta { font-size: 11.5px; padding: 6px 10px; }
}
</style>
"""


def inject_branding():
    st.html(_BRAND_CSS)


def render_navbar(active: str = "Home"):
    pages = [
        ("Home",           "/",               "Home"),
        ("Instructions",   "/Instructions",   "Instructions"),
        ("Contact_Us",     "/Contact_Us",     "Contact Us"),
    ]
    links = "".join(
        f'<a href="{href}" class="nav-link{" active" if key == active else ""}">{label}</a>'
        for key, href, label in pages
    )
    run_sim_active = ' nav-cta-active' if active == "Run_Simulation" else ''
    st.html(
        f'<div class="navbar">'
        f'<a href="/" class="navbar-brand">Underwater Detection</a>'
        f'<div class="navbar-links">'
        f'{links}'
        f'<a href="/Run_Simulation" class="nav-cta{run_sim_active}">Run Simulation</a>'
        f'</div>'
        f'</div>'
    )


def render_footer():
    st.html("""
    <div style="
        background:#0B3C5D; color:rgba(255,255,255,0.5);
        text-align:center; padding:48px 6rem;
        font-family:'Inter',sans-serif; font-size:13.5px;
        width:100vw; margin-left:calc(-50vw + 50%);
        margin-top:5rem;
    ">
        <div style="font-family:'Poppins',sans-serif;font-weight:600;color:#ffffff;
                    font-size:14px;margin-bottom:10px;letter-spacing:0.01em;">
            An Enhanced YOLOv12 Architecture with Dual-Branch Network<br>for Underwater Object Detection
        </div>
        <div style="margin:6px 0;">University of Mindanao</div>
        <div style="margin:6px 0;">© 2026 All Rights Reserved</div>
    </div>
    """)
