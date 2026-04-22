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

st.html("""
<div style="padding:56px 0 36px;">
    <div style="font-family:'Inter',sans-serif;font-size:11.5px;font-weight:600;
                color:#1FA3A3;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:12px;">
        Get in Touch
    </div>
    <h1 style="font-family:'Poppins',sans-serif;font-size:38px;font-weight:700;
               color:#0B3C5D;margin-bottom:12px;letter-spacing:-0.02em;">Contact Us</h1>
    <p style="font-family:'Inter',sans-serif;font-size:16px;color:#64748b;max-width:580px;line-height:1.7;">
        For inquiries about this research, the dataset, or the detection system, feel free to reach out.
    </p>
</div>
""")

contact_col, info_col = st.columns([3, 2], gap="large")

with contact_col:
    st.html("""
    <div style="font-family:'Poppins',sans-serif;font-size:16px;font-weight:600;
                color:#0B3C5D;margin-bottom:20px;">Research Team</div>
    """)
    for role, institution, email in [
        ("Researcher", "University of Mindanao", "researcher@umindanao.edu.ph"),
        ("Thesis Adviser", "University of Mindanao", "adviser@umindanao.edu.ph"),
    ]:
        st.html(f"""
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
                    padding:24px;margin-bottom:12px;display:flex;gap:18px;align-items:flex-start;
                    box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <div style="background:#f0fafa;border:1px solid #c7eaea;border-radius:50%;
                        width:44px;height:44px;display:flex;align-items:center;
                        justify-content:center;font-size:18px;flex-shrink:0;">👤</div>
            <div>
                <div style="font-family:'Poppins',sans-serif;font-weight:600;font-size:15px;
                            color:#0B3C5D;margin-bottom:3px;">{role}</div>
                <div style="font-family:'Inter',sans-serif;font-size:13.5px;
                            color:#94a3b8;margin-bottom:6px;">{institution}</div>
                <a href="mailto:{email}" style="font-family:'Inter',sans-serif;font-size:13.5px;
                           color:#1FA3A3;font-weight:500;text-decoration:none;
                           border-bottom:1px solid #1FA3A3;padding-bottom:1px;">{email}</a>
            </div>
        </div>
        """)

with info_col:
    st.html("""
    <div style="background:#0B3C5D;border-radius:16px;padding:32px;color:#ffffff;height:100%;">
        <div style="font-family:'Inter',sans-serif;font-size:11px;font-weight:600;
                    color:#1FA3A3;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:20px;">
            Institution
        </div>
        <div style="font-family:'Poppins',sans-serif;font-size:18px;font-weight:600;
                    margin-bottom:6px;">University of Mindanao</div>
        <div style="font-family:'Inter',sans-serif;font-size:13.5px;
                    color:rgba(255,255,255,0.45);line-height:1.6;margin-bottom:28px;">
            Bolton St., Davao City, Philippines
        </div>
        <div style="border-top:1px solid rgba(255,255,255,0.1);padding-top:24px;">
            <div style="font-family:'Inter',sans-serif;font-size:11px;font-weight:600;
                        color:#1FA3A3;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:14px;">
                About This System
            </div>
            <p style="font-family:'Inter',sans-serif;font-size:13.5px;
                      color:rgba(255,255,255,0.5);line-height:1.7;margin:0;">
                This prototype was developed as part of a thesis study on enhanced underwater object
                detection using a dual-branch YOLOv12 architecture. Intended for academic and
                research purposes only.
            </p>
        </div>
    </div>
    """)

render_footer()
