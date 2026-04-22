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
            This prototype was developed as part of a thesis study on enhanced underwater object detection
            using a dual-branch YOLOv12 architecture. It is intended for academic and research purposes.
        </p>
    </div>
    """, unsafe_allow_html=True)

render_footer()
