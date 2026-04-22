import streamlit as st

from utils import inject_branding, render_footer, render_navbar

st.set_page_config(
    page_title="Contact Us · Underwater Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_branding()
render_navbar("Contact_Us")

st.html("""
<div style="padding:16px 0 48px;">
    <div style="font-family:'Inter',sans-serif;font-size:13px;font-weight:600;
                color:#1FA3A3;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:14px;">
        Get in Touch
    </div>
    <h1 style="font-family:'Poppins',sans-serif;font-size:42px;font-weight:700;
               color:#0B3C5D;margin-bottom:14px;letter-spacing:-0.02em;">Contact Us</h1>
    <p style="font-family:'Inter',sans-serif;font-size:17px;color:#64748b;max-width:600px;line-height:1.75;">
        For questions, feedback, or research-related inquiries, please contact any member of the research team.
    </p>
</div>
""")

contact_col, info_col = st.columns([3, 2], gap="large")

researchers = [
    (
        "Kristoffe Bien P. Montelibano",
        "BS in Computer Science",
        "University of Mindanao, Davao City",
        "k.montelibano.534284@umindanao.edu.ph",
    ),
    (
        "Aaron John I. Olanolan",
        "BS in Computer Science",
        "University of Mindanao, Davao City",
        "a.olanolan.533790@umindanao.edu.ph",
    ),
    (
        "Al-abass D. Ibrahim",
        "BS in Computer Science",
        "University of Mindanao, Davao City",
        "a.ibrahim.533550@umindanao.edu.ph",
    ),
]

with contact_col:
    st.html("""
    <div style="font-family:'Inter',sans-serif;font-size:13px;font-weight:600;
                color:#1FA3A3;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:24px;">
        Research Team
    </div>
    """)
    for name, degree, institution, email in researchers:
        initials = "".join(p[0] for p in name.split() if p[0].isupper())[:2]
        st.html(f"""
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;
                    padding:28px 32px;margin-bottom:14px;
                    display:flex;gap:22px;align-items:center;
                    box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <div style="background:#0B3C5D;border-radius:50%;
                        width:54px;height:54px;display:flex;align-items:center;
                        justify-content:center;font-family:'Poppins',sans-serif;
                        font-size:17px;font-weight:700;color:#ffffff;flex-shrink:0;
                        letter-spacing:0.02em;">{initials}</div>
            <div style="flex:1;min-width:0;">
                <div style="font-family:'Poppins',sans-serif;font-weight:600;font-size:17px;
                            color:#0B3C5D;margin-bottom:3px;">{name}</div>
                <div style="font-family:'Inter',sans-serif;font-size:14px;
                            color:#64748b;margin-bottom:10px;">{degree} &nbsp;·&nbsp; {institution}</div>
                <a href="mailto:{email}"
                   style="font-family:'Inter',sans-serif;font-size:14px;font-weight:500;
                          color:#1FA3A3;text-decoration:none;
                          border-bottom:1px solid rgba(31,163,163,0.4);padding-bottom:1px;
                          word-break:break-all;">{email}</a>
            </div>
        </div>
        """)

with info_col:
    st.markdown(
        """<style>
        [data-testid="stHorizontalBlock"] { align-items: stretch !important; }
        [data-testid="stColumn"]:last-child {
            display: flex !important;
            flex-direction: column !important;
        }
        [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"] {
            flex: 1 !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-end !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )
    st.html("""
    <div style="background:#0B3C5D;border-radius:16px;padding:40px;color:#ffffff;">
        <div style="font-family:'Inter',sans-serif;font-size:13px;font-weight:600;
                    color:#1FA3A3;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:20px;">
            Institution
        </div>
        <div style="font-family:'Poppins',sans-serif;font-size:20px;font-weight:600;
                    margin-bottom:8px;">University of Mindanao</div>
        <div style="font-family:'Inter',sans-serif;font-size:15px;
                    color:rgba(255,255,255,0.45);line-height:1.65;margin-bottom:32px;">
            Bolton St., Davao City, Philippines
        </div>
        <div style="border-top:1px solid rgba(255,255,255,0.1);padding-top:28px;margin-bottom:28px;">
            <div style="font-family:'Inter',sans-serif;font-size:13px;font-weight:600;
                        color:#1FA3A3;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:16px;">
                Degree Program
            </div>
            <div style="font-family:'Poppins',sans-serif;font-size:16px;font-weight:600;
                        color:#ffffff;margin-bottom:6px;">BS in Computer Science</div>
            <div style="font-family:'Inter',sans-serif;font-size:14px;
                        color:rgba(255,255,255,0.4);line-height:1.65;">
                College of Computing Education
            </div>
        </div>
        <div style="border-top:1px solid rgba(255,255,255,0.1);padding-top:28px;">
            <div style="font-family:'Inter',sans-serif;font-size:13px;font-weight:600;
                        color:#1FA3A3;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:16px;">
                About This System
            </div>
            <p style="font-family:'Inter',sans-serif;font-size:14px;
                      color:rgba(255,255,255,0.45);line-height:1.75;margin:0;">
                Developed as part of a thesis study on enhanced underwater object detection
                using a dual-branch YOLOv12 architecture. Intended for academic and research purposes only.
            </p>
        </div>
    </div>
    """)

render_footer()
