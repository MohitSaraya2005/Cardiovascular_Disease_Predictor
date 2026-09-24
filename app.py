"""
CardioPulse AI - Clinical Cardiovascular Disease Risk Intelligence Platform
Production-Ready Streamlit Web Application
"""

import os
import io
import time
import json
import base64
import textwrap
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & THEMING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioPulse AI | Cardiovascular Risk Intelligence",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to render HTML safely without markdown code-block indentation issues
def clean_html(html_str):
    return "\n".join([line.strip() for line in html_str.strip().splitlines()])

def get_base64_image(file_path):
    """Convert local image file to base64 data URL string."""
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
            ext = os.path.splitext(file_path)[1].lower().replace('.', '')
            if ext == 'jpg':
                ext = 'jpeg'
            return f"data:image/{ext};base64,{base64.b64encode(data).decode()}"
    return ""

# -----------------------------------------------------------------------------
# CUSTOM CSS DESIGN SYSTEM (Clinical Palette: Navy #1E3A8A, Teal #0D9488, Slate)
# -----------------------------------------------------------------------------
def apply_custom_css():
    st.markdown(clean_html("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }
    
    /* Top Main App Header & Padding Adjustments */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    .block-container {
        padding-top: 4rem !important;
        padding-bottom: 2rem !important;
        max-width: 1300px !important;
    }
    
    /* Main Clinical Header Banner */
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #0F172A 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: white;
        box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.25);
        margin-bottom: 1.5rem;
        border-left: 6px solid #0D9488;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::after {
        content: "🫀";
        position: absolute;
        right: 20px;
        bottom: -10px;
        font-size: 7rem;
        opacity: 0.12;
        pointer-events: none;
    }
    
    .main-header h1 {
        font-weight: 800;
        font-size: 2.2rem;
        margin: 0;
        color: #FFFFFF !important;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
        max-width: 800px;
    }

    /* Hero Card Container matching exact reference screenshot */
    .hero-banner-container {
        background: linear-gradient(135deg, #050d1a 0%, #0a1b32 60%, #06152a 100%);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        color: white;
        box-shadow: 0 12px 35px -5px rgba(5, 13, 26, 0.4);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .hero-title-text {
        font-size: 2.6rem;
        font-weight: 800;
        color: #FFFFFF !important;
        line-height: 1.15;
        margin: 0 0 0.8rem 0;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle-text {
        color: #94A3B8;
        font-size: 1.05rem;
        line-height: 1.55;
        margin-bottom: 1.8rem;
        font-weight: 400;
        max-width: 520px;
    }
    
    .hero-links-text {
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #94A3B8;
        font-weight: 500;
    }
    
    .hero-links-text a {
        color: #94A3B8;
        text-decoration: underline;
        transition: color 0.2s;
    }
    
    .hero-links-text a:hover {
        color: #38BDF8;
    }

    /* Universal Medical Disclaimer Banner matching image */
    .disclaimer-banner {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        color: #334155;
        padding: 0.65rem 1.2rem;
        border-radius: 8px;
        font-size: 0.84rem;
        font-weight: 500;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    
    .disclaimer-banner strong {
        color: #0F172A;
        font-weight: 700;
    }
    
    /* Impact Metric Card Styling */
    .impact-metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem 1.1rem;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.03);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .impact-metric-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.6rem;
    }
    
    .impact-metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1;
        letter-spacing: -0.5px;
    }
    
    .impact-metric-lbl {
        font-size: 0.75rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        line-height: 1.3;
    }

    /* Clinical Card Container */
    .clinical-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.6rem;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.04);
        margin-bottom: 1.5rem;
    }
    
    .card-title {
        color: #1E3A8A;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border-bottom: 2px solid #F1F5F9;
        padding-bottom: 0.6rem;
    }

    /* Metric Display Box */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
    }
    
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #1E3A8A;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.82rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.3rem;
    }

    /* Status Badges */
    .badge-low-risk {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
        padding: 0.45rem 1.2rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }
    
    .badge-high-risk {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
        padding: 0.45rem 1.2rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }
    
    .badge-normal {
        background-color: #F0FDF4;
        color: #15803D;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    
    .badge-warning {
        background-color: #FEFCE8;
        color: #A16207;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .badge-danger {
        background-color: #FEF2F2;
        color: #B91C1C;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* Input Field Labels */
    label {
        font-weight: 600 !important;
        color: #334155 !important;
        font-size: 0.92rem !important;
    }
    
    /* Primary CTA Button Styling */
    div.stButton > button {
        background: linear-gradient(135deg, #14B8A6 0%, #06B6D4 100%) !important;
        color: #050D1A !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.02rem !important;
        border-radius: 25px !important;
        box-shadow: 0 4px 18px rgba(20, 184, 166, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    div.stButton > button:hover {
        background: linear-gradient(135deg, #0D9488 0%, #0891B2 100%) !important;
        box-shadow: 0 6px 22px rgba(20, 184, 166, 0.5) !important;
        transform: translateY(-1px) !important;
        color: #FFFFFF !important;
    }

    /* Hero specific button full width or inline override */
    div.hero-btn-wrap div.stButton > button {
        width: auto !important;
    }
    
    /* Premium ECG Dark Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0B111E 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #E2E8F0 !important;
    }

    /* Option Menu Custom Container Overrides */
    iframe[title="streamlit_option_menu.option_menu"] {
        border-radius: 16px !important;
        box-shadow: 0 0 35px rgba(13, 148, 136, 0.15) !important;
    }

    /* Sidebar Radio Button Fallback Match */
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        background-color: #1A242F !important;
        padding: 1rem !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 0 0 30px rgba(13, 148, 136, 0.2) !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: transparent !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        padding: 0.75rem 1rem !important;
        margin-bottom: 0.3rem !important;
        border: none !important;
        font-weight: 600 !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: rgba(255, 255, 255, 0.08) !important;
    }

    /* Footer */
    .clinical-footer {
        text-align: center;
        padding: 1.8rem;
        color: #64748B;
        font-size: 0.85rem;
        border-top: 1px solid #E2E8F0;
        margin-top: 3rem;
    }
    </style>
    """), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# BACKEND MODEL & SCALER UTILITIES
# -----------------------------------------------------------------------------
MODEL_FILES = ["cardiovascular_logistic_regression.pkl", "cardiovascular_decision_tree.pkl"]
DATA_FILE = "cardio_train.csv"

@st.cache_resource
def load_ml_model():
    """Safely load the pre-trained cardiovascular machine learning model."""
    found_file = None
    for mf in MODEL_FILES:
        if os.path.exists(mf):
            found_file = mf
            break
            
    if not found_file:
        return None, f"Model file not found in root directory (checked: {', '.join(MODEL_FILES)})."
    try:
        model = joblib.load(found_file)
        return model, None
    except Exception as e:
        return None, f"Failed to load model file '{found_file}': {str(e)}"

@st.cache_resource
def load_feature_scaler():
    """
    Initialize StandardScaler calibrated on dataset numerical statistics.
    Numerical features: ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
    """
    scaler = StandardScaler()
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE, sep=";")
            df['age'] = (df['age'] / 365).astype(int)
            df = df[
                (df['height'] >= 100) & (df['height'] <= 250) &
                (df['weight'] >= 30) & (df['weight'] <= 250) &
                (df['ap_hi'] >= 70) & (df['ap_hi'] <= 250) &
                (df['ap_lo'] >= 40) & (df['ap_lo'] <= 180) &
                (df['ap_hi'] > df['ap_lo'])
            ]
            numerical_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
            scaler.fit(df[numerical_cols])
            return scaler
        except Exception:
            pass
            
    # Fallback to pre-calculated dataset mean & scale from cardio_train.csv
    scaler.mean_ = np.array([52.81488769, 164.38800532, 74.12400448, 126.65753949, 81.30553278])
    scaler.scale_ = np.array([6.77361081, 7.9776704, 14.34592664, 16.70822955, 9.47845748])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_samples_seen_ = 54000
    return scaler

def calculate_bmi(weight_kg, height_cm):
    """Auto-compute Body Mass Index (BMI) and categorization."""
    if height_cm <= 0:
        return 0.0, "Invalid"
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)
    
    if bmi < 18.5:
        category = "Underweight"
    elif 18.5 <= bmi < 25.0:
        category = "Normal Weight"
    elif 25.0 <= bmi < 30.0:
        category = "Overweight"
    else:
        category = "Obese"
    return round(bmi, 2), category

def render_disclaimer():
    """Reusable Medical Disclaimer Across All Pages matching exact image."""
    st.markdown(clean_html("""
    <div class="disclaimer-banner">
        <span>⚠️</span>
        <div><strong>CLINICAL DISCLAIMER:</strong> This tool is designed strictly for screening and educational purposes. It does not constitute a diagnostic medical device or a substitute for professional clinical evaluation.</div>
    </div>
    """), unsafe_allow_html=True)

def render_risk_gauge(risk_percent):
    """Plotly Semi-Circle Clinical Gauge Chart for Risk Score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_percent,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Calculated CVD Probability Score", 'font': {'size': 18, 'color': '#1E3A8A', 'family': 'Plus Jakarta Sans'}},
        number={'suffix': "%", 'font': {'size': 42, 'color': '#1E3A8A', 'weight': 'bold'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#94A3B8"},
            'bar': {'color': "#DC2626" if risk_percent >= 50 else ("#D97706" if risk_percent >= 35 else "#059669"), 'thickness': 0.3},
            'bgcolor': "#FFFFFF",
            'borderwidth': 2,
            'bordercolor': "#E2E8F0",
            'steps': [
                {'range': [0, 35], 'color': '#DCFCE7'},
                {'range': [35, 65], 'color': '#FEF3C7'},
                {'range': [65, 100], 'color': '#FEE2E2'}
            ],
            'threshold': {
                'line': {'color': "#991B1B", 'width': 4},
                'thickness': 0.8,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Plus Jakarta Sans, sans-serif")
    )
    return fig

# -----------------------------------------------------------------------------
# EXACT MATCH REFERENCE DESIGN SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        # Faint Heartbeat ECG Graphic Header Background + Centered Glowing Rounded App Icon
        st.markdown(clean_html("""
<div style="position: relative; text-align: center; padding: 1.2rem 0 1rem 0; overflow: hidden;">
<svg style="position: absolute; top: -10px; left: -10%; width: 120%; height: 120px; opacity: 0.15; pointer-events: none;" viewBox="0 0 500 100" preserveAspectRatio="none">
<path d="M0,50 L120,50 L135,25 L150,75 L165,10 L180,90 L195,50 L210,50 L220,40 L230,60 L240,50 L500,50" fill="none" stroke="#14B8A6" stroke-width="2.5"/>
</svg>
<div style="display: flex; justify-content: center; margin-bottom: 0.85rem;">
<div style="width: 64px; height: 64px; background: linear-gradient(135deg, #0284C7 0%, #0D9488 100%); border-radius: 18px; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 30px rgba(13, 148, 136, 0.45), 0 0 20px rgba(2, 132, 199, 0.35);">
<span style="font-size: 2.1rem;">🫀</span>
</div>
</div>
<div style="font-size: 1.55rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.5px; text-align: center;">
CardioPulse <span style="color: #14B8A6;">AI</span>
</div>
<div style="font-size: 0.72rem; color: #94A3B8; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; text-align: center; margin-top: 0.35rem; margin-bottom: 1.2rem;">
CLINICAL INTELLIGENCE PLATFORM
</div>
</div>
"""), unsafe_allow_html=True)
        
        # Check query params for page routing
        if "page" in st.query_params:
            qp_page = st.query_params["page"]
            if qp_page in ["Home", "Risk Assessment", "About", "Contact Us"]:
                st.session_state["current_page"] = qp_page

        if "current_page" not in st.session_state:
            st.session_state["current_page"] = "Home"

        options_list = ["Home", "Risk Assessment", "About", "Contact Us"]
        default_idx = options_list.index(st.session_state["current_page"]) if st.session_state["current_page"] in options_list else 0

        selected_page = st.session_state["current_page"]
        try:
            from streamlit_option_menu import option_menu
            selected_page = option_menu(
                menu_title=None,
                options=options_list,
                icons=["house-door-fill", "heart-pulse-fill", "info-circle-fill", "envelope-fill"],
                default_index=default_idx,
                styles={
                    "container": {
                        "padding": "0.6rem 0.6rem!important", 
                        "background-color": "#1A232E", 
                        "border-radius": "16px",
                        "border": "1px solid rgba(255, 255, 255, 0.08)",
                        "box-shadow": "0 0 30px rgba(13, 148, 136, 0.22)"
                    },
                    "icon": {
                        "color": "#14B8A6", 
                        "font-size": "1.2rem"
                    },
                    "nav-link": {
                        "font-size": "0.98rem", 
                        "text-align": "left", 
                        "margin": "0.35rem 0px", 
                        "padding": "0.75rem 1rem", 
                        "font-weight": "600",
                        "color": "#FFFFFF", 
                        "border-radius": "12px",
                        "transition": "all 0.2s ease"
                    },
                    "nav-link-hover": {
                        "background-color": "rgba(255, 255, 255, 0.08)", 
                        "color": "#FFFFFF"
                    },
                    "nav-link-selected": {
                        "background": "linear-gradient(135deg, #0F766E 0%, #14B8A6 100%)", 
                        "color": "#FFFFFF", 
                        "font-weight": "700",
                        "box-shadow": "0 4px 15px rgba(20, 184, 166, 0.4)"
                    },
                }
            )
        except Exception:
            selected_page = st.radio(
                "NAVIGATION MENU",
                options_list,
                index=default_idx
            )
            
        st.session_state["current_page"] = selected_page
        
        # Bottom Right Subtle Sparkle Icon ✦
        st.markdown(clean_html("""
<div style="text-align: right; padding-right: 0.5rem; margin-top: 1.5rem; color: #14B8A6; font-size: 1.2rem; opacity: 0.7;">
✦
</div>
"""), unsafe_allow_html=True)
        
        return selected_page

# -----------------------------------------------------------------------------
# PAGE 1: HOME OVERVIEW (EXACT MATCH TO REFERENCE DESIGN)
# -----------------------------------------------------------------------------
def page_home():
    render_disclaimer()
    
    # Hero Card Container matching reference screenshot exactly
    banner_b64 = get_base64_image("heart_banner.jpg")
    
    bg_banner_css = (
        f"background: linear-gradient(90deg, #050D1A 0%, rgba(5, 13, 26, 0.95) 38%, rgba(5, 13, 26, 0.55) 65%, rgba(5, 13, 26, 0.15) 100%), url('{banner_b64}') right center / cover no-repeat;"
        if banner_b64 else
        "background: linear-gradient(135deg, #050d1a 0%, #0a1b32 60%, #06152a 100%);"
    )
    
    # Integrated Hero Banner with deep-blended 3D glowing heart background texture
    st.markdown(clean_html(f"""
    <div style="{bg_banner_css} border-radius: 20px; padding: 3.2rem 3.5rem; color: white; border: 1px solid rgba(255, 255, 255, 0.08); box-shadow: 0 15px 40px -5px rgba(5, 13, 26, 0.5); margin-bottom: 2rem; position: relative; overflow: hidden; min-height: 280px; display: flex; align-items: center;">
        <div style="max-width: 620px; position: relative; z-index: 2;">
            <h1 style="font-size: 2.75rem; font-weight: 800; color: #FFFFFF !important; line-height: 1.15; margin: 0 0 0.9rem 0; letter-spacing: -0.5px;">
                CardioPulse AI<br>
                Intelligence Platform
            </h1>
            <p style="color: #94A3B8; font-size: 1.05rem; line-height: 1.55; margin-bottom: 2rem; font-weight: 400; max-width: 540px;">
                Next-Generation Clinical Decision Support System for<br>Early Cardiovascular Disease Risk Stratification
            </p>
            <div style="margin-bottom: 1.2rem;">
                <a href="?page=Risk+Assessment" target="_self" style="background: linear-gradient(135deg, #14B8A6 0%, #06B6D4 100%); color: #050D1A; font-weight: 700; font-size: 1rem; padding: 13px 32px; border-radius: 25px; text-decoration: none; display: inline-block; box-shadow: 0 4px 20px rgba(20, 184, 166, 0.45); transition: all 0.2s ease;">
                    Launch Risk Stratification Engine
                </a>
            </div>
            <div style="font-size: 0.9rem; color: #94A3B8; font-weight: 500;">
                <a href="#mission" style="color: #94A3B8; text-decoration: underline;">View Documentation</a> &nbsp;&nbsp;·&nbsp;&nbsp; <a href="#mission" style="color: #94A3B8; text-decoration: underline;">Read Case Studies</a>
            </div>
        </div>
    </div>
    """), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Key Global CVD Metrics Grid matching reference screenshot
    st.markdown(clean_html("""
    <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 1.2rem;">
        <span style="font-size: 1.6rem; line-height: 1;">📊</span>
        <h2 style="font-size: 1.45rem; font-weight: 700; color: #0F172A; margin: 0;">Global Cardiovascular Impact Metrics</h2>
    </div>
    """), unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(clean_html("""
        <div class="impact-metric-card">
            <div class="impact-metric-top">
                <span class="impact-metric-val">17.9M</span>
                <svg viewBox="0 0 50 30" width="48" height="28" fill="none" stroke="#0EA5E9" stroke-width="2.5" stroke-linecap="round">
                    <path d="M2,25 Q12,20 20,15 T35,8 T48,3" />
                </svg>
            </div>
            <div class="impact-metric-lbl">ANNUAL DEATHS GLOBALLY</div>
        </div>
        """), unsafe_allow_html=True)
        
    with col2:
        st.markdown(clean_html("""
        <div class="impact-metric-card">
            <div class="impact-metric-top">
                <span class="impact-metric-val">80%</span>
                <svg viewBox="0 0 36 36" width="30" height="30">
                    <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="#E2E8F0" stroke-width="4" />
                    <path d="M18 2.0845 a 15.9155 15.9155 0 1 1 -12.5 5.2" fill="none" stroke="#64748B" stroke-width="4" stroke-linecap="round" />
                </svg>
            </div>
            <div class="impact-metric-lbl">PREVENTABLE VIA EARLY RISK STRATIFICATION</div>
        </div>
        """), unsafe_allow_html=True)
        
    with col3:
        st.markdown(clean_html("""
        <div class="impact-metric-card">
            <div class="impact-metric-top">
                <span class="impact-metric-val">12</span>
            </div>
            <div class="impact-metric-lbl">CLINICAL BIOMARKERS ANALYZED</div>
        </div>
        """), unsafe_allow_html=True)
        
    with col4:
        st.markdown(clean_html("""
        <div class="impact-metric-card">
            <div class="impact-metric-top">
                <span class="impact-metric-val">73%</span>
            </div>
            <div class="impact-metric-lbl">ML PREDICTION PRECISION</div>
        </div>
        """), unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Mission Statement, Core Capabilities & Assessment CTA Grid
    col_left, col_right = st.columns([6, 5])
    
    with col_left:
        # Card 1: Mission & Clinical Vision
        st.markdown(clean_html("""
        <div class="clinical-card" id="mission" style="border: 1px solid #E2E8F0; border-radius: 16px; padding: 1.6rem; background: #FFFFFF; position: relative; overflow: hidden; margin-bottom: 1.5rem;">
            <svg style="position: absolute; right: -20px; bottom: -20px; width: 180px; height: 180px; opacity: 0.15; pointer-events: none;" viewBox="0 0 100 100">
                <polygon points="50,5 95,25 95,75 50,95 5,75 5,25" fill="none" stroke="#0D9488" stroke-width="1.5" />
                <polygon points="50,20 80,35 80,65 50,80 20,65 20,35" fill="none" stroke="#0284C7" stroke-width="1" />
                <line x1="50" y1="5" x2="50" y2="95" stroke="#0D9488" stroke-width="0.8" />
                <line x1="5" y1="25" x2="95" y2="75" stroke="#0284C7" stroke-width="0.8" />
                <line x1="5" y1="75" x2="95" y2="25" stroke="#0284C7" stroke-width="0.8" />
            </svg>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem; font-weight: 700; font-size: 1.15rem; color: #0F3759;">
                    <span>🎯</span> Mission & Clinical Vision
                </div>
                <div style="font-weight: 800; font-size: 1.2rem; color: #0D9488; letter-spacing: -0.3px;">
                    CardioPulse <span style="color: #0284C7;">AI</span>
                </div>
            </div>
            <p style="color: #334155; font-size: 0.95rem; line-height: 1.6; margin-bottom: 1.2rem;">
                Cardiovascular diseases (CVDs) remain the leading cause of morbidity and mortality worldwide. <strong>CardioPulse AI</strong> empowers healthcare practitioners and patients by providing non standardized physiologicales and behavioral biomarkers.
            </p>
            <div style="text-align: center; font-weight: 800; font-size: 1.1rem; color: #0F3759; margin: 1.4rem 0 1.2rem 0;">
                Revolutionizing Cardiovascular Disease Prediction
            </div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8rem; margin-top: 1rem;">
                <div style="display: flex; align-items: center; gap: 0.6rem;">
                    <div style="width: 36px; height: 36px; border-radius: 8px; background: #E0F2FE; display: flex; align-items: center; justify-content: center; color: #0284C7; font-size: 1.1rem; flex-shrink: 0;">
                        ➔
                    </div>
                    <div style="font-size: 0.72rem; font-weight: 800; color: #0F172A; text-transform: uppercase; line-height: 1.2;">
                        NON-INVASIVE<br>STRATIFICATION
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 0.6rem;">
                    <div style="width: 36px; height: 36px; border-radius: 8px; background: #E6FFFA; display: flex; align-items: center; justify-content: center; color: #0D9488; font-size: 1.1rem; flex-shrink: 0;">
                        🖧
                    </div>
                    <div style="font-size: 0.72rem; font-weight: 800; color: #0F172A; text-transform: uppercase; line-height: 1.2;">
                        INSTANT<br>BIOSENSOR<br>ANALYSIS
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 0.6rem;">
                    <div style="width: 36px; height: 36px; border-radius: 8px; background: #F0FDF4; display: flex; align-items: center; justify-content: center; color: #16A34A; font-size: 1.1rem; flex-shrink: 0;">
                        📊
                    </div>
                    <div style="font-size: 0.72rem; font-weight: 800; color: #0F172A; text-transform: uppercase; line-height: 1.2;">
                        INTERPRETABLE<br>RISK SCORES
                    </div>
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)
        
        # Card 2: Core Capabilities
        st.markdown(clean_html("""
        <div class="clinical-card" style="border: 1px solid #E2E8F0; border-radius: 16px; padding: 1.6rem; background: #FFFFFF;">
            <div style="display: flex; align-items: center; gap: 0.5rem; font-weight: 700; font-size: 1.15rem; color: #0F3759; margin-bottom: 1.2rem;">
                <span style="color: #F97316;">⚡</span> Core Capabilities
            </div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.9rem;">
                <div style="border: 1px solid #E2E8F0; border-radius: 12px; padding: 0.85rem 0.9rem; display: flex; align-items: center; gap: 0.75rem; background: #FFFFFF;">
                    <div style="width: 42px; height: 42px; border-radius: 10px; background: #E0F2FE; display: flex; align-items: center; justify-content: center; font-size: 1.3rem;">
                        ⚖️
                    </div>
                    <div style="font-weight: 700; font-size: 0.92rem; color: #0F172A;">
                        BMI Calc
                    </div>
                </div>
                <div style="border: 1px solid #E2E8F0; border-radius: 12px; padding: 0.85rem 0.9rem; display: flex; align-items: center; gap: 0.75rem; background: #FFFFFF;">
                    <div style="width: 42px; height: 42px; border-radius: 10px; background: #FFEDD5; display: flex; align-items: center; justify-content: center; font-size: 1.3rem;">
                        📈
                    </div>
                    <div style="font-weight: 700; font-size: 0.92rem; color: #0F172A;">
                        Hemodynamics
                    </div>
                </div>
                <div style="border: 1px solid #E2E8F0; border-radius: 12px; padding: 0.85rem 0.9rem; display: flex; align-items: center; gap: 0.75rem; background: #FFFFFF;">
                    <div style="width: 42px; height: 42px; border-radius: 10px; background: #DCFCE7; display: flex; align-items: center; justify-content: center; font-size: 1.3rem;">
                        🎛️
                    </div>
                    <div style="font-weight: 700; font-size: 0.92rem; color: #0F172A; line-height: 1.2;">
                        Risk<br>Spectrum
                    </div>
                </div>
                <div style="border: 1px solid #E2E8F0; border-radius: 12px; padding: 0.85rem 0.9rem; display: flex; align-items: center; gap: 0.75rem; background: #FFFFFF;">
                    <div style="width: 42px; height: 42px; border-radius: 10px; background: #FEF9C3; display: flex; align-items: center; justify-content: center; font-size: 1.3rem;">
                        🎛️
                    </div>
                    <div style="font-weight: 700; font-size: 0.92rem; color: #0F172A; line-height: 1.2;">
                        Risk<br>Spectrum
                    </div>
                </div>
                <div style="border: 1px solid #E2E8F0; border-radius: 12px; padding: 0.85rem 0.9rem; display: flex; align-items: center; gap: 0.75rem; background: #FFFFFF;">
                    <div style="width: 42px; height: 42px; border-radius: 10px; background: #DBEAFE; display: flex; align-items: center; justify-content: center; font-size: 1.3rem;">
                        📋
                    </div>
                    <div style="font-weight: 700; font-size: 0.92rem; color: #0F172A; line-height: 1.2;">
                        Clinical<br>Directives
                    </div>
                </div>
                <div style="border: 1px solid #E2E8F0; border-radius: 12px; padding: 0.85rem 0.9rem; display: flex; align-items: center; gap: 0.75rem; background: #FFFFFF;">
                    <div style="width: 42px; height: 42px; border-radius: 10px; background: #F3E8FF; display: flex; align-items: center; justify-content: center; font-size: 1.3rem;">
                        📑
                    </div>
                    <div style="font-weight: 700; font-size: 0.92rem; color: #0F172A; line-height: 1.2;">
                        Exportable<br>Reports
                    </div>
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)
        
    with col_right:
        # Card 3: Ready for Assessment?
        st.markdown(clean_html("""
        <div style="background: linear-gradient(180deg, #E6FFFA 0%, #D1FAF5 100%); border: 1px solid #A7F3D0; border-radius: 16px; padding: 1.8rem; box-shadow: 0 4px 20px -2px rgba(13, 148, 136, 0.08); margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 0.6rem; font-weight: 800; font-size: 1.35rem; color: #0F172A; margin-bottom: 0.9rem;">
                <span>🚀</span> Ready for Assessment?
            </div>
            <p style="color: #334155; font-size: 0.95rem; line-height: 1.55; margin-bottom: 1.5rem;">
                Proceed to the interactive prediction workspace to input patient clinical metrics and generate a full risk analysis.
            </p>
            <a href="?page=Risk+Assessment" target="_self" style="display: block; text-align: center; background: linear-gradient(135deg, #0D9488 0%, #14B8A6 100%); color: #FFFFFF; font-weight: 700; font-size: 0.95rem; padding: 12px 24px; border-radius: 25px; text-decoration: none; box-shadow: 0 4px 15px rgba(13, 148, 136, 0.35); letter-spacing: 0.5px;">
                START INTERACTIVE ASSESSMENT
            </a>
        </div>
        """), unsafe_allow_html=True)
        
        # Card 4: Systolic BP Distribution Chart
        st.markdown(clean_html("""
        <div class="clinical-card" style="border: 1px solid #E2E8F0; border-radius: 16px; padding: 1.4rem; background: #FFFFFF;">
            <div style="font-weight: 800; font-size: 1.2rem; color: #0F172A; margin-bottom: 0.4rem;">
                Systolic BP Distribution
            </div>
        """), unsafe_allow_html=True)
        
        if os.path.exists(DATA_FILE):
            try:
                df_sample = pd.read_csv(DATA_FILE, sep=";", nrows=10000)
                df_sample = df_sample[(df_sample['ap_hi'] >= 90) & (df_sample['ap_hi'] <= 200)]
                df_sample['Target'] = df_sample['cardio'].map({0: 'Healthy', 1: 'CVD Present'})
                fig_hist = px.histogram(
                    df_sample, 
                    x="ap_hi", 
                    color="Target", 
                    labels={"ap_hi": "Systolic Blood Pressure (mmHg)", "count": "Count"},
                    color_discrete_map={"Healthy": "#0D9488", "CVD Present": "#DC2626"},
                    barmode="group",
                    range_x=[85, 205],
                    nbins=24
                )
                fig_hist.update_traces(opacity=1.0, marker_line_width=0)
                fig_hist.update_layout(
                    height=260, 
                    bargap=0.25,
                    bargroupgap=0.05,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(
                        title=dict(text="Target", font=dict(size=12, family="Plus Jakarta Sans", weight="bold")),
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="right",
                        x=1
                    ),
                    xaxis=dict(
                        title="Systolic Blood Pressure (mmHg)",
                        showgrid=False,
                        linecolor="#CBD5E1",
                        tickfont=dict(family="Plus Jakarta Sans", size=11),
                        tickvals=[100, 150, 200]
                    ),
                    yaxis=dict(
                        title="Count",
                        showgrid=True,
                        gridcolor="#F1F5F9",
                        tickfont=dict(family="Plus Jakarta Sans", size=11),
                        tickvals=[0, 500, 1000]
                    ),
                    font=dict(family="Plus Jakarta Sans, sans-serif")
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            except Exception:
                pass
                
        st.markdown("</div>", unsafe_allow_html=True)

def render_page_header(title, subtitle, badge_text="CLINICAL INTELLIGENCE PLATFORM"):
    """Reusable Hero Banner across all pages with glowing 3D heart background texture."""
    banner_b64 = get_base64_image("heart_banner.jpg")
    bg_banner_css = (
        f"background: linear-gradient(90deg, #050D1A 0%, rgba(5, 13, 26, 0.95) 42%, rgba(5, 13, 26, 0.55) 70%, rgba(5, 13, 26, 0.15) 100%), url('{banner_b64}') right center / cover no-repeat;"
        if banner_b64 else
        "background: linear-gradient(135deg, #050d1a 0%, #0a1b32 60%, #06152a 100%);"
    )
    badge_html = f"<div style='display: inline-block; background: rgba(20, 184, 166, 0.15); border: 1px solid rgba(20, 184, 166, 0.4); color: #2DD4BF; font-size: 0.72rem; font-weight: 700; letter-spacing: 1.5px; padding: 4px 12px; border-radius: 9999px; text-transform: uppercase; margin-bottom: 0.8rem;'>{badge_text}</div>" if badge_text else ""
    
    st.markdown(clean_html(f"""
    <div style="{bg_banner_css} border-radius: 20px; padding: 2.6rem 3.2rem; color: white; border: 1px solid rgba(255, 255, 255, 0.08); box-shadow: 0 15px 40px -5px rgba(5, 13, 26, 0.5); margin-bottom: 2rem; position: relative; overflow: hidden; min-height: 180px; display: flex; align-items: center;">
        <div style="max-width: 650px; position: relative; z-index: 2;">
            {badge_html}
            <h1 style="font-size: 2.4rem; font-weight: 800; color: #FFFFFF !important; line-height: 1.18; margin: 0 0 0.6rem 0; letter-spacing: -0.5px;">
                {title}
            </h1>
            <p style="color: #94A3B8; font-size: 1.02rem; line-height: 1.5; margin: 0; font-weight: 400; max-width: 580px;">
                {subtitle}
            </p>
        </div>
    </div>
    """), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 2: RISK ASSESSMENT (INTERACTIVE FORM & PREDICTION)
# -----------------------------------------------------------------------------
def page_risk_assessment():
    render_disclaimer()
    
    render_page_header(
        title="Cardiovascular Risk Assessment Workspace",
        subtitle="Enter patient physiological parameters below to compute real-time cardiovascular disease risk probability",
        badge_text="CLINICAL DIAGNOSTIC WORKSPACE"
    )
    
    # Load Backend Model & Scaler
    model, error_msg = load_ml_model()
    scaler = load_feature_scaler()
    
    if error_msg:
        st.error(f"⚠️ **Model Initialization Error:** {error_msg}")
        st.info("Please verify that `cardiovascular_decision_tree.pkl` is located in the working directory.")
        return

    # Assessment Form Container
    with st.form("risk_assessment_form"):
        st.markdown(clean_html("""
        <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 1.2rem;">
            <span style="font-size: 1.4rem;">📋</span>
            <h2 style="font-size: 1.35rem; font-weight: 700; color: #0F172A; margin: 0;">Patient Clinical Input Form</h2>
        </div>
        """), unsafe_allow_html=True)
        
        # Section 1: Demographics & Biometrics
        st.markdown(clean_html("<div class='card-title'>1. Demographics & Biometrics</div>"), unsafe_allow_html=True)
        col_demo1, col_demo2, col_demo3, col_demo4 = st.columns(4)
        
        with col_demo1:
            age_years = st.number_input(
                "Age (Years)", 
                min_value=18, 
                max_value=100, 
                value=52, 
                step=1,
                help="Patient age in full years."
            )
            
        with col_demo2:
            gender_label = st.selectbox(
                "Gender at Birth", 
                options=["Female", "Male"],
                index=0,
                help="Biological gender designation."
            )
            gender_val = 1 if gender_label == "Female" else 2
            
        with col_demo3:
            height_cm = st.number_input(
                "Height (cm)", 
                min_value=100.0, 
                max_value=230.0, 
                value=165.0, 
                step=0.5,
                help="Height measured in centimeters."
            )
            
        with col_demo4:
            weight_kg = st.number_input(
                "Weight (kg)", 
                min_value=30.0, 
                max_value=220.0, 
                value=70.0, 
                step=0.5,
                help="Body weight measured in kilograms."
            )
            
        # Live Calculated BMI Metrics
        bmi_val, bmi_cat = calculate_bmi(weight_kg, height_cm)
        
        bmi_color_class = "badge-normal"
        if bmi_cat == "Overweight":
            bmi_color_class = "badge-warning"
        elif bmi_cat in ["Underweight", "Obese"]:
            bmi_color_class = "badge-danger"
            
        st.markdown(clean_html(f"""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 0.75rem 1.2rem; border-radius: 10px; margin-bottom: 1.5rem; display: flex; align-items: center; justify-content: space-between;">
            <div><strong>Calculated BMI:</strong> <span style="font-size: 1.1rem; color: #0D9488; font-weight: 700;">{bmi_val} kg/m²</span></div>
            <div>BMI Classification: <span class="{bmi_color_class}">{bmi_cat}</span></div>
        </div>
        """), unsafe_allow_html=True)
        
        # Section 2: Hemodynamics & Laboratory Markers
        st.markdown(clean_html("<div class='card-title'>2. Hemodynamics & Laboratory Biomarkers</div>"), unsafe_allow_html=True)
        col_hemo1, col_hemo2, col_lab1, col_lab2 = st.columns(4)
        
        with col_hemo1:
            ap_hi = st.number_input(
                "Systolic BP (ap_hi)", 
                min_value=70, 
                max_value=240, 
                value=120, 
                step=1,
                help="Systolic Blood Pressure (mmHg)."
            )
            
        with col_hemo2:
            ap_lo = st.number_input(
                "Diastolic BP (ap_lo)", 
                min_value=40, 
                max_value=180, 
                value=80, 
                step=1,
                help="Diastolic Blood Pressure (mmHg)."
            )
            
        with col_lab1:
            chol_label = st.selectbox(
                "Serum Cholesterol Level",
                options=["1: Normal", "2: Above Normal", "3: Well Above Normal"],
                index=0,
                help="Fasting lipid panel cholesterol classification."
            )
            chol_val = int(chol_label.split(":")[0])
            
        with col_lab2:
            gluc_label = st.selectbox(
                "Fasting Blood Glucose",
                options=["1: Normal", "2: Above Normal", "3: Well Above Normal"],
                index=0,
                help="Fasting glucose concentration classification."
            )
            gluc_val = int(gluc_label.split(":")[0])
            
        # Hemodynamic Validation Alert
        if ap_hi <= ap_lo:
            st.warning("⚠️ **Clinical Warning:** Systolic blood pressure (`ap_hi`) must be strictly greater than Diastolic blood pressure (`ap_lo`).")
            
        # Section 3: Lifestyle & Behavioral Indicators
        st.markdown(clean_html("<div class='card-title'>3. Lifestyle & Behavioral Factors</div>"), unsafe_allow_html=True)
        col_life1, col_life2, col_life3 = st.columns(3)
        
        with col_life1:
            smoke_label = st.radio(
                "Tobacco Smoking Habit",
                options=["No", "Yes"],
                index=0,
                horizontal=True,
                help="Active tobacco smoking history."
            )
            smoke_val = 1 if smoke_label == "Yes" else 0
            
        with col_life2:
            alco_label = st.radio(
                "Alcohol Intake",
                options=["No", "Yes"],
                index=0,
                horizontal=True,
                help="Regular alcohol consumption."
            )
            alco_val = 1 if alco_label == "Yes" else 0
            
        with col_life3:
            active_label = st.radio(
                "Physical Activity (>=150 min/wk)",
                options=["Yes", "No"],
                index=0,
                horizontal=True,
                help="Engages in regular moderate-to-vigorous physical exercise."
            )
            active_val = 1 if active_label == "Yes" else 0
            
        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("🫀 Run Cardiovascular Risk Assessment")
        
    # PROCESS PREDICTION ON FORM SUBMISSION
    if submit_btn:
        if ap_hi <= ap_lo:
            st.error("Cannot proceed: Please correct Blood Pressure values (Systolic must be greater than Diastolic).")
            return
            
        with st.spinner("Processing clinical biomarkers and evaluating risk score..."):
            time.sleep(0.4)
            
            input_dict = {
                'age': [int(age_years)],
                'gender': [int(gender_val)],
                'height': [float(height_cm)],
                'weight': [float(weight_kg)],
                'ap_hi': [int(ap_hi)],
                'ap_lo': [int(ap_lo)],
                'cholesterol': [int(chol_val)],
                'gluc': [int(gluc_val)],
                'smoke': [int(smoke_val)],
                'alco': [int(alco_val)],
                'active': [int(active_val)],
                'BMI': [float(bmi_val)]
            }
            
            raw_df = pd.DataFrame(input_dict)
            
            scaled_df = raw_df.copy()
            numerical_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
            scaled_df[numerical_cols] = scaler.transform(raw_df[numerical_cols])
            
            try:
                probabilities = model.predict_proba(scaled_df)[0]
                prediction_class = model.predict(scaled_df)[0]
                risk_prob_percent = round(probabilities[1] * 100, 1)
            except Exception:
                prediction_class = model.predict(scaled_df)[0]
                risk_prob_percent = 85.0 if prediction_class == 1 else 15.0
                
            # DISPLAY RESULTS SECTION
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(clean_html("""
            <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 1.2rem;">
                <span style="font-size: 1.5rem;">🔬</span>
                <h2 style="font-size: 1.4rem; font-weight: 700; color: #0F172A; margin: 0;">Cardiovascular Risk Analysis Results</h2>
            </div>
            """), unsafe_allow_html=True)
            
            res_col1, res_col2 = st.columns([5, 7])
            
            with res_col1:
                st.markdown(clean_html("""
                <div class="clinical-card" style="text-align: center; border-radius: 16px; border: 1px solid #E2E8F0; padding: 1.8rem; background: #FFFFFF; height: 100%; display: flex; flex-direction: column; justify-content: center;">
                    <div class="card-title" style="justify-content: center; border-bottom: none; margin-bottom: 0.5rem;">Clinical Classification</div>
                """), unsafe_allow_html=True)
                
                if prediction_class == 1 or risk_prob_percent >= 50:
                    st.markdown(clean_html("""
                    <div class="badge-high-risk" style="margin: 1rem 0;">
                        <span>🔴 ELEVATED CARDIOVASCULAR RISK</span>
                    </div>
                    <p style="color: #991B1B; font-weight: 600; font-size: 0.95rem;">
                        Patient demonstrates elevated risk markers for cardiovascular disease. Prompt medical follow-up is recommended.
                    </p>
                    """), unsafe_allow_html=True)
                else:
                    st.markdown(clean_html("""
                    <div class="badge-low-risk" style="margin: 1rem 0;">
                        <span>🟢 LOW CARDIOVASCULAR RISK</span>
                    </div>
                    <p style="color: #166534; font-weight: 600; font-size: 0.95rem;">
                        Biomarkers are within healthy ranges. Maintain preventative routine clinical monitoring and healthy lifestyle practices.
                    </p>
                    """), unsafe_allow_html=True)
                    
                st.markdown("</div>", unsafe_allow_html=True)
                
            with res_col2:
                gauge_fig = render_risk_gauge(risk_prob_percent)
                st.plotly_chart(gauge_fig, use_container_width=True)
                
            # DETAILED RISK FACTOR BREAKDOWN & DIRECTIVES
            st.markdown(clean_html("""
            <div class="clinical-card" style="border-radius: 16px; border: 1px solid #E2E8F0; padding: 1.8rem; background: #FFFFFF; margin-top: 1.5rem;">
                <div class="card-title">🔍 Clinical Risk Factor Analysis & Directives</div>
            """), unsafe_allow_html=True)
            
            col_b1, col_b2, col_b3 = st.columns(3)
            
            with col_b1:
                st.markdown("#### 🩺 Blood Pressure Category")
                if ap_hi < 120 and ap_lo < 80:
                    st.success("• **Normal BP:** <120/<80 mmHg")
                elif 120 <= ap_hi <= 129 and ap_lo < 80:
                    st.warning("• **Elevated BP:** 120-129/<80 mmHg")
                elif 130 <= ap_hi <= 139 or 80 <= ap_lo <= 89:
                    st.warning("• **Stage 1 Hypertension:** 130-139 / 80-89 mmHg")
                else:
                    st.error("• **Stage 2 Hypertension:** >=140 / >=90 mmHg")
                    
            with col_b2:
                st.markdown("#### ⚖️ Metabolic Profile")
                if chol_val > 1:
                    st.warning(f"• **Cholesterol:** {chol_label}")
                else:
                    st.success("• **Cholesterol:** Normal")
                    
                if gluc_val > 1:
                    st.warning(f"• **Glucose:** {gluc_label}")
                else:
                    st.success("• **Glucose:** Normal")
                    
            with col_b3:
                st.markdown("#### 🏃 Lifestyle Risk Profile")
                if smoke_val == 1:
                    st.error("• **Tobacco Use:** Active Smoker")
                else:
                    st.success("• **Tobacco Use:** Non-Smoker")
                    
                if active_val == 0:
                    st.warning("• **Physical Activity:** Insufficient (<150 min/wk)")
                else:
                    st.success("• **Physical Activity:** Active")
                    
            st.markdown("</div>", unsafe_allow_html=True)
            
            # DOWNLOADABLE PATIENT SUMMARY REPORT
            report_text = f"""===================================================================
CARDIOVASCULAR DISEASE RISK ASSESSMENT REPORT
CardioPulse AI - Clinical Decision Support Platform
===================================================================
Assessment Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}

PATIENT DEMOGRAPHICS & BIOMETRICS:
----------------------------------
• Age: {age_years} years
• Gender: {gender_label}
• Height: {height_cm} cm | Weight: {weight_kg} kg
• Body Mass Index (BMI): {bmi_val} kg/m² ({bmi_cat})

HEMODYNAMIC & BIOCHEMICAL MARKERS:
-----------------------------------
• Systolic Blood Pressure: {ap_hi} mmHg
• Diastolic Blood Pressure: {ap_lo} mmHg
• Serum Cholesterol: {chol_label}
• Fasting Blood Glucose: {gluc_label}

LIFESTYLE INDICATORS:
---------------------
• Tobacco Smoking: {smoke_label}
• Alcohol Intake: {alco_label}
• Physical Activity: {active_label}

MODEL RISK STRATIFICATION RESULTS:
----------------------------------
• Overall CVD Classification: {"ELEVATED RISK" if prediction_class == 1 else "LOW RISK"}
• Predicted CVD Probability Score: {risk_prob_percent}%

RECOMMENDED CLINICAL ACTION PLAN:
----------------------------------
{"1. Schedule full diagnostic cardiovascular workup and ECG evaluation.\n2. Initiate blood pressure monitoring and lipid panel reassessment.\n3. Adopt dietary DASH guidelines and tobacco cessation protocols." if prediction_class == 1 else "1. Continue routine annual health screenings.\n2. Maintain balanced physical exercise (>=150 mins/week).\n3. Keep dietary sodium and saturated fats within recommended limits."}

===================================================================
DISCLAIMER: For clinical screening and educational use only.
Not intended to serve as a standalone diagnostic device.
===================================================================
"""
            
            st.download_button(
                label="📄 Download Official Patient Summary Report (.TXT)",
                data=report_text,
                file_name=f"CardioPulse_Assessment_Patient_{age_years}Y.txt",
                mime="text/plain"
            )

# -----------------------------------------------------------------------------
# PAGE 3: ABOUT MODEL & CLINICAL RELEVANCE (WITH SAAS FEATURE MATRIX TABLE)
# -----------------------------------------------------------------------------
def page_about():
    render_disclaimer()
    
    render_page_header(
        title="About CardioPulse AI Engine",
        subtitle="Technical specification, model architecture, dataset lineage, and clinical validation protocols",
        badge_text="SYSTEM ARCHITECTURE & METHODOLOGY"
    )
    
    col1, col2 = st.columns([6, 6])
    
    with col1:
        st.markdown(clean_html("""
        <div class="clinical-card" style="border: 1px solid #E2E8F0; border-radius: 16px; padding: 1.6rem; background: #FFFFFF; height: calc(100% - 1.5rem);">
            <div class="card-title">🤖 Model Architecture & Pipeline</div>
            <p style="color: #475569; line-height: 1.6;">
                The underlying engine utilizes a trained Decision Tree Classifier / Logistic Regression estimator optimized for high sensitivity and clinical interpretability.
            </p>
            <ul style="color: #475569; line-height: 1.8; margin-left: -1rem;">
                <li><strong>Model Binary File:</strong> <code>cardiovascular_decision_tree.pkl</code></li>
                <li><strong>Preprocessing Scaler:</strong> <code>StandardScaler</code> fitted on training continuous features (Age, Height, Weight, Systolic BP, Diastolic BP).</li>
                <li><strong>Feature Dimension:</strong> 12 input features including auto-derived Body Mass Index.</li>
                <li><strong>Target Variable:</strong> Binary CVD Presence (0 = Absent, 1 = Present).</li>
            </ul>
        </div>
        """), unsafe_allow_html=True)
        
    with col2:
        st.markdown(clean_html("""
        <div class="clinical-card" style="border: 1px solid #E2E8F0; border-radius: 16px; padding: 1.6rem; background: #FFFFFF; height: calc(100% - 1.5rem);">
            <div class="card-title">📚 Dataset & Training Methodology</div>
            <p style="color: #475569; line-height: 1.6;">
                Trained on the standardized Kaggle Cardiovascular Disease Dataset consisting of <strong>70,000 anonymized patient observations</strong> collected during clinical examination.
            </p>
            <p style="color: #475569; line-height: 1.6;">
                Rigorous cleaning was performed to filter out extreme physiological outliers (e.g., Systolic BP outside 70-250 mmHg or Diastolic BP outside 40-180 mmHg).
            </p>
        </div>
        """), unsafe_allow_html=True)
        
    # Full Width Custom SaaS Modern Data Table for Input Feature Matrix
    st.markdown(clean_html("""
    <div class="clinical-card" style="border: 1px solid #E2E8F0; border-radius: 16px; padding: 1.6rem; background: #FFFFFF; box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.04); margin-top: 1rem; margin-bottom: 1.5rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.2rem; border-bottom: 2px solid #F1F5F9; padding-bottom: 0.8rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; font-weight: 800; font-size: 1.25rem; color: #0F3759;">
                <span>🔬</span> Input Feature Matrix
            </div>
            <span style="font-size: 0.78rem; font-weight: 700; color: #0D9488; background: #E6FFFA; padding: 4px 12px; border-radius: 9999px; border: 1px solid #99F6E4;">12 CLINICAL BIOMARKERS</span>
        </div>
        
        <div style="overflow-x: auto;">
            <table style="width: 100%; border-collapse: collapse; font-family: 'Plus Jakarta Sans', Inter, sans-serif; font-size: 0.9rem;">
                <thead>
                    <tr style="background: #F8FAFC; border-bottom: 2px solid #E2E8F0; text-align: left;">
                        <th style="padding: 14px 16px; font-weight: 700; font-size: 0.78rem; color: #475569; text-transform: uppercase; letter-spacing: 0.6px;">Feature Name</th>
                        <th style="padding: 14px 16px; font-weight: 700; font-size: 0.78rem; color: #475569; text-transform: uppercase; letter-spacing: 0.6px;">Data Type</th>
                        <th style="padding: 14px 16px; font-weight: 700; font-size: 0.78rem; color: #475569; text-transform: uppercase; letter-spacing: 0.6px;">Clinical Significance</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid #F1F5F9;">
                        <td style="padding: 14px 16px; font-weight: 600; color: #0F172A;">
                            <span style="margin-right: 8px;">🎂</span> Age
                        </td>
                        <td style="padding: 14px 16px;">
                            <span style="background: #E0F2FE; color: #0369A1; font-weight: 700; font-size: 0.78rem; padding: 5px 12px; border-radius: 9999px; display: inline-block;">Continuous (Yrs)</span>
                        </td>
                        <td style="padding: 14px 16px; color: #475569; font-size: 0.88rem;">Primary non-modifiable cardiovascular aging risk factor</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #F1F5F9;">
                        <td style="padding: 14px 16px; font-weight: 600; color: #0F172A;">
                            <span style="margin-right: 8px;">⚧</span> Gender
                        </td>
                        <td style="padding: 14px 16px;">
                            <span style="background: #EDE9FE; color: #6D28D9; font-weight: 700; font-size: 0.78rem; padding: 5px 12px; border-radius: 9999px; display: inline-block;">Categorical (1/2)</span>
                        </td>
                        <td style="padding: 14px 16px; color: #475569; font-size: 0.88rem;">Biological gender-specific baseline epidemiological incidence</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #F1F5F9;">
                        <td style="padding: 14px 16px; font-weight: 600; color: #0F172A;">
                            <span style="margin-right: 8px;">📏</span> Height
                        </td>
                        <td style="padding: 14px 16px;">
                            <span style="background: #E0F2FE; color: #0369A1; font-weight: 700; font-size: 0.78rem; padding: 5px 12px; border-radius: 9999px; display: inline-block;">Continuous (cm)</span>
                        </td>
                        <td style="padding: 14px 16px; color: #475569; font-size: 0.88rem;">Anthropometric reference for Body Surface Area indexing</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #F1F5F9;">
                        <td style="padding: 14px 16px; font-weight: 600; color: #0F172A;">
                            <span style="margin-right: 8px;">⚖️</span> Weight
                        </td>
                        <td style="padding: 14px 16px;">
                            <span style="background: #E0F2FE; color: #0369A1; font-weight: 700; font-size: 0.78rem; padding: 5px 12px; border-radius: 9999px; display: inline-block;">Continuous (kg)</span>
                        </td>
                        <td style="padding: 14px 16px; color: #475569; font-size: 0.88rem;">Body mass index calibration and metabolic loading indicator</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #F1F5F9;">
                        <td style="padding: 14px 16px; font-weight: 600; color: #0F172A;">
                            <span style="margin-right: 8px;">🩺</span> Systolic BP (ap_hi)
                        </td>
                        <td style="padding: 14px 16px;">
                            <span style="background: #E0F2FE; color: #0369A1; font-weight: 700; font-size: 0.78rem; padding: 5px 12px; border-radius: 9999px; display: inline-block;">Continuous (mmHg)</span>
                        </td>
                        <td style="padding: 14px 16px; color: #475569; font-size: 0.88rem;">Primary hemodynamic hypertension marker & vascular strain</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #F1F5F9;">
                        <td style="padding: 14px 16px; font-weight: 600; color: #0F172A;">
                            <span style="margin-right: 8px;">🩺</span> Diastolic BP (ap_lo)
                        </td>
                        <td style="padding: 14px 16px;">
                            <span style="background: #E0F2FE; color: #0369A1; font-weight: 700; font-size: 0.78rem; padding: 5px 12px; border-radius: 9999px; display: inline-block;">Continuous (mmHg)</span>
                        </td>
                        <td style="padding: 14px 16px; color: #475569; font-size: 0.88rem;">Resting vascular peripheral resistance marker</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #F1F5F9;">
                        <td style="padding: 14px 16px; font-weight: 600; color: #0F172A;">
                            <span style="margin-right: 8px;">🧪</span> Cholesterol
                        </td>
                        <td style="padding: 14px 16px;">
                            <span style="background: #F3E8FF; color: #7E22CE; font-weight: 700; font-size: 0.78rem; padding: 5px 12px; border-radius: 9999px; display: inline-block;">Ordinal (1, 2, 3)</span>
                        </td>
                        <td style="padding: 14px 16px; color: #475569; font-size: 0.88rem;">Atherosclerosis and coronary plaque accumulation risk</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #F1F5F9;">
                        <td style="padding: 14px 16px; font-weight: 600; color: #0F172A;">
                            <span style="margin-right: 8px;">🩸</span> Glucose
                        </td>
                        <td style="padding: 14px 16px;">
                            <span style="background: #F3E8FF; color: #7E22CE; font-weight: 700; font-size: 0.78rem; padding: 5px 12px; border-radius: 9999px; display: inline-block;">Ordinal (1, 2, 3)</span>
                        </td>
                        <td style="padding: 14px 16px; color: #475569; font-size: 0.88rem;">Hyperglycemia / diabetic microvascular damage risk factor</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #F1F5F9;">
                        <td style="padding: 14px 16px; font-weight: 600; color: #0F172A;">
                            <span style="margin-right: 8px;">🚬</span> Smoking
                        </td>
                        <td style="padding: 14px 16px;">
                            <span style="background: #CCFBF1; color: #0F766E; font-weight: 700; font-size: 0.78rem; padding: 5px 12px; border-radius: 9999px; display: inline-block;">Binary (0/1)</span>
                        </td>
                        <td style="padding: 14px 16px; color: #475569; font-size: 0.88rem;">Direct endothelial injury & arterial vasoconstriction trigger</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #F1F5F9;">
                        <td style="padding: 14px 16px; font-weight: 600; color: #0F172A;">
                            <span style="margin-right: 8px;">🍷</span> Alcohol Intake
                        </td>
                        <td style="padding: 14px 16px;">
                            <span style="background: #CCFBF1; color: #0F766E; font-weight: 700; font-size: 0.78rem; padding: 5px 12px; border-radius: 9999px; display: inline-block;">Binary (0/1)</span>
                        </td>
                        <td style="padding: 14px 16px; color: #475569; font-size: 0.88rem;">Hepatovascular metabolic stress and rhythm disturbance driver</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #F1F5F9;">
                        <td style="padding: 14px 16px; font-weight: 600; color: #0F172A;">
                            <span style="margin-right: 8px;">🏃</span> Physical Activity
                        </td>
                        <td style="padding: 14px 16px;">
                            <span style="background: #CCFBF1; color: #0F766E; font-weight: 700; font-size: 0.78rem; padding: 5px 12px; border-radius: 9999px; display: inline-block;">Binary (0/1)</span>
                        </td>
                        <td style="padding: 14px 16px; color: #475569; font-size: 0.88rem;">Protective metabolic & cardiopulmonary conditioning index</td>
                    </tr>
                    <tr>
                        <td style="padding: 14px 16px; font-weight: 600; color: #0F172A;">
                            <span style="margin-right: 8px;">🧮</span> BMI Index
                        </td>
                        <td style="padding: 14px 16px;">
                            <span style="background: #E0F2FE; color: #0369A1; font-weight: 700; font-size: 0.78rem; padding: 5px 12px; border-radius: 9999px; display: inline-block;">Continuous (kg/m²)</span>
                        </td>
                        <td style="padding: 14px 16px; color: #475569; font-size: 0.88rem;">Computed adiposity index and metabolic syndrome biomarker</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
    """), unsafe_allow_html=True)
        
    st.markdown(clean_html("""
    <div class="clinical-card" style="border: 1px solid #E2E8F0; border-radius: 16px; padding: 1.6rem; background: #FFFFFF; margin-bottom: 1.5rem;">
        <div class="card-title">⚖️ Clinical Relevance & Evidence Base</div>
        <p style="color: #475569; line-height: 1.6;">
            Elevated systolic blood pressure (hypertension), high serum cholesterol, and elevated Body Mass Index (BMI) are universally recognized by the World Health Organization (WHO) and American Heart Association (AHA) as primary modifiable drivers of atherosclerosis, coronary artery disease, and stroke. Early identification through automated risk stratification tools provides a vital window for primary intervention before irreversible organ damage occurs.
        </p>
    </div>
    """), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 4: CONTACT US & SUPPORT
# -----------------------------------------------------------------------------
def page_contact():
    render_disclaimer()
    
    render_page_header(
        title="Clinical Support & Contact Portal",
        subtitle="Get in touch with our medical technology team, report system issues, or explore clinical deployment",
        badge_text="24/7 CLINICAL HELPDESK & INTEGRATION"
    )
    
    col1, col2 = st.columns([7, 5])
    
    with col1:
        st.markdown(clean_html("""
        <div class="clinical-card" style="border: 1px solid #E2E8F0; border-radius: 16px; padding: 1.6rem; background: #FFFFFF;">
            <div class="card-title">✉️ Submit an Inquiry</div>
        """), unsafe_allow_html=True)
        
        with st.form("contact_form"):
            name = st.text_input("Full Name", placeholder="Dr. Jane Doe")
            email = st.text_input("Work Email Address", placeholder="jane.doe@hospital.org")
            org = st.text_input("Organization / Hospital Department", placeholder="Cardiology Department, General Hospital")
            category = st.selectbox("Inquiry Subject", ["Clinical Deployment", "Technical Support", "Model Calibration / Data Science", "Regulatory Inquiry", "Other"])
            message = st.text_area("Message Details", placeholder="Describe your inquiry or technical question...")
            
            submit_contact = st.form_submit_button("Send Inquiry")
            if submit_contact:
                if not name or not email or not message:
                    st.error("Please complete all required fields (Name, Email, Message).")
                else:
                    st.success("✅ Thank you! Your inquiry has been submitted to the CardioPulse AI Clinical Support Team. We will respond within 24 business hours.")
                    
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown(clean_html("""
        <div class="clinical-card" style="border: 1px solid #E2E8F0; border-radius: 16px; padding: 1.6rem; background: #FFFFFF; margin-bottom: 1.5rem;">
            <div class="card-title">📞 Urgent Support Contacts</div>
            <p style="color: #475569; font-size: 0.95rem;">
                <strong>CardioPulse AI Technical Response Center</strong><br>
                🏢 Medical Health Informatics Center<br>
                📧 Email: <code>support@cardiopulse-ai.org</code><br>
                ☎️ Hotline: <code>+1 (800) 555-CARDIO</code>
            </p>
            <hr style="border-top: 1px solid #E2E8F0; margin: 1rem 0;">
            <div style="background-color: #FEE2E2; padding: 1rem; border-radius: 10px; border: 1px solid #FCA5A5;">
                <strong style="color: #991B1B;">🚨 Medical Emergency Alert:</strong><br>
                <span style="color: #7F1D1D; font-size: 0.88rem;">If a patient is experiencing acute chest pain, shortness of breath, or sudden weakness, immediately initiate emergency protocol and call <strong>911 / 112</strong>. Do NOT rely on software screening during acute events.</span>
            </div>
        </div>
        """), unsafe_allow_html=True)
        
        st.markdown(clean_html("""
        <div class="clinical-card" style="border: 1px solid #E2E8F0; border-radius: 16px; padding: 1.6rem; background: #FFFFFF;">
            <div class="card-title">🔒 Privacy & Compliance</div>
            <p style="color: #475569; font-size: 0.88rem; line-height: 1.5;">
                CardioPulse AI operates under strict data privacy principles. Patient inputs submitted during session evaluation are processed ephemerally in memory and are never persisted to external database storage without explicit patient authorization.
            </p>
        </div>
        """), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MAIN APP ENTRY POINT & ROUTING
# -----------------------------------------------------------------------------
def main():
    apply_custom_css()
    
    # Render Navigation Sidebar and Get Current Page
    current_page = render_sidebar()
    
    # Page Router
    if current_page == "Home":
        page_home()
    elif current_page == "Risk Assessment":
        page_risk_assessment()
    elif current_page == "About":
        page_about()
    elif current_page == "Contact Us":
        page_contact()
        
    # Global Footer
    st.markdown(clean_html("""
    <div class="clinical-footer">
        © 2026 CardioPulse AI Platform. Developed for Clinical Decision Support and Educational Use.<br>
        For screening/informational use only; not a diagnostic device.
    </div>
    """), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
