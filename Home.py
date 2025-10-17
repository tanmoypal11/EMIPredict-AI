import streamlit as st
from PIL import Image
from datetime import datetime
from streamlit_lottie import st_lottie
import requests

# -----------------------------------------------
# Page Configuration
# -----------------------------------------------
st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------
# Helper function to load Lottie animations
# -----------------------------------------------
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie animation for FinTech feel
lottie_fintech = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")

# -----------------------------------------------
# Header Section
# -----------------------------------------------
st.markdown("<h1 style='text-align: center; color: #1F77B4;'>💰 EMIPredict AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Intelligent Financial Risk Assessment Platform</h4>", unsafe_allow_html=True)

# Display animation or banner
st_lottie(lottie_fintech, height=300, key="fintech_animation")

st.markdown("---")

# -----------------------------------------------
# Project Overview Section
# -----------------------------------------------
st.header("🔍 Project Overview")
st.write("""
**EMIPredict AI** is an intelligent **Financial Risk Assessment Platform** that leverages Machine Learning 
to predict:
- **EMI Eligibility (Classification Problem)** — determines whether a customer is *Eligible*, *High Risk*, or *Not Eligible*.
- **Maximum EMI Amount (Regression Problem)** — estimates the maximum safe EMI a person can afford monthly.

This platform enables banks, fintech companies, and loan officers to make faster, more data-driven decisions while reducing risk.
""")

# -----------------------------------------------
# Key Highlights
# -----------------------------------------------
st.markdown("### 🚀 Key Highlights")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Dataset Size", value="400,000+", delta="Financial Profiles")
with col2:
    st.metric(label="Input Features", value="22", delta="Demographic + Financial Variables")
with col3:
    st.metric(label="ML Models", value="6+", delta="Classification + Regression")


# -----------------------------------------------
# Business Impact Section
# -----------------------------------------------
st.markdown("---")
st.header("🏦 Business Impact")
st.write("""
- 🔹 **Financial Institutions** — Automate loan approval and reduce manual underwriting time by **80%**.
- 🔹 **FinTech Platforms** — Integrate instant EMI eligibility checks for digital lending.
- 🔹 **Credit Agencies** — Enable standardized and transparent credit risk assessment.
""")

# -----------------------------------------------
# Footer
# -----------------------------------------------
st.markdown("---")
st.markdown(
    f"<p style='text-align: center; color: gray;'>© {datetime.now().year} EMIPredict AI | Developed by Tanmoy Pal</p>",
    unsafe_allow_html=True
)
