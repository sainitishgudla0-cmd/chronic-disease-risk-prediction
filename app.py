"""Streamlit entry point: page setup, navigation and the disclaimer on every page."""

import streamlit as st

from chronic_risk.ui import theme

st.set_page_config(page_title="Chronic Disease Risk Explorer", page_icon="🩺", layout="wide")
theme.apply_theme()

navigation = st.navigation(
    {
        "Start": [st.Page("app_pages/home.py", title="Home", icon="🏠", default=True)],
        "Estimates": [
            st.Page("app_pages/diabetes.py", title="Diabetes", icon="🩸"),
            st.Page("app_pages/heart.py", title="Heart disease", icon="🫀"),
            st.Page("app_pages/kidney.py", title="Kidney disease", icon="🧪"),
        ],
        "Learn": [
            st.Page("app_pages/models.py", title="Models & data", icon="📊"),
            st.Page("app_pages/about.py", title="Privacy & about", icon="🔒"),
        ],
    }
)
navigation.run()

st.divider()
st.caption(theme.DISCLAIMER)
