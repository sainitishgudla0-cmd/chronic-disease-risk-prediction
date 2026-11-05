"""Landing page."""

import streamlit as st

from chronic_risk.pipelines import PIPELINES
from chronic_risk.ui import theme

st.title("Chronic Disease Risk Explorer")
theme.lede(
    "See how machine learning estimates relate to public health datasets for diabetes, heart "
    "disease and chronic kidney disease, and learn how much those estimates can and can't tell "
    "you."
)
theme.notice(theme.DISCLAIMER)

st.subheader("Choose a condition")
SUMMARIES = {
    "diabetes": "Everyday health and lifestyle questions from a large US telephone survey.",
    "heart": "Clinical test results from people referred for coronary angiography.",
    "kidney": "Health history, blood and urine results from a small single-hospital dataset.",
}
for column, (key, pipeline) in zip(st.columns(3), PIPELINES.items(), strict=True):
    with column, st.container(border=True):
        st.markdown(
            f'<div class="cr-card-title">{pipeline.title}</div>'
            f'<div class="cr-muted">{SUMMARIES[key]}</div>',
            unsafe_allow_html=True,
        )
        st.page_link(
            f"app_pages/{key}.py",
            label=f"Start {pipeline.title.lower()} estimate",
            icon=":material/arrow_forward:",
        )

st.subheader("How to read an estimate")
EXPLAINERS = (
    (
        "It's a comparison",
        "The number reflects how often people with similar answers had the outcome in that "
        "dataset. It says nothing about what will happen to you.",
    ),
    (
        "The data has limits",
        "Each dataset was collected from specific people, at a specific time. Each page says "
        "who was included and who was not.",
    ),
    (
        "There are no risk levels",
        "You won't see categories of risk here, because these models don't support clinical "
        "categories. Only a clinician can interpret your health.",
    ),
)
for column, (title, body) in zip(st.columns(3), EXPLAINERS, strict=True):
    with column:
        st.markdown(f"**{title}**")
        st.markdown(f'<div class="cr-muted">{body}</div>', unsafe_allow_html=True)

st.subheader("Your privacy")
st.markdown(
    "No account or identifying details are needed. Your answers are used only to calculate the "
    "estimate on the page, and are not saved to a file or database."
)
st.page_link("app_pages/about.py", label="Read how privacy works in this app")
