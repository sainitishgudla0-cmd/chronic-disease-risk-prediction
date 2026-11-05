"""Privacy, responsible use and attribution."""

import streamlit as st

from chronic_risk.pipelines import PIPELINES
from chronic_risk.ui import theme

st.title("Privacy & about")
theme.notice(theme.DISCLAIMER)

st.subheader("What happens to your answers")
st.markdown(
    """
- The app asks for no name, contact details, ID numbers or account.
- Answers are used to calculate one estimate. The app does not write them to a file or database,
  does not log them, and keeps no prediction history.
- While your browser tab is open, your answers stay in the app server's temporary session memory
  so the form keeps working. They are gone when the session ends. **Clear answers** removes them
  sooner.
- Usage statistics from the app framework are turned off in this project's configuration. If the
  app is hosted by a third party, that host's own terms and infrastructure logs also apply.
"""
)

st.subheader("Using this responsibly")
st.markdown(
    """
- These are educational demonstrations of machine learning, and they have not been clinically
  validated or approved for medical use.
- Do not use an estimate to start, stop or change any treatment, or to delay seeking care.
- If you think you may be having a medical emergency, contact your local emergency services.
"""
)

st.subheader("Data sources and credits")
st.markdown(
    "Models were trained on public, de-identified research datasets. They are not stored in this "
    "project's repository, and nobody's records can be viewed through this app."
)
for pipeline in PIPELINES.values():
    with st.container(border=True):
        st.markdown(f"**{pipeline.title}: [{pipeline.dataset_name}]({pipeline.dataset_url})**")
        st.markdown(f"License: [{pipeline.license_name}]({pipeline.license_url})")
        st.caption(pipeline.citation)
