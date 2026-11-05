"""Shared styling and small HTML components. Styles work in both light and dark themes."""

from __future__ import annotations

from html import escape

import streamlit as st

DISCLAIMER = (
    "Educational only. This is not a diagnosis or treatment recommendation. "
    "Please talk to a qualified healthcare professional about your health."
)

_CSS = """
<style>
.block-container { max-width: 1080px; padding-top: 2.5rem; }
h1 { font-weight: 700; letter-spacing: -0.02em; }
h2, h3 { font-weight: 600; letter-spacing: -0.01em; }
.cr-lede { font-size: 1.15rem; line-height: 1.55; opacity: .85; max-width: 46rem;
  margin-bottom: 1rem; }
.cr-notice { border-left: 4px solid #0f766e; background: rgba(15,118,110,.09);
  border-radius: 0 8px 8px 0; padding: .75rem 1rem; margin: .5rem 0 1rem; line-height: 1.5; }
.cr-urgent { border-left-color: #c2410c; background: rgba(194,65,12,.09); }
.cr-hero { border: 1px solid rgba(128,128,128,.3); border-radius: 12px; padding: 1.25rem 1.5rem;
  margin-bottom: 1rem; }
.cr-hero-label { font-size: .9rem; font-weight: 600; opacity: .75; }
.cr-hero-value { font-size: 3rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1.15; }
.cr-hero-text { line-height: 1.5; max-width: 44rem; }
.cr-meter { position: relative; height: 12px; border-radius: 6px; margin: 1.1rem 0 2.4rem;
  background: rgba(128,128,128,.22); }
.cr-meter-fill { position: absolute; inset: 0 auto 0 0; border-radius: 6px; background: #0f766e; }
.cr-meter-tick { position: absolute; top: -5px; width: 3px; height: 22px; border-radius: 2px;
  background: currentColor; transform: translateX(-50%); }
.cr-meter-label { position: absolute; top: 20px; font-size: .8rem; opacity: .8;
  white-space: nowrap; transform: translateX(-50%); }
.cr-meter-label.is-left { transform: none; }
.cr-meter-label.is-right { transform: translateX(-100%); }
.cr-card-title { font-size: 1.15rem; font-weight: 600; margin-bottom: .25rem; }
.cr-muted { opacity: .75; font-size: .92rem; line-height: 1.5; }
@media (max-width: 640px) {
  .block-container { padding: 1.5rem 1rem 3rem; }
  .cr-hero-value { font-size: 2.4rem; }
  .cr-hero { padding: 1rem; }
}
</style>
"""


def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def lede(text: str) -> None:
    st.markdown(f'<p class="cr-lede">{escape(text)}</p>', unsafe_allow_html=True)


def notice(text: str, urgent: bool = False) -> None:
    role = "alert" if urgent else "note"
    extra = " cr-urgent" if urgent else ""
    st.markdown(
        f'<div class="cr-notice{extra}" role="{role}">{escape(text)}</div>',
        unsafe_allow_html=True,
    )


def format_percent(probability: float) -> str:
    """Whole percentages only, because more digits would imply precision the models lack."""
    percent = probability * 100
    if percent < 1:
        return "Under 1%"
    if percent > 99:
        return "Over 99%"
    return f"About {percent:.0f}%"


def hero_result(probability: float, base_rate: float, outcome_phrase: str) -> None:
    """Show the estimate as a number with a meter. Deliberately no risk bands or level colors."""
    value = format_percent(probability)
    estimate = min(max(probability * 100, 0.0), 100.0)
    base = min(max(base_rate * 100, 0.0), 100.0)
    placement = "is-left" if base < 12 else "is-right" if base > 88 else ""
    text = (
        f"The model estimates that {value.lower()} of records resembling these answers "
        f"{escape(outcome_phrase)}."
    )
    st.markdown(
        f"""<div class="cr-hero">
<div class="cr-hero-label">Model estimate for these answers</div>
<div class="cr-hero-value">{escape(value)}</div>
<div class="cr-hero-text">{text}</div>
<div class="cr-meter" role="img"
aria-label="Estimate {estimate:.0f} percent; dataset rate {base:.0f} percent">
<div class="cr-meter-fill" style="width:{estimate:.1f}%"></div>
<div class="cr-meter-tick" style="left:{base:.1f}%"></div>
<div class="cr-meter-label {placement}" style="left:{base:.1f}%">
Rate in this dataset: {base:.0f}%</div>
</div>
<div class="cr-muted">The dataset rate reflects who was included in the data. It is not a typical
rate for the general population, and neither number is a clinical risk category.</div>
</div>""",
        unsafe_allow_html=True,
    )
