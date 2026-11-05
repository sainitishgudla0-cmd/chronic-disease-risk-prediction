"""Run the real Streamlit scripts headlessly with Streamlit's own test harness."""

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from chronic_risk.inference import ArtifactError
from chronic_risk.paths import PROJECT_ROOT
from chronic_risk.pipelines import PIPELINES
from chronic_risk.ui import assessment

TIMEOUT = 60
APP = str(PROJECT_ROOT / "app.py")


def _open(page=None):
    app = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    if page:
        app.switch_page(f"app_pages/{page}.py").run()
    return app


def _button(app, label):
    return next(b for b in app.button if b.label == label)


def _markdown(app):
    return "\n".join(m.value for m in app.markdown)


def test_home_page_renders_without_error():
    app = _open()
    assert not app.exception
    assert app.title[0].value == "Chronic Disease Risk Explorer"
    assert any("Educational only" in c.value for c in app.caption)


def test_about_page_covers_privacy_and_every_citation():
    app = _open("about")
    assert not app.exception
    captions = " ".join(c.value for c in app.caption)
    for pipeline in PIPELINES.values():
        assert pipeline.citation in captions


@pytest.mark.parametrize("key", PIPELINES)
def test_page_starts_empty_with_a_helpful_prompt(key):
    app = _open(key)
    assert not app.exception
    assert any("Get my estimate" in i.value for i in app.info)
    assert 'class="cr-hero"' not in _markdown(app)
    assert all(s.value is None for s in app.selectbox)
    assert all(n.value is None for n in app.number_input)


@pytest.mark.parametrize("key", PIPELINES)
def test_typical_values_produce_an_estimate_with_caveats(key):
    app = _open(key)
    _button(app, "Fill with typical values").click().run()
    assert not app.exception
    _button(app, "Get my estimate").click().run()
    assert not app.exception and not app.error
    text = _markdown(app)
    assert 'class="cr-hero"' in text and "Model estimate for these answers" in text
    assert "qualified healthcare professional" in text
    assert PIPELINES[key].urgent_note in text


@pytest.mark.parametrize("key", PIPELINES)
def test_empty_submission_shows_validation_messages_and_no_estimate(key):
    app = _open(key)
    _button(app, "Get my estimate").click().run()
    assert not app.exception
    assert app.error and "Please check" in app.error[0].value
    assert 'class="cr-hero"' not in _markdown(app)
    assert "this is required" in _markdown(app)


def test_clear_answers_empties_the_form():
    app = _open("heart")
    _button(app, "Fill with typical values").click().run()
    assert any(n.value is not None for n in app.number_input)
    _button(app, "Clear answers").click().run()
    assert all(n.value is None for n in app.number_input)
    assert all(s.value is None for s in app.selectbox)


def test_blank_optional_values_trigger_a_visible_warning():
    app = _open("heart")
    _button(app, "Fill with typical values").click().run()
    app.selectbox(key="heart__thal").set_value(None)
    _button(app, "Get my estimate").click().run()
    assert not app.exception
    assert any("left blank" in w.value and "Thallium" in w.value for w in app.warning)


def test_missing_model_shows_guidance_instead_of_crashing(monkeypatch):
    def unavailable(_):
        raise ArtifactError("The model file for Heart disease is missing.")

    monkeypatch.setattr(assessment, "load_artifact", unavailable)
    st.cache_resource.clear()
    try:
        app = _open("heart")
        assert not app.exception
        assert any("unavailable" in e.value for e in app.error)
        assert not app.selectbox
    finally:
        monkeypatch.undo()
        st.cache_resource.clear()


def test_prediction_failure_shows_a_generic_message(monkeypatch):
    secret_value = "sensitive-input-value"

    def explode(*_):
        raise RuntimeError(secret_value)

    app = _open("heart")
    _button(app, "Fill with typical values").click().run()
    monkeypatch.setattr(assessment, "predict", explode)
    _button(app, "Get my estimate").click().run()
    assert not app.exception
    assert any("Nothing was saved" in e.value for e in app.error)
    assert secret_value not in str(app)


@pytest.mark.parametrize("key", PIPELINES)
def test_models_page_renders_each_condition(key):
    app = _open("models")
    app.radio(key="models_condition").set_value(key).run()
    assert not app.exception
    assert any(s.value == "Limitations" for s in app.subheader)
    assert any(s.value == "Calibration" for s in app.subheader)


def test_models_page_flags_the_suspicious_kidney_score():
    app = _open("models")
    app.radio(key="models_condition").set_value("kidney").run()
    text = _markdown(app)
    assert "raise suspicion" in text and "Missingness alone" in text


def test_models_page_shows_outside_sites_for_heart():
    app = _open("models")
    app.radio(key="models_condition").set_value("heart").run()
    assert any(s.value == "Testing at other hospitals" for s in app.subheader)
