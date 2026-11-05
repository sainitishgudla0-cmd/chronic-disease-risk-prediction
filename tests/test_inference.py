import json
import shutil

import pytest

from chronic_risk.inference import (
    ArtifactError,
    InputProblem,
    load_artifact,
    load_report,
    predict,
    validate_inputs,
)
from chronic_risk.paths import MANIFEST_PATH, MODELS_DIR
from chronic_risk.pipelines import PIPELINES


@pytest.fixture(params=list(PIPELINES))
def artifact(request):
    return load_artifact(request.param)


def typical(artifact):
    return dict(artifact.report["reference_values"])


def test_estimate_is_a_probability(artifact):
    assert 0.0 <= predict(artifact, typical(artifact)).probability <= 1.0


def test_answers_equal_to_reference_do_not_contribute(artifact):
    result = predict(artifact, typical(artifact))
    assert all(c.points == 0 for c in result.contributions)
    assert result.imputed == () and result.unusual == ()


def test_contribution_is_the_change_from_the_reference_value(artifact):
    inputs = typical(artifact)
    feature = next(f for f in artifact.pipeline.features if f.choices)
    inputs[feature.name] = next(v for v, _ in feature.choices if v != inputs[feature.name])
    changed = predict(artifact, inputs)
    baseline = predict(artifact, typical(artifact))
    item = next(c for c in changed.contributions if c.feature == feature.name)
    expected = (changed.probability - baseline.probability) * 100
    assert item.points == pytest.approx(expected, abs=0.01)


def test_invalid_choice_is_rejected(artifact):
    inputs = typical(artifact)
    feature = next(f for f in artifact.pipeline.features if f.choices)
    inputs[feature.name] = "not-an-option"
    with pytest.raises(InputProblem, match="listed options"):
        predict(artifact, inputs)


def test_missing_required_value_is_rejected(artifact):
    inputs = typical(artifact)
    feature = next(f for f in artifact.pipeline.features if f.required)
    inputs[feature.name] = None
    with pytest.raises(InputProblem, match="required"):
        predict(artifact, inputs)


@pytest.mark.parametrize("value", [-1000.0, 1e9, float("inf")])
def test_impossible_numbers_are_rejected(artifact, value):
    inputs = typical(artifact)
    feature = next(f for f in artifact.pipeline.features if not f.choices)
    inputs[feature.name] = value
    with pytest.raises(InputProblem):
        predict(artifact, inputs)


@pytest.mark.parametrize("key", ["heart", "kidney"])
def test_optional_inputs_can_be_blank_and_are_reported(key):
    artifact = load_artifact(key)
    inputs = typical(artifact)
    blanked = [f for f in artifact.pipeline.features if not f.required][:3]
    for feature in blanked:
        inputs[feature.name] = None
    result = predict(artifact, inputs)
    assert 0 <= result.probability <= 1
    assert set(result.imputed) == {f.label for f in blanked}


@pytest.mark.parametrize("key", ["heart", "kidney"])
def test_too_few_answers_are_refused(key):
    pipeline = PIPELINES[key]
    inputs = {f.name: None for f in pipeline.features}
    assert any("at least" in problem for problem in validate_inputs(pipeline, inputs))


def test_values_far_from_training_data_are_flagged():
    artifact = load_artifact("diabetes")
    inputs = typical(artifact) | {"BMI": 90.0}
    assert "Body mass index (BMI)" in predict(artifact, inputs).unusual


def test_higher_cardiometabolic_burden_raises_diabetes_estimate():
    artifact = load_artifact("diabetes")
    base = typical(artifact)
    burden = base | {"HighBP": 1, "HighChol": 1, "BMI": 38.0, "GenHlth": 4}
    assert predict(artifact, burden).probability > predict(artifact, base).probability


def _copy_models(tmp_path):
    models = tmp_path / "models"
    shutil.copytree(MODELS_DIR, models)
    return models, models / MANIFEST_PATH.name


def test_tampered_model_is_not_loaded(tmp_path):
    models, manifest = _copy_models(tmp_path)
    with (models / "heart.joblib").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ArtifactError, match="integrity"):
        load_artifact("heart", models, manifest)


def test_missing_model_has_helpful_message(tmp_path):
    models, manifest = _copy_models(tmp_path)
    (models / "heart.joblib").unlink()
    with pytest.raises(ArtifactError, match="chronic_risk.train"):
        load_artifact("heart", models, manifest)


def test_scikit_learn_mismatch_is_refused(tmp_path):
    models, manifest = _copy_models(tmp_path)
    content = json.loads(manifest.read_text())
    content["sklearn_version"] = "0.0.1"
    manifest.write_text(json.dumps(content))
    with pytest.raises(ArtifactError, match="0.0.1"):
        load_artifact("heart", models, manifest)


def test_corrupt_report_has_helpful_message(tmp_path):
    (tmp_path / "heart.json").write_text("{not json")
    with pytest.raises(ArtifactError, match="unreadable"):
        load_report("heart", tmp_path)
