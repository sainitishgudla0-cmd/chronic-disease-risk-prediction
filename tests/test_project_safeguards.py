"""Guards for the project's commitments on privacy, honest output, and committed files."""

import json
import re
import tomllib

import joblib
import pytest

from chronic_risk.inference import load_artifact, predict
from chronic_risk.paths import MODELS_DIR, PROJECT_ROOT
from chronic_risk.pipelines import PIPELINES

UI_FILES = [
    PROJECT_ROOT / "app.py",
    *sorted((PROJECT_ROOT / "app_pages").glob("*.py")),
    *sorted((PROJECT_ROOT / "chronic_risk" / "ui").glob("*.py")),
]
RUNTIME_FILES = [*UI_FILES, PROJECT_ROOT / "chronic_risk" / "inference.py"]
SKIP_DIRS = {".venv", ".git", "data", "__pycache__", ".ruff_cache", ".pytest_cache"}


def _user_facing_text():
    parts = [path.read_text(encoding="utf-8") for path in UI_FILES]
    for pipeline in PIPELINES.values():
        parts += [
            pipeline.outcome_phrase,
            pipeline.target_definition,
            pipeline.population,
            pipeline.result_caution,
            pipeline.urgent_note,
            *pipeline.limitations,
        ]
        parts += [f"{f.label} {f.help}" for f in pipeline.features]
    return "\n".join(parts)


def _project_files():
    for path in PROJECT_ROOT.rglob("*"):
        if path.is_file() and not SKIP_DIRS & set(path.relative_to(PROJECT_ROOT).parts):
            yield path


@pytest.mark.parametrize("path", RUNTIME_FILES, ids=lambda p: p.name)
def test_runtime_code_never_writes_or_logs(path):
    pattern = (
        r"\bopen\(|\.to_csv|\.to_json|\.to_pickle|\.write_text|\.write_bytes|joblib\.dump"
        r"|\blogging\b|\bprint\(|sqlite|requests\.|urlopen"
    )
    assert not re.search(pattern, path.read_text(encoding="utf-8"))


def test_prediction_creates_no_files():
    before = set(_project_files())
    artifact = load_artifact("heart")
    predict(artifact, dict(artifact.report["reference_values"]))
    assert set(_project_files()) == before


def test_no_input_asks_for_an_identifier():
    banned = re.compile(
        r"\b(name|e-?mail|phone|address|ssn|social security|date of birth|mrn|zip|postcode)\b",
        re.IGNORECASE,
    )
    for pipeline in PIPELINES.values():
        for feature in pipeline.features:
            assert not banned.search(f"{feature.name} {feature.label}"), feature.name


def test_no_risk_category_labels_are_presented():
    text = _user_facing_text().lower()
    for level in ("high", "low", "medium", "moderate", "elevated", "at"):
        assert f"{level} risk" not in text, level


def test_cutoffs_are_described_as_not_medical():
    source = (PROJECT_ROOT / "app_pages" / "models.py").read_text(encoding="utf-8")
    assert "none is medically established" in source


def test_every_assessment_shows_disclaimer_and_urgent_guidance():
    source = (PROJECT_ROOT / "chronic_risk" / "ui" / "assessment.py").read_text(encoding="utf-8")
    assert "theme.DISCLAIMER" in source and "urgent_note" in source and "result_caution" in source
    assert "theme.DISCLAIMER" in (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    for pipeline in PIPELINES.values():
        assert pipeline.urgent_note and pipeline.result_caution


def test_gitignore_protects_data_and_secrets():
    lines = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for rule in ("data/", "*.csv", ".env", ".streamlit/secrets.toml", ".venv/"):
        assert rule in lines


def test_config_disables_usage_stats_and_error_details():
    text = (PROJECT_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    config = tomllib.loads(text)
    assert config["browser"]["gatherUsageStats"] is False
    assert config["client"]["showErrorDetails"] == "none"


def test_repository_has_no_hard_coded_secrets():
    pattern = re.compile(
        r"(api[_-]?key|secret|password|token)\s*[:=]\s*[\"'][^\"']{8,}"
        r"|sk-[a-zA-Z0-9]{20}|ghp_\w{20}",
        re.IGNORECASE,
    )
    for path in _project_files():
        if path.suffix in {".py", ".toml", ".yml", ".md", ".json", ".txt"}:
            assert not pattern.search(path.read_text(encoding="utf-8")), path


@pytest.mark.parametrize("key", PIPELINES)
def test_committed_model_files_hold_no_records(key):
    path = MODELS_DIR / f"{key}.joblib"
    assert path.stat().st_size < 1_000_000
    report = json.loads((MODELS_DIR / f"{key}.json").read_text(encoding="utf-8"))
    # Anything list-shaped must be an aggregate, far smaller than the dataset's row count.
    for name in ("calibration_table", "roc", "subgroups", "cutoffs"):
        assert len(report[name]) <= 60
    assert all(row["n"] >= 5 for row in report["calibration_table"])
    model = joblib.load(path)
    assert not hasattr(model, "X_") and not hasattr(model, "_fit_X")


def test_no_dataset_files_outside_the_ignored_data_folder():
    suffixes = {".csv", ".data", ".arff", ".xpt", ".rar", ".zip"}
    assert not [p for p in _project_files() if p.suffix.lower() in suffixes]


def test_requirements_are_pinned_or_bounded():
    for name in ("requirements.txt", "requirements-dev.txt"):
        for line in (PROJECT_ROOT / name).read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "-r")):
                assert re.search(r"(==|<)", line), line
