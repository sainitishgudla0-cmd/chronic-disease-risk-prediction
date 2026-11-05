import hashlib
import io
import urllib.error

import pytest

from chronic_risk import data
from chronic_risk.pipelines import PIPELINES
from chronic_risk.schema import DataSource

PAYLOAD = b"a,b\n1,2\n"
GOOD = DataSource("https://example.org/f.csv", "f.csv", hashlib.sha256(PAYLOAD).hexdigest())


def _serve(monkeypatch, payload=PAYLOAD):
    calls = []

    def fake(request, timeout):
        calls.append(request.full_url)
        return io.BytesIO(payload)

    monkeypatch.setattr(data.urllib.request, "urlopen", fake)
    return calls


def test_downloads_once_then_reuses_verified_copy(monkeypatch, tmp_path):
    calls = _serve(monkeypatch)
    assert data.ensure_source(GOOD, tmp_path).read_bytes() == PAYLOAD
    data.ensure_source(GOOD, tmp_path)
    assert len(calls) == 1


def test_checksum_mismatch_is_rejected_and_leaves_nothing_behind(monkeypatch, tmp_path):
    _serve(monkeypatch, b"tampered")
    with pytest.raises(data.DataDownloadError, match="Checksum mismatch"):
        data.ensure_source(GOOD, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_corrupted_local_copy_is_replaced(monkeypatch, tmp_path):
    (tmp_path / "f.csv").write_bytes(b"corrupt")
    _serve(monkeypatch)
    assert data.ensure_source(GOOD, tmp_path).read_bytes() == PAYLOAD


def test_network_failure_gives_a_clear_error(monkeypatch, tmp_path):
    def fail(request, timeout):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(data.urllib.request, "urlopen", fail)
    with pytest.raises(data.DataDownloadError, match="Check your connection"):
        data.ensure_source(GOOD, tmp_path)


def test_plain_http_is_refused(tmp_path):
    source = DataSource("http://example.org/f.csv", "f.csv", "0" * 64)
    with pytest.raises(data.DataDownloadError, match="non-HTTPS"):
        data.ensure_source(source, tmp_path)


@pytest.mark.parametrize("pipeline", PIPELINES.values(), ids=PIPELINES)
def test_every_pipeline_documents_and_pins_its_source(pipeline):
    assert pipeline.sources and pipeline.citation and pipeline.license_name
    assert pipeline.limitations and pipeline.target_definition and pipeline.population
    for source in pipeline.sources:
        assert source.url.startswith("https://") and len(source.sha256) == 64
