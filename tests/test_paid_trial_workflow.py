"""Isolation and integration contracts for the one paid test."""
import json
from pathlib import Path

import pytest
import yaml

from movie_pipeline import gemini, gemini_editing as editing
from scripts import paid_trial_preflight as preflight


@pytest.fixture
def paid(monkeypatch, tmp_path):
    for name, value in {
        "GEMINI_PAID_TRIAL": "true", "GEMINI_FREE_ONLY": "false", "AI_BACKEND": "gemini",
        "GEMINI_API_KEY": "offline-test-key", "GEMINI_SCRIPT_MODEL": "gemini-4-carbon",
        "GEMINI_TEXT_MODEL": "gemini-3.8-flash", "GEMINI_VISION_MODEL": "gemini-3.8-flash",
        "GEMINI_VIDEO_MODEL": "gemini-3.8-flash", "GEMINI_RESEARCH_MODE": "public_sources",
        "GITHUB_RUN_ATTEMPT": "1", "NARRATOR": "Warguy", "TTS_MODEL": "original",
        "GEMINI_VOICE_WARGUY": "", "GEMINI_VOICE_KIT_LASER": "", "GEMINI_VOICE_FEMALE": "",
        "GEMINI_PAID_BUDGET_PATH": str(tmp_path / "budget.json"),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(preflight, "REPORT", tmp_path / "preflight.json")


def test_single_billed_job_exclusive_pro_key_and_guard():
    trial = yaml.safe_load(Path(".github/workflows/paid-gemini-trial.yml").read_text())
    assert set(trial["jobs"]) == {"request", "produce"}
    assert "env" not in trial["jobs"]["request"]
    job = trial["jobs"]["produce"]
    env = job["env"]
    assert env["GEMINI_API_KEY"] == "${{ secrets.GEMINI_API_KEY_PRO }}"
    assert env["GEMINI_API_KEY_2"] == env["GEMINI_API_KEY_3"] == ""
    assert env["GEMINI_VOICE_WARGUY"] == env["GEMINI_VOICE_KIT_LASER"] == env["GEMINI_VOICE_FEMALE"] == ""
    assert env["TTS_MODEL"] == "original"
    assert env["GEMINI_SCRIPT_MODEL"] == "gemini-4-carbon"
    assert env["GEMINI_TEXT_MODEL"] == env["GEMINI_VISION_MODEL"] == env["GEMINI_VIDEO_MODEL"] == "gemini-3.8-flash"
    assert env["GEMINI_PAID_MAX_USD"] == "5" and env["GEMINI_MAX_RETRIES"] == "0"
    steps = job["steps"]
    assert any(s.get("uses") == "./.github/actions/resilient-production" for s in steps)
    assert any(s.get("with", {}).get("path") == "work/models" for s in steps)
    assert any(s.get("with", {}).get("cache") == "pip" for s in steps)
    packages = [s for s in steps if s.get("with", {}).get("path") == "output/"]
    assert len(packages) == 1 and packages[0]["with"]["retention-days"] == 2
    assert any(s.get("with", {}).get("path") == "work/paid-thumbnails/" for s in steps)
    assert "Telegram" not in str(steps)


def test_regular_workflow_remains_free():
    ordinary = yaml.safe_load(Path(".github/workflows/generate-video.yml").read_text())
    for job in ordinary["jobs"].values():
        env = job.get("env", {})
        if "GEMINI_FREE_ONLY" in env:
            assert env["GEMINI_FREE_ONLY"] == "true"
            assert "GEMINI_PAID_TRIAL" not in env and "GEMINI_API_KEY_PRO" not in str(env)


@pytest.mark.parametrize("name,value", [
    ("GITHUB_RUN_ATTEMPT", "2"), ("GEMINI_PAID_TRIAL", "false"), ("GEMINI_FREE_ONLY", "true"),
    ("GEMINI_API_KEY", ""), ("GEMINI_SCRIPT_MODEL", "gemini-3.8-flash"),
    ("NARRATOR", "Female"), ("TTS_MODEL", "turbo"), ("GEMINI_VOICE_WARGUY", "different-voice"),
])
def test_preflight_rejects_wrong_contract(paid, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        preflight.verify_config()


def test_real_contract_check_is_required_after_metadata(paid, monkeypatch):
    calls = []
    class Client:
        def generate_json(self, prompt, schema, **kwargs):
            calls.append(kwargs)
            return {"status": "ready"}
    monkeypatch.setattr(preflight, "listed_models", lambda: {m: True for m in preflight.MODELS})
    monkeypatch.setattr(gemini, "get_client", lambda: Client())
    assert preflight.main() == 0
    assert len(calls) == 1 and calls[0]["model"] == "gemini-4-carbon"
    assert calls[0]["max_output_tokens"] == 32768
    assert json.loads(preflight.REPORT.read_text())["inference_verified"] is True


def test_preflight_cannot_hide_failure_or_echo_http_internals(paid, monkeypatch, capsys):
    class Client:
        def generate_json(self, *args, **kwargs):
            raise RuntimeError("sensitive-HTTP-internals")
    monkeypatch.setattr(preflight, "listed_models", lambda: {m: True for m in preflight.MODELS})
    monkeypatch.setattr(gemini, "get_client", lambda: Client())
    assert preflight.main() == 1
    assert "sensitive" not in capsys.readouterr().out + preflight.REPORT.read_text()


def test_paid_video_failure_never_reuploads_or_switches_models(paid, monkeypatch, tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"actual-file-bytes")
    source.with_suffix(".json").write_text(json.dumps({"source": "https://www.youtube.com/watch?v=abcdefghijk"}))
    monkeypatch.setattr(editing, "_ATTEMPTED", set())
    monkeypatch.setattr(editing, "_FAILURES", {})
    monkeypatch.setattr(editing, "_URL_REVIEWS", {})
    monkeypatch.setattr(editing, "REPORT", {})
    monkeypatch.setattr(editing, "duration", lambda _: 40.0)
    monkeypatch.setattr(editing, "_budget_left", lambda: True)
    monkeypatch.setattr(editing, "_video_pause", lambda: None)
    calls = []
    class Client:
        vision_model = "gemini-3.5-flash-lite"
        def generate_json(self, *args, **kwargs):
            calls.append(kwargs)
            raise gemini.GeminiUnavailable("read_timeout")
        def upload_video(self, *args, **kwargs):
            pytest.fail("Paid failure must not cause another billed route")
    monkeypatch.setattr(gemini, "get_client", lambda: Client())
    assert editing.video_models(Client()) == ["gemini-3.8-flash"]
    assert editing.analyse_source(source) is None
    assert editing.analyse_source(source) is None
    assert len(calls) == 1
    assert "https://www.youtube.com/watch?v=abcdefghijk" in editing._ATTEMPTED


def test_paid_semantic_fallback_is_reported(paid, monkeypatch):
    from movie_pipeline.script import Writer
    events = []
    monkeypatch.setattr(Writer, "ask", lambda *a, **k: {"paragraphs": []})
    monkeypatch.setattr(gemini, "record_event", lambda *a, **k: events.append((a, k)))
    result = Writer().narration("Five shows", "Mixed list", "rank 1", {"title": "Test", "rank": 1, "overview": "A detective follows a missing case."}, None)
    assert result
    assert any(row[0][:4] == ("script", "local-template", "fallback", "paid_narration_semantic_validation") for row in events)
