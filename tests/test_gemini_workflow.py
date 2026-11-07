"""Cloud routing is checked without API calls or large model downloads."""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.error

import pytest
import yaml

from scripts import gemini_preflight, preflight


def workflow():
    return yaml.load(Path(".github/workflows/generate-video.yml").read_text(), Loader=yaml.BaseLoader)


def test_backend_choice_secret_alias_and_models_reach_every_pipeline_job():
    data = workflow()
    dispatch = data["on"]["workflow_dispatch"]["inputs"]["ai_backend"]
    assert dispatch["options"] == ["Gemini with local fallback", "Local"]
    assert dispatch["default"] == "Gemini with local fallback"
    assert data["on"]["workflow_call"]["inputs"]["ai_backend"] == {
        "type": "string", "default": "Gemini with local fallback"}
    for name in ("prepare", "plan", "script", "footage", "produce"):
        job = data["jobs"][name]
        env = job["env"]
        assert env["AI_BACKEND"] == "${{ inputs.ai_backend == 'Local' && 'local' || 'gemini' }}"
        assert env["GEMINI_API_KEY"] == "${{ secrets.GEMINI_API || secrets.GEMINI_API_KEY }}"
        assert env["GEMINI_TEXT_MODEL"] == "${{ vars.GEMINI_TEXT_MODEL || 'gemini-3.8-flash' }}"
        assert env["GEMINI_VISION_MODEL"] == "${{ vars.GEMINI_VISION_MODEL || 'gemini-3.8-flash' }}"
        assert env["GEMINI_RESEARCH_MODEL"] == "${{ vars.GEMINI_RESEARCH_MODEL || 'gemini-2.5-flash' }}"
        assert env["GEMINI_TTS_MODEL"] == "${{ vars.GEMINI_TTS_MODEL || 'gemini-3.8-flash-tts' }}"
        for narrator in ("WARGUY", "KIT_LASER", "FEMALE"):
            variable = "GEMINI_VOICE_" + narrator
            assert env[variable] == "${{ vars." + variable + " }}"
        assert env["SCRIPT_MODEL"] == "gemma4:12b"
        assert env["GEMINI_FREE_ONLY"] == "true"
        assert "GEMINI_MAX_REQUESTS" in env and "GEMINI_MAX_VIDEO_SECONDS" in env
        assert env["GEMINI_USAGE_REPORT"].startswith("work/gemini-usage-")
        probes = [i for i, step in enumerate(job["steps"]) if step.get("run") == "python -m scripts.gemini_preflight"]
        assert len(probes) == 1
        for i, step in enumerate(job["steps"]):
            command = step.get("run", "")
            if "bootstrap_ollama.sh" in command or "pip install torch==" in command:
                assert probes[0] < i
            assert "${{ inputs." not in command
        upload = next(step for step in job["steps"] if step.get("name") == "Bounded Gemini usage and capability diagnostics")
        assert upload["if"] == "always() && !cancelled()"
        assert "work/gemini-usage-*.jsonl" in upload["with"]["path"]


def test_tmdb_is_optional_only_for_selected_gemini_with_a_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API", raising=False)
    monkeypatch.setenv("AI_BACKEND", "gemini")
    assert not preflight.gemini_configured()
    monkeypatch.setenv("GEMINI_API", "alias-key")
    assert preflight.gemini_configured()
    monkeypatch.setenv("AI_BACKEND", "local")
    assert not preflight.gemini_configured()
    monkeypatch.setenv("VIDEO_TITLE", "Two films")
    monkeypatch.setenv("VIDEO_BRIEF", "A clear countdown")
    monkeypatch.setenv("MOVIE_LIST", "Arrival (2016); Inception (2010)")
    monkeypatch.setenv("NARRATOR", "Warguy")
    monkeypatch.setenv("VIDEO_TYPE", "Movies")
    monkeypatch.setenv("VIDEO_MINUTES", "Auto")
    monkeypatch.delenv("TMDB_READ_TOKEN", raising=False)
    with pytest.raises(preflight.PipelineError, match="TMDB_READ_TOKEN"):
        preflight.main()
    monkeypatch.setenv("AI_BACKEND", "gemini")
    checked = []
    monkeypatch.setattr(preflight, "voice_reference", lambda name, path: checked.append(name))
    assert preflight.main() == 0
    assert checked  # Cloud speech does not bypass the selected voice reference check.


def probe_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_BACKEND", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "private-key-do-not-log")
    monkeypatch.delenv("GEMINI_API", raising=False)
    monkeypatch.delenv("TMDB_READ_TOKEN", raising=False)
    report = tmp_path / "preflight.json"
    monkeypatch.setenv("GEMINI_PREFLIGHT_REPORT", str(report))
    return report


def test_metadata_probe_authenticates_via_header_without_content_generation(monkeypatch, tmp_path, capsys):
    report = probe_env(monkeypatch, tmp_path)
    calls = []
    def urlopen(request, timeout):
        calls.append(request)
        assert timeout == 20
        assert request.get_header("X-goog-api-key") == "private-key-do-not-log"
        return io.BytesIO(json.dumps({"models": [{
            "name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]
        }]}).encode())
    monkeypatch.setattr(gemini_preflight.urllib.request, "urlopen", urlopen)
    assert gemini_preflight.main() == 0
    assert len(calls) == 1 and calls[0].get_method() == "GET"
    assert "key=" not in calls[0].full_url
    result = json.loads(report.read_text())
    assert result["authentication"] == "verified"
    assert result["models"]["research"]["status"] == "available"
    assert result["models"]["text"]["status"] == "unavailable"
    assert result["billing_verified"] is False
    assert "private-key" not in report.read_text() + capsys.readouterr().out


@pytest.mark.parametrize("status", [401, 403, 429, 503])
def test_auth_and_quota_failure_use_local_tmdb_and_never_print_raw_errors(monkeypatch, tmp_path, capsys, status):
    report = probe_env(monkeypatch, tmp_path)
    env_file = tmp_path / "github-env"
    monkeypatch.setenv("GITHUB_ENV", str(env_file))
    monkeypatch.setenv("TMDB_READ_TOKEN", "private-tmdb-token")
    def denied(*args, **kwargs):
        raise urllib.error.HTTPError("https://private-key-do-not-log", status,
                                     "private-key-do-not-log", {}, io.BytesIO(b"private-key-do-not-log"))
    monkeypatch.setattr(gemini_preflight.urllib.request, "urlopen", denied)
    assert gemini_preflight.main() == 0
    assert env_file.read_text() == "AI_BACKEND=local\n"
    assert json.loads(report.read_text())["fallback"] == "local-with-tmdb"
    output = capsys.readouterr()
    assert "private-key" not in report.read_text() + output.out + output.err
    assert "private-tmdb" not in report.read_text() + output.out + output.err


def test_failed_auth_without_research_fallback_stops_before_model_setup(monkeypatch, tmp_path, capsys):
    report = probe_env(monkeypatch, tmp_path)
    def unavailable(*args, **kwargs):
        raise OSError("private-key-do-not-log")
    monkeypatch.setattr(gemini_preflight.urllib.request, "urlopen", unavailable)
    assert gemini_preflight.main() == 1
    assert json.loads(report.read_text())["fallback"] == "unavailable-without-tmdb"
    output = capsys.readouterr()
    assert "private-key" not in report.read_text() + output.out + output.err


def test_local_selection_does_not_contact_gemini(monkeypatch, tmp_path):
    report = probe_env(monkeypatch, tmp_path)
    monkeypatch.setenv("AI_BACKEND", "local")
    def forbidden(*args, **kwargs):
        pytest.fail("Local selection must not call the cloud")
    monkeypatch.setattr(gemini_preflight.urllib.request, "urlopen", forbidden)
    assert gemini_preflight.main() == 0
    assert json.loads(report.read_text())["authentication"] == "skipped-local-selection"


@pytest.mark.parametrize("backend,has_key,expected_pull", [
    ("gemini", True, False), ("local", True, True), ("gemini", False, True),
])
def test_bootstrap_keeps_cloud_model_download_lazy_and_local_route_eager(tmp_path, backend, has_key, expected_pull):
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    calls = tmp_path / "calls"
    fake = {
        "python3": '#!/bin/bash\nif [ "$2" = scripts.install_ollama ]; then printf "%s\\n" "$FAKE_BIN"; else printf "%s\\n" "$*" >> "$FAKE_CALLS"; fi\n',
        "curl": "#!/bin/bash\nexit 0\n",
        "ollama": '#!/bin/bash\nprintf "%s\\n" "$*" >> "$FAKE_CALLS"\n[ "$1" != show ]\n',
        "timeout": '#!/bin/bash\nshift 2\nexec "$@"\n',
    }
    for name, content in fake.items():
        path = binary_dir / name
        path.write_text(content)
        path.chmod(0o755)
    env = dict(os.environ, PATH=str(binary_dir) + os.pathsep + os.environ["PATH"],
               FAKE_BIN=str(binary_dir), FAKE_CALLS=str(calls), AI_BACKEND=backend)
    env.pop("GEMINI_API_KEY", None)
    env.pop("GEMINI_API", None)
    if has_key:
        env["GEMINI_API"] = "private-key-do-not-log"
    result = subprocess.run([shutil.which("bash"), "scripts/bootstrap_ollama.sh"], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    commands = calls.read_text()
    assert ("pull gemma4:12b" in commands) == expected_pull
    assert "private-key" not in commands + result.stdout + result.stderr
    assert ("scripts.runner_disk --stage script-model" in commands) == expected_pull
