from pathlib import Path
import subprocess

import yaml


def workflow():
    return yaml.load(Path(".github/workflows/generate-video.yml").read_text(), Loader=yaml.BaseLoader)


def test_production_uses_guard_and_durable_checkpoints_without_changing_outputs():
    steps = workflow()["jobs"]["produce"]["steps"]
    production = next(step for step in steps if step.get("name") == "Produce the complete package")
    assert production["uses"] == "./.github/actions/resilient-production"
    restore = next(step for step in steps if step.get("name") == "Restore the latest checkpoint from this run")
    assert restore["with"]["mode"] == "restore"
    assert steps.index(restore) < steps.index(production)
    assert not any("bootstrap_ollama.sh" in step.get("run", "") for step in steps)
    upload = next(step for step in steps if step.get("name") == "Downloadable video package")
    assert upload["with"]["path"] == "output/"
    assert upload["with"]["retention-days"] == "2"
    assert "if" not in upload
    diagnostics = next(step for step in steps if step.get("name") == "Runner resource diagnostics")
    assert diagnostics["if"] == "always()"
    assert diagnostics["with"]["path"] == "work/runner-diagnostics/"
    assert diagnostics["with"]["retention-days"] == "2"


def test_guarded_entry_starts_server_before_cli_and_has_owned_cleanup():
    source = Path("scripts/produce_guarded.sh").read_text()
    assert source.index("bash scripts/bootstrap_ollama.sh") < source.index("python -m movie_pipeline.cli")
    assert "trap cleanup_ollama EXIT" in source
    assert 'b"serve" not in args' in source
    assert "OLLAMA_PID_FILE" in source
    assert subprocess.run(["bash", "-n", "scripts/produce_guarded.sh"], capture_output=True).returncode == 0


def test_bootstrap_pid_record_does_not_change_model_cache_location():
    source = Path("scripts/bootstrap_ollama.sh").read_text()
    assert 'OLLAMA_PID_FILE' in source
    assert 'OLLAMA_LOG_FILE' in source
    assert 'OLLAMA_MODELS' not in source

def test_guarded_production_reports_status_and_stops_its_server(tmp_path, monkeypatch):
    import json
    import os
    import sys
    import time

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    tool = tmp_path / "ollama"
    tool.write_text("#!" + sys.executable + "\nimport time\ntime.sleep(60)\n")
    tool.chmod(0o755)
    bootstrap = scripts / "bootstrap_ollama.sh"
    bootstrap.write_text(
        'set -eu\n"$STUB_OLLAMA" serve >/dev/null 2>&1 &\n'
        'printf "%s\\n" "$!" > "$OLLAMA_PID_FILE"\n'
        'cp "$OLLAMA_PID_FILE" server.pid\n'
    )
    (scripts / "run_logged.sh").write_text('exit "$STUB_EXIT"\n')
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("STUB_OLLAMA", str(tool))
    monkeypatch.setenv("STUB_EXIT", "23")
    result = subprocess.run(
        ["bash", str(Path("scripts/produce_guarded.sh").resolve())],
        cwd=tmp_path, env=os.environ.copy(), capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 23, result.stderr
    pid = int((tmp_path / "server.pid").read_text())
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        status = Path(f"/proc/{pid}/stat")
        if not status.exists() or status.read_text().split()[2] == "Z":
            break
        time.sleep(.05)
    else:
        raise AssertionError("Owned Ollama process survived production failure")
    assert not list(tmp_path.glob("movie-production-ollama-*.pid"))
