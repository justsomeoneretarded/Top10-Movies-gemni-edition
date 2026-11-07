from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import install_ollama


@pytest.fixture
def install_env(tmp_path, monkeypatch):
    monkeypatch.setattr(install_ollama.platform, "system", lambda: "Linux")
    monkeypatch.setattr(install_ollama.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(install_ollama.shutil, "which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("GITHUB_PATH", str(tmp_path / "github-path"))
    archive_bytes = b"small verified fixture, never a real download"
    monkeypatch.setattr(install_ollama, "ARCHIVE_SHA256", hashlib.sha256(archive_bytes).hexdigest())
    sleeps = []
    monkeypatch.setattr(install_ollama.time, "sleep", sleeps.append)
    return {"temp": tmp_path, "bytes": archive_bytes, "calls": [], "sleeps": sleeps}


def fake_commands(monkeypatch, environment, *, failure=None, fail_first_download=False):
    curl_attempts = 0

    def run(command, **kwargs):
        nonlocal curl_attempts
        environment["calls"].append((command, kwargs))
        if command[0] == "curl":
            curl_attempts += 1
            target = Path(command[command.index("--output") + 1])
            target.write_bytes(environment["bytes"])
            if failure == "download":
                target.write_bytes(b"partial")
                raise subprocess.CalledProcessError(22, command)
            if failure == "checksum" or (fail_first_download and curl_attempts == 1):
                target.write_bytes(b"wrong checksum")
            return subprocess.CompletedProcess(command, 0)
        if command[0] == "tar":
            if failure == "extract":
                raise subprocess.CalledProcessError(2, command)
            destination = Path(command[command.index("--directory") + 1])
            (destination / "bin").mkdir()
            binary = destination / "bin" / "ollama"
            binary.write_text("fixture")
            binary.chmod(0o755)
            libraries = destination / "lib" / "ollama"
            libraries.mkdir(parents=True)
            (libraries / "llama-server").write_text("CPU runtime")
            (libraries / "libggml-cpu-test.so").write_text("CPU backend")
            return subprocess.CompletedProcess(command, 0)
        assert command[-1] == "--version"
        version = "0.0.0" if failure == "version" else install_ollama.VERSION
        return subprocess.CompletedProcess(command, 0, f"ollama version is {version}\n", "")

    monkeypatch.setattr(install_ollama.subprocess, "run", run)


def previous_install(environment):
    root = environment["temp"] / "movie-ranks-ollama"
    previous = root / "releases" / "previous-working-release"
    (previous / "bin").mkdir(parents=True)
    (previous / "bin" / "ollama").write_text("keep this working version")
    (previous / ".release-sha256").write_text("previous-release-checksum")
    (root / "current").symlink_to(Path("releases") / previous.name)
    return root, previous


def test_pinned_download_cpu_exclusions_atomic_install_and_path(install_env, monkeypatch):
    fake_commands(monkeypatch, install_env)
    binary_dir = install_ollama.install_ollama()
    root = install_env["temp"] / "movie-ranks-ollama"
    assert binary_dir == root / "current" / "bin"
    payload = (root / "current").resolve()
    assert payload.parent == root / "releases"
    assert (payload / "lib" / "ollama" / "libggml-cpu-test.so").is_file()
    assert (install_env["temp"] / "github-path").read_text() == f"{binary_dir}\n"
    assert not list(root.glob(".install-*"))
    assert not list(root.rglob("*.tar.zst"))
    curl, options = next(call for call in install_env["calls"] if call[0][0] == "curl")
    assert curl[-1] == (
        "https://github.com/ollama/ollama/releases/download/v0.40.14/"
        "ollama-linux-amd64.tar.zst"
    )
    assert curl[curl.index("--max-time") + 1] == "600"
    assert curl[curl.index("--connect-timeout") + 1] == "20"
    assert curl[curl.index("--max-filesize") + 1] == str(2 * 1024**3)
    assert options["timeout"] == 630
    tar, options = next(call for call in install_env["calls"] if call[0][0] == "tar")
    assert "--zstd" in tar
    assert set(argument for argument in tar if argument.startswith("--exclude=")) == {
        "--exclude=lib/ollama/cuda_v*", "--exclude=./lib/ollama/cuda_v*",
        "--exclude=lib/ollama/vulkan", "--exclude=./lib/ollama/vulkan",
    }
    assert "--no-same-owner" in tar
    assert options["timeout"] == 300
    assert install_env["sleeps"] == []


def test_reuses_verified_install_without_download(install_env, monkeypatch):
    fake_commands(monkeypatch, install_env)
    original = install_ollama.install_ollama()
    install_env["calls"].clear()
    assert install_ollama.install_ollama() == original
    assert all(command[-1] == "--version" for command, _ in install_env["calls"])
    assert (install_env["temp"] / "github-path").read_text().splitlines() == [
        str(original), str(original),
    ]


def test_checksum_failure_retries_whole_archive_then_succeeds(install_env, monkeypatch):
    fake_commands(monkeypatch, install_env, fail_first_download=True)
    install_ollama.install_ollama()
    assert sum(command[0] == "curl" for command, _ in install_env["calls"]) == 2
    assert install_env["sleeps"] == [2]
    assert not list(install_env["temp"].rglob("*.tar.zst"))


@pytest.mark.parametrize("failure", ["download", "checksum", "extract", "version"])
def test_failed_replacement_preserves_prior_release_and_cleans_temp(
    install_env, monkeypatch, failure
):
    root, previous = previous_install(install_env)
    original_target = os.readlink(root / "current")
    fake_commands(monkeypatch, install_env, failure=failure)
    with pytest.raises((RuntimeError, subprocess.SubprocessError)):
        install_ollama.install_ollama()
    assert os.readlink(root / "current") == original_target
    assert (previous / "bin" / "ollama").read_text() == "keep this working version"
    assert list((root / "releases").iterdir()) == [previous]
    assert not list(root.glob(".install-*"))
    assert not list(root.rglob("*.tar.zst"))
    if failure in {"download", "checksum"}:
        assert sum(command[0] == "curl" for command, _ in install_env["calls"]) == 3
        assert install_env["sleeps"] == [2, 4]
        assert not any(command[0] == "tar" for command, _ in install_env["calls"])


def test_failed_activation_preserves_prior_release(install_env, monkeypatch):
    root, previous = previous_install(install_env)
    fake_commands(monkeypatch, install_env)
    def fail_replace(*args):
        raise OSError("activation failure")
    monkeypatch.setattr(install_ollama.os, "replace", fail_replace)
    with pytest.raises(OSError, match="activation failure"):
        install_ollama.install_ollama()
    assert (root / "current").resolve() == previous
    assert list((root / "releases").iterdir()) == [previous]
    assert not list(root.glob(".install-*"))
    assert not list(root.glob(".current-*"))


def test_refuses_symlinked_owned_root_without_touching_target(install_env, monkeypatch):
    target = install_env["temp"] / "unrelated-cache"
    target.mkdir()
    (target / "keep").write_text("preserve")
    (install_env["temp"] / "movie-ranks-ollama").symlink_to(target)
    fake_commands(monkeypatch, install_env)
    with pytest.raises(RuntimeError, match="symlinked Ollama"):
        install_ollama.install_ollama()
    assert (target / "keep").read_text() == "preserve"
    assert install_env["calls"] == []


def test_missing_zstd_fails_before_download(install_env, monkeypatch):
    monkeypatch.setattr(
        install_ollama.shutil, "which", lambda command: None if command == "zstd" else command
    )
    fake_commands(monkeypatch, install_env)
    with pytest.raises(RuntimeError, match="missing: zstd"):
        install_ollama.install_ollama()
    assert install_env["calls"] == []


def test_installer_main_outputs_only_path_or_clear_failure(monkeypatch, capsys):
    monkeypatch.setattr(install_ollama, "install_ollama", lambda: Path("/tmp/owned/current/bin"))
    assert install_ollama.main() == 0
    captured = capsys.readouterr()
    assert captured.out == "/tmp/owned/current/bin\n"
    assert captured.err == ""
    def fail():
        raise RuntimeError("checksum failure")
    monkeypatch.setattr(install_ollama, "install_ollama", fail)
    assert install_ollama.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "checksum failure" in captured.err


BOOTSTRAP = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_ollama.sh"

# All bootstrap executables are local fakes. The absolute Python shebang avoids
# recursing through the fake python3, and no subprocess can perform a download.
COMMAND_SHIM = r'''
import json
import os
from pathlib import Path
import sys

command = Path(sys.argv[0]).name
arguments = sys.argv[1:]
with open(os.environ["CALL_LOG"], "a") as log:
    log.write(json.dumps({
        "command": command, "args": arguments,
        "host": os.environ.get("OLLAMA_HOST"),
    }) + "\n")
if command == "python3":
    if arguments[:2] == ["-m", "scripts.install_ollama"]:
        if arguments[2:] == ["--existing"] and os.environ.get("INSTALL_PRESENT") != "true":
            sys.exit(3)
        print(os.environ["MOCK_INSTALL_BIN"])
    elif arguments[:3] == ["-m", "scripts.runner_disk", "--stage"]:
        if arguments[3] == os.environ.get("FAIL_DISK_STAGE"):
            sys.exit(9)
    else:
        sys.exit(99)
elif command == "curl":
    ready = os.environ.get("HEALTHY", "true") == "true"
    ready = ready or Path(os.environ["SERVER_MARKER"]).exists()
    sys.exit(0 if ready and os.environ.get("FAIL_SERVER") != "true" else 22)
elif command == "ollama":
    if arguments[0] == "serve":
        if os.environ.get("FAIL_SERVER") == "true":
            print("server fixture failed", file=sys.stderr)
            sys.exit(7)
        Path(os.environ["SERVER_MARKER"]).touch()
    elif arguments[0] == "show":
        sys.exit(0 if os.environ.get("MODEL_PRESENT") == "true" else 1)
    elif arguments[0] == "pull":
        count_file = Path(os.environ["PULL_COUNT"])
        count = int(count_file.read_text()) + 1 if count_file.exists() else 1
        count_file.write_text(str(count))
        sys.exit(23 if count <= int(os.environ.get("FAIL_PULLS", "0")) else 0)
    else:
        sys.exit(98)
elif command == "timeout":
    while arguments and arguments[0].startswith("--"):
        arguments.pop(0)
    arguments.pop(0)
    os.execvp(arguments[0], arguments)
elif command == "sleep":
    pass
else:
    sys.exit(97)
'''


@pytest.fixture
def bootstrap_env(tmp_path):
    commands = tmp_path / "commands"
    installed = tmp_path / "installed" / "bin"
    commands.mkdir()
    installed.mkdir(parents=True)
    shim = tmp_path / "command-shim"
    shim.write_text(f"#!{sys.executable}\n" + COMMAND_SHIM)
    shim.chmod(0o755)
    for command in ("python3", "curl", "timeout", "sleep"):
        (commands / command).symlink_to(shim)
    (installed / "ollama").symlink_to(shim)
    environment = {
        **os.environ,
        "PATH": str(commands) + os.pathsep + os.environ["PATH"],
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_PATH": str(tmp_path / "github-path"),
        "CALL_LOG": str(tmp_path / "calls.jsonl"),
        "MOCK_INSTALL_BIN": str(installed),
        "SERVER_MARKER": str(tmp_path / "server-ready"),
        "PULL_COUNT": str(tmp_path / "pull-count"),
        "HEALTHY": "true",
        "MODEL_PRESENT": "false",
        "INSTALL_PRESENT": "false",
        "FAIL_PULLS": "0",
        "OLLAMA_HOST": "127.0.0.1:11435",
        "OLLAMA_URL": "http://127.0.0.1:11435",
        "SCRIPT_MODEL": "fixture:12b",
    }
    for variable in ("FAIL_DISK_STAGE", "FAIL_SERVER"):
        environment.pop(variable, None)
    return environment


def run_bootstrap(environment, *arguments):
    return subprocess.run(
        ["bash", str(BOOTSTRAP), *arguments],
        env=environment, capture_output=True, text=True, timeout=20,
    )


def bootstrap_calls(environment):
    log = Path(environment["CALL_LOG"])
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def disk_stages(calls):
    return [
        call["args"][-1] for call in calls
        if call["command"] == "python3" and call["args"][:2] == ["-m", "scripts.runner_disk"]
    ]


def ollama_calls(calls, subcommand):
    return [
        call for call in calls
        if call["command"] == "ollama" and call["args"][0] == subcommand
    ]


def test_healthy_custom_port_reuses_server_and_installed_cli_path(bootstrap_env):
    result = run_bootstrap(bootstrap_env)
    assert result.returncode == 0, result.stderr
    calls = bootstrap_calls(bootstrap_env)
    assert disk_stages(calls) == ["ollama-install", "script-model", "ready"]
    assert not ollama_calls(calls, "serve")
    assert ollama_calls(calls, "pull")[0]["args"] == ["pull", "fixture:12b"]
    assert ollama_calls(calls, "pull")[0]["host"] == "127.0.0.1:11435"
    health = next(call for call in calls if call["command"] == "curl")
    assert health["args"][-1] == "http://127.0.0.1:11435/api/tags"
    assert "--max-time" in health["args"]


def test_cached_model_uses_only_ready_reserve(bootstrap_env):
    bootstrap_env["MODEL_PRESENT"] = "true"
    result = run_bootstrap(bootstrap_env)
    assert result.returncode == 0, result.stderr
    calls = bootstrap_calls(bootstrap_env)
    assert disk_stages(calls) == ["ollama-install", "ready"]
    assert not ollama_calls(calls, "pull")


def test_serve_only_never_shows_or_pulls_script_model(bootstrap_env):
    result = run_bootstrap(bootstrap_env, "--serve-only")
    assert result.returncode == 0, result.stderr
    calls = bootstrap_calls(bootstrap_env)
    assert disk_stages(calls) == ["ollama-install", "ready"]
    assert not ollama_calls(calls, "show")
    assert not ollama_calls(calls, "pull")


def test_model_pull_retries_are_bounded_and_preserve_final_failure(bootstrap_env):
    bootstrap_env["FAIL_PULLS"] = "3"
    result = run_bootstrap(bootstrap_env)
    assert result.returncode == 23
    calls = bootstrap_calls(bootstrap_env)
    assert len(ollama_calls(calls, "pull")) == 3
    timeouts = [
        call["args"] for call in calls
        if call["command"] == "timeout" and "pull" in call["args"]
    ]
    assert timeouts == [["--kill-after=15s", "1200s", "ollama", "pull", "fixture:12b"]] * 3
    sleeps = [call["args"] for call in calls if call["command"] == "sleep"]
    assert sleeps == [["5"], ["10"]]
    assert disk_stages(calls) == ["ollama-install", "script-model"]
    assert "after three bounded attempts" in result.stderr


def test_model_pull_recovers_after_transient_failure(bootstrap_env):
    bootstrap_env["FAIL_PULLS"] = "1"
    result = run_bootstrap(bootstrap_env)
    assert result.returncode == 0, result.stderr
    calls = bootstrap_calls(bootstrap_env)
    assert len(ollama_calls(calls, "pull")) == 2
    assert disk_stages(calls)[-1] == "ready"


@pytest.mark.parametrize("stage", ["ollama-install", "script-model", "ready"])
def test_disk_failure_stops_before_unsafe_next_work(bootstrap_env, stage):
    bootstrap_env["FAIL_DISK_STAGE"] = stage
    if stage == "ready":
        bootstrap_env["MODEL_PRESENT"] = "true"
    result = run_bootstrap(bootstrap_env)
    assert result.returncode == 9
    calls = bootstrap_calls(bootstrap_env)
    assert not ollama_calls(calls, "pull")
    if stage == "ollama-install":
        assert not any(call["args"] == ["-m", "scripts.install_ollama"] for call in calls)


def test_server_start_failure_is_reported_without_model_pull(bootstrap_env):
    bootstrap_env["HEALTHY"] = "false"
    bootstrap_env["FAIL_SERVER"] = "true"
    result = run_bootstrap(bootstrap_env)
    assert result.returncode == 1
    calls = bootstrap_calls(bootstrap_env)
    assert len(ollama_calls(calls, "serve")) == 1
    assert not ollama_calls(calls, "pull")
    assert "did not become ready" in result.stderr


def test_unknown_bootstrap_stage_fails_before_mutation(bootstrap_env):
    result = run_bootstrap(bootstrap_env, "--unknown-stage")
    assert result.returncode == 2
    assert bootstrap_calls(bootstrap_env) == []
    assert "Usage:" in result.stderr

def test_ready_install_reuses_with_less_space_than_a_fresh_install(bootstrap_env):
    bootstrap_env["INSTALL_PRESENT"]="true"
    bootstrap_env["MODEL_PRESENT"]="true"
    bootstrap_env["FAIL_DISK_STAGE"]="ollama-install"
    result=run_bootstrap(bootstrap_env)
    assert result.returncode==0,result.stderr
    calls=bootstrap_calls(bootstrap_env)
    assert disk_stages(calls)==["ready"]
    assert not any(call["args"]==["-m","scripts.install_ollama"] for call in calls)
    assert not ollama_calls(calls,"pull")

def test_existing_probe_never_downloads_when_install_is_missing(install_env,monkeypatch,capsys):
    fake_commands(monkeypatch,install_env)
    assert install_ollama.main(["--existing"])==3
    assert install_env["calls"]==[]
    assert capsys.readouterr().out==""

def test_existing_probe_reuses_verified_version(install_env,monkeypatch,capsys):
    fake_commands(monkeypatch,install_env)
    binary=install_ollama.install_ollama()
    install_env["calls"].clear()
    assert install_ollama.main(["--existing"])==0
    assert capsys.readouterr().out==str(binary)+"\n"
    assert all(command[-1]=="--version" for command,_ in install_env["calls"])
