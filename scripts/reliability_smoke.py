"""Small real-runner checks; no TTS, rendering, paid APIs or production requests."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def guarded_checks() -> None:
    root = Path("work/runner-smoke")
    root.mkdir(parents=True, exist_ok=True)
    cases = [
        ("normal", [sys.executable, __file__, "child-check", str(os.getuid())], 0),
        ("failure", [sys.executable, "-c", "raise SystemExit(23)"], 23),
        ("memory-pressure", [sys.executable, "-c",
                             "items=[]\nwhile True: items.append(bytearray(16*1024*1024))"], None),
    ]
    report = []
    for name, command, expected in cases:
        folder = root / name
        environment = os.environ.copy()
        environment["RUNNER_GUARD_SENTINEL"] = "preserved-by-smoke"
        started = time.monotonic()
        result = subprocess.run(
            [sys.executable, "-m", "scripts.runner_guard", "--log-dir", str(folder),
             "--memory-limit-mib", "192", "--swap-limit-mib", "0", "--", *command],
            capture_output=True, text=True, timeout=60, env=environment,
        )
        events = [json.loads(line) for line in (folder / "resources.jsonl").read_text().splitlines()]
        if expected is None:
            if result.returncode in (0, 78):
                raise RuntimeError("Memory-pressure check did not exercise the protected workload")
        elif result.returncode != expected:
            raise RuntimeError(f"Runner guard {name} returned {result.returncode}, expected {expected}")
        if not any(event.get("event") == "start" and event.get("mode") == "systemd_scope" for event in events):
            raise RuntimeError("Required cgroup isolation was not demonstrated")
        if not any(event.get("event") == "finish" for event in events):
            raise RuntimeError("Missing final resource diagnostic")
        report.append({"case": name, "exit_code": result.returncode,
                       "seconds": round(time.monotonic() - started, 2),
                       "runner_survived": True, "final": events[-1]})
        print("RUNNER_CHECK " + json.dumps(report[-1]), flush=True)
    signal_folder = root / "signal"
    marker = root / "signal-child.json"
    process = subprocess.Popen(
        [sys.executable, "-m", "scripts.runner_guard", "--log-dir", str(signal_folder),
         "--memory-limit-mib", "192", "--swap-limit-mib", "0", "--",
         sys.executable, __file__, "signal-fixture", str(marker.resolve())],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 30
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.1)
        if not marker.exists():
            raise RuntimeError("Signal fixture never entered the protected scope")
        identities = json.loads(marker.read_text())
        process.terminate()
        if process.wait(timeout=35) != 143:
            raise RuntimeError("Guard did not preserve the cancellation signal")
        for pid in identities:
            status = Path(f"/proc/{pid}/stat")
            if status.exists() and status.read_text().split()[2] != "Z":
                raise RuntimeError("A protected descendant survived cancellation")
        report.append({"case": "signal", "exit_code": 143, "descendants_stopped": True})
        print("RUNNER_CHECK " + json.dumps(report[-1]), flush=True)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=35)
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def child_check(expected_uid: str) -> None:
    if os.getuid() != int(expected_uid):
        raise RuntimeError("Scope changed the production user")
    if os.getenv("RUNNER_GUARD_SENTINEL") != "preserved-by-smoke":
        raise RuntimeError("Scope dropped the production environment")
    cgroup = Path("/proc/self/cgroup").read_text()
    if "movie-production-" not in cgroup:
        raise RuntimeError("Workload is outside the protected scope")
    if not Path("scripts/runner_guard.py").is_file():
        raise RuntimeError("Scope changed the working directory")


def checkpoint_fixture() -> None:
    folder = Path("work/jobs") / ("c" * 20)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "input.json").write_text(json.dumps({"fixture": "checkpoint-roundtrip"}))
    (folder / "script.json").write_text(json.dumps({"fixture": "checkpoint-roundtrip", "sections": []}))
    for index in range(12):
        print(f"Checkpoint fixture progress {index + 1}/12", flush=True)
        time.sleep(1)


def research_check() -> None:
    from movie_pipeline.core import FilmInput
    from movie_pipeline.research import Research

    research = Research()
    research.selected_kind = research.preferred_kind = "tv"
    resolved = research.resolve(FilmInput("Maxton Hall", 2024))
    if resolved.get("id") != 250670 or resolved.get("kind") != "tv" or resolved.get("year") != 2024:
        raise RuntimeError("Live short-title identity did not match the verified production")
    print("LIVE_TITLE_IDENTITY " + json.dumps({
        "title": resolved["title"], "year": resolved["year"],
        "kind": resolved["kind"], "id": resolved["id"],
        "identity_source": resolved.get("identity_source"),
    }), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("guard", "child-check", "checkpoint-fixture", "signal-fixture", "research"))
    parser.add_argument("argument", nargs="?")
    args = parser.parse_args()
    if args.mode == "guard":
        guarded_checks()
    elif args.mode == "child-check":
        child_check(args.argument)
    elif args.mode == "checkpoint-fixture":
        checkpoint_fixture()
    elif args.mode == "signal-fixture":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        Path(args.argument).write_text(json.dumps([os.getpid(), child.pid]))
        time.sleep(60)
    else:
        research_check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
