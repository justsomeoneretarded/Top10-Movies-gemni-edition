"""Contain a complete production workload while keeping the Actions runner outside it.

The default requires a working systemd cgroup-v2 scope. No command or environment
values are written to diagnostics. Watchdog-only mode is explicit and forbidden
in GitHub Actions because a polling monitor cannot contain a sudden allocation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

GIB = 1024 ** 3
MIB = 1024 ** 2
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "TTS_THREADS",
)
CGROUP_ROOT = Path("/sys/fs/cgroup")
# Redirect workload errors to stdout so systemd's own stderr can be suppressed:
# its launch errors can otherwise repeat command arguments containing secrets.
CHILD_SHELL = ('exec 2>&1; printf "500\\n" > /proc/self/oom_score_adj '
               '2>/dev/null || :; exec "$@"')


class GuardSetupError(RuntimeError):
    """A fixed, non-sensitive setup reason."""


@dataclass(frozen=True)
class Limits:
    effective_memory: int
    host_reserve: int
    memory_high: int
    memory_max: int
    swap_max: int
    cpu_quota_percent: float
    threads: int


def read_meminfo(path: Path = Path("/proc/meminfo")) -> dict[str, int]:
    result = {}
    try:
        for line in path.read_text().splitlines():
            name, _, value = line.partition(":")
            if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                result[name] = int(value.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return result


def cgroup_ancestors(root: Path = CGROUP_ROOT,
                     membership: Path = Path("/proc/self/cgroup")) -> list[Path]:
    """Read every unified ancestor, including inherited container/job limits."""
    try:
        relative = next(line.split(":", 2)[2] for line in membership.read_text().splitlines()
                        if line.startswith("0::"))
        current = (root / relative.lstrip("/")).resolve()
        root = root.resolve()
        if current != root and root not in current.parents:
            return []
        result = []
        while True:
            result.append(current)
            if current == root:
                return result
            current = current.parent
    except (OSError, StopIteration):
        return []


def inherited_capacities(ancestors: list[Path]) -> tuple[list[int], list[float], list[int]]:
    memory, cpus, swap = [], [], []
    for parent in ancestors:
        for filename, values in (("memory.max", memory), ("memory.swap.max", swap)):
            try:
                raw = (parent / filename).read_text().strip()
                if raw != "max":
                    value = int(raw)
                    if value >= 0:
                        values.append(value)
            except (OSError, ValueError):
                pass
        try:
            quota, period = (parent / "cpu.max").read_text().split()
            if quota != "max" and int(period) > 0 and int(quota) > 0:
                cpus.append(int(quota) / int(period))
        except (OSError, ValueError):
            pass
    return memory, cpus, swap


def calculate_limits(mem_total: int, swap_total: int, affinity_cpus: int,
                     memory_caps=(), cpu_caps=(), swap_caps=()) -> Limits:
    effective = min([mem_total, *memory_caps])
    reserve = max(2 * GIB, math.ceil(effective * .20))
    maximum = effective - reserve
    if maximum < 512 * MIB:
        raise GuardSetupError("insufficient_memory_capacity")
    cpu_capacity = min([float(max(1, affinity_cpus)), *cpu_caps])
    # A fractional inherited CPU quota must never be increased by moving scope.
    workload_cpus = max(1.0, cpu_capacity - 1.0) if cpu_capacity >= 1 else cpu_capacity
    quota = workload_cpus * 100
    return Limits(effective, reserve,
                  maximum - min(GIB, max(128 * MIB, maximum // 10)), maximum,
                  max(0, min([swap_total, maximum // 2, *swap_caps])),
                  quota, max(1, math.floor(workload_cpus)))


def restrict_limits(limits: Limits, memory_limit_mib=None, swap_limit_mib=None) -> Limits:
    """Reduce-only overrides support small, real cgroup smoke tests in CI."""
    maximum = limits.memory_max
    if memory_limit_mib is not None:
        requested = memory_limit_mib * MIB
        if memory_limit_mib <= 0 or requested > maximum:
            raise GuardSetupError("invalid_memory_limit_override")
        maximum = requested
    swap = min(limits.swap_max, maximum // 2)
    if swap_limit_mib is not None:
        requested = swap_limit_mib * MIB
        if swap_limit_mib < 0 or requested > swap:
            raise GuardSetupError("invalid_swap_limit_override")
        swap = requested
    return replace(limits, memory_max=maximum,
                   memory_high=min(limits.memory_high, max(1, maximum * 9 // 10)),
                   swap_max=swap)


def detect_limits() -> Limits:
    info = read_meminfo()
    if not info.get("MemTotal"):
        raise GuardSetupError("memory_capacity_unavailable")
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity = os.cpu_count() or 1
    return calculate_limits(info["MemTotal"], info.get("SwapTotal", 0), affinity,
                            *inherited_capacities(cgroup_ancestors()))


def workload_environment(limits: Limits) -> dict[str, str]:
    environment = os.environ.copy()
    for name in THREAD_VARIABLES:
        try:
            requested = int(environment.get(name, limits.threads))
        except ValueError:
            requested = limits.threads
        environment[name] = str(max(1, min(requested, limits.threads)))
    return environment


def scope_command(command: list[str], unit: str, limits: Limits) -> list[str]:
    tools = {name: shutil.which(name) for name in ("sudo", "systemd-run", "setpriv", "env")}
    if not all(tools.values()) or not (CGROUP_ROOT / "cgroup.controllers").is_file():
        raise GuardSetupError("cgroup_tools_unavailable")
    return [
        tools["sudo"], "-n", "-E", tools["systemd-run"], "--scope", "--quiet",
        "--unit=" + unit,
        "--property=MemoryHigh=" + str(limits.memory_high),
        "--property=MemoryMax=" + str(limits.memory_max),
        "--property=MemorySwapMax=" + str(limits.swap_max),
        "--property=MemoryOOMGroup=yes",
        "--property=CPUQuota=" + format(limits.cpu_quota_percent, ".12g") + "%",
        "--property=CPUWeight=20", "--property=IOWeight=20",
        "--property=TasksMax=2048", "--",
        tools["setpriv"], "--reuid=" + str(os.getuid()), "--regid=" + str(os.getgid()),
        "--init-groups", tools["env"], "PATH=" + os.environ.get("PATH", os.defpath),
        "HOME=" + str(Path.home()), "USER=" + os.environ.get("USER", ""),
        "LOGNAME=" + os.environ.get("LOGNAME", ""),
        "/bin/sh", "-c", CHILD_SHELL, "runner-guard", *command,
    ]


def probe_scope(limits: Limits, environment: dict[str, str]) -> None:
    probe = scope_command(["/usr/bin/true"], "movie-guard-probe-" + uuid.uuid4().hex + ".scope",
                          limits)
    try:
        result = subprocess.run(probe, env=environment, capture_output=True, timeout=30,
                                check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise GuardSetupError("scope_probe_unavailable") from None
    if result.returncode:
        raise GuardSetupError("scope_probe_rejected")


def systemctl(arguments: list[str]):
    try:
        return subprocess.run(["sudo", "-n", "systemctl", *arguments],
                              capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None


def scope_path(unit: str) -> Path | None:
    result = systemctl(["show", "--property=ControlGroup", "--value", unit])
    if result is None or result.returncode:
        return None
    relative = result.stdout.strip()
    if not relative.startswith("/"):
        return None
    candidate = (CGROUP_ROOT / relative.lstrip("/")).resolve()
    root = CGROUP_ROOT.resolve()
    return candidate if candidate != root and root in candidate.parents else None


def numeric_file(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def counter_file(path: Path) -> dict[str, int]:
    result = {}
    try:
        for line in path.read_text().splitlines():
            key, value = line.split()
            if key.replace("_", "").isalnum():
                result[key] = int(value)
    except (OSError, ValueError):
        pass
    return result


def pressure(path: Path) -> dict:
    result = {}
    try:
        for line in path.read_text().splitlines():
            name, *values = line.split()
            if name not in {"some", "full"}:
                continue
            row = {}
            for value in values:
                key, raw = value.split("=", 1)
                if key in {"avg10", "avg60", "avg300", "total"}:
                    row[key] = float(raw) if key != "total" else int(raw)
            result[name] = row
    except (OSError, ValueError):
        pass
    return result


def cgroup_sample(path: Path | None) -> dict:
    if path is None:
        return {}
    result = {}
    for name in ("memory.current", "memory.peak", "memory.swap.current", "pids.current"):
        value = numeric_file(path / name)
        if value is not None:
            result[name] = value
    for name in ("memory.events", "cpu.stat"):
        values = counter_file(path / name)
        if values:
            result[name] = values
    return result


def disk_sample() -> dict:
    disks = {}
    seen = set()
    for label, location in (("workspace", Path.cwd()),
                            ("runner_temp", Path(os.getenv("RUNNER_TEMP", "/tmp")))):
        try:
            device = location.stat().st_dev
            if device in seen:
                continue
            seen.add(device)
            usage = shutil.disk_usage(location)
            disks[label] = {"free_bytes": usage.free, "total_bytes": usage.total}
        except OSError:
            pass
    return disks


def sample(path: Path | None) -> dict:
    try:
        load = list(os.getloadavg())
    except OSError:
        load = []
    return {"memory": read_meminfo(), "load_average": load, "disk": disk_sample(),
            "pressure": {name: pressure(Path("/proc/pressure") / name)
                         for name in ("cpu", "memory", "io")},
            "cgroup": cgroup_sample(path)}


class Diagnostics:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.stream = (directory / "resources.jsonl").open("a", encoding="utf-8")

    def write(self, event: str, **values):
        self.stream.write(json.dumps({"event": event, "time_unix": time.time(), **values},
                                     sort_keys=True) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()


def signal_workload(child, unit: str | None, signum: int):
    if unit:
        systemctl(["kill", "--kill-whom=all", "--signal=" + str(signum), unit])
    if child.poll() is None:
        try:
            os.killpg(child.pid, signum)
        except (ProcessLookupError, PermissionError):
            pass


def stop_workload(child, unit: str | None, signum: int = signal.SIGTERM,
                  grace: float = 20):
    signal_workload(child, unit, signum)
    try:
        child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        signal_workload(child, unit, signal.SIGKILL)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    # The launcher may have exited before a daemonized descendant. Scope-wide
    # cleanup also reaches descendants which have changed their process group.
    if unit:
        systemctl(["kill", "--kill-whom=all", "--signal=KILL", unit])


def exit_code(returncode: int) -> int:
    return 128 - returncode if returncode < 0 else returncode


def run_guard(command: list[str], log_dir: Path, *, allow_watchdog_fallback=False,
              memory_limit_mib=None, swap_limit_mib=None) -> int:
    diagnostics = Diagnostics(log_dir)
    child = None
    unit = None
    previous_handlers = {}
    received_signal = [0]
    last_cgroup = {}
    started = time.monotonic()
    try:
        limits = restrict_limits(detect_limits(), memory_limit_mib, swap_limit_mib)
        environment = workload_environment(limits)
        diagnostics.write("limits", **asdict(limits))
        try:
            probe_scope(limits, environment)
            mode = "systemd_scope"
            unit = "movie-production-" + uuid.uuid4().hex + ".scope"
            launch = scope_command(command, unit, limits)
        except GuardSetupError as exc:
            if not allow_watchdog_fallback or os.getenv("GITHUB_ACTIONS") == "true":
                raise
            mode = "watchdog_only"
            # This mode is intentionally unavailable in production Actions.
            launch = ["/bin/sh", "-c", CHILD_SHELL, "runner-guard", *command]
            diagnostics.write("isolation_unavailable", reason=str(exc))
            print("Runner guard: explicit local watchdog mode; cgroup isolation unavailable.",
                  flush=True)
        threshold = max(512 * MIB, limits.host_reserve // 2)
        available = read_meminfo().get("MemAvailable")
        if available is not None and available < threshold:
            raise GuardSetupError("insufficient_available_memory")
        if any(row["free_bytes"] < GIB for row in disk_sample().values()):
            raise GuardSetupError("insufficient_available_disk")
        diagnostics.write("start", mode=mode)
        print("Runner guard: " + mode + "; resource diagnostics enabled.", flush=True)

        def handle_signal(signum, _frame):
            received_signal[0] = signum

        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.signal(signum, handle_signal)
        try:
            child = subprocess.Popen(launch, env=environment, start_new_session=True,
                                     stderr=subprocess.DEVNULL if unit else None)
        except OSError:
            raise GuardSetupError("workload_launch_unavailable") from None
        # The real application is launched exactly once. A failure here never
        # retries it, changes mode, or runs it outside the successfully probed scope.
        next_sample = 0.0
        low_memory_samples = 0
        low_disk_samples = 0
        cgroup = None
        while child.poll() is None:
            if received_signal[0]:
                diagnostics.write("signal", signal=received_signal[0])
                stop_workload(child, unit, received_signal[0])
                diagnostics.write("finish", exit_code=128 + received_signal[0], cgroup=last_cgroup)
                return 128 + received_signal[0]
            now = time.monotonic()
            if unit and cgroup is None:
                cgroup = scope_path(unit)
            current_cgroup = cgroup_sample(cgroup)
            if current_cgroup:
                last_cgroup = current_cgroup
            if now >= next_sample:
                row = sample(cgroup)
                if row["cgroup"]:
                    last_cgroup = row["cgroup"]
                diagnostics.write("sample", elapsed_seconds=round(now - started, 3), **row)
                next_sample = now + 30
            available = read_meminfo().get("MemAvailable")
            low_memory_samples = low_memory_samples + 1 if (
                available is not None and available < threshold) else 0
            if low_memory_samples >= 3:
                diagnostics.write("host_memory_pressure", available_bytes=available,
                                  threshold_bytes=threshold)
                stop_workload(child, unit)
                diagnostics.write("finish", exit_code=75, cgroup=last_cgroup)
                return 75
            disks = disk_sample()
            low_disk_samples = low_disk_samples + 1 if any(
                row["free_bytes"] < GIB for row in disks.values()) else 0
            if low_disk_samples >= 3:
                diagnostics.write("host_disk_pressure", threshold_bytes=GIB, disk=disks)
                stop_workload(child, unit)
                diagnostics.write("finish", exit_code=75, cgroup=last_cgroup)
                return 75
            time.sleep(1)
        result = exit_code(child.returncode)
        if received_signal[0]:
            result = 128 + received_signal[0]
        final = cgroup_sample(cgroup)
        if final:
            last_cgroup = final
        diagnostics.write("finish", exit_code=result,
                          elapsed_seconds=round(time.monotonic() - started, 3),
                          cgroup=last_cgroup)
        return result
    except GuardSetupError as exc:
        diagnostics.write("setup_failed", reason=str(exc))
        print("Runner guard: setup failed (" + str(exc) + "); workload stopped.", file=sys.stderr)
        return 78
    finally:
        if child is not None:
            stop_workload(child, unit, grace=5)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        diagnostics.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=Path("work/runner-diagnostics"))
    parser.add_argument("--allow-watchdog-fallback", action="store_true")
    parser.add_argument("--memory-limit-mib", type=int)
    parser.add_argument("--swap-limit-mib", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a workload command is required after --")
    try:
        return run_guard(command, args.log_dir,
                         allow_watchdog_fallback=args.allow_watchdog_fallback,
                         memory_limit_mib=args.memory_limit_mib,
                         swap_limit_mib=args.swap_limit_mib)
    except OSError:
        # Never print subprocess exceptions: their command fields may be secret.
        print("Runner guard: diagnostics or process control unavailable.", file=sys.stderr)
        return 78


if __name__ == "__main__":
    raise SystemExit(main())
