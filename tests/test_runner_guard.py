"""Resource policy and supervisor behavior, without privileged test setup."""
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

from scripts import runner_guard as guard


def test_limits_reserve_host_memory_and_one_cpu():
    limits = guard.calculate_limits(16 * guard.GIB, 12 * guard.GIB, 4)
    assert limits.host_reserve >= 2 * guard.GIB
    assert limits.host_reserve >= limits.effective_memory * .20
    assert limits.memory_max + limits.host_reserve == limits.effective_memory
    assert 0 < limits.memory_high < limits.memory_max
    assert limits.swap_max <= 12 * guard.GIB
    assert limits.swap_max <= limits.memory_max // 2
    assert limits.cpu_quota_percent == 300
    assert limits.threads == 3


def test_effective_ancestor_limits_are_never_increased():
    limits = guard.calculate_limits(16 * guard.GIB, 12 * guard.GIB, 4,
                                   [8 * guard.GIB], [2.5], [guard.GIB])
    assert limits.effective_memory == 8 * guard.GIB
    assert limits.memory_max == 6 * guard.GIB
    assert limits.cpu_quota_percent == 150
    assert limits.threads == 1
    assert limits.swap_max == guard.GIB
    fractional = guard.calculate_limits(8 * guard.GIB, 0, 4, cpu_caps=[.5])
    assert fractional.cpu_quota_percent == 50
    assert fractional.swap_max == 0
    tiny = guard.calculate_limits(8 * guard.GIB, 0, 4, cpu_caps=[.005])
    assert tiny.cpu_quota_percent == .5


def test_small_runner_fails_before_launch():
    with pytest.raises(guard.GuardSetupError, match="insufficient_memory_capacity"):
        guard.calculate_limits(2 * guard.GIB, 0, 1)


def test_all_unified_ancestors_are_inspected(tmp_path):
    root = tmp_path / "cgroups"
    leaf = root / "job" / "nested"
    leaf.mkdir(parents=True)
    membership = tmp_path / "membership"
    membership.write_text("0::/job/nested\n")
    (root / "memory.max").write_text(str(16 * guard.GIB))
    (root / "memory.swap.max").write_text("max")
    (root / "cpu.max").write_text("max 100000")
    (root / "job" / "memory.max").write_text(str(8 * guard.GIB))
    (root / "job" / "memory.swap.max").write_text("0")
    (leaf / "memory.max").write_text("max")
    (leaf / "cpu.max").write_text("150000 100000")
    ancestors = guard.cgroup_ancestors(root, membership)
    assert ancestors == [leaf, leaf.parent, root]
    memory, cpus, swap = guard.inherited_capacities(ancestors)
    assert min(memory) == 8 * guard.GIB
    assert cpus == [1.5]
    assert swap == [0]


def test_smoke_test_overrides_can_only_reduce_capacity():
    limits = guard.calculate_limits(16 * guard.GIB, 12 * guard.GIB, 4)
    smoke = guard.restrict_limits(limits, 192, 0)
    assert smoke.memory_max == 192 * guard.MIB
    assert 0 < smoke.memory_high < smoke.memory_max
    assert smoke.swap_max == 0
    assert smoke.host_reserve == limits.host_reserve
    assert smoke.cpu_quota_percent == limits.cpu_quota_percent
    for memory, swap in [(0, None), (-1, None), (100000, None),
                         (None, -1), (None, 100000)]:
        with pytest.raises(guard.GuardSetupError):
            guard.restrict_limits(limits, memory, swap)


def test_threads_are_bounded_without_logging_environment(monkeypatch):
    limits = guard.calculate_limits(16 * guard.GIB, 0, 4)
    monkeypatch.setenv("TTS_THREADS", "500")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("MKL_NUM_THREADS", "invalid")
    monkeypatch.setenv("APP_TOKEN", "environment-secret")
    environment = guard.workload_environment(limits)
    assert environment["TTS_THREADS"] == "3"
    assert environment["OMP_NUM_THREADS"] == "2"
    assert environment["MKL_NUM_THREADS"] == "3"
    assert environment["APP_TOKEN"] == "environment-secret"


def test_scope_restores_invoking_identity_and_path(monkeypatch, tmp_path):
    (tmp_path / "cgroup.controllers").write_text("memory cpu")
    monkeypatch.setattr(guard, "CGROUP_ROOT", tmp_path)
    monkeypatch.setattr(guard.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(guard.os, "getuid", lambda: 1001)
    monkeypatch.setattr(guard.os, "getgid", lambda: 1002)
    monkeypatch.setenv("PATH", "/custom/python/bin:/usr/bin")
    limits = guard.calculate_limits(16 * guard.GIB, 0, 4)
    command = guard.scope_command(["python", "-m", "movie_pipeline.cli"], "test.scope", limits)
    assert command[:5] == ["/usr/bin/sudo", "-n", "-E", "/usr/bin/systemd-run", "--scope"]
    assert "--property=MemoryOOMGroup=yes" in command
    assert "--property=MemoryMax=" + str(limits.memory_max) in command
    assert "--property=MemorySwapMax=0" in command
    assert "--property=CPUQuota=300%" in command
    assert "--reuid=1001" in command
    assert "--regid=1002" in command
    assert "--init-groups" in command
    assert "PATH=/custom/python/bin:/usr/bin" in command
    assert "HOME=" + str(Path.home()) in command
    assert command[-3:] == ["python", "-m", "movie_pipeline.cli"]


def test_probe_never_runs_real_application_or_exposes_failure_text(monkeypatch):
    calls = []
    monkeypatch.setattr(guard, "scope_command",
                        lambda command, unit, limits: ["scoped", *command])

    def rejected(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1, stderr="secret-command-argument")

    monkeypatch.setattr(guard.subprocess, "run", rejected)
    limits = guard.calculate_limits(16 * guard.GIB, 0, 4)
    with pytest.raises(guard.GuardSetupError, match="^scope_probe_rejected$"):
        guard.probe_scope(limits, {})
    assert calls == [["scoped", "/usr/bin/true"]]


class FakeChild:
    pid = 123456

    def __init__(self, status=0, polls=2):
        self.status = status
        self.returncode = None
        self.remaining = polls

    def poll(self):
        self.remaining -= 1
        if self.remaining <= 0 and self.returncode is None:
            self.returncode = self.status
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = self.status
        return self.returncode


@pytest.fixture
def supervisor(monkeypatch):
    state = SimpleNamespace(launches=[], stops=[], child=FakeChild())
    monkeypatch.setattr(guard, "detect_limits",
                        lambda: guard.calculate_limits(16 * guard.GIB, 0, 4))
    monkeypatch.setattr(guard, "read_meminfo", lambda: {"MemAvailable": 10 * guard.GIB})
    monkeypatch.setattr(guard, "probe_scope", lambda limits, environment: None)
    monkeypatch.setattr(guard, "scope_command", lambda command, unit, limits: command)
    monkeypatch.setattr(guard, "scope_path", lambda unit: None)
    monkeypatch.setattr(guard, "cgroup_sample", lambda path: {})
    monkeypatch.setattr(guard, "sample", lambda path: {"cgroup": {}})
    monkeypatch.setattr(guard, "disk_sample",
                        lambda: {"workspace": {"free_bytes": 20 * guard.GIB}})
    monkeypatch.setattr(guard.time, "sleep", lambda seconds: None)

    def launch(command, **kwargs):
        state.launches.append((command, kwargs))
        return state.child

    def stop(child, unit, signum=signal.SIGTERM, grace=20):
        state.stops.append((unit, signum))
        if child.returncode is None:
            child.returncode = -signum

    monkeypatch.setattr(guard.subprocess, "Popen", launch)
    monkeypatch.setattr(guard, "stop_workload", stop)
    return state


@pytest.mark.parametrize("status,expected", [(0, 0), (7, 7), (78, 78), (137, 137),
                                            (-9, 137), (-15, 143)])
def test_real_workload_executes_once_and_preserves_status(supervisor, tmp_path, status, expected):
    supervisor.child = FakeChild(status)
    assert guard.run_guard(["application", "argument-secret"], tmp_path) == expected
    assert len(supervisor.launches) == 1
    assert len(supervisor.stops) == 1


def test_missing_isolation_fails_closed_and_redacts_secrets(supervisor, monkeypatch,
                                                          tmp_path, capsys):
    monkeypatch.setenv("APP_TOKEN", "environment-secret")

    def unavailable(*args):
        raise guard.GuardSetupError("scope_probe_rejected")

    monkeypatch.setattr(guard, "probe_scope", unavailable)
    assert guard.run_guard(["application", "argument-secret"], tmp_path) == 78
    assert supervisor.launches == []
    recorded = (tmp_path / "resources.jsonl").read_text()
    output = str(capsys.readouterr())
    for secret in ("argument-secret", "environment-secret", "APP_TOKEN"):
        assert secret not in recorded + output
    assert "setup_failed" in recorded


def test_actions_cannot_opt_into_watchdog_only(supervisor, monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    def unavailable(*args):
        raise guard.GuardSetupError("scope_probe_rejected")

    monkeypatch.setattr(guard, "probe_scope", unavailable)
    assert guard.run_guard(["application"], tmp_path, allow_watchdog_fallback=True) == 78
    assert supervisor.launches == []


def test_explicit_local_watchdog_executes_once(supervisor, monkeypatch, tmp_path):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    def unavailable(*args):
        raise guard.GuardSetupError("scope_probe_rejected")

    monkeypatch.setattr(guard, "probe_scope", unavailable)
    supervisor.child = FakeChild(7)
    assert guard.run_guard(["application"], tmp_path, allow_watchdog_fallback=True) == 7
    assert len(supervisor.launches) == 1
    assert "watchdog_only" in (tmp_path / "resources.jsonl").read_text()


def test_sigterm_is_forwarded_and_reported(supervisor, monkeypatch, tmp_path):
    handlers = {}

    def install(signum, handler):
        old = handlers.get(signum)
        handlers[signum] = handler
        return old

    monkeypatch.setattr(guard.signal, "signal", install)
    supervisor.child = FakeChild(polls=100)
    monkeypatch.setattr(guard.time, "sleep",
                        lambda seconds: handlers[signal.SIGTERM](signal.SIGTERM, None))
    assert guard.run_guard(["application"], tmp_path) == 143
    assert supervisor.stops[0][1] == signal.SIGTERM
    assert len(supervisor.launches) == 1
    assert '"exit_code": 143' in (tmp_path / "resources.jsonl").read_text()


def test_sustained_host_pressure_stops_only_workload(supervisor, monkeypatch, tmp_path):
    available = iter([10 * guard.GIB, 100 * guard.MIB, 100 * guard.MIB, 100 * guard.MIB])
    monkeypatch.setattr(guard, "read_meminfo", lambda: {"MemAvailable": next(available)})
    supervisor.child = FakeChild(polls=100)
    assert guard.run_guard(["application"], tmp_path) == 75
    assert len(supervisor.launches) == 1
    assert supervisor.stops[0][0].startswith("movie-production-")
    assert "host_memory_pressure" in (tmp_path / "resources.jsonl").read_text()


def test_sustained_disk_pressure_stops_before_filling_runner(supervisor, monkeypatch, tmp_path):
    available = iter([20 * guard.GIB, 100 * guard.MIB, 100 * guard.MIB, 100 * guard.MIB])
    monkeypatch.setattr(guard, "disk_sample",
                        lambda: {"workspace": {"free_bytes": next(available)}})
    supervisor.child = FakeChild(polls=100)
    assert guard.run_guard(["application"], tmp_path) == 75
    assert len(supervisor.launches) == 1
    assert "host_disk_pressure" in (tmp_path / "resources.jsonl").read_text()


def test_last_cgroup_evidence_survives_disappearing_scope(supervisor, monkeypatch, tmp_path):
    values = iter([{"memory.events": {"oom_kill": 1}}, {}, {}])
    monkeypatch.setattr(guard, "cgroup_sample", lambda path: next(values, {}))
    supervisor.child = FakeChild(137)
    assert guard.run_guard(["application"], tmp_path) == 137
    assert '"oom_kill": 1' in (tmp_path / "resources.jsonl").read_text()


def test_scope_cleanup_reaches_daemonized_descendants(monkeypatch):
    calls = []
    child = FakeChild(polls=100)
    monkeypatch.setattr(guard, "systemctl", lambda args: calls.append(args))
    monkeypatch.setattr(guard.os, "killpg", lambda pid, signum: calls.append((pid, signum)))
    guard.stop_workload(child, "test.scope")
    assert ["kill", "--kill-whom=all", "--signal=15", "test.scope"] in calls
    assert ["kill", "--kill-whom=all", "--signal=KILL", "test.scope"] in calls
    assert (child.pid, signal.SIGTERM) in calls


def test_scope_cleanup_tolerates_root_launcher_process_group(monkeypatch):
    calls = []
    child = FakeChild(polls=100)
    monkeypatch.setattr(guard, "systemctl", lambda args: calls.append(args))

    def denied(pid, signum):
        raise PermissionError("root-owned launcher")

    monkeypatch.setattr(guard.os, "killpg", denied)
    guard.stop_workload(child, "test.scope")
    assert ["kill", "--kill-whom=all", "--signal=KILL", "test.scope"] in calls


def test_disk_sampling_counts_each_filesystem_once(monkeypatch, tmp_path):
    temporary = tmp_path / "temp"
    temporary.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUNNER_TEMP", str(temporary))
    disks = guard.disk_sample()
    assert set(disks) == {"workspace"}
    assert disks["workspace"]["free_bytes"] > 0


def test_ffmpeg_filter_parallelism_preserves_quality_arguments(monkeypatch):
    from movie_pipeline import core

    calls = []
    monkeypatch.setattr(core, "run", lambda args: calls.append(args))
    core.ffmpeg("-i", "input.mp4", "-c:v", "libx264", "-crf", "18", "output.mp4")
    args = calls[0]
    assert args[args.index("-filter_threads") + 1] == "2"
    assert args[args.index("-filter_complex_threads") + 1] == "2"
    assert args[-7:] == ["-i", "input.mp4", "-c:v", "libx264", "-crf", "18", "output.mp4"]
