"""Keep GitHub-hosted model setup away from disk exhaustion without deleting caches."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

GIB = 1024**3
# Only unused SDKs on GitHub's disposable Ubuntu machines. Never delete the
# workspace, Python/tool cache, pip cache, model cache, or a self-hosted SDK.
UNUSED_SDKS = (
    Path("/usr/share/dotnet"),
    Path("/usr/local/lib/android"),
    Path("/opt/ghc"),
    Path("/opt/hostedtoolcache/CodeQL"),
)
MINIMUM_GIB = {
    "plan": 18,
    "script": 24,
    "footage": 14,
    "produce": 24,
    "ollama-install": 5,
    "script-model": 12,  # Gemma 4 12B is about 8 GB; keep room for logs/runtime.
    "voice-warm": 15,    # Leave room for the script model before optional voice installs.
    "ready": 3,
}

def github_hosted() -> bool:
    return (sys.platform.startswith("linux")
            and os.getenv("GITHUB_ACTIONS") == "true"
            and os.getenv("RUNNER_ENVIRONMENT") == "github-hosted")

def protected_paths(workspace: Path) -> list[Path]:
    paths = [workspace, Path.home()/".cache"/"pip",
             workspace/"work"/"models", workspace/"work"/"ollama-models",
             workspace/"work"/"jobs", workspace/"output",
             Path(sys.executable)]
    for name in ("PIP_CACHE_DIR","HF_HOME","TORCH_HOME","OLLAMA_MODELS","pythonLocation"):
        if os.getenv(name):
            paths.append(Path(os.environ[name]))
    return [path.resolve() for path in paths]

def safe_sdk(path: Path, protected: list[Path]) -> bool:
    if path not in UNUSED_SDKS or not path.is_absolute() or path.is_symlink():
        return False
    resolved = path.resolve()
    return not any(resolved == other or resolved in other.parents or other in resolved.parents
                   for other in protected)

def reclaim_unused_sdks(workspace: Path) -> list[str]:
    if not github_hosted():
        print("Disk cleanup: keeping SDKs on this non-hosted or unrecognized runner.",flush=True)
        return []
    protected = protected_paths(workspace)
    removed=[]
    for path in UNUSED_SDKS:
        if not path.exists() or not safe_sdk(path,protected):
            continue
        command = ["rm","-rf","--",str(path)]
        if os.geteuid() != 0:
            command = ["sudo","-n",*command]
        try:
            result=subprocess.run(command,check=False,capture_output=True,text=True,timeout=180)
            if result.returncode == 0:
                removed.append(str(path))
            else:
                print(f"Disk cleanup: unused SDK could not be removed: {path}",flush=True)
        except (OSError,subprocess.TimeoutExpired):
            print(f"Disk cleanup: unused SDK removal unavailable: {path}",flush=True)
    return removed

def existing_parent(path: Path) -> Path:
    path=path.expanduser().resolve()
    while not path.exists() and path.parent != path:
        path=path.parent
    return path

def volumes(workspace: Path) -> list[dict]:
    # Check every filesystem that the actual installations write to. A free /mnt
    # does not establish that /, the workspace, or the model directory has room.
    locations=[workspace,Path(sys.prefix),Path.home(),Path(os.getenv("RUNNER_TEMP","/tmp"))]
    locations += [Path(os.environ[name]) for name in
                  ("PIP_CACHE_DIR","HF_HOME","TORCH_HOME","OLLAMA_MODELS") if os.getenv(name)]
    seen=set();result=[]
    for location in locations:
        path=existing_parent(location)
        device=path.stat().st_dev
        if device in seen:
            continue
        seen.add(device)
        usage=shutil.disk_usage(path)
        row={"path":str(path),"free_gib":round(usage.free/GIB,3),"free_bytes":usage.free}
        try:
            row["free_inodes"]=os.statvfs(path).f_favail
        except (AttributeError,OSError):
            pass
        result.append(row)
    return result

def prepare(stage: str, workspace: Path | None = None, *, optional_voice: bool = False) -> dict:
    if stage not in MINIMUM_GIB:
        raise ValueError(f"Unknown disk preparation stage: {stage}")
    if optional_voice and stage != "voice-warm":
        raise ValueError("Optional mode is only for voice warm-up.")
    workspace=(workspace or Path.cwd()).resolve()
    before=volumes(workspace)
    removed=reclaim_unused_sdks(workspace)
    after=volumes(workspace)
    required=MINIMUM_GIB[stage]*GIB
    enough=all(item["free_bytes"]>=required for item in after)
    report={"stage":stage,"required_gib":MINIMUM_GIB[stage],"before":before,
            "after":after,"removed_unused_sdks":removed,"enough":enough}
    print("Runner disk: "+json.dumps(report,sort_keys=True),flush=True)
    if optional_voice and not enough:
        env_file=os.getenv("GITHUB_ENV")
        if env_file:
            with open(env_file,"a",encoding="utf-8") as stream:
                stream.write("VOICE_WARM=false\n")
        print("Optional voice warm-up deferred to final production to preserve script-model space; "
              "the selected narrator and final voice quality are unchanged.",flush=True)
    elif not enough:
        free=min(item["free_gib"] for item in after)
        raise RuntimeError(f"Not enough working disk before {stage}: {free:.1f} GiB free; "
                           f"{MINIMUM_GIB[stage]} GiB required. Caches were preserved. "
                           "Stopped before a large download could fill the runner.")
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=MINIMUM_GIB,required=True)
    parser.add_argument("--optional-voice",action="store_true")
    args=parser.parse_args()
    try:
        prepare(args.stage,optional_voice=args.optional_voice)
    except (RuntimeError,ValueError) as exc:
        print(f"Runner setup: {exc}",file=sys.stderr)
        return 1
    return 0

if __name__=="__main__":
    raise SystemExit(main())
