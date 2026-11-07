"""Storage checks use simulated disks and sandbox paths, never runner SDK deletion."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import runner_disk as disk

def usage(free):
    return SimpleNamespace(free=free*disk.GIB)

def test_non_hosted_runner_never_deletes_sdks(monkeypatch,tmp_path):
    monkeypatch.setenv("GITHUB_ACTIONS","true")
    monkeypatch.setenv("RUNNER_ENVIRONMENT","self-hosted")
    monkeypatch.setattr(disk.subprocess,"run",lambda *a,**k:pytest.fail("must not delete SDKs"))
    assert disk.reclaim_unused_sdks(tmp_path)==[]

def test_hosted_cleanup_only_removes_allowlisted_unused_sdk(monkeypatch,tmp_path):
    sdk=tmp_path/"unused-sdk";sdk.mkdir()
    cache=tmp_path/"models";cache.mkdir()
    link=tmp_path/"sdk-link";link.symlink_to(cache,target_is_directory=True)
    monkeypatch.setattr(disk,"UNUSED_SDKS",(sdk,cache,link))
    monkeypatch.setattr(disk,"github_hosted",lambda:True)
    monkeypatch.setattr(disk,"protected_paths",lambda workspace:[cache.resolve()])
    calls=[]
    monkeypatch.setattr(disk.subprocess,"run",lambda args,**kw:calls.append(args) or SimpleNamespace(returncode=0))
    assert disk.reclaim_unused_sdks(tmp_path)==[str(sdk)]
    assert len(calls)==1 and calls[0][-3:]==["-rf","--",str(sdk)]
    assert cache.exists() and link.exists()

def test_parent_of_protected_python_or_model_cache_is_not_a_cleanup_target(monkeypatch,tmp_path):
    sdk=tmp_path/"sdk";sdk.mkdir()
    protected=sdk/"python-cache"
    monkeypatch.setattr(disk,"UNUSED_SDKS",(sdk,))
    assert not disk.safe_sdk(sdk,[protected])
    assert not disk.safe_sdk(Path("/"),[protected])

def test_disk_is_measured_after_cleanup_and_cache_contents_untouched(monkeypatch,tmp_path):
    model=tmp_path/"work/models/weights.bin";model.parent.mkdir(parents=True);model.write_bytes(b"weights")
    pip=tmp_path/"pip-cache";pip.mkdir();(pip/"wheel.whl").write_bytes(b"cached")
    monkeypatch.setenv("PIP_CACHE_DIR",str(pip))
    free=[2]
    monkeypatch.setattr(disk.shutil,"disk_usage",lambda p:usage(free[0]))
    def cleanup(workspace):
        free[0]=30
        return ["/unused/sdk"]
    monkeypatch.setattr(disk,"reclaim_unused_sdks",cleanup)
    report=disk.prepare("script",tmp_path)
    assert report["before"][0]["free_gib"]==2
    assert report["after"][0]["free_gib"]==30 and report["enough"]
    assert model.read_bytes()==b"weights" and (pip/"wheel.whl").read_bytes()==b"cached"

def test_insufficient_disk_stops_before_model_download(monkeypatch,tmp_path):
    monkeypatch.setattr(disk.shutil,"disk_usage",lambda p:usage(4))
    monkeypatch.setattr(disk,"reclaim_unused_sdks",lambda workspace:[])
    with pytest.raises(RuntimeError,match="before script-model"):
        disk.prepare("script-model",tmp_path)

@pytest.mark.parametrize("model", ["original", "turbo", "nano"])
def test_low_space_voice_warmup_defers_without_changing_final_model(monkeypatch,tmp_path,model):
    env=tmp_path/"job-env";env.write_text("")
    monkeypatch.setenv("GITHUB_ENV",str(env))
    monkeypatch.setenv("TTS_MODEL",model)
    monkeypatch.setattr(disk.shutil,"disk_usage",lambda p:usage(13))
    monkeypatch.setattr(disk,"reclaim_unused_sdks",lambda workspace:[])
    result=disk.prepare("voice-warm",tmp_path,optional_voice=True)
    assert not result["enough"]
    assert env.read_text()=="VOICE_WARM=false\n"
    assert __import__("os").environ["TTS_MODEL"]==model

def test_unknown_optional_stage_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        disk.prepare("ready",tmp_path,optional_voice=True)

def test_missing_download_directory_uses_existing_parent(tmp_path):
    assert disk.existing_parent(tmp_path/"new"/"models")==tmp_path

def test_all_model_setup_jobs_prepare_disk_before_caches_or_installs():
    import yaml
    workflow=yaml.load(Path(".github/workflows/generate-video.yml").read_text(),Loader=yaml.BaseLoader)
    for name in ("plan","script","footage","produce"):
        steps=workflow["jobs"][name]["steps"]
        guard=next(i for i,s in enumerate(steps) if s.get("run")==f"python3 -m scripts.runner_disk --stage {name}")
        setup=next(i for i,s in enumerate(steps) if s.get("uses","").startswith("actions/setup-python"))
        assert guard<setup
        assert "zstd" in next(s["run"] for s in steps if s.get("name")=="System dependencies")
    steps=workflow["jobs"]["script"]["steps"]
    restore=next(i for i,s in enumerate(steps) if s.get("name")=="Restore voice model cache")
    guard=next(i for i,s in enumerate(steps) if s.get("name")=="Keep script-model space before optional voice setup")
    install=next(i for i,s in enumerate(steps) if s.get("name")=="Install the voice model")
    assert restore<guard<install
    assert "--optional-voice" in steps[guard]["run"]

def test_footage_starts_same_bounded_installer_without_pulling_script_model():
    import yaml
    workflow=yaml.load(Path(".github/workflows/generate-video.yml").read_text(),Loader=yaml.BaseLoader)
    steps=workflow["jobs"]["footage"]["steps"]
    startup=next(s for s in steps if s.get("name")=="Start local Ollama for scene descriptions")
    assert startup["run"]=="bash scripts/bootstrap_ollama.sh --serve-only"

def test_four_gib_is_enough_to_reuse_ready_models_but_not_to_install(monkeypatch,tmp_path):
    monkeypatch.setattr(disk.shutil,"disk_usage",lambda p:usage(4))
    monkeypatch.setattr(disk,"reclaim_unused_sdks",lambda workspace:[])
    assert disk.prepare("ready",tmp_path)["enough"]
    with pytest.raises(RuntimeError,match="before ollama-install"):
        disk.prepare("ollama-install",tmp_path)
