"""Discard downloaded and rendered media; retain a small, bounded speech/script cache."""
from __future__ import annotations

import shutil
from pathlib import Path

MAX_CHECKPOINT_BYTES = 2_000_000_000


def clean_work(root: Path = Path("work"), keep_jobs: int = 2) -> dict:
    # These paths are pipeline-owned scratch directories. Never touch output/ or models/.
    footage = root / "footage"
    if footage.exists():
        shutil.rmtree(footage)
    jobs = root / "jobs"
    retained = []
    if jobs.exists():
        for job in jobs.iterdir():
            if not job.is_dir() or job.is_symlink():
                continue
            for path in job.glob("render-*"):
                if path.is_dir():
                    shutil.rmtree(path)
            for path in job.rglob("*"):
                if path.is_file() and (path.suffix in {".mp4", ".tmp"} or path.name.endswith(".dry.wav") or path.name in {"music.wav", "effects.wav", "music.shaped.wav", "voice-before-pacing.wav", "voice-after-pacing.wav", "finished-mix.wav", "ending-music-only.wav", "background-only.wav", "edit-without-music.wav", "comparison-original.wav", "comparison-augmented.wav"}):
                    path.unlink(missing_ok=True)
            modified = max((p.stat().st_mtime for p in job.rglob("*") if p.is_file()), default=job.stat().st_mtime)
            size = sum(p.stat().st_size for p in job.rglob("*") if p.is_file())
            retained.append((modified, size, job))
        retained.sort(reverse=True)
        total = 0
        count = 0
        for _modified, size, job in retained:
            if count >= keep_jobs or total+size > MAX_CHECKPOINT_BYTES:
                shutil.rmtree(job)
            else:
                total += size
                count += 1
    else:
        count = total = 0
    print(f"Removed downloaded trailers and rendered intermediates; retained {count} job checkpoints ({total/1e6:.1f} MB).")
    return {"jobs": count, "bytes": total}


if __name__ == "__main__":
    clean_work()
