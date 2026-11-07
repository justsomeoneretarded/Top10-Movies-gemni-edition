"""Generate the episode's instrumental score with ACE-Step 1.5 on the CPU runner.

Usage: python scripts/generate_music.py <ACE-Step checkout> <genre> <seconds> <output.wav>
The genre prompt comes from movie_pipeline.soundtrack.GENRE_MUSIC. Any failure leaves no
file, so production falls back to the bundled licensed tracks.
"""
from __future__ import annotations

import shutil
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def notice(title: str, message: str, level: str = "notice"):
    print(f"::{level} title={title}::{message.replace(chr(10), '%0A')}", flush=True)


def main() -> int:
    root, genre, seconds, output = sys.argv[1], sys.argv[2].strip().lower(), float(sys.argv[3]), Path(sys.argv[4])
    from movie_pipeline.soundtrack import GENRE_MUSIC
    caption = GENRE_MUSIC.get(genre, GENRE_MUSIC["neutral"])
    sys.path.insert(0, root)
    try:
        from acestep.handler import AceStepHandler
        from acestep.inference import GenerationConfig, GenerationParams, generate_music
        started = time.time()
        dit = AceStepHandler()
        status, ok = dit.initialize_service(project_root=root, config_path="acestep-v15-turbo", device="cpu")
        if not ok:
            raise RuntimeError(f"initialize_service: {status}")
        work = output.parent / "ace-output"
        params = GenerationParams(caption=caption, instrumental=True, lyrics="[Instrumental]",
                                  duration=seconds, inference_steps=8, seed=int(hashlib.sha256((genre+os.getenv("MUSIC_TOPIC","")).encode()).hexdigest()[:7],16), thinking=False)
        result = generate_music(dit, None, params, GenerationConfig(batch_size=1, audio_format="wav"), save_dir=str(work))
        if not result.success:
            raise RuntimeError(result.error)
        produced = sorted(work.rglob("*.wav"), key=lambda path: path.stat().st_size)
        if not produced:
            raise RuntimeError("no audio written")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(produced[-1], output)
        output.with_suffix(".json").write_text(json.dumps({"genre":genre,"prompt":caption,"topic":os.getenv("MUSIC_TOPIC",""),"seconds":seconds,"model":"ACE-Step 1.5"}))
        shutil.rmtree(work, ignore_errors=True)
        notice("Genre score", f"{genre}: {seconds:.0f}s generated in {(time.time()-started)/60:.1f} min. Prompt: {caption}")
        return 0
    except Exception as exc:  # noqa: BLE001 -- bundled music remains the fallback
        notice("Genre score unavailable", f"{type(exc).__name__}: {str(exc)[:500]}; the bundled track will be used.", "warning")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
