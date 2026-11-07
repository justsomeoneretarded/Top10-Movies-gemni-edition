"""Measure free/open models on the GitHub-hosted CPU runner.

Each subcommand writes samples to ``bench-output/<name>/`` and prints GitHub
annotations with timings, so results are readable from the run page.
Failures are reported as annotations instead of crashing the whole matrix.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

OUT = Path("bench-output")
REFERENCE = Path("assets/voices/warguy.ogg")

LINES = {
    "hype": "Number one. This is the movie that broke my brain in the best possible way, and it still holds up.",
    "normal": "A thief who steals secrets from dreams gets one last job: plant an idea instead of taking one.",
    "dramatic": "But the deeper they go, the harder it becomes to tell what's real... and what isn't.",
    "joke": "Like and subscribe. It's free. My movie opinions are the expensive part. [chuckle]",
}

FACTS = {
    "title": "Inception", "year": 2010, "rank": 1, "director": "Christopher Nolan",
    "cast": ["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Elliot Page", "Tom Hardy"],
    "genres": ["Action", "Science Fiction", "Adventure"], "runtime_minutes": 148,
    "overview": "Cobb, a skilled thief who commits corporate espionage by infiltrating the subconscious of his targets, "
                "is offered a chance to regain his old life as payment for a task considered to be impossible: "
                "inception, the implantation of another person's idea into a target's subconscious.",
    "tagline": "Your mind is the scene of the crime.",
    "production_notes": "The rotating hallway fight was filmed practically in a full-size rotating set.",
    "reception": "Won four Academy Awards, including Best Cinematography and Best Visual Effects.",
}


def note(title: str, message: str, level: str = "notice"):
    message = message.replace("%", "%25").replace("\r", "").replace("\n", "%0A")
    print(f"::{level} title={title}::{message}", flush=True)


def guarded(name):
    def wrap(func):
        def run(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # report and continue with other models
                note(f"{name} failed", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}", "error")
                return None
        return run
    return wrap


def save_wave(path: Path, samples, rate: int):
    import numpy as np
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(samples, dtype="float32").squeeze()
    sf.write(path, data, rate)
    return len(data) / rate


def reference_transcript() -> str:
    from faster_whisper import WhisperModel
    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(REFERENCE))
    return " ".join(segment.text.strip() for segment in segments)


# ---------------------------------------------------------------- LLMs
def llm(models: list[str]):
    import requests
    url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    prompt = (
        "Write the narration for the #1 entry of a YouTube top-10 movie countdown. "
        "Spoken, casual, punchy, 110-140 words, no markdown. Start with 'Number one:' and the title. "
        "Use a one-line hook, the premise, one or two specific facts from the data, and who should watch. "
        "Never invent facts beyond this JSON: " + json.dumps(FACTS)
    )
    out = OUT / "llm"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in models:
        @guarded(f"LLM {model}")
        def one():
            started = time.time()
            pull = requests.post(f"{url}/api/pull", json={"model": model, "stream": False}, timeout=3600)
            pull.raise_for_status()
            pulled = time.time() - started
            response = requests.post(f"{url}/api/generate", json={
                "model": model, "prompt": prompt, "stream": False, "think": False,
                "options": {"temperature": 0.7, "num_ctx": 8192, "num_predict": 600}}, timeout=3600)
            response.raise_for_status()
            data = response.json()
            text = data.get("response", "").strip()
            speed = data.get("eval_count", 0) / max(data.get("eval_duration", 1) / 1e9, 1e-6)
            prompt_speed = data.get("prompt_eval_count", 0) / max(data.get("prompt_eval_duration", 1) / 1e9, 1e-6)
            total = data.get("total_duration", 0) / 1e9
            safe = model.replace(":", "_").replace("/", "_")
            (out / f"{safe}.txt").write_text(text + "\n")
            rows.append({"model": model, "tokens_per_second": round(speed, 2),
                         "prompt_tokens_per_second": round(prompt_speed, 2),
                         "seconds_for_review": round(total, 1), "pull_seconds": round(pulled, 1),
                         "words": len(text.split())})
            note(f"LLM {model}", f"{speed:.1f} tokens/s generating, {prompt_speed:.0f} tokens/s reading; "
                 f"one review took {total:.0f}s ({len(text.split())} words). Pull {pulled:.0f}s.\n\n{text[:900]}")
            requests.post(f"{url}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
        one()
    (out / "results.json").write_text(json.dumps(rows, indent=2))


# ---------------------------------------------------------------- Chatterbox
def chatterbox():
    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    out = OUT / "chatterbox"
    rows = []

    @guarded("Chatterbox Turbo")
    def turbo():
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        started = time.time()
        model = ChatterboxTurboTTS.from_pretrained(device="cpu")
        model.prepare_conditionals(str(REFERENCE))
        load = time.time() - started
        for style, text in LINES.items():
            torch.manual_seed(7)
            begin = time.time()
            with torch.inference_mode():
                wave = model.generate(text)
            spent = time.time() - begin
            seconds = save_wave(out / f"turbo-{style}.wav", wave.cpu().numpy(), model.sr)
            rows.append({"model": "chatterbox-turbo", "style": style, "audio_seconds": round(seconds, 2),
                         "compute_seconds": round(spent, 2), "realtime_factor": round(spent / seconds, 2)})
        note("Chatterbox Turbo", f"load {load:.0f}s; " + "; ".join(
            f"{r['style']}: {r['realtime_factor']}x realtime" for r in rows if r["model"] == "chatterbox-turbo"))
        del model

    @guarded("Chatterbox original")
    def original():
        from chatterbox.tts import ChatterboxTTS
        started = time.time()
        model = ChatterboxTTS.from_pretrained(device="cpu")
        load = time.time() - started
        settings = {"hype": (0.75, 0.35), "normal": (0.5, 0.5), "dramatic": (0.8, 0.3)}
        for style, (exaggeration, cfg) in settings.items():
            torch.manual_seed(7)
            begin = time.time()
            with torch.inference_mode():
                wave = model.generate(LINES[style], audio_prompt_path=str(REFERENCE),
                                      exaggeration=exaggeration, cfg_weight=cfg)
            spent = time.time() - begin
            seconds = save_wave(out / f"original-{style}.wav", wave.cpu().numpy(), model.sr)
            rows.append({"model": "chatterbox-original", "style": style, "exaggeration": exaggeration,
                         "audio_seconds": round(seconds, 2), "compute_seconds": round(spent, 2),
                         "realtime_factor": round(spent / seconds, 2)})
        note("Chatterbox original", f"load {load:.0f}s; " + "; ".join(
            f"{r['style']} (exaggeration {r['exaggeration']}): {r['realtime_factor']}x realtime"
            for r in rows if r["model"] == "chatterbox-original"))
        del model

    turbo()
    original()
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(rows, indent=2))


# ---------------------------------------------------------------- Qwen3-TTS
def qwen_tts(sizes: list[str]):
    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    out = OUT / "qwen3-tts"
    rows = []
    ref_text = reference_transcript()
    note("Qwen3-TTS reference transcript", ref_text)
    for size in sizes:
        @guarded(f"Qwen3-TTS {size}")
        def one():
            from qwen_tts import Qwen3TTSModel
            started = time.time()
            model = Qwen3TTSModel.from_pretrained(f"Qwen/Qwen3-TTS-12Hz-{size}-Base", device_map="cpu",
                                                  dtype=torch.float32)
            prompt = model.create_voice_clone_prompt(ref_audio=str(REFERENCE), ref_text=ref_text,
                                                     x_vector_only_mode=False)
            load = time.time() - started
            for style, text in LINES.items():
                text = text.replace(" [chuckle]", "")
                begin = time.time()
                wavs, rate = model.generate_voice_clone(text=text, language="English", voice_clone_prompt=prompt)
                spent = time.time() - begin
                seconds = save_wave(out / f"{size}-{style}.wav", wavs[0], rate)
                rows.append({"model": f"qwen3-tts-{size}", "style": style, "audio_seconds": round(seconds, 2),
                             "compute_seconds": round(spent, 2), "realtime_factor": round(spent / seconds, 2)})
            note(f"Qwen3-TTS {size}", f"load {load:.0f}s; " + "; ".join(
                f"{r['style']}: {r['realtime_factor']}x realtime" for r in rows if r["model"] == f"qwen3-tts-{size}"))
        one()
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(rows, indent=2))


# ---------------------------------------------------------------- ACE-Step
def acestep(project_root: str, seconds: float):
    out = (OUT / "acestep").resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    sys.path.insert(0, project_root)

    @guarded("ACE-Step 1.5")
    def run():
        from acestep.handler import AceStepHandler
        from acestep.inference import GenerationConfig, GenerationParams, generate_music
        started = time.time()
        dit = AceStepHandler()
        status, ok = dit.initialize_service(project_root=project_root, config_path="acestep-v15-turbo", device="cpu")
        if not ok:
            raise RuntimeError(f"initialize_service: {status}")
        load = time.time() - started
        prompts = {
            "tense-trailer": "cinematic orchestral trailer underscore, tense staccato strings, low brass, "
                             "taiko pulses, building suspense, no vocals",
            "warm-drama": "emotional cinematic piano and soft strings, hopeful, slow build, no vocals",
        }
        for name, caption in prompts.items():
            begin = time.time()
            params = GenerationParams(caption=caption, instrumental=True, lyrics="[Instrumental]",
                                      duration=seconds, inference_steps=8, seed=11, thinking=False)
            config = GenerationConfig(batch_size=1, audio_format="wav")
            result = generate_music(dit, None, params, config, save_dir=str(out / name))
            spent = time.time() - begin
            if not result.success:
                raise RuntimeError(result.error)
            rows.append({"prompt": name, "audio_seconds": seconds, "compute_seconds": round(spent, 1),
                         "realtime_factor": round(spent / seconds, 2)})
        note("ACE-Step 1.5 (2B turbo, CPU)", f"load {load:.0f}s; " + "; ".join(
            f"{r['prompt']}: {r['compute_seconds']}s for {seconds:.0f}s of music ({r['realtime_factor']}x realtime)"
            for r in rows))

    run()
    (out / "results.json").write_text(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("llm").add_argument("models", nargs="+")
    sub.add_parser("chatterbox")
    sub.add_parser("qwen-tts").add_argument("sizes", nargs="+")
    music = sub.add_parser("acestep")
    music.add_argument("project_root")
    music.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()
    if args.command == "llm":
        llm(args.models)
    elif args.command == "chatterbox":
        chatterbox()
    elif args.command == "qwen-tts":
        qwen_tts(args.sizes)
    else:
        acestep(args.project_root, args.seconds)


if __name__ == "__main__":
    main()
