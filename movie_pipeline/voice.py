from __future__ import annotations

import gc
import os
import re
import json
from pathlib import Path

from .core import PipelineError, duration, ffmpeg, key


# Spoken forms for titles text-to-speech tends to mangle. Captions keep the written form.
PRONUNCIATIONS = {
    r"\bM3GAN\b": "Megan", r"\bSe7en\b": "Seven", r"\bThir13en\b": "Thirteen", r"\bWALL[-·]E\b": "Wall-E",
    r"\bvs\.?(?=\s)": "versus", r"\bPt\.\s*": "Part ", r"\bDr\.\s": "Doctor ", r"\bMr\.\s": "Mister ", r"\bMrs\.\s": "Missus ",
    r"\bSt\.\s(?=[A-Z])": "Saint ", r"\b(Part|Chapter|Episode)\s+II\b": r"\1 Two", r"\b(Part|Chapter|Episode)\s+III\b": r"\1 Three",
    r"\b(Part|Chapter|Episode)\s+IV\b": r"\1 Four", r"\b(Part|Chapter|Episode)\s+V\b": r"\1 Five", r"&": " and ",
    r"\b(\d{4})[–-](\d{4})\b": r"\1 to \2", r"\b(\d{4})[–-](\d{2})\b": r"\1 to \2",
}

# Complete thoughts are voiced together. Extra boundary silence is measured,
# rather than assigned mechanically by the final punctuation mark.
REVEAL_LEAD = 0.18

DEFAULT_TTS_MODEL = "original"
TTS_MODELS = ("original", "turbo", "nano")
# A single original model/reference voices every delivery. CFG stays positive:
# the pinned upstream original inference requires its conditioned/unconditioned batch.
ORIGINAL_DELIVERY = {
    "normal": (0.50, 0.50, 0.80),
    "calm": (0.40, 0.55, 0.75),
    "hype": (0.65, 0.45, 0.80),
    "dramatic": (0.60, 0.45, 0.80),
}
ORIGINAL_PHRASE_WORDS = 45  # upstream caps each generation at ~40 seconds of speech


def selected_tts_model() -> str:
    """One canonical selection for synthesis, warm-up, checkpoints and reporting."""
    name = os.getenv("TTS_MODEL", DEFAULT_TTS_MODEL).strip().casefold()
    if name not in TTS_MODELS:
        raise PipelineError("TTS_MODEL must be original, turbo or nano.")
    return name


def generation_settings(model: str, delivery: str, expression: bool = True) -> dict:
    if model != "original":
        return {}
    values = ORIGINAL_DELIVERY.get(delivery if expression else "normal", ORIGINAL_DELIVERY["normal"])
    return dict(zip(("exaggeration", "cfg_weight", "temperature"), values))


# Broadcast narrator chain, kept light so the voice stays natural: low cut, de-esser, gentle
# compression, a little warmth and presence, then loudness. (No gate: it clipped breaths and
# word tails; no soft clipper: it added a processed edge.)
VOICE_CHAIN = ("asetpts=PTS-STARTPTS,highpass=f=70,"
               "deesser=i=0.15:m=0.25:f=0.5:s=o,"
               "acompressor=threshold=-18dB:ratio=1.6:attack=15:release=180:makeup=1:knee=6,"
               "equalizer=f=3000:t=q:w=0.8:g=0.8,"
               "adelay={lead}:all=1,apad=pad_dur=0.26,loudnorm=I=-18:TP=-2:LRA=9")


def custom_pronunciations() -> dict[str, str]:
    """Your own respellings from pronunciations.json at the repository root, e.g.
    {"Villeneuve": "Vil-nuv", "Chalamet": "Shala-may"}; whole words, case-sensitive."""
    path = Path(__file__).resolve().parents[1] / "pronunciations.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {rf"\b{re.escape(str(word))}\b": str(spoken) for word, spoken in data.items() if str(word).strip()}


def spoken_form(text: str) -> str:
    """The text as the narrator should say it (respellings for known trouble titles and abbreviations)."""
    for pattern, replacement in {**PRONUNCIATIONS, **custom_pronunciations()}.items():
        text = re.sub(pattern, replacement, text)
    return " ".join(text.split())


def sentences(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘])", text) if part.strip()]
    merged = []
    for part in parts:
        # Very short fragments ('Number 3: Sinners.') stay separate only when they are reveals.
        if merged and len(part.split()) < 3 and not re.match(r"Number\s+\w+", part):
            merged[-1] += " " + part
        else:
            merged.append(part)
    return merged or [text]


def _bounded_thoughts(text: str, maximum: int = 90) -> list[str]:
    """Bound exceptional TTS input without dropping words or editing the script."""
    if len(text.split()) <= maximum:
        return [text]
    clauses = re.split(r"(?<=[,;:—])\s+", text)
    chunks: list[str] = []
    for clause in clauses:
        words = clause.split()
        while words:
            if chunks and len(chunks[-1].split()) + len(words) <= maximum:
                chunks[-1] += " " + " ".join(words)
                break
            if len(words) <= maximum:
                chunks.append(" ".join(words))
                break
            chunks.append(" ".join(words[:maximum]))
            words = words[maximum:]
    return chunks


def phrases(parts: list[str], reveal: bool = False, limit: int = 45, *, maximum: int = 90) -> list[str]:
    """Group complete thoughts; bound unusually long sentences for TTS transport."""
    grouped: list[str] = []
    for part in (chunk for text in parts for chunk in _bounded_thoughts(text, maximum=maximum)):
        last = grouped[-1] if grouped else None
        if (last is not None and len(re.findall(r"[.!?]", last)) < 3 and len(last.split()) + len(part.split()) <= limit):
            grouped[-1] = f"{last} {part}"
        else:
            grouped.append(part)
    return grouped


def thought_pause(previous, following, rate: int, text: str, delivery: str = "normal") -> float:
    """Add only the breath room missing between two complete model phrases."""
    import numpy as np

    if rate <= 0 or not len(previous) or not len(following):
        return 0.0

    def margins(samples):
        signal = np.asarray(samples)
        loud = np.flatnonzero(np.abs(signal) > max(1e-4, float(np.max(np.abs(signal)))*.02))
        return (len(signal)/rate, len(signal)/rate) if not loud.size else (loud[0]/rate, (len(signal)-1-loud[-1])/rate)

    existing = margins(previous)[1] + margins(following)[0]
    word_seconds = len(previous)/rate/max(1, len(text.split()))
    preference = {"calm": 1.15, "dramatic": 1.1, "hype": .9}.get(delivery, 1.0)
    desired = min(.28, max(.10, word_seconds*.55*preference))
    return max(0.0, desired-existing)


def take_score(samples, rate: int, words: int) -> float:
    """Higher is better: natural speed, no long dead air inside, lively (not flat) energy."""
    import numpy as np
    if samples.size < rate * 0.3:
        return float("-inf")
    frame = max(1, rate // 50)
    usable = samples[: samples.size // frame * frame].reshape(-1, frame)
    energy = 20 * np.log10(np.sqrt((usable ** 2).mean(axis=1)) + 1e-6)
    voiced = energy > energy.max() - 35
    first, last = (np.argmax(voiced), len(voiced) - np.argmax(voiced[::-1])) if voiced.any() else (0, len(voiced))
    inside = ~voiced[first:last]
    longest, run = 0, 0
    for silent in inside:
        run = run + 1 if silent else 0
        longest = max(longest, run)
    seconds = (last - first) / 50
    wpm = words * 60 / max(0.3, seconds)
    liveliness = float(np.std(energy[first:last][voiced[first:last]])) if voiced[first:last].any() else 0.0
    # Prefer clean, complete speech; do not reward forced excitement or one exact speaking rate.
    pace_cost = max(0, 90-wpm, wpm-230) / 70
    clipping_cost = float(np.mean(np.abs(samples) >= .99)) * 40
    return -pace_cost - clipping_cost - max(0, longest / 50 - 0.9) * 3 + min(liveliness, 5) / 10


class Narrator:
    def __init__(self, reference: Path, cache: Path):
        import torch
        self.model_name = selected_tts_model()
        torch.set_num_threads(int(os.getenv("TTS_THREADS", str(os.cpu_count() or 2))))
        if self.model_name == "original":
            from chatterbox.tts import ChatterboxTTS
            self.model = ChatterboxTTS.from_pretrained(device="cpu")
        else:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            self.model = ChatterboxTurboTTS.from_pretrained(device="cpu", nano=self.model_name == "nano")
        # Original conditioning uses the same reference at every delivery. Never
        # select a separate "-dark" recording or load another model for one line.
        with torch.inference_mode():
            if self.model_name == "original":
                self.model.prepare_conditionals(str(reference), exaggeration=.5)
            else:
                self.model.prepare_conditionals(str(reference))
        self.reference = reference
        self.reference_hash = key(reference.read_bytes().hex())
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.emotion_enabled = (self.model_name == "original"
            and os.getenv("VOICE_EMOTION", "true").strip().lower() == "true")
        self.takes = max(1, min(3, int(os.getenv("VOICE_TAKES", "2"))))
        self.deliveries: dict[str, str] = {}

    def _generate(self, text: str, delivery: str, seed: int):
        import torch
        torch.manual_seed(seed)
        settings = generation_settings(self.model_name, delivery, self.emotion_enabled)
        with torch.inference_mode():
            wave = self.model.generate(text, **settings)
            return wave.squeeze().detach().cpu().numpy(), self.model.sr

    @staticmethod
    def _trim(samples, rate: int):
        import numpy as np
        loud = np.flatnonzero(np.abs(samples) > max(1e-4, float(np.abs(samples).max()) * 0.02))
        if not loud.size:
            return samples[:0]
        margin = int(rate * 0.04)
        return samples[max(0, loud[0] - margin): min(samples.size, loud[-1] + margin)]

    def speak(self, text: str, attempt_offset: int = 0, *, force: bool = False, delivery: str | None = None) -> Path:
        import numpy as np
        import soundfile as sf
        if delivery:
            self.deliveries[text] = delivery
        delivery = delivery or self.deliveries.get(text, "normal")
        settings = generation_settings(self.model_name, delivery, self.emotion_enabled)
        maximum = ORIGINAL_PHRASE_WORDS if self.model_name == "original" else 90
        baseline = key("voice-v8-original-consistent", spoken_form(text), self.reference_hash,
                       self.model_name, delivery, settings, maximum, self.takes)
        token = key("voice-retry-v1", baseline, attempt_offset) if force else baseline
        output = self.cache / f"{token}.wav"
        suffix = 0
        if not force and output.exists() and duration(output) > 0.5:
            return output
        if force:
            # A later use of the same text must not overwrite already-verified
            # audio or captions belonging to another paragraph.
            while output.exists() or output.with_suffix(".captions.json").exists():
                suffix += 1
                output = self.cache / f"{token}-{suffix}.wav"
        reveal = bool(re.match(r"\s*Number\s+\w+\s*:", text))
        for attempt in range(3):
            pieces, rate = [], None
            previous_text = ""
            seed_base = int(baseline[:8], 16) + attempt * 7919 + attempt_offset + suffix * 1009
            try:
                # Sentences are voiced in natural phrases of up to three sentences, so intonation flows
                # across a full stop instead of resetting on every sentence.
                for index, sentence in enumerate(phrases(sentences(spoken_form(text)), reveal, maximum=maximum)):
                    best, best_score = None, float("-inf")
                    # Keep complete takes and the same selected model through bounded retries.
                    for take in range(self.takes):
                        samples, rate = self._generate(sentence, delivery, seed_base + index * 131 + take * 17)
                        if not np.isfinite(samples).all() or not samples.size or np.max(np.abs(samples)) < 1e-5:
                            continue
                        samples = self._trim(samples.astype(np.float32), rate)
                        score = take_score(samples, rate, len(sentence.split()))
                        if score > best_score:
                            best, best_score = samples, score
                    if best is None:
                        raise ValueError("silent take")
                    if pieces:
                        gap = thought_pause(pieces[-1], best, rate, previous_text, delivery)
                        if gap:
                            pieces.append(np.zeros(round(rate*gap), dtype=np.float32))
                    pieces.append(best)
                    previous_text = sentence
            except (ValueError, RuntimeError) as exc:
                print(f"Voice attempt {attempt+1}: unusable take ({type(exc).__name__}); retrying.", flush=True)
                continue
            samples = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
            raw = output.with_suffix(".raw.wav")
            sf.write(raw, samples, rate, subtype="PCM_24")
            # Keep the first/last phonemes clear of encoder and edit boundaries; a reveal gets a
            # short silence first so the rank lands. Padding adds real silence, never trims speech.
            lead = int((REVEAL_LEAD if reveal else 0.16) * 1000)
            ffmpeg("-i", raw, "-af", VOICE_CHAIN.format(lead=lead), "-ar", "48000", "-ac", "1", output)
            # Keep the unprocessed take for matched diagnostics; cleanup removes these copies.
            try:
                import shutil
                shutil.copyfile(raw, output.with_suffix(".dry.wav"))
            except OSError:
                pass
            raw.unlink(missing_ok=True)
            seconds = duration(output)
            # Rate is diagnostic only. Actual missing/repeated speech is checked by transcription.
            if 0.15 <= seconds <= max(30, len(text.split())*2+10):
                wpm = len(text.split()) * 60 / seconds
                print(f"Voice generated ({self.model_name}, {delivery}): "
                      f"{len(text.split())} words, {seconds:.1f}s, {wpm:.0f} words/min; transcript verification follows.", flush=True)
                return output
            print(f"Voice attempt {attempt+1}: unusable audio duration {seconds:.1f}s; retrying.", flush=True)
            output.unlink(missing_ok=True)
        raise PipelineError("Voice generation returned empty, silent, invalid, or runaway audio after three attempts. Speaking speed alone does not cause this failure.")

    def close(self):
        del self.model
        gc.collect()


def captions(audio: Path, model, *, vad_filter: bool = True, beam_size: int = 3,
             condition_on_previous_text: bool = False) -> list[dict]:
    segments, _ = model.transcribe(str(audio), language="en", beam_size=beam_size, word_timestamps=True,
                                 vad_filter=vad_filter, condition_on_previous_text=condition_on_previous_text)
    result = []
    for segment in segments:
        words = segment.words or []
        chunk = []
        for word in words:
            chunk.append(word)
            if len(" ".join(w.word.strip() for w in chunk)) >= 55 or word.end - chunk[0].start >= 3.5:
                result.append({"start": chunk[0].start, "end": chunk[-1].end, "text": " ".join(w.word.strip() for w in chunk)})
                chunk = []
        if chunk:
            result.append({"start": chunk[0].start, "end": chunk[-1].end, "text": " ".join(w.word.strip() for w in chunk)})
        if not words and segment.text.strip():
            result.append({"start": segment.start, "end": segment.end, "text": segment.text.strip()})
    return result


def spoken_coverage(expected: str, cues: list[dict]) -> float:
    from difflib import SequenceMatcher

    from .core import normalized
    a = normalized(spoken_form(expected)).split()
    b = normalized(" ".join(c["text"] for c in cues)).split()
    matched = sum(block.size for block in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
    return matched / max(1, len(a))


_NUMBER_WORDS = dict(zip(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"],
    range(20),
))
_NUMBER_WORDS.update(dict(zip(["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"], range(20, 100, 10))))


def _under_hundred(words: list[str]) -> int | None:
    values = [_NUMBER_WORDS.get(word, int(word) if word.isdigit() else -1) for word in words]
    if len(values) == 1 and 0 <= values[0] < 100:
        return values[0]
    if len(values) == 2 and values[0] in range(20, 100, 10) and 0 < values[1] < 10:
        return sum(values)
    return None


def _under_thousand(words: list[str]) -> int | None:
    if len(words) >= 2 and words[1] == "hundred":
        prefix = _under_hundred(words[:1])
        suffix = _under_hundred(words[2:]) if len(words) > 2 else 0
        if prefix is not None and prefix > 0 and suffix is not None:
            return prefix*100 + suffix
    return _under_hundred(words)


def _number_value(words: list[str]) -> int | None:
    """Parse valid number phrases; adjacent independent numbers never get added."""
    words = [word for word in words if word != "and"]
    if len(words) == 1:
        if words[0].isdigit():
            return int(words[0])
        return {"hundred": 100, "thousand": 1000}.get(words[0], _NUMBER_WORDS.get(words[0]))
    first = _under_hundred(words[:1])
    if first in (18, 19, 20):
        remainder = _under_hundred(words[1:])
        if remainder is not None and 10 <= remainder < 100:
            return first*100 + remainder
    if words.count("thousand") == 1:
        split = words.index("thousand")
        prefix = _under_thousand(words[:split])
        suffix = _under_thousand(words[split+1:]) if split+1 < len(words) else 0
        if prefix is not None and prefix > 0 and suffix is not None:
            return prefix*1000 + suffix
    return _under_thousand(words)


def _speech_tokens(text: str) -> list[str]:
    """Ignore punctuation and tolerate ASR contraction/number formatting changes."""
    text = text.casefold().replace("’", "'")
    for contraction, expanded in {
        "can't": "can not", "cannot": "can not", "won't": "will not", "let's": "let us",
        "i'm": "i am", "it's": "it is", "that's": "that is", "there's": "there is",
        "here's": "here is", "what's": "what is", "who's": "who is", "he's": "he is", "she's": "she is",
    }.items():
        text = re.sub(r"\b" + re.escape(contraction) + r"\b", expanded, text)
    for ending, expanded in (("n't", " not"), ("'re", " are"), ("'ve", " have"), ("'ll", " will"), ("'d", " would")):
        text = re.sub(r"\b(\w+)" + re.escape(ending) + r"\b", r"\1" + expanded, text)
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    # Sentence/clause punctuation ends a number phrase. Hyphens still permit
    # ordinary compounds such as twenty-six.
    words = re.findall(r"\w+|[.!?;:,\n]", text)
    tokens = []
    index = 0
    while index < len(words):
        if not re.fullmatch(r"\w+", words[index]):
            index += 1
            continue
        start = index
        numeric = []
        while index < len(words):
            word = words[index]
            if word in _NUMBER_WORDS or word in ("hundred", "thousand") or word.isdigit():
                numeric.append(word)
                index += 1
            elif numeric and word == "and" and index+1 < len(words) and words[index+1] in _NUMBER_WORDS:
                index += 1
            else:
                break
        if numeric:
            # Partition a numeric run into valid phrases, preserving lists such
            # as one two three instead of treating them as the number six.
            while numeric:
                for stop in range(len(numeric), 0, -1):
                    value = _number_value(numeric[:stop])
                    if value is not None:
                        tokens.append(str(value))
                        numeric = numeric[stop:]
                        break
                else:
                    tokens.append(numeric.pop(0))
        else:
            tokens.append(words[start])
            index += 1
    return tokens


def _number_spelling(value: int) -> str:
    small = list(_NUMBER_WORDS)
    if value < 20:
        return small[value]
    if value < 100:
        return small[18+value//10] + (_number_spelling(value % 10) if value % 10 else "")
    if value < 1000:
        return _number_spelling(value//100) + "hundred" + (_number_spelling(value % 100) if value % 100 else "")
    if value < 10000:
        return _number_spelling(value//1000) + "thousand" + (_number_spelling(value % 1000) if value % 1000 else "")
    return str(value)


def _aligned_tokens(planned: list[str], heard: list[str]) -> tuple[set[int], set[int]]:
    """Monotone word alignment with at most a three-token spelling join/split."""
    def spellings(token):
        if not token.isdigit():
            return {token}
        value = int(token)
        forms = {_number_spelling(value)}
        if 1000 <= value <= 2099:
            century, remainder = divmod(value, 100)
            prefix = _number_spelling(century)
            if remainder == 0:
                forms.add(prefix+"hundred")
            elif remainder < 10:
                # Preserve the zero: "twenty one" means 21, not 2001.
                forms.update(prefix+zero+_number_spelling(remainder) for zero in ("oh", "zero"))
            else:
                forms.add(prefix+_number_spelling(remainder))
        return forms

    def spans(tokens):
        result = {}
        for start in range(len(tokens)):
            forms = {""}
            for size in range(1, min(3, len(tokens)-start)+1):
                forms = {prefix+suffix for prefix in forms for suffix in spellings(tokens[start+size-1])}
                result[start, size] = forms
        return result

    left, right = spans(planned), spans(heard)
    scores = [[(0, 0) for _ in range(len(heard)+1)] for _ in range(len(planned)+1)]
    trace = {}

    def update(i, j, score, step):
        if score > scores[i][j]:
            scores[i][j] = score
            trace[(i, j)] = step

    for i in range(len(planned)+1):
        for j in range(len(heard)+1):
            score = scores[i][j]
            if i < len(planned):
                update(i+1, j, score, (i, j, 0, 0))
            if j < len(heard):
                update(i, j+1, score, (i, j, 0, 0))
            for a, b in ((1, 1), (1, 2), (1, 3), (2, 1), (3, 1)):
                if i+a <= len(planned) and j+b <= len(heard) and not left[i, a].isdisjoint(right[j, b]):
                    update(i+a, j+b, (score[0]+a, score[1]+b), (i, j, a, b))
    expected_matches, heard_matches = set(), set()
    position = (len(planned), len(heard))
    while position in trace:
        i, j, a, b = trace[position]
        expected_matches.update(range(i, i+a))
        heard_matches.update(range(j, j+b))
        position = (i, j)
    return expected_matches, heard_matches


def speech_check(expected: str, cues: list[dict]) -> dict:
    """Report completeness evidence without requiring word-perfect ASR.

    The original coverage and boundary requirements are retained. Bounded
    spelling joins tolerate ASR word segmentation; transcript precision rejects
    substantial extra/repeated speech instead of accepting a complete sentence
    merely because it appears somewhere inside a runaway recording.
    """
    # The verifier must compare the pronunciation actually sent to TTS. Keep
    # the written transcript unchanged in diagnostics and captions.
    planned = _speech_tokens(spoken_form(expected))
    transcript = " ".join(cue.get("text", "") for cue in cues)
    heard = _speech_tokens(spoken_form(transcript))
    absent_edge = {"accepted": False, "matched": 0, "required": max(1, (min(8, len(planned))+1)//2),
                   "unmatched_tokens": len(planned)}
    result = {"accepted": False, "coverage": 0.0, "precision": 0.0,
              "reason": "empty_expected" if not planned else "empty_transcript", "heard": transcript,
              "opening": dict(absent_edge), "ending": dict(absent_edge),
              "longest_unmatched_phrase": len(planned)}
    if not planned or not heard:
        return result
    matched, heard_matches = _aligned_tokens(planned, heard)
    result["coverage"] = len(matched)/len(planned)
    result["precision"] = len(heard_matches)/len(heard)
    longest_missing = current_missing = 0
    for index in range(len(planned)):
        current_missing = 0 if index in matched else current_missing+1
        longest_missing = max(longest_missing, current_missing)
    result["longest_unmatched_phrase"] = longest_missing
    edge_allowance = min(2, max(1, len(planned)//8))
    window = min(8, len(planned))
    required = max(1, (window+1)//2)
    for name, indices, omitted in (
        ("opening", range(window), min(matched) if matched else len(planned)),
        ("ending", range(len(planned)-window, len(planned)), len(planned)-1-max(matched) if matched else len(planned)),
    ):
        count = sum(index in matched for index in indices)
        result[name] = {"accepted": omitted <= edge_allowance and count >= required,
                        "matched": count, "required": required, "unmatched_tokens": omitted}
    if result["coverage"] < .72:
        result["reason"] = "coverage"
    elif not result["opening"]["accepted"]:
        result["reason"] = "opening"
    elif not result["ending"]["accepted"]:
        result["reason"] = "ending"
    elif longest_missing >= 6:
        result["reason"] = "missing_or_reordered_phrase"
    elif len(heard) > len(planned)+2 and result["precision"] < .72:
        result["reason"] = "extra_or_repeated_speech"
    else:
        result.update(accepted=True, reason="accepted")
    return result


def speech_verified(expected: str, cues: list[dict]) -> bool:
    """Compatibility wrapper for callers that only need the acceptance decision."""
    return speech_check(expected, cues)["accepted"]


def paced_take(source: Path, speed: float) -> Path:
    """Pitch-preserving narration pacing before edit and subtitle assembly."""
    from .render import _strict_audio_decode
    if speed == 1:
        return source
    output = source.with_name(source.stem+f"-pace-v3-{speed:.3f}.wav")
    if not output.exists():
        tempo = f"atempo={speed}"  # Preserve articulation without spectral/formant reconstruction.
        ffmpeg("-i", source, "-af", f"{tempo},asetpts=N/SR/TB",
               "-ar", "48000", "-ac", "1", "-c:a", "pcm_s24le", output)
    _strict_audio_decode(output)
    return output
