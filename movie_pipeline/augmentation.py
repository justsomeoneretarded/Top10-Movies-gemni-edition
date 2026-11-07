"""Optional A/B augmentation. This is not anonymization or claim protection.

Both exports keep identical frame counts, audio lengths, chapters and captions.
Visual fields are fixed across frames to avoid random per-frame flicker. Audio
processing uses bounded overlapping windows instead of loading a full film.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from .core import PipelineError, ffmpeg, probe, save_json


def perlin_noise(height: int, width: int, cells: int = 8, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, (cells + 1, cells + 1))
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    y, x = np.meshgrid(np.linspace(0, cells, height, endpoint=False),
                       np.linspace(0, cells, width, endpoint=False), indexing="ij")
    ix, iy = x.astype(int), y.astype(int)
    fx, fy = x - ix, y - iy
    def dot(dx, dy):
        g = gradients[iy + dy, ix + dx]
        return g[..., 0] * (fx - dx) + g[..., 1] * (fy - dy)
    def fade(t):
        return t**3 * (t * (t * 6 - 15) + 10)
    u, v = fade(fx), fade(fy)
    return ((1-v) * ((1-u)*dot(0, 0) + u*dot(1, 0))
            + v * ((1-u)*dot(0, 1) + u*dot(1, 1))).astype(np.float32)


class FrameAugmenter:
    """Subpixel warp, faint gradient noise and 1% local disjoint pixel swaps."""
    def __init__(self, height: int, width: int, seed: int = 42):
        self.shape = height, width
        y, x = np.mgrid[:height, :width].astype(np.float32)
        def field(offset):
            return cv2.resize(perlin_noise(90, 160, seed=seed+offset),
                              (width, height), interpolation=cv2.INTER_CUBIC)
        self.map_x, self.map_y = x + .6 * field(1), y + .6 * field(2)
        self.noise = field(3)[..., None] * 1.5
        rng = np.random.default_rng(seed)
        occupied = set()
        first, second = [], []
        # Local swaps limit damage; arbitrary distant swaps create colored specks.
        target = (height * width // 100) // 2
        for value in rng.integers(0, height * width, size=target * 4):
            a = int(value)
            if a in occupied:
                continue
            yy, xx = divmod(a, width)
            ny = int(np.clip(yy+rng.integers(-2, 3), 0, height-1))
            nx = int(np.clip(xx+rng.integers(-2, 3), 0, width-1))
            b = ny * width + nx
            if b == a or b in occupied:
                continue
            first.append(a)
            second.append(b)
            occupied.update((a, b))
            if len(first) >= target:
                break
        self.first, self.second = np.array(first, dtype=np.intp), np.array(second, dtype=np.intp)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[:2] != self.shape or frame.dtype != np.uint8:
            raise ValueError("Expected a matching uint8 frame")
        result = cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
        result = np.clip(np.rint(result.astype(np.float32)+self.noise), 0, 255).astype(np.uint8)
        flat = result.reshape(-1, result.shape[-1])
        flat[self.first], flat[self.second] = flat[self.second].copy(), flat[self.first].copy()
        return result


def augment_audio_block(samples: np.ndarray, rate: int) -> np.ndarray:
    """Smooth variable tempo plus approximate +0.15-semitone envelope shift.

    Endpoints are anchored: no accumulating subtitle or chapter drift. The
    maximum local tempo deviation is 0.8%; pitch is retained by a phase vocoder.
    This is a spectral-envelope approximation, not a speech vocoder.
    """
    import librosa
    from scipy.ndimage import gaussian_filter1d, map_coordinates

    n_fft, hop = 2048, 256
    result = []
    for channel in samples.T:
        spectrum = librosa.stft(channel, n_fft=n_fft, hop_length=hop)
        n = spectrum.shape[1]
        t = np.linspace(0, 1, n)
        positions = np.arange(n) + .008 * (n-1)/(2*np.pi) * np.sin(2*np.pi*t)
        positions = np.clip(positions, 0, n-1)
        padded = np.pad(spectrum, ((0, 0), (0, 2)))
        phase = np.angle(spectrum[:, 0])
        advance = 2*np.pi*hop*np.arange(spectrum.shape[0])/n_fft
        output = np.empty_like(spectrum)
        for j, step in enumerate(positions):
            i = int(step)
            fraction = step-i
            a, b = padded[:, i], padded[:, i+1]
            magnitude = (1-fraction)*np.abs(a)+fraction*np.abs(b)
            output[:, j] = magnitude*np.exp(1j*phase)
            residual = np.angle(b)-np.angle(a)-advance
            residual -= 2*np.pi*np.round(residual/(2*np.pi))
            phase += advance+residual
        envelope = gaussian_filter1d(np.log(np.maximum(np.abs(output), 1e-7)), 12, axis=0)
        f, time = np.meshgrid(np.arange(output.shape[0])/2**(.15/12), np.arange(n), indexing="ij")
        shifted = map_coordinates(envelope, [f, time], order=1, mode="nearest")
        gain = np.clip(np.exp(shifted-envelope), .8, 1.25)
        result.append(librosa.istft(output*gain, hop_length=hop, length=len(channel)))
    return np.stack(result, axis=1).astype(np.float32)


def augment_audio(source: Path, target: Path, *, protected_intervals: list[tuple[float, float]] | None = None):
    import soundfile as sf
    with sf.SoundFile(source) as reader, sf.SoundFile(target, "w", samplerate=reader.samplerate,
                  channels=reader.channels, subtype="PCM_24") as writer:
        rate = reader.samplerate
        block, overlap = 8*rate, rate
        step = block-overlap
        pending = None
        position = 0
        while position < len(reader):
            reader.seek(position)
            samples = reader.read(block, dtype="float32", always_2d=True)
            weight = None
            if protected_intervals:
                time = (np.arange(len(samples))+position)/rate
                weight = np.ones(len(samples), dtype=np.float32)
                for low, high in protected_intervals:
                    if high < position/rate-.12 or low > (position+len(samples))/rate+.12:
                        continue
                    # Transition outside speech; every protected sample is untouched.
                    allowed = np.clip((low-time)/.12, 0, 1)+np.clip((time-high)/.12, 0, 1)
                    weight = np.minimum(weight, allowed)
            changed = samples.copy() if weight is not None and not np.any(weight) else augment_audio_block(samples, rate)
            if weight is not None:
                changed = samples+(changed-samples)*weight[:, None]
            if pending is not None:
                count = min(len(pending), len(changed), overlap)
                weight = np.linspace(0, 1, count, dtype=np.float32)[:, None]
                changed[:count] = pending[:count]*(1-weight)+changed[:count]*weight
            if position+block >= len(reader):
                writer.write(np.clip(changed, -.99, .99))
                break
            writer.write(np.clip(changed[:step], -.99, .99))
            pending = changed[step:]
            position += step


def export_augmented(source: Path, target: Path, work: Path, *, protected_intervals: list[tuple[float, float]] | None = None):
    """Stream modified frames to FFmpeg; never create a raw intermediate video."""
    from .render import _strict_audio_decode, validate_audio_video_duration
    work.mkdir(parents=True, exist_ok=True)
    metadata = probe(source)
    video = next(s for s in metadata["streams"] if s["codec_type"] == "video")
    expected = float(metadata["format"]["duration"])
    width, height = video["width"], video["height"]
    rate = video.get("avg_frame_rate", "30/1")
    original, changed = work / "comparison-original.wav", work / "comparison-augmented.wav"
    temporary = target.with_name(target.stem+".partial.mp4")
    cap = cv2.VideoCapture(str(source))
    process = None
    frames = 0
    try:
        ffmpeg("-i", source, "-vn", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s24le", original)
        augment_audio(original, changed, protected_intervals=protected_intervals)
        augmenter = FrameAugmenter(height, width)
        with (work / "augmentation-encode.log").open("w+") as log:
            process = subprocess.Popen([
                "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", rate, "-i", "pipe:0", "-i", str(changed),
                "-i", str(source), "-map", "0:v", "-map", "1:a", "-map_metadata", "2", "-map_chapters", "2",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                "-threads", "2", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(temporary)
            ], stdin=subprocess.PIPE, stderr=log)
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    process.stdin.write(augmenter(frame).tobytes())
                    frames += 1
                    if frames % 1800 == 0:
                        print(f"Augmentation: {frames} frames processed.", flush=True)
            finally:
                process.stdin.close()
            if process.wait() != 0:
                log.seek(0)
                raise PipelineError("Comparison encode failed: "+log.read()[-1000:])
        # Check full source frame coverage, including truncated-decode failures.
        expected_frames = int(video.get("nb_frames", frames))
        if frames != expected_frames:
            raise PipelineError("Comparison export did not decode all source frames.")
        _strict_audio_decode(temporary)
        validate_audio_video_duration(temporary, expected)
        ffmpeg("-xerror", "-i", temporary, "-f", "null", "-")
        temporary.replace(target)
        report = {"frames": frames, "seconds": expected, "seed": 42,
                  "warp_pixels": .6, "noise_amplitude_8bit": 1.5,
                  "swap_fraction": 2*len(augmenter.first)/(width*height),
                  "variable_tempo": .008, "formant_envelope_semitones": .15,
                  "same_timeline": True, "narration_protected": protected_intervals is not None,
                  "protected_intervals": protected_intervals or [], "privacy_guarantee": False}
        save_json(work / "augmentation-report.json", report)
        return report
    finally:
        cap.release()
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        original.unlink(missing_ok=True)
        changed.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
