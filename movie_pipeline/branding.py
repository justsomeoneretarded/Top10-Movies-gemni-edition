"""MOVIE RANKS: an original folded-M light entrance and two-hit sonic ident.

Frames are streamed to FFmpeg at the export resolution; the animation never
materializes an uncompressed frame sequence on a runner's disk.
"""
from __future__ import annotations

import math
from contextlib import suppress
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw

IDENT_SECONDS = 2.3
IDENT_FPS = 30


def _ease(start: float, end: float, value: float) -> float:
    x = min(1.0, max(0.0, (value-start)/(end-start)))
    return x*x*(3-2*x)


def _logo_layers(width: int, height: int) -> tuple[Image.Image, Image.Image]:
    """Native-resolution facets retain clean edges even on a 4K canvas."""
    from .render import brand_font

    symbol = Image.new("RGBA", (width, height))
    wordmark = Image.new("RGBA", (width, height))
    unit = height/720
    left, top = width/2-158*unit, 142*unit
    shapes = [
        ([(0,0),(76,0),(76,278),(0,294)], (126,3,18)),
        ([(76,0),(158,137),(158,234),(0,0)], (232,13,34)),
        ([(158,137),(240,0),(316,0),(158,234)], (181,5,25)),
        ([(240,0),(316,0),(316,294),(240,278)], (126,3,18)),
    ]
    # Small, bounded arrays per facet rather than a float64 full-frame texture.
    for points, base_color in shapes:
        pts = [(round(left+x*unit), round(top+y*unit)) for x,y in points]
        box = (min(x for x,y in pts), min(y for x,y in pts),
               max(x for x,y in pts)+1, max(y for x,y in pts)+1)
        bw, bh = box[2]-box[0], box[3]-box[1]
        yy, xx = np.mgrid[0:bh,0:bw].astype(np.float32)
        light = 1.25-.57*yy/max(1,bh-1)+.13*np.exp(-((xx/max(1,bw)-.13)/.14)**2)
        rgb = np.clip(np.asarray(base_color, dtype=np.float32)*light[:,:,None],0,255).astype("uint8")
        facet = Image.fromarray(rgb, "RGB").convert("RGBA")
        mask = Image.new("L", (bw,bh))
        ImageDraw.Draw(mask).polygon([(x-box[0],y-box[1]) for x,y in pts],fill=255)
        symbol.paste(facet, box[:2], mask)
        ImageDraw.Draw(symbol).line([*pts,pts[0]], fill=(255,101,107,220),
                                   width=max(1,round(.85*unit)), joint="curve")
    # The top and inside folds catch the light, rather than a thick neon outline.
    d = ImageDraw.Draw(symbol)
    d.line([(left,top),(left+76*unit,top),(left+158*unit,top+137*unit)],
           fill=(255,159,146,235),width=max(1,round(1.1*unit)))
    d.line([(left+240*unit,top),(left+316*unit,top)],
           fill=(255,119,119,230),width=max(1,round(unit)))
    ImageDraw.Draw(wordmark).text(
        (width/2,top+342*unit), "MOVIE RANKS",
        font=brand_font("body-medium",max(1,round(25*unit))),
        anchor="mm",fill=(237,237,242,255),stroke_width=0)
    return symbol, wordmark


def logo_image(path: Path, width: int = 1280, height: int = 720):
    """The clean resolved logo, also usable independently of the animation."""
    canvas = Image.new("RGBA",(width,height),(5,5,8,255))
    symbol, wordmark = _logo_layers(width,height)
    canvas.alpha_composite(symbol)
    canvas.alpha_composite(wordmark)
    picture = canvas.convert("RGB")
    picture.save(path)
    return picture


def _ribbons(t: float, width: int, height: int) -> Image.Image:
    """Slowly spreading colored light filaments; no flashes or random jitter.

    Light is deliberately calculated at low resolution and smoothly enlarged.
    The logo itself is rendered at native resolution. This bounds the expensive
    light field independently of a 1080p/1440p/4K export.
    """
    lw = min(480,width)
    lh = max(2,round(height*lw/width))
    yy, xx = np.mgrid[0:lh,0:lw].astype(np.float32)
    xx /= max(1,lw-1)
    yy /= max(1,lh-1)
    field = np.zeros((lh,lw,3),dtype=np.float32)
    strength = _ease(.04,.29,t)*(1-_ease(.70,1.53,t))
    travel = _ease(.02,1.35,t)
    colors = ((1,.07,.05),(1,.26,.035),(1,.035,.26),
              (.58,.055,1),(.13,.28,1),(.035,.7,1))
    for i in range(25):
        offset = (i-12)/12
        center = .5+offset*(.055+.45*travel)
        slant = .011*math.sin(i*1.81)*(yy-.5)
        spread = .0015+.0025*((i*7)%11)/10
        core = np.exp(-((xx-center-slant)/spread)**2)
        bloom = .19*np.exp(-((xx-center-slant)/(spread*4.5))**2)
        y_peak = .45+.18*math.sin(i*.7+t*1.7)
        length = .17+.83*np.exp(-((yy-y_peak)/.43)**2)
        energy = (core+bloom)*length*(.48+.28*math.sin(i*2.1)**2)
        field += energy[:,:,None]*np.asarray(colors[i%len(colors)],dtype=np.float32)
    # A barely visible red aura anchors the resolved M against true dark edges.
    aura = .06*np.exp(-(((xx-.5)/.23)**2+((yy-.45)/.39)**2))*_ease(.25,.95,t)
    field *= strength
    field[:,:,0] += aura
    field[:,:,1] += aura*.045
    field[:,:,2] += aura*.10
    return Image.fromarray(np.clip(field*205,0,255).astype("uint8"),"RGB").resize(
        (width,height),Image.Resampling.BICUBIC)


def ident_frame(time: float, width: int = 1280, height: int = 720,
                seconds: float = IDENT_SECONDS, *, _layers=None) -> Image.Image:
    """Render a deterministic animation frame for previews and visual QA."""
    t = max(0.0,min(IDENT_SECONDS,float(time)*IDENT_SECONDS/seconds))
    symbol, wordmark = _layers if _layers is not None else _logo_layers(width,height)
    base = Image.new("RGB",(width,height),(3,3,6))
    lights = _ribbons(t,width,height)
    # Screen blend keeps colored light luminous without opaque graphic bars.
    canvas = ImageChops.screen(base,lights).convert("RGBA")
    reveal = _ease(.20,.86,t)
    if reveal:
        # A narrow soft sweep moves through the M as its facets resolve.
        sheen_position = -.45+1.9*_ease(.18,1.10,t)
        sheen = Image.new("RGBA",canvas.size,(255,185,167,0))
        if t < 1.10:
            sw, sh = max(2,width//6), max(2,height//6)
            yy, xx = np.mgrid[0:sh,0:sw].astype(np.float32)
            sweep = np.exp(-((xx/max(1,sw-1)-.14*yy/max(1,sh-1)-sheen_position)/.055)**2)
            sheen.putalpha(Image.fromarray((sweep*52).astype("uint8"),"L").resize(
                canvas.size,Image.Resampling.BILINEAR))
            sheen.putalpha(Image.composite(sheen.getchannel("A"),Image.new("L",canvas.size),
                                          symbol.getchannel("A")))
        visible = symbol.copy()
        visible.putalpha(visible.getchannel("A").point(lambda v:round(v*reveal)))
        canvas.alpha_composite(visible)
        if t < 1.10:
            canvas.alpha_composite(sheen)
    label_alpha = _ease(.78,1.13,t)
    if label_alpha:
        label = wordmark.copy()
        label.putalpha(label.getchannel("A").point(lambda v:round(v*label_alpha)))
        canvas.alpha_composite(label)
    fade = 1-_ease(IDENT_SECONDS-.18,IDENT_SECONDS,t)
    return Image.blend(Image.new("RGB",canvas.size),canvas.convert("RGB"),fade)


def ident_audio(seconds: float = IDENT_SECONDS, rate: int = 48000) -> np.ndarray:
    """Original, warm two-hit sting; no borrowed sound or synthetic noise bed.

    Smooth attack/release envelopes avoid clicks. Upper harmonics retain impact
    on phone speakers without adding a piercing high-frequency beep.
    """
    if not math.isfinite(seconds) or seconds <= 0 or rate <= 0:
        raise ValueError("Ident duration and sample rate must be positive.")
    t = np.arange(round(rate*seconds),dtype=np.float64)/rate
    audio = np.zeros((len(t),2),dtype=np.float64)
    time_scale = seconds/IDENT_SECONDS
    for at,fundamental,gain,pan in ((.23,58.27,.29,-.10),(.79,77.78,.30,.10)):
        x = np.maximum(0,t-at*time_scale)
        attack = 1-np.exp(-x/.008)
        active = t >= at*time_scale
        body = (np.sin(2*np.pi*fundamental*x)*np.exp(-x/ .26)
                +.43*np.sin(2*np.pi*fundamental*2*x)*np.exp(-x/.32)
                +.17*np.sin(2*np.pi*fundamental*4*x)*np.exp(-x/.23))
        resonance = .10*np.sin(2*np.pi*fundamental*3*x)*np.exp(-x/.53)
        hit = gain*attack*(body+resonance)*active
        audio[:,0] += hit*(1-pan)
        audio[:,1] += hit*(1+pan)
    # Leave a genuinely quiet tail; the final sample is zero, so concatenation
    # with the first scene cannot produce an impulsive endpoint.
    release = 1-np.clip((t-(seconds-.20))/.20,0,1)
    audio *= release[:,None]**2
    if len(audio):
        audio[-1] = 0
    peak = float(np.max(np.abs(audio))) if len(audio) else 0
    if peak > .65:
        audio *= .65/peak
    return audio.astype(np.float32)


def render_ident(output: Path, work: Path, seconds: float = IDENT_SECONDS):
    from .core import PipelineError
    from .render import LOSSLESS_AUDIO
    from .video_settings import export_size

    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Ident duration must be positive.")
    work.mkdir(parents=True,exist_ok=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    width,height = export_size()
    wav = work/"ident.wav"
    samples = ident_audio(seconds)
    with wave.open(str(wav),"wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        stream.writeframes((samples*32767).astype("<i2").tobytes())
    frames = max(1,round(seconds*IDENT_FPS))
    # Streaming bounds scratch space to one small WAV, even for a 4K ident.
    command = ["ffmpeg","-hide_banner","-loglevel","error","-y",
               "-f","rawvideo","-pixel_format","rgb24","-video_size",f"{width}x{height}",
               "-framerate",str(IDENT_FPS),"-i","pipe:0","-i",str(wav),
               "-map","0:v:0","-map","1:a:0","-t",str(seconds),
               "-c:v","libx264","-preset","veryfast","-crf","18",
               "-pix_fmt","yuv420p","-threads","2",*LOSSLESS_AUDIO,str(output)]
    layers = _logo_layers(width,height)
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=errors)
        try:
            assert process.stdin is not None
            for frame_index in range(frames):
                process.stdin.write(ident_frame(frame_index*seconds/max(1,frames-1),width,height,seconds,
                                                _layers=layers).tobytes())
            process.stdin.close()
            code = process.wait(timeout=120)
        except BaseException:
            if process.stdin is not None:
                with suppress(OSError):
                    process.stdin.close()
            if process.poll() is None:
                process.kill()
            process.wait()
            errors.seek(0)
            detail = errors.read().decode("utf-8",errors="replace")[-2400:]
            if detail:
                raise PipelineError(f"Movie Ranks ident encoding failed: {detail}") from None
            raise
        if code:
            errors.seek(0)
            raise PipelineError("Movie Ranks ident encoding failed: "+
                                errors.read().decode("utf-8",errors="replace")[-2400:])
    return seconds
