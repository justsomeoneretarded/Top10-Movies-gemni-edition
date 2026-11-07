"""Create a small private voice reference and optionally set its GitHub Actions secret."""
import argparse
import base64
import os
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("name", choices=["warguy", "kit-laser", "female"])
parser.add_argument("audio", type=Path)
parser.add_argument("--start", type=float, default=5)
parser.add_argument("--seconds", type=float, default=12)
parser.add_argument("--repo", help="owner/repo; requires authenticated gh and permission to manage secrets")
parser.add_argument("--output", type=Path, default=Path("assets/voices"))
args = parser.parse_args()
if not 5 <= args.seconds <= 20 or args.start < 0:
    parser.error("Use a 5–20 second clean section and nonnegative start time.")
args.output.mkdir(parents=True, exist_ok=True)
reference = args.output / f"{args.name}.ogg"
subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(args.start), "-i", str(args.audio), "-t", str(args.seconds), "-af", "loudnorm=I=-20:TP=-2:LRA=9", "-ar", "24000", "-ac", "1", "-c:a", "libopus", "-b:a", "20k", "-vbr", "on", str(reference)], check=True)
encoded = base64.b64encode(reference.read_bytes()).decode()
if len(encoded) >= 48000:
    raise SystemExit("Reference exceeds GitHub's secret-size budget. Use a shorter section.")
secret = "VOICE_" + args.name.upper().replace("-", "_") + "_B64"
if args.repo:
    subprocess.run(["gh", "secret", "set", secret, "--repo", args.repo], input=encoded, text=True, check=True)
    print(f"Configured {secret}; reference remains private.")
else:
    destination = args.output / f"{args.name}.secrets.env"
    destination.write_text(secret + "=" + encoded + "\n")
    os.chmod(destination, 0o600)
    print(f"Saved private reference and {secret} in {destination}; never commit this file.")
