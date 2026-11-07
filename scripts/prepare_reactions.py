"""Fetch and validate additional reactions only while preparing this feature branch.

Normal production reads bundled files and never downloads reaction assets.
"""
from __future__ import annotations
import base64
import hashlib
import io
import json
from pathlib import Path

import requests
from PIL import Image

SOURCE_REF = "3cd940af8cd795393388f117f4e8c091cde72d0c"
CANDIDATES = [{"path":"homer-disappear-bush.gif","sha":"bae881b7572901037c24c0d097ed1bc923eaf563"},{"path":"picard-facepalm.gif","sha":"c7c95455aa4dbbeb351c052fbad88ccdfb90002e"},{"path":"keanu-thumbs-up.gif","sha":"4c1d2763616721428cba4f410c89fb89839660f5"},{"path":"dramatic-pug.gif","sha":"2572823fde947dd79b808167c3a9089b98d88173"},{"path":"mic-drop-bryan-cranston.gif","sha":"48c251b84357a886127fa2c074da32dc5a2ba2a3"},{"path":"dancing-carlton.gif","sha":"30b5ca262b753380f63772c6dff99dc88e329400"},{"path":"cool-story-bro.gif","sha":"f021c26dd99488f5d972372d3eca5ead152674b3"},{"path":"banging-head-on-table.gif","sha":"c1ce0fc0d95531781d447077bc609fe988b97687"}]


def main():
    root = Path("assets/memes")
    root.mkdir(parents=True, exist_ok=True)
    manifest = root/"extra-reactions.json"
    if manifest.exists():
        rows = json.loads(manifest.read_text())
        for row in rows:
            data = base64.b64decode("".join((root/row["asset"]).read_text().split()), validate=True)
            assert hashlib.sha256(data).hexdigest() == row["sha256"]
        print("Bundled additional reactions already validated:", len(rows))
        return
    rows = []
    for item in CANDIDATES:
        url = f"https://raw.githubusercontent.com/tlberglund/animated-gifs/{SOURCE_REF}/{item['path']}"
        response = requests.get(url, timeout=45)
        response.raise_for_status()
        data = response.content
        git_sha = hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
        assert git_sha == item["sha"], item["path"]
        assert 0 < len(data) <= 2_000_000, item["path"]
        with Image.open(io.BytesIO(data)) as image:
            if min(image.size) < 180 or image.width*image.height > 8_000_000:
                print("Skipping unsuitable reaction size:", item["path"], image.size)
                continue
            assert image.format == "GIF" and image.n_frames > 1
            image.seek(image.n_frames//2)
            image.load()
            dimensions, frames = image.size, image.n_frames
        name = item["path"]+".b64"
        (root/name).write_text(base64.b64encode(data).decode()+"\n")
        rows.append({"asset": name, "sha256": hashlib.sha256(data).hexdigest(),
            "source": f"https://github.com/tlberglund/animated-gifs/blob/{SOURCE_REF}/{item['path']}",
            "attribution": f"Reaction GIF: {item['path']}; source: tlberglund/animated-gifs.",
            "width": dimensions[0], "height": dimensions[1], "frames": frames, "bytes": len(data)})
        print("Validated reaction:", json.dumps(rows[-1]))
        if len(rows) >= 6:
            break
    assert len(rows) >= 4, "Not enough additional full-size reaction images"
    manifest.write_text(json.dumps(rows, indent=2)+"\n")



if __name__ == "__main__":
    main()
