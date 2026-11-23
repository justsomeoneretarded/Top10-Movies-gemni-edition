"""Read-only inspection of an existing finished artifact; no production or model calls."""
import base64
import io
import json
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

root = Path("inspected").resolve()
files = [p for p in root.rglob("*") if p.is_file() and not p.is_symlink()]
print("PACKAGE_FILES " + json.dumps([{"path":str(p.relative_to(root)),"bytes":p.stat().st_size} for p in files]))
def scrub(value):
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if re.search(r"(?i)secret|password|cookie|authorization|api[_-]?key|access[_-]?token|download_url|upload_url",str(k)) else scrub(v) for k,v in value.items()}
    if isinstance(value,list):
        return [scrub(v) for v in value]
    if isinstance(value,str):
        value = re.sub(r"AIza[0-9A-Za-z_-]{20,}|gh[pousr]_[0-9A-Za-z_]{20,}|Bearer\\s+[^\\s]+","[REDACTED]",value)
        value = re.sub(r"([?&](?:key|token|sig|signature|api_key)=)[^&\\s]+",r"\\1[REDACTED]",value,flags=re.I)
        return value[:6000]
    return value
names = ["run-manifest.json","production-quality-report.json","script-quality-report.json",
"quality-report.json","duration-report.json","ai-production-report.json","gemini-edit-report.json",
"native-scenes.json","audio-review.json","soundtrack-report.json","thumbnail-sources.json",
"visual-repair-report.json","quality-review.json"]
for name in names:
    matches = [p for p in files if p.name == name]
    if not matches:
        print("REPORT_MISSING "+name)
    for p in matches:
        if p.stat().st_size > 3*1024*1024:
            print("REPORT_TOO_LARGE "+name)
            continue
        try:
            data = scrub(json.loads(p.read_text()))
        except (OSError,ValueError) as exc:
            print("REPORT_UNREADABLE "+name+" "+type(exc).__name__)
            continue
        print("REPORT_BEGIN "+name)
        print(json.dumps(data,ensure_ascii=True))
        print("REPORT_END "+name)
for name in ["final-script.txt","recommended-title.txt","description.txt"]:
    for p in [p for p in files if p.name == name]:
        print("TEXT_BEGIN "+name)
        print(json.dumps(scrub(p.read_text()[:24000]),ensure_ascii=True))
        print("TEXT_END "+name)
thumbs = sorted([p for p in files if re.fullmatch(r"thumbnail-(?:0[1-9]|10)\\.jpg",p.name)],key=lambda p:p.name)
print("THUMBNAIL_COUNT "+str(len(thumbs)))
if thumbs:
    canvas = Image.new("RGB",(1500,2*195),"#171717")
    draw = ImageDraw.Draw(canvas)
    for i,p in enumerate(thumbs[:10]):
        with Image.open(p) as opened:
            pic=ImageOps.contain(opened.convert("RGB"),(292,164),Image.Resampling.LANCZOS)
            x,y=(i%5)*300,(i//5)*195
            canvas.paste(pic,(x+(300-pic.width)//2,y))
            draw.text((x+6,y+168),p.name,fill="white")
    buffer=io.BytesIO()
    canvas.save(buffer,"JPEG",quality=86)
    print("THUMBNAIL_IMAGE_BEGIN")
    print(base64.b64encode(buffer.getvalue()).decode())
    print("THUMBNAIL_IMAGE_END")
