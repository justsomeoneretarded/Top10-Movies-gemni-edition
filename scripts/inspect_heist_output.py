"""Read existing output artifacts only; no production or model calls."""
import base64,json,math,re,shutil,subprocess,zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path,PurePosixPath
from PIL import Image,ImageDraw,ImageFont,ImageOps
root=Path("payload").resolve();out=Path("comparison-output");out.mkdir(exist_ok=True)
for archive in list(root.rglob("*.zip"))[:4]:
    with zipfile.ZipFile(archive) as z:
        rows=z.infolist()
        if len(rows)>2000 or sum(x.file_size for x in rows)>1_500_000_000: raise RuntimeError("Unexpected archive size")
        for row in rows:
            p=PurePosixPath(row.filename)
            if p.is_absolute() or ".." in p.parts or ((row.external_attr>>16)&0o170000)==0o120000: raise RuntimeError("Unsafe path")
            dest=root/"unpacked"/archive.stem/str(p)
            if not dest.resolve().is_relative_to(root): raise RuntimeError("Unsafe destination")
            if row.is_dir():dest.mkdir(parents=True,exist_ok=True)
            else:
                dest.parent.mkdir(parents=True,exist_ok=True)
                with z.open(row) as src,dest.open("wb") as target: shutil.copyfileobj(src,target)
files=sorted(p for p in root.rglob("*") if p.is_file())
print("COMPARE_FILES "+json.dumps([{"path":str(p.relative_to(root)),"bytes":p.stat().st_size} for p in files][:500]),flush=True)
def clean(v):
    if isinstance(v,dict):return{k:("[redacted]" if re.search(r"secret|cookie|authorization|api.?key|access.?token",k,re.I) else clean(x)) for k,x in v.items()}
    if isinstance(v,list):return[clean(x) for x in v]
    if isinstance(v,str):
        v=re.sub(r"(?i)([?&](?:key|token|api_key|access_token)=)[^&\s]+",r"\1[redacted]",v)
        return re.sub(r"AIza[A-Za-z0-9_-]{20,}","[redacted]",v)
    return v
data={};budget=1_500_000
for p in files:
    if p.suffix.lower()==".json":
        try:
            obj=clean(json.loads(p.read_text()));data[p.name]=obj
            text=json.dumps(obj,ensure_ascii=True)
            if len(text)<=250000 and len(text)<budget:
                print("COMPARE_JSON "+json.dumps({"path":str(p.relative_to(root)),"data":obj},ensure_ascii=True),flush=True);budget-=len(text)
            else: print("COMPARE_LARGE "+json.dumps({"path":str(p.relative_to(root)),"keys":list(obj)[:40] if isinstance(obj,dict) else "array"}),flush=True)
        except (ValueError,UnicodeError,OSError):pass
    if p.suffix.lower() in {".srt",".vtt"} or p.name in {"final-script.txt","script.txt","recommended-title.txt","description.txt"}:
        try:print("COMPARE_TEXT "+json.dumps({"path":str(p.relative_to(root)),"text":clean(p.read_text()[:60000])},ensure_ascii=True),flush=True)
        except(UnicodeError,OSError):pass
videos=[p for p in files if p.name=="final-video.mp4"] or sorted((p for p in files if p.suffix.lower()==".mp4"),key=lambda p:p.stat().st_size,reverse=True)
if not videos:raise RuntimeError("No finished video")
video=videos[0]
probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_format","-show_streams","-of","json",str(video)]))
print("COMPARE_PROBE "+json.dumps(clean(probe)),flush=True)
duration=float(probe["format"]["duration"])
chapters=data.get("quality-summary.json",{}).get("chapters",[])
if not chapters:
    for v in data.values():
        if isinstance(v,dict) and isinstance(v.get("chapters"),list):chapters=v["chapters"];break
if not chapters:raise RuntimeError("No chapter report")
print("COMPARE_CHAPTERS "+json.dumps(chapters),flush=True)
font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",16)
def emit_sheet(name,times,labels=None):
    folder=out/name;folder.mkdir(exist_ok=True)
    times=[min(max(0,float(t)),duration-.1) for t in times]
    def frame(task):
        i,t=task;dest=folder/f"{i:03}.jpg"
        subprocess.run(["ffmpeg","-v","error","-nostdin","-threads","1","-ss",f"{t:.6f}","-i",str(video),"-frames:v","1","-vf","scale=320:180","-q:v","3","-y",str(dest)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        return dest
    with ThreadPoolExecutor(max_workers=2) as pool:paths=list(pool.map(frame,enumerate(times)))
    sheet=Image.new("RGB",(1280,math.ceil(len(paths)/4)*208),(18,18,18));draw=ImageDraw.Draw(sheet)
    for i,(p,t) in enumerate(zip(paths,times)):
        x=i%4*320;y=i//4*208
        with Image.open(p) as im:sheet.paste(im.convert("RGB"),(x,y))
        minutes,seconds=divmod(t,60)
        label=(labels[i]+" " if labels else "")+f"{int(minutes):02}:{seconds:05.2f}"
        draw.text((x+5,y+184),label,font=font,fill="white")
    path=out/f"{name}.jpg";sheet.save(path,quality=82,optimize=True)
    print("COMPARE_IMAGE "+json.dumps({"name":name,"times":times,"mime":"image/jpeg","base64":base64.b64encode(path.read_bytes()).decode()}),flush=True)
film_rows=[(i,c) for i,c in enumerate(chapters) if str(c.get("title","")).startswith("#")]
first=float(film_rows[0][1]["start"]) if film_rows else 18
emit_sheet("main-intro",list(range(min(30,int(math.ceil(first)))))+[max(0,first-.12)])
body_times=[];body_labels=[]
for i,c in film_rows:
    name=re.sub(r"[^a-z0-9]+","-",str(c["title"]).lower()).strip("-")
    start=float(c["start"])
    emit_sheet(name,[start+j for j in range(14)]+[start+13.85])
    end=float(chapters[i+1]["start"]) if i+1<len(chapters) else duration
    for fraction in (.23,.42,.64,.83):
        body_times.append(start+(end-start)*fraction);body_labels.append(str(c["title"]).split("(")[0][:24])
emit_sheet("body-samples",body_times,body_labels)
thumbs=sorted(p for p in files if re.fullmatch(r"thumbnail-\d+\.jpg",p.name))[:10]
if thumbs:
    sheet=Image.new("RGB",(1600,math.ceil(len(thumbs)/5)*208),(18,18,18));draw=ImageDraw.Draw(sheet)
    for i,p in enumerate(thumbs):
        x=i%5*320;y=i//5*208
        with Image.open(p) as im:sheet.paste(ImageOps.fit(im.convert("RGB"),(320,180)),(x,y))
        draw.text((x+5,y+184),p.name,font=font,fill="white")
    path=out/"thumbnail-options.jpg";sheet.save(path,quality=86,optimize=True)
    print("COMPARE_IMAGE "+json.dumps({"name":"thumbnail-options","mime":"image/jpeg","base64":base64.b64encode(path.read_bytes()).decode()}),flush=True)
print("COMPARE_EXTRACTION_COMPLETE",flush=True)
