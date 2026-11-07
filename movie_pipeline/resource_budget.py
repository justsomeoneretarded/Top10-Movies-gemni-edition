"""Thread-safe reservations for optional downloads; never touch model caches."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import shutil
import threading

class DownloadBudget:
    def __init__(self, reserve_bytes=800_000_000, headroom_bytes=2_000_000_000):
        self.reserve_bytes = reserve_bytes
        self.headroom_bytes = headroom_bytes
        self.pending = 0
        self.lock = threading.Lock()

    @contextmanager
    def claim(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        with self.lock:
            accepted = shutil.disk_usage(root).free - self.pending >= self.reserve_bytes + self.headroom_bytes
            if accepted:
                self.pending += self.reserve_bytes
        try:
            yield accepted
        finally:
            if accepted:
                with self.lock:
                    self.pending -= self.reserve_bytes

# Shared within one production process: two titles must not download the same
# discovered upload concurrently, or accept it under different source identities.
_source_lock=threading.Lock()
_source_claims={}

def _source_key(url):
    from urllib.parse import urlsplit,parse_qs
    parsed=urlsplit(str(url))
    if parsed.hostname in {"youtube.com","www.youtube.com","m.youtube.com"}:
        return "youtube:"+parse_qs(parsed.query).get("v",[str(url)])[0]
    if parsed.hostname=="youtu.be":
        return "youtube:"+parsed.path.strip("/")
    return str(url).split("#",1)[0]

def reserve_source(root, url, owner):
    identity=(str(Path(root).resolve()),_source_key(url))
    owner=str(Path(owner).resolve())
    with _source_lock:
        existing=_source_claims.get(identity)
        if existing is not None and existing!=owner:
            return False
        _source_claims[identity]=owner
        return True

def release_source(root,url,owner):
    identity=(str(Path(root).resolve()),_source_key(url))
    with _source_lock:
        if _source_claims.get(identity)==str(Path(owner).resolve()):
            _source_claims.pop(identity,None)

def seed_sources(root,paths):
    import json
    for path in paths:
        try:
            source=json.loads(Path(path).with_suffix(".json").read_text()).get("source")
            if source:
                reserve_source(root,source,path)
        except (OSError,ValueError,TypeError):
            continue
