"""Shared editorial intent and bounded discovery plans. Metadata never invents an episode genre."""
from __future__ import annotations
import re
from pathlib import Path

GENRES = {
    "scifi": r"\bsci[ -]?fi\b|\bscience fiction\b",
    "action": r"\baction\b", "adventure": r"\badventure\b",
    "comedy": r"\bcomed(?:y|ies)\b|\bfunny\b|\bhilarious\b",
    "horror": r"\bhorror\b|\bscary\b", "thriller": r"\bthrillers?\b",
    "romance": r"\bromance\b|\bromantic\b|\brom[ -]?com\b",
    "fantasy": r"\bfantasy\b", "drama": r"\bdramas?\b",
    "documentary": r"\bdocumentar(?:y|ies)\b|\bdocuseries\b",
    "family": r"\bfamily\b|\bkids\b", "crime": r"\bcrime\b",
    "mystery": r"\bmyster(?:y|ies)\b", "animation": r"\banimation\b|\banimated\b",
    "war": r"\bwar\b", "western": r"\bwesterns?\b",
}
# These are concepts, not global search terms: only a title's own genres enable them.
SCENE_CONCEPTS = {
    "action": ("action scene", "fight scene", "chase scene"),
    "adventure": ("adventure scene", "chase scene", "stunt scene"),
    "comedy": ("funny scene", "physical comedy scene"),
    "horror": ("scary scene", "suspense scene"),
    "thriller": ("tense scene", "suspense scene"),
    "romance": ("romantic scene", "emotional scene"),
    "drama": ("emotional scene", "dramatic scene"),
    "scifi": ("science fiction scene",), "fantasy": ("fantasy scene", "battle scene"),
    "crime": ("crime scene", "confrontation scene"),
    "family": ("memorable scene",), "documentary": ("best clip",),
    "mystery": ("mystery scene","confrontation scene"), "animation": ("best scene",),
    "war": ("battle scene","rescue scene"), "western": ("showdown scene","chase scene"),
}

def explicit_genre(title: str = "", brief: str = "") -> str:
    """A single requested genre wins; a mixed or unspecified list is neutral."""
    title_matches=[name for name,pattern in GENRES.items() if re.search(pattern,title,re.I)]
    if title_matches:
        return title_matches[0] if len(title_matches)==1 else "neutral"
    if re.search(r"\b(?:neutral|mixed|general|broad)\s+(?:genre|list|selection|countdown|music|instrumental|recommendation)",brief,re.I):
        return "neutral"
    matches=[]
    for name,pattern in GENRES.items():
        if re.fullmatch(pattern,brief.strip(),re.I):
            return name
        for match in re.finditer(pattern,brief,re.I):
            before=brief[max(0,match.start()-45):match.start()]
            after=brief[match.end():match.end()+45]
            # A per-film scene request is not an instruction to theme the episode.
            requested=(re.match(r"\s+(?:movies|films|shows|series|recommendations|countdown|theme|music|tone)\b",after,re.I)
                       or re.search(r"\b(?:genre|theme|tone|music)\s*(?:is|:|=|should be)?\s*$",before,re.I)
                       or re.search(r"\b(?:focus on|list of|video about|countdown of)\s*$",before,re.I))
            if requested:
                matches.append(name)
    matches=list(dict.fromkeys(matches))
    return matches[0] if len(matches)==1 else "neutral"


def title_genres(film: dict) -> list[str]:
    text = " ".join(str(v.get("name", "") if isinstance(v, dict) else v) for v in film.get("genres", []))
    return [name for name, pattern in GENRES.items() if re.search(pattern, text, re.I)]

def season_hint(film: dict) -> str:
    value = film.get("requested_season") or film.get("season")
    if isinstance(value, dict):
        value = value.get("season_number")
    return f"season {value}" if str(value).isdigit() and 0 < int(value) < 100 else ""

def context_terms(film: dict, limit: int = 2) -> list[str]:
    """Use this title's verified keyword context, not just a fixed genre vocabulary."""
    keywords=film.get("keywords",[])
    if isinstance(keywords,dict):
        keywords=keywords.get("keywords",keywords.get("results",[]))
    terms=[]
    for item in keywords if isinstance(keywords,list) else []:
        text=str(item.get("name","") if isinstance(item,dict) else item).strip()
        if (not 1<=len(text.split())<=4 or len(text)>55
                or re.search(r"based on|sequel|franchise|adaptation|spoiler|ending|murderer identity",text,re.I)):
            continue
        if text and text.casefold() not in {t.casefold() for t in terms}:
            terms.append(text)
        if len(terms)>=limit:
            break
    return terms

def search_queries(film: dict, *, primary: bool = False, limit: int = 7) -> list[str]:
    title=str(film["title"]).strip()
    base=f"{title} {film['year']}"
    season=season_hint(film)
    stem=" ".join(part for part in (base,season) if part)
    generic=[f"{stem} official trailer",f"{stem} trailer",f"{stem} best scene"]
    year_free=f"{title} {season} best scene".replace("  "," ")
    concepts=[SCENE_CONCEPTS[g] for g in title_genres(film) if g in SCENE_CONCEPTS]
    relevant=[f"{stem} {words[0]}" for words in concepts]
    contextual=[f"{stem} {term} scene" for term in context_terms(film)]
    required=generic if primary else [generic[2],*([year_free] if film.get("kind")=="tv" else [])]
    if primary and film.get("kind")=="tv":
        required.append(year_free)
    # Preserve the generic discovery families, then spend a small budget on
    # verified per-title genres and its own premise keywords.
    queries=[*required,*relevant[:1],*contextual[:1]]
    if not primary:
        queries += generic[:2]
    queries += [*relevant[1:],*contextual[1:]]
    for depth in (1,2):
        queries += [f"{stem} {words[depth]}" for words in concepts if depth<len(words)]
    return list(dict.fromkeys(queries))[:limit]

def image_queries(film: dict, limit: int = 3) -> list[str]:
    base=f"{film['title']} {film.get('year','')} {season_hint(film)}".strip()
    cast=film.get("cast",[])
    actor=str(cast[0]) if cast and isinstance(cast[0],str) else ""
    subject=f"{base} {actor}".strip()
    relevant=[f"{subject} {SCENE_CONCEPTS[g][0]} image" for g in title_genres(film) if g in SCENE_CONCEPTS]
    contextual=[f"{base} {term} image" for term in context_terms(film)]
    return list(dict.fromkeys([f"{subject} best scene image",*relevant[:1],*contextual,
                               *relevant[1:],f"{base} best image"]))[:limit]

def source_identity(path) -> str:
    """Legacy path fallback; callers should register actual title IDs where possible."""
    return re.sub(r"-extra(?:-\d+)?$", "", Path(str(path)).stem)

def clean_review(paragraphs: list[str], availability: str = "", bridge: str = "") -> list[str]:
    """Remove low-value whole sentences, never fill their place to meet a word count."""
    result = []
    provider = r"(?:you can (?:find|watch|stream)|(?:it|this (?:movie|show|series)) is (?:listed|available)|stream(?:ing)? (?:it|on)|watch it)\b"
    comparison = r"(?:if you (?:liked|loved|enjoyed)|fans of .{0,80} (?:will|should|need to))\b"
    for paragraph in paragraphs:
        if paragraph.strip() in {availability, bridge} - {""}:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
        kept = [s for s in sentences if not re.match(provider, s, re.I) and not re.match(comparison, s, re.I)]
        text = " ".join(kept).strip()
        if text:
            result.append(text)
    return result
