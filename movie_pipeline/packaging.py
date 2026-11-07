"""Plain-language metadata and grounded availability copy; no model calls or hard failures."""
from __future__ import annotations

import re
from collections import Counter

from .availability import availability_line
from .core import key

MIDROLLS = (
    "Like and subscribe. It's free. My movie opinions are the expensive part.",
    "Like and subscribe. Your watchlist needs help, and my ego needs supervision.",
    "Like and subscribe. Consider it insurance against another terrible movie night.",
    "Hit like and subscribe. Your finger could use the exercise.",
    "Like and subscribe. The button has been looking lonely lately.",
    "Subscribe. It's the only commitment I'm asking you to make today.",
    "Like and subscribe. My Wi-Fi deserves better than an audience of three.",
    "Subscribe. Consider it your post-credit scene. Back to the movies.",
)


def midroll_text(seed: str) -> str:
    """Most options request both actions; all stay short enough for a quick break."""
    return MIDROLLS[int(key(seed, "midroll-v1")[:8], 16) % len(MIDROLLS)]


def _provider_names(values) -> list[str]:
    if not isinstance(values, list):
        return []
    names = []
    for value in values:
        name = value if isinstance(value, str) else value.get("provider_name", value.get("name", "")) if isinstance(value, dict) else ""
        name = " ".join(str(name).split()).strip()
        if name and name not in names:
            names.append(name)
    return names


def _availability(film: dict) -> dict:
    data = film.get("watch_availability")
    return data if isinstance(data, dict) else {}


def refresh_availability(script: dict, films: list[dict]) -> dict:
    """Keep availability in metadata/description, never in the spoken review."""
    from .editorial_policy import clean_review
    for section in script.get("sections", []):
        index = section.get("film")
        if section.get("kind") != "film" or not isinstance(index, int) or not 0 <= index < len(films):
            continue
        original = section.get("paragraphs", [])
        labels = section.get("delivery", [])
        clean, delivery = [], []
        for position, paragraph in enumerate(original):
            kept = clean_review([paragraph], section.get("availability_text", ""), section.get("bridge", ""))
            clean.extend(kept)
            delivery.extend([labels[position] if position < len(labels) else "normal"] * len(kept))
        section["paragraphs"] = clean
        section["delivery"] = delivery
        section["availability_text"] = ""
        section.pop("bridge", None)
    return script


def featured_heading(films: list[dict]) -> str:
    kinds = {film.get("kind", "movie") for film in films}
    if kinds == {"tv"}:
        return "Featured shows / series"
    if kinds == {"movie"}:
        return "Featured movies"
    return "Featured movies and shows / series"


def chapter_time(seconds: float) -> str:
    total = max(0, int(seconds))
    if total >= 3600:
        return f"{total//3600}:{total//60%60:02}:{total%60:02}"
    return f"{total//60:02}:{total%60:02}"


def topic_hashtags(title: str, films: list[dict]) -> list[str]:
    """Choose 3–5 actual topic tags; these are relevance hints, not a ranking guarantee."""
    shows_only = bool(films) and all(film.get("kind") == "tv" for film in films)
    tags = ["#TVRecommendations" if shows_only else "#MovieRecommendations", "#WhatToWatch"]
    mapping = {
        "action": "ActionShows" if shows_only else "ActionMovies",
        "horror": "HorrorShows" if shows_only else "HorrorMovies",
        "science fiction": "SciFi", "sci-fi": "SciFi", "sci fi": "SciFi",
        "thriller": "Thrillers", "comedy": "Comedy", "drama": "Drama",
        "mystery": "Mystery", "animation": "Animation", "fantasy": "Fantasy",
    }
    topic = title.casefold()
    genres = Counter(str(genre).casefold() for film in films for genre in film.get("genres", []))
    candidates = [name for name in mapping if name in topic]
    candidates += [name for name, _count in genres.most_common() if name in mapping]
    for genre in candidates:
        tag = "#"+mapping[genre]
        if tag not in tags:
            tags.append(tag)
        if len(tags) == 5:
            break
    if len(tags) < 3:
        tags.append("#TVShows" if shows_only else "#MoviesToWatch")
    return tags


def description_hook(script: dict, films: list[dict]) -> str:
    """The first ~150 characters: what YouTube shows in search and above '...more'.

    It must carry the search phrase and the biggest names, so it never reuses narration.
    The writer's hook is used when it names at least two of the films; otherwise a plain
    sentence is built from the title and the top picks.
    """
    package = script.get("package", {})
    hook = " ".join(str(package.get("hook", "")).split())
    titles = [film["title"] for film in films if film.get("title")]
    if hook and len(hook) <= 240 and sum(title.casefold() in hook.casefold() for title in titles) >= min(2, len(titles)):
        return hook
    title = " ".join(str(package.get("title", "")).split()).rstrip(".!?")
    names = titles[:3]
    featuring = (", ".join(names[:-1]) + " and " + names[-1]) if len(names) > 1 else (names[0] if names else "")
    kind = "show" if films and all(film.get("kind") == "tv" for film in films) else "movie"
    first = f"{title}, featuring {featuring}." if title and featuring else (f"{title}." if title else "")
    return (first + f" What makes each {kind} worth your time, and where you can stream it.").strip()


def where_to_watch(film: dict) -> str:
    """Recognised services only: streaming first, otherwise rent/buy."""
    from .availability import services
    data = _availability(film)
    stream = services(data, ("subscription", "free", "ads"), limit=2)
    if stream:
        return " and ".join(stream)
    store = services(data, ("rent", "buy"), limit=2)
    return ("rent or buy on " + " and ".join(store)) if store else ""


def build_description(script: dict, films: list[dict], chapters: list[dict], trailer_sources: list[str],
                      credit: str = "") -> str:
    """Search-ready hook, the countdown with where to watch, chapters, a comment prompt, credits, hashtags."""
    package = script.get("package", {})
    chunks = [description_hook(script, films)]
    summary = str(package.get("description", "")).strip()
    # Old checkpoint copy may contain the exact paragraph the owner asked to remove.
    summary = re.sub(r"Film information:\s*TMDB\.\s*This product uses the TMDB API but is not endorsed or certified by TMDB\.?", "", summary, flags=re.IGNORECASE).strip()
    if summary and summary.casefold() not in chunks[0].casefold():
        chunks.append(summary)
    if chapters:
        chunks.append("Chapters\n" + "\n".join(f"{chapter_time(c['start'])} {c['title']}" for c in chapters))
    rows = []
    for rank in range(len(films), 0, -1):  # same order as the video: countdown to number one
        film = films[rank - 1]
        where = where_to_watch(film)
        rows.append(f"#{rank} {film['title']} ({film['year']})" + (f" — {where}" if where else ""))
    heading = featured_heading(films)
    dates = sorted({str(_availability(f).get("checked_at", ""))[:10] for f in films if _availability(f).get("checked_at")})
    note = f"\nWhere to watch: US listings{', checked ' + ', '.join(dates) if dates else ''}; availability can change." \
        if any(where_to_watch(f) for f in films) else ""
    chunks.append(heading + "\n" + "\n".join(rows) + note)
    question = str(package.get("pinned_comment", "")).strip()
    if question:
        chunks.append(f"{question} Tell me in the comments.")
    chunks.append("Subscribe for more countdowns and honest recommendations.")
    sources = "Trailer sources\n" + "\n".join(dict.fromkeys(trailer_sources)) if trailer_sources else ""
    chunks.append("\n".join(part for part in (
        "Trailer clips are used for commentary and review.", sources, credit.strip()) if part))
    # YouTube shows the first three hashtags above the title; they go last in the text.
    tags = topic_hashtags(str(package.get("title", "")), films)
    generic = {"#MovieRecommendations", "#TVRecommendations", "#WhatToWatch"}
    chunks.append(" ".join([t for t in tags if t not in generic] + [t for t in tags if t in generic]))
    return "\n\n".join(chunk for chunk in chunks if chunk) + "\n"
