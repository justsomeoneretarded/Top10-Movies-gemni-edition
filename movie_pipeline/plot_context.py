"""Plot material for spoken reviews; full metadata stays with research and packaging."""
from __future__ import annotations

from copy import deepcopy
import re

STORY_FIELDS = (
    "title", "year", "kind", "genres", "keywords", "characters", "character_context",
    "overview", "premise", "plot", "plot_summary", "synopsis", "storyline", "story",
    "wiki_plot", "wiki_synopsis", "research_plot", "wiki_summary", "summary",
    "research_notes", "grounded_notes",
)
CONTEXT_FIELDS = (
    "rank", "titles", "other_titles", "count", "shared_genres", "requested_years",
    "opening_word_goal", "topic", "comment_question", "opening_hook_idea",
    "open_loop_idea", "list_angle", "video_promise", "episode_plan",
    "previous_narration_context", "available_scenes", "requested_creative_context",
)
_IDENTITY_FIELDS = {"title", "year", "kind", "genres", "keywords", "characters"}
_METADATA_FIELDS = {
    "directors", "director", "creators", "creator", "cast", "cast_profiles",
    "actors", "actor", "credits", "production", "wiki_production", "reception",
    "wiki_reception", "ratings", "rating", "tmdb_rating", "tmdb_votes", "tagline",
    "sources", "source", "research_sources", "url",
}
_CREDIT_HEAD = re.compile(
    r"^(?:(?:the|this) (?:film|movie|show|series) (?:is|was) )?"
    r"(?:written|directed|created|produced|co-created|co-written|co-directed)"
    r"(?: and (?:written|directed|created|produced))? by\s+", re.I)
_STARRING_HEAD = re.compile(
    r"^(?:(?:the|this) (?:film|movie|show|series) )?"
    r"(?:stars?|starring|features? performances? by)\s+", re.I)
_CATALOGUE = re.compile(
    r"^.{1,100}?\s+is\s+(?:an?\s+)?(?:18|19|20)\d{2}\b.{0,220}?"
    r"\b(?:film|movie|series|show)\b.{0,120}?"
    r"\b(?:directed|written|created|produced|starring)\b", re.I)
_STORY_CONTINUATION = re.compile(
    r"\b((?:the|this) (?:film|movie|show|series|story)\s+"
    r"(?:follows|centres on|centers on|is about)\b.*)$", re.I)


def _story_sentence(sentence: str) -> str:
    text = sentence.strip()
    credit = _CREDIT_HEAD.match(text)
    starring = _STARRING_HEAD.match(text)
    catalogue = _CATALOGUE.match(text)
    if not (credit or starring or catalogue):
        return text
    continuation = _STORY_CONTINUATION.search(text)
    if continuation:
        retained = continuation.group(1)
        return retained[:1].upper() + retained[1:]
    if starring or catalogue:
        # Preserve a mixed actor/role premise without speaking the actor credit.
        role = re.search(r"\bas\s+((?:a|an|the)\s+.+)", text, re.I)
        if role:
            return "The story follows " + role.group(1)
    if credit:
        tail = text[credit.end():]
        explicit = bool(re.match(
            r"^(?:the|this) (?:film|movie|show|series) (?:is|was) ", text, re.I))
        # A passive followed by a purpose or an independent subject may describe
        # something inside the story: "Created by Umbrella, the virus spreads."
        # Retain ambiguous prose for the meaning-led editor instead of guessing.
        if not explicit and (re.search(r"\bto\s+\w+", tail, re.I) or
                             re.search(r"[,;:]\s*(?:the|a|an|he|she|they|it|his|her|their)\b", tail, re.I)):
            return text
        if not explicit and re.match(r"(?:a|an|the|his|her|their)\s+", tail, re.I):
            return text
    return ""


def story_text(value: str) -> str:
    """Keep plot in mixed source text without replacing creative vocabulary."""
    if not isinstance(value, str):
        return ""
    return " ".join(clean for sentence in re.split(r"(?<=[.!?])\s+", value.strip())
                    if (clean := _story_sentence(sentence)))


def remove_credit_sentences(paragraphs: list[str]) -> list[str]:
    """Remove structural production credits while preserving plot paragraphs."""
    return [clean for paragraph in paragraphs if isinstance(paragraph, str)
            and (clean := story_text(paragraph))]


def character_names(film: dict) -> list[str]:
    """Read fictional names from original profiles, never actor-as-role strings."""
    result = []
    supplied = film.get("characters", [])
    for value in supplied if isinstance(supplied, list) else []:
        if isinstance(value, str) and value.strip() and value.strip() not in result:
            result.append(value.strip())
    profiles = film.get("cast_profiles")
    for person in profiles if isinstance(profiles, list) else []:
        value = person.get("character") if isinstance(person, dict) else None
        if isinstance(value, str) and value.strip() and value.strip() not in result:
            result.append(value.strip())
    return result[:8]


def _story_value(value):
    if isinstance(value, str):
        return story_text(value)
    if isinstance(value, list):
        return [clean for item in value if (clean := _story_value(item)) not in (None, "", [], {})]
    if isinstance(value, dict):
        return {name: clean for name, item in value.items()
                if name not in _METADATA_FIELDS
                and (clean := _story_value(item)) not in (None, "", [], {})}
    return deepcopy(value)


def plot_material(facts: dict, *, include_creative_context: bool = False) -> dict:
    """Project story and section controls; creative background is explicit opt-in."""
    result = {}
    for name in (*STORY_FIELDS, *CONTEXT_FIELDS):
        value = facts.get(name)
        if value in (None, "", [], {}):
            continue
        if name not in _IDENTITY_FIELDS and name not in {"available_scenes", "requested_creative_context"}:
            value = _story_value(value)
        if value not in (None, "", [], {}):
            result[name] = deepcopy(value)
    people = character_names(facts)
    if people:
        result["characters"] = people
    if include_creative_context and facts.get("creative_context"):
        result["creative_context"] = deepcopy(facts["creative_context"])
    return result
