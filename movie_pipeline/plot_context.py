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


# Limit automatic cast-tail cuts to complete lists of full names. A named
# companion with an action or purpose remains plot, as does an ambiguous bare
# first name; the meaning-led editor can decide those cases.
_NAME_WORD = r"(?:(?:Mr|Mrs|Ms|Dr|Prof|St)\.|[A-Z][A-Za-z'’\-]*|[A-Z]\.)"
_PERSON_NAME = rf"{_NAME_WORD}(?:\s+{_NAME_WORD})+"
_CAST_TAIL = re.compile(
    rf",\s+(?i:alongside)\s+{_PERSON_NAME}"
    rf"(?:\s*(?:,\s*(?:(?i:and)\s+)?|(?i:and)\s+|&\s*){_PERSON_NAME})+"
    r"(?P<punctuation>[.!?]?)$")
_NAMED_ROLE = re.compile(
    rf"^{_NAME_WORD}(?:\s+{_NAME_WORD})*(?:,\s*|\s+)"
    r"(?i:a|an|the|who|whose)\s+\S")
_INITIAL_END = re.compile(r"(?:^|\s)(?:[A-Za-z]\.)+$")
_TITLE_END = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|St)\.$", re.I)
_SUFFIX_END = re.compile(r"\b(?:Jr|Sr)\.$", re.I)


def _story_sentences(value: str) -> list[str]:
    """Keep initials and titles attached to the following name or role."""
    result = []
    for part in re.split(r"(?<=[.!?])\s+", value.strip()):
        if result and (
                _INITIAL_END.search(result[-1]) or _TITLE_END.search(result[-1]) or
                (_SUFFIX_END.search(result[-1]) and part[:1].islower())):
            result[-1] += " " + part
        else:
            result.append(part)
    return result


def _role_plot(text: str) -> str:
    role = re.search(r"\bas\s+(.+)", text, re.I)
    if not role:
        return ""
    premise = role.group(1)
    cast_tail = _CAST_TAIL.search(premise)
    if cast_tail:
        premise = premise[:cast_tail.start()].rstrip() + cast_tail["punctuation"]
    if re.match(r"^(?:a|an|the)\s+\S", premise, re.I) or _NAMED_ROLE.match(premise):
        return "The story follows " + premise
    # "Stars Actor as Ada" is still only a credit, not a supplied storyline.
    return ""


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
        if premise := _role_plot(text):
            return premise
    if credit:
        tail = text[credit.end():]
        explicit = bool(re.match(
            r"^(?:the|this) (?:film|movie|show|series) (?:is|was) ", text, re.I))
        # A passive followed by a purpose or an independent subject may describe
        # something inside the story: "Created by Umbrella, the virus spreads."
        # Retain ambiguous prose for the meaning-led editor instead of guessing.
        production_subject = bool(re.search(
            r"[,;:]\s*(?:the|this) (?:film|movie|show|series|story)\b", tail, re.I))
        if not explicit and not production_subject and (
                re.search(r"\bto\s+\w+", tail, re.I) or
                re.search(r"[,;:]\s*(?!and\b|&)\S", tail, re.I)):
            return text
        if not explicit and re.match(r"(?:a|an|the|his|her|their)\s+", tail, re.I):
            return text
    return ""


def story_text(value: str) -> str:
    """Keep plot in mixed source text without replacing creative vocabulary."""
    if not isinstance(value, str):
        return ""
    return " ".join(clean for sentence in _story_sentences(value)
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
