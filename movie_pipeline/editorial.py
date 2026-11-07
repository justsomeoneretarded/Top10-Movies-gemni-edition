"""Small timeline decisions based on measured speech rather than guessed word counts."""
from __future__ import annotations


def subscription_boundary(sections: list[dict], lengths: dict[tuple[int, int], float], beat_lengths: dict[int, float], cta_seconds: float, ending_hold: float = 3, *, opening_seconds: float = 0) -> tuple[int, float]:
    durations = [sum(lengths[(i,p)] for p in range(len(section['paragraphs']))) + beat_lengths.get(i, 0)
                 for i, section in enumerate(sections)]
    total = sum(durations)+cta_seconds+ending_hold+opening_seconds
    elapsed = opening_seconds
    choices = []
    films_seen = 0
    for i, (section, seconds) in enumerate(zip(sections, durations)):
        if section['kind'] == 'film' and films_seen:
            fraction = elapsed/max(total, .1)
            choices.append((i, fraction))
        elapsed += seconds
        films_seen += section['kind'] == 'film'
    # Whole reviews can leave no boundary in the requested range. Preserve the review,
    # choose the nearest boundary, and never fail or insert halfway through a review.
    if not choices:
        return max(1, len(sections)-1), elapsed/max(total, .1)
    inside = [choice for choice in choices if .30 <= choice[1] <= .50]
    return min(inside or choices, key=lambda choice: abs(choice[1]-.40))


def place_intro_meme(plan: list[dict], text: str, cues: list[dict], meme: dict) -> tuple[list[dict], bool]:
    """Replace one existing readable beat near the matching joke, without adding cuts."""
    import re
    pattern = r'\babsolute\s+(?:cinema|masterpieces?)\b'
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return plan, False
    cue = next((cue for cue in cues if re.search(pattern, cue['text'], re.IGNORECASE)), None)
    total = sum(shot['duration'] for shot in plan)
    when = cue['start'] if cue else total*match.start()/max(1,len(text))
    elapsed = 0.0
    choices = []
    for index, shot in enumerate(plan):
        if shot['duration'] >= 2.5:
            distance = 0 if elapsed <= when < elapsed+shot['duration'] else min(abs(elapsed-when), abs(elapsed+shot['duration']-when))+.001
            choices.append((distance, index))
        elapsed += shot['duration']
    if not choices:
        return plan, False
    _, index = min(choices)
    result = [dict(shot) for shot in plan]
    result[index].update(source=meme['path'], image=True, still=True, start=0,
                         visual_reason='relevant-intro-meme')
    return result, True
