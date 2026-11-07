from movie_pipeline.core import write_srt
from movie_pipeline.subtitles import MAX_CUE_CHARS, script_cues, two_lines

HEARD = [{"start": 0.2, "end": 3.0, "text": "Number one, inception directed by"},
         {"start": 3.0, "end": 6.1, "text": "christopher nolan came out in twenty ten."},
         {"start": 6.3, "end": 9.0, "text": "It won four academy awards"}]
SCRIPT = "Number 1: Inception, directed by Christopher Nolan, came out in 2010. It won four Academy Awards."


def test_captions_use_exact_script_spelling_with_recognized_timing():
    cues = script_cues(SCRIPT, HEARD)
    assert " ".join(cue["text"] for cue in cues) == SCRIPT
    assert cues[0]["start"] == 0.2 and cues[-1]["end"] == 9.0
    assert all(len(cue["text"]) <= MAX_CUE_CHARS for cue in cues)
    assert all(a["end"] <= b["start"] for a, b in zip(cues, cues[1:]))
    assert "Christopher Nolan" in cues[0]["text"] and "2010." in cues[0]["text"]


def test_unreliable_recognition_keeps_original_cues():
    garbled = [{"start": 0, "end": 2, "text": "completely different words here"}]
    assert script_cues("Number 1: Inception is great.", garbled) == garbled
    assert script_cues("", HEARD) == HEARD


def test_long_captions_become_two_balanced_lines(tmp_path):
    assert two_lines("Short caption") == "Short caption"
    first, second = two_lines("Number 1: Inception, directed by Christopher Nolan, came out in 2010.").split("\n")
    assert len(first) <= 42 and len(second) <= 42
    path = tmp_path / "subtitles.srt"
    write_srt(path, script_cues(SCRIPT, HEARD))
    assert "Inception, directed by\nChristopher Nolan" in path.read_text() or "\n" in path.read_text().split("-->")[1]
