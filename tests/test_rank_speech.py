"""Short title announcements use the same pronunciation contract as TTS."""
import pytest

from movie_pipeline.voice import speech_check


@pytest.fixture(autouse=True)
def repository_pronunciations_are_not_test_inputs(monkeypatch):
    monkeypatch.setattr("movie_pipeline.voice.custom_pronunciations", lambda: {})


def check(expected, heard):
    return speech_check(expected, [{"start": 0, "end": 3, "text": heard}])


@pytest.mark.parametrize("written,spoken", [
    ("M3GAN", "Megan"),
    ("M3GAN", "M3GAN"),
    ("Se7en", "Seven"),
    ("Dr. Strangelove", "Doctor Strangelove"),
    ("Mr. Bean", "Mister Bean"),
    ("Thir13en", "Thirteen"),
])
def test_known_title_pronunciations_verify_complete_short_announcements(written, spoken):
    result = check(f"Number 1: {written}.", f"Number one: {spoken}.")
    assert result["accepted"]
    assert result["coverage"] == 1
    assert result["precision"] == 1


@pytest.mark.parametrize("written,spoken", [
    ("1917", "nineteen seventeen"),
    ("1917", "one thousand nine hundred seventeen"),
    ("1923", "nineteen twenty-three"),
    ("1923", "one thousand nine hundred twenty-three"),
    ("2020", "twenty twenty"),
    ("2020", "two thousand twenty"),
    ("2001", "two thousand one"),
    ("2001", "two thousand and one"),
    ("2001", "twenty oh one"),
    ("2001", "twenty zero one"),
    ("1507", "fifteen oh seven"),
    ("1507", "one thousand five hundred seven"),
])
def test_cardinal_and_year_readings_preserve_the_exact_numeric_title(written, spoken):
    result = check(f"Number 1: {written}.", f"Number one: {spoken}.")
    assert result["accepted"], result
    assert result["coverage"] == 1
    assert result["precision"] == 1


@pytest.mark.parametrize("written,heard", [
    ("M3GAN", "Number one: Arrival."),
    ("1917", "Number one: nineteen twenty-three."),
    ("2001", "Number one: twenty one."),
    ("2001", "Number one: two thousand two."),
    ("2020", "Number one: twenty twenty-one."),
    ("M3GAN", "Number one."),
])
def test_wrong_or_missing_title_is_not_rescued_by_pronunciation_aliases(written, heard):
    assert not check(f"Number 1: {written}.", heard)["accepted"]


def test_pronunciation_fix_does_not_accept_missing_review_words():
    expected = "Number 1: M3GAN. A lonely scientist creates a friend who refuses to obey."
    heard = "Number one: Megan. A friend refuses."
    assert not check(expected, heard)["accepted"]


def test_pronunciation_fix_does_not_accept_runaway_extra_speech():
    heard = "Number one: Megan. Thank you for listening thank you for listening thank you for listening."
    assert not check("Number 1: M3GAN.", heard)["accepted"]


def test_diagnostics_keep_the_written_recognizer_transcript():
    heard = "Number one: M3GAN."
    assert check("Number 1: M3GAN.", heard)["heard"] == heard
