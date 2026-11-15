from movie_pipeline.voice import speech_check


def test_sound_alike_names_pass_the_speech_check():
    assert speech_check("Number 5: Dredd.", [{"start": 0, "end": 2, "text": "number five dread"}])["accepted"]
    assert speech_check("Number 4: Carrie.", [{"start": 0, "end": 2, "text": "Number four, carry."}])["accepted"]
    assert not speech_check("Number 5: Dredd.", [{"start": 0, "end": 2, "text": "number five bread"}])["accepted"]
    assert not speech_check("Number 4: Carrie.", [{"start": 0, "end": 2, "text": "number four, cat"}])["accepted"]
