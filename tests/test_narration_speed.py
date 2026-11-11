from movie_pipeline.sound_design import narration_speed


def test_speed_dropdown_values_and_safe_default(monkeypatch):
    for value, expected in (("1", 1.0), ("1.1", 1.1), ("1.2", 1.2), ("1.3", 1.3), ("", 1.2), ("2", 1.3)):
        monkeypatch.setenv("NARRATION_SPEED", value)
        assert narration_speed() == expected
    monkeypatch.delenv("NARRATION_SPEED")
    assert narration_speed() == 1.2
