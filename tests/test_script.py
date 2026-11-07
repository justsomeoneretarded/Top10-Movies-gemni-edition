from unittest.mock import Mock

import pytest

from movie_pipeline.core import PipelineError
from movie_pipeline.script import SCRIPT_POLICY_VERSION, Writer

CLOSING = "a short closing thought and one audience question"
OPENING = "a punchy opening hook, without listing every title"
NATURAL_CLOSING = "That wraps up our picks for tonight. Which story sounds like your kind of trouble? Tell us below, then like and subscribe for more recommendations. Check out our other lists before you spend another evening arguing with the remote. Your next movie night deserves better."


def test_fifty_word_closing_does_not_abort_video():
    writer = Writer()
    text = NATURAL_CLOSING
    writer.ask = Mock(return_value={"paragraphs": [text]})
    assert writer.narration("Top 5", "Action", CLOSING, {}, 35) == [text]
    assert writer.ask.call_count == 1


@pytest.mark.parametrize("draft", [{"paragraphs": []}, {"paragraphs": ["Too short."]}, {"paragraphs": ["word " * 200]}])
def test_invalid_closing_uses_bounded_fallback(draft):
    writer = Writer()
    writer.ask = Mock(return_value=draft)
    result = writer.narration("Top 5", "Action", CLOSING, {}, 35)
    assert writer.ask.call_count == 3
    assert result and all(p.strip() for p in result)
    assert "like" in result[0].lower() and "subscribe" in result[0].lower()


def test_short_complete_review_is_kept_without_padding_retries():
    writer = Writer()
    writer.ask = Mock(return_value={"paragraphs": ["A short recommendation."]})
    assert writer.narration("Top 5", "Action", "rank 1: Example", {}, 216) == ["A short recommendation."]
    assert writer.ask.call_count == 1


def test_completed_section_is_reused_after_later_failure(tmp_path):
    first = Writer(cache_dir=tmp_path)
    text = NATURAL_CLOSING
    first.ask = Mock(return_value={"paragraphs": [text]})
    first.narration("Top 5", "Action", CLOSING, {}, 35)
    resumed = Writer(cache_dir=tmp_path)
    resumed.ask = Mock(side_effect=AssertionError("Should use checkpoint"))
    assert resumed.narration("Top 5", "Action", CLOSING, {}, 35) == [text]
    resumed.ask.assert_not_called()


@pytest.mark.parametrize("draft", [{"paragraphs": []}, {"paragraphs": ["word " * 200]}])
def test_invalid_opening_uses_fallback_instead_of_aborting(draft):
    writer = Writer()
    writer.ask = Mock(return_value=draft)
    result = writer.narration("Top 5", "Action", "a punchy opening hook, without listing every title", {}, 65)
    assert writer.ask.call_count == 3
    assert 6 <= len(" ".join(result).split()) <= 24
    assert "action" in " ".join(result)


def test_opening_can_be_natural_length():
    writer = Writer()
    text = "Five action films bring big risks and desperate characters. Here's the story behind each pick, without the big reveals."
    writer.ask = Mock(return_value={"paragraphs": [text]})
    result = writer.narration("Top 5", "Action", "a punchy opening hook, without listing every title", {}, 65)
    assert " ".join(result) == text
    assert all(len(p.split()) <= 65 for p in result)
    assert writer.ask.call_count == 1


def test_empty_movie_text_is_still_an_error():
    writer = Writer()
    writer.ask = Mock(return_value={"paragraphs": []})
    with pytest.raises(PipelineError, match="No usable narration text"):
        writer.narration("Top 5", "Action", "rank 1: Example", {}, 216)


FILMS = ["The Odyssey", "Spider-Man: Brand New Day", "Michael", "Toy Story 5", "Project Hail Mary"]


def test_intro_does_not_reveal_the_full_list_even_when_model_ignores_its_role():
    writer = Writer()
    bad = "Project Hail Mary is five. Toy Story 5 is four. Michael is three. Spider-Man: Brand New Day is two. The Odyssey is one. Which is your pick?"
    writer.ask = Mock(return_value={"paragraphs": [bad]})
    result = writer.narration("Top 5 Movies of 2026", "Action", "a punchy opening hook, without listing every title", {"titles":FILMS}, 65)
    assert writer.ask.call_count == 3
    assert not any(title in " ".join(result) for title in FILMS)
    assert "5 action movies" in " ".join(result)
    # The title list is kept for checking the response, not fed back as material to enumerate.
    assert all(title not in writer.ask.call_args_list[0].args[0] for title in FILMS)


def test_outro_is_a_goodbye_instead_of_a_second_countdown():
    writer = Writer()
    writer.ask = Mock(return_value={"paragraphs": ["Who's one? Let's count down. At number five, Project Hail Mary. Number four is Toy Story 5. Number three is Michael. Our movie list is about to begin."]})
    result = writer.narration("Top 5 Movies of 2026", "Action", CLOSING, {"titles":FILMS}, 35)
    text = " ".join(result)
    assert writer.ask.call_count == 3
    assert "subscribe" in text and "other recommendations" in text
    assert "Let's count down" not in text and "Who's one" not in text
    assert all(title not in text for title in FILMS)


def test_single_review_cannot_turn_into_the_whole_countdown():
    writer = Writer()
    writer.ask = Mock(return_value={"paragraphs": ["We're counting down the top five. Project Hail Mary is a space adventure. Next up, Toy Story 5 is family fun. Michael has music, then Spider-Man: Brand New Day brings action. Finally, The Odyssey is number one."]})
    facts = {"title":"Project Hail Mary", "rank":5, "overview":"A teacher wakes up aboard a spaceship with no memory of how he got there.",
             "other_titles":FILMS[:-1]}
    result = writer.narration("Top 5 Movies of 2026", "Keep it relaxed", "rank 5: Project Hail Mary", facts, 180)
    text = " ".join(result)
    assert writer.ask.call_count == 3
    assert "Project Hail Mary" in text and "teacher" in text
    assert all(title not in text for title in FILMS[:-1])
    assert all(len(paragraph.split()) <= 65 for paragraph in result)
    prompt = writer.ask.call_args_list[0].args[0]
    assert "SINGLE REVIEW" in prompt and all(title not in prompt for title in FILMS[:-1])


def test_one_movie_comparison_and_repeated_same_rank_remain_usable():
    writer = Writer()
    text = "Number five is Project Hail Mary. A teacher wakes up on a spaceship, far from home. This setup might appeal to fans of The Odyssey. At rank 5, the focus stays on this one adventure."
    writer.ask = Mock(return_value={"paragraphs":[text]})
    result = writer.narration("Top 5 Movies", "Relaxed", "rank 5: Project Hail Mary",
                              {"title":"Project Hail Mary", "rank":5, "other_titles":FILMS[:-1]}, 40)
    assert result == [text]
    assert writer.ask.call_count == 1


BAD_INTRO = "Tired of endless scrolling? Before you lose another night hunting for something real, here’s 1.0.0. But hold up—this list starts at 10, not one. Why? Because the real deal? It’s the last show on the list. Seriously, you won’t believe what’s coming next."


@pytest.mark.parametrize("bad", [BAD_INTRO, "Here comes 1.0-point-dot. Seriously, you won't believe what's next.", "Hold up! Why? The real deal is coming next."])
def test_empty_bait_and_spoken_numeric_debris_receive_useful_nonfatal_fallback(bad):
    writer = Writer()
    writer.ask = Mock(return_value={"paragraphs": [bad]})
    result = writer.narration("Top 10 Horror Shows", "Keep it relaxed", OPENING,
                              {"count": 10, "kind": "shows", "genres": ["Horror"]}, 65)
    text = " ".join(result)
    assert 6 <= len(text.split()) <= 28
    assert "10 horror shows" in text
    assert "1.0" not in text and "list starts" not in text
    assert writer.ask.call_count == 3


def test_semantic_retry_can_recover_without_rewriting_good_casual_language():
    writer = Writer()
    good = "Five action movies, with desperate heroes making dangerous choices. Here's the setup for each, without the big reveals."
    writer.ask = Mock(side_effect=[{"paragraphs": [BAD_INTRO]}, {"paragraphs": [good]}])
    assert writer.narration("Top 5 Action Movies", "Casual", OPENING, {"count": 5}, 18) == [good]
    assert writer.ask.call_count == 2


def test_repeated_review_padding_falls_back_to_verified_premise():
    writer = Writer()
    writer.ask = Mock(return_value={"paragraphs": ["This is something special and you should watch it. " * 8]})
    facts = {"title": "Example", "rank": 2, "overview": "A teacher wakes up aboard a spaceship with no memory of how he got there."}
    result = writer.narration("Top 5 Movies", "Casual", "rank 2: Example", facts, 140)
    assert writer.ask.call_count == 3
    assert result == ["Number 2: Example.", facts["overview"]]
    assert "something special" not in " ".join(result)


def test_outro_cannot_promise_a_final_show_after_every_review_finished():
    writer = Writer()
    writer.ask = Mock(return_value={"paragraphs": ["The last show we'll mention is a real surprise. Stay tuned and find out soon."]})
    result = writer.narration("Top 10 Shows", "Casual", CLOSING, {}, 30)
    assert "subscribe" in " ".join(result)
    assert "find out soon" not in " ".join(result)


def cached_script():
    films = [{"title": "Example", "kind": "tv", "genres": ["Horror"],
              "overview": "A teacher discovers an empty town where everyone disappears after sunset."}]
    safe = "At number one, Example traps a teacher in an empty town. This guy picked a seriously bad place to work."
    available = "You can watch it on Hulu."
    script = {"package": {"title": "Top shows"}, "sections": [
        {"kind": "intro", "film": 0, "paragraphs": [BAD_INTRO]},
        {"kind": "film", "film": 0, "paragraphs": [safe, available], "availability_text": available},
        {"kind": "outro", "film": 0, "paragraphs": [NATURAL_CLOSING]},
    ]}
    return films, script


def test_editorial_cache_migration_refreshes_stale_reviews_and_reuses_current_policy():
    films, script = cached_script()
    writer = Writer()
    writer.ask = Mock(return_value={"paragraphs": ["One horror show traps a teacher in a disappearing town. Here's the setup, without the big reveals."]})
    result = writer.refresh_editorial("Top Horror Shows", "Casual", films, script)
    assert writer.ask.call_count >= 3
    assert result["package"]["script_policy_version"] == SCRIPT_POLICY_VERSION
    assert result["sections"][1]["paragraphs"] != script["sections"][1]["paragraphs"][:1]
    assert script["sections"][0]["paragraphs"] == [BAD_INTRO]
    writer.ask.reset_mock()
    assert writer.refresh_editorial("Top Horror Shows", "Casual", films, result) == result
    writer.ask.assert_not_called()


def test_editorial_repair_excludes_checked_availability_from_spoken_script():
    films, script = cached_script()
    script["package"]["script_policy_version"] = SCRIPT_POLICY_VERSION
    script["sections"][0]["paragraphs"] = ["One horror show traps a teacher in a disappearing town. Here's the setup, without the big reveals."]
    script["sections"][1]["paragraphs"][0] = "This is something special and you should watch it. " * 8
    writer = Writer()
    writer.ask = Mock(return_value={"paragraphs": ["At number one, Example follows a teacher in a town where everyone disappears after sunset. That is a rough first day."]})
    result = writer.refresh_editorial("Top Horror Shows", "Casual", films, script)
    assert writer.ask.call_count >= 1
    assert all("Hulu" not in call.args[0] for call in writer.ask.call_args_list)
    assert result["sections"][1]["paragraphs"].count("You can watch it on Hulu.") == 0
    assert result["sections"][1]["availability_text"] == ""


def test_measured_intro_retry_honors_shorter_goal_without_modifying_body():
    films, script = cached_script()
    writer = Writer()
    writer.ask = Mock(return_value={"paragraphs": ["One horror show. Here's the story behind the scares."]})
    result = writer.refresh_intro("Top Horror Shows", "Casual", films, script, target_words=20)
    assert result["sections"][1:] == script["sections"][1:]
    assert "About 20 words" in writer.ask.call_args.args[0]
    assert result["package"]["script_policy_version"] == SCRIPT_POLICY_VERSION


def test_opening_model_unavailable_uses_useful_local_fallback():
    import requests
    writer = Writer()
    writer.ask = Mock(side_effect=requests.ConnectionError("offline"))
    result = writer.narration("Top 5 Action Movies", "Casual", OPENING, {"count": 5}, 18)
    assert "5 action movies" in " ".join(result)
    assert writer.ask.call_count == 3


def test_mixed_list_fallback_describes_variety_without_claiming_one_shared_genre():
    films = [{"title": "Space", "genres": ["Science Fiction"]},
             {"title": "Laughs", "genres": ["Comedy"]}]
    writer = Writer()
    writer.ask = Mock(return_value={"paragraphs": []})
    script = {"sections": [{"kind": "intro", "paragraphs": [BAD_INTRO]}], "package": {}}
    result = writer.refresh_editorial("Top 2 Movies", "Pick something for tonight", films, script)
    text = " ".join(result["sections"][0]["paragraphs"])
    assert "2 amazing movies" in text
    assert "2 sci-fi movies" not in text


def test_editorial_repair_with_sparse_facts_never_fails_just_over_bad_draft():
    writer = Writer()
    writer.ask = Mock(return_value={"paragraphs": ["This is something special and you should watch it. " * 8]})
    result = writer.narration("Top Movies", "Casual", "rank 2: Example",
                              {"title": "Example", "rank": 2, "genres": ["Fantasy"]}, 150)
    assert "Example" in " ".join(result)
    assert result == ["Number 2: Example."]
    assert "something special" not in " ".join(result)


def test_rank_lead_in_is_short_without_removing_the_actual_premise():
    from movie_pipeline.script import short_rank_opening
    assert short_rank_opening('At number 7 on our action countdown is Arrival. A translator faces an impossible choice.', 7, 'Arrival') == 'Number 7: Arrival. A translator faces an impossible choice.'
    assert short_rank_opening('Arrival follows a translator.', 7, 'Arrival') == 'Number 7: Arrival follows a translator.'
    original = 'A translator faces a choice in Arrival.'
    assert short_rank_opening(original, 7, 'Arrival') == original


def test_long_opening_is_trimmed_keeping_hook_and_tease():
    from movie_pipeline.script import fit_opening
    draft = ["Some movies feel like a plane ticket. These three drop you in jungles, deserts and lost cities. "
             "Each one has heroes in way over their heads. Every one of them is perfect for a lazy weekend. "
             "And number one is the trip you'll want to take twice."]
    fitted = fit_opening(draft, 30)
    assert fitted and len(fitted[0].split()) <= 30
    assert fitted[0].startswith("Some movies feel like a plane ticket.")
    assert fitted[0].endswith("number one is the trip you'll want to take twice.")
    assert fit_opening(["One very long sentence " * 10 + "."], 20) == []


def test_saved_fallback_opening_is_replaced_on_the_next_run():
    from unittest.mock import Mock

    from movie_pipeline.script import Writer, _fallback_intro, _intro_facts, is_fallback_opening
    films = [{"title": "Dune", "year": 2024, "kind": "movie", "genres": ["Adventure"]},
             {"title": "Jumanji", "year": 2017, "kind": "movie", "genres": ["Adventure"]}]
    facts = _intro_facts("2 Adventure Movies", "", films)
    fallback = _fallback_intro("2 Adventure Movies", "", facts)
    assert is_fallback_opening(fallback, "2 Adventure Movies", "", facts)
    assert not is_fallback_opening(["Some movies feel like a plane ticket. And number one is the trip you'll take twice."], "2 Adventure Movies", "", facts)
    writer = Writer()
    writer.narration = Mock(return_value=["Some movies feel like a plane ticket. And number one is the trip you'll take twice."])
    script = {"sections": [{"kind": "intro", "film": 0, "paragraphs": fallback}], "package": {"script_policy_version": __import__("movie_pipeline.script", fromlist=["x"]).SCRIPT_POLICY_VERSION}}
    refreshed = writer.refresh_editorial("2 Adventure Movies", "", films, script)
    assert refreshed["sections"][0]["paragraphs"][0].startswith("Some movies feel like a plane ticket.")


def test_plan_retries_a_cut_off_answer_with_more_room(tmp_path, monkeypatch):
    from movie_pipeline import script as module
    writer = module.Writer(cache_dir=tmp_path)
    budgets = []

    def ask(prompt, schema, **options):
        budgets.append(options["num_predict"])
        if len(budgets) == 1:
            raise ValueError("Unterminated string")  # a truncated JSON answer
        return {"promise": "p", "angle": "a", "opening_hook": "h", "open_loop": "o", "comment_question": "q", "entries": []}

    monkeypatch.setattr(writer, "ask", ask)
    films = [{"title": f"Show {i}", "year": 2026} for i in range(10)]
    plan = writer.plan("10 New Shows of 2026", "brief", films)
    assert plan["angle"] == "a" and len(budgets) == 2 and budgets[0] >= 3300 and budgets[1] > budgets[0]
    assert list(tmp_path.glob("*.json"))


def test_mixed_genre_brief_does_not_become_a_scifi_countdown():
    from movie_pipeline.script import _fallback_intro, _intro_facts, _theme

    title = "10 New Shows of 2026 That Are So Freaking Good"
    brief = ("Range across medical drama, fantasy, spy thriller, sci-fi mystery, "
             "superhero crime, horror and political conspiracy so there is something for everyone.")
    films = [{"title": "The Pitt", "kind": "tv", "genres": ["Drama"]},
             {"title": "Severance", "kind": "tv", "genres": ["Science Fiction", "Drama"]}]
    facts = _intro_facts(title, brief, films)
    facts["count"] = 10
    assert _theme(title, brief, facts) == ""
    assert _fallback_intro(title, brief, facts) == ["10 amazing shows worth adding to your watchlist."]


@pytest.mark.parametrize("topic,brief,expected", [
    ("Top 5 Horror Shows", "Mix scares, mystery and drama.", "horror"),
    ("Top 10 Shows", "Horror, sci-fi and crime.", ""),
    ("Top 10 Sci-Fi and Fantasy Shows", "", ""),
    ("Top 5 Movies", "Action movies only.", "action"),
])
def test_explicit_theme_wins_but_mixed_requests_stay_general(topic, brief, expected):
    from movie_pipeline.script import _theme

    assert _theme(topic, brief, {"shared_genres": ["Drama"]}) == expected


def test_script_no_longer_rejects_figures_against_source_metadata():
    writer = Writer()
    draft = "Number 1: Example. The story jumps to 2099."
    writer.ask = Mock(return_value={"paragraphs": [draft]})
    assert writer.narration("Top 2", "Casual", "rank 1: Example", {"title": "Example", "rank": 1}, 100) == [draft]
    assert writer.ask.call_count == 1


def test_model_failure_for_mixed_brief_keeps_general_opening():
    writer = Writer()
    writer.ask = Mock(return_value={"paragraphs": []})
    result = writer.narration("Top 10 Shows", "Sci-fi, horror, crime and drama.", OPENING,
                              {"count": 10, "kind": "shows", "shared_genres": ["Drama"]}, 28)
    assert "10 amazing shows" in " ".join(result)
    assert "sci-fi shows" not in " ".join(result)
