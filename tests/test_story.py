import pytest

from movie_pipeline.script import (DELIVERIES, Writer, aligned_delivery, film_facts,
                                   rank_weights)

FILMS = [
    {"title": "Inception", "year": 2010, "kind": "movie", "genres": ["Science Fiction"],
     "overview": "A thief who steals secrets from dreams is offered a chance to plant an idea instead.",
     "tagline": "Your mind is the scene of the crime.", "runtime_minutes": 148,
     "wiki_production": "The rotating hallway fight was filmed practically on a full-size rotating set.",
     "wiki_reception": "It won four Academy Awards.",
     "cast_profiles": [{"name": "Leonardo DiCaprio", "character": "Cobb"}], "trailers": [{"key": "x"}],
     "watch_availability": {"flatrate": ["Max"]}},
    {"title": "Arrival", "year": 2016, "kind": "movie", "genres": ["Science Fiction", "Drama"],
     "overview": "A linguist is recruited to communicate with alien visitors."},
]


def test_film_facts_keep_story_material_and_drop_media_records():
    facts = film_facts(FILMS[0])
    assert facts["cast"] == ["Leonardo DiCaprio as Cobb"]
    assert facts["wiki_production"].startswith("The rotating hallway")
    assert "trailers" not in facts and "watch_availability" not in facts


def test_narration_material_keeps_plot_but_omits_database_readouts():
    from movie_pipeline.script import narration_material
    material = narration_material({**film_facts(FILMS[0]), "tagline": "Dream bigger",
                                   "tmdb_rating": 8.5, "tmdb_votes": 1000})
    assert material["overview"] == FILMS[0]["overview"]
    assert not {"tagline", "tmdb_rating", "tmdb_votes"} & material.keys()


def test_escalation_gives_number_one_the_most_time_without_changing_the_total():
    weights = rank_weights(10)
    assert weights[0] == max(weights) and weights[-1] == min(weights)
    assert sum(weights) == pytest.approx(10)
    assert rank_weights(1) == [1.0]


def test_older_scripts_get_sensible_delivery_labels():
    section = {"kind": "film", "film": 0, "paragraphs": ["Number 1: Inception.", "More.", "On Max."],
               "availability_text": "On Max."}
    assert aligned_delivery(section) == ["hype", "normal", "normal"]
    assert aligned_delivery({"kind": "outro", "paragraphs": ["Bye."], "delivery": ["shouting"]}) == ["normal"]


def fake_model(edit_text=None, unsupported=None):
    calls = {"plan": 0, "edit": 0, "narration": 0, "bridges": 0}

    def ask(prompt, schema, **options):
        props = schema["properties"]
        if "bridges" in props:
            calls["bridges"] += 1
            return {"bridges": ["If that one bent your brain, the next one goes even further."]}
        if "entries" in props:
            calls["plan"] += 1
            return {"angle": "Movies that bend reality.", "opening_hook": "Some movies mess with your head on purpose.",
                    "open_loop": "Number one changed how blockbusters think.",
                    "entries": [{"rank": 1, "hook": "A heist inside a dream.", "angle": "Practical spectacle",
                                 "standout_facts": ["Real rotating hallway set"], "why_here": "Biggest swing",
                                 "who_for": "Puzzle lovers"},
                                {"rank": 2, "hook": "First contact, told through language.", "angle": "Quiet sci-fi",
                                 "standout_facts": ["Linguist lead"], "why_here": "Emotional", "who_for": "Thinkers"}],
                    "comment_question": "Which movie messed with your head the most?"}
        if props["paragraphs"]["items"].get("type") == "object":
            calls["edit"] += 1
            draft = prompt.split("Draft narration: ", 1)[1].split("\n", 1)[0]
            import json
            paragraphs = json.loads(draft)
            if edit_text:
                paragraphs = edit_text
            return {"paragraphs": [{"text": text, "delivery": "dramatic"} for text in paragraphs],
                    "unsupported_claims": unsupported or []}
        calls["narration"] += 1
        if "OPENING ONLY" in prompt:
            return {"paragraphs": ["Two sci-fi movies that play with your mind. Stick around, number one changed blockbusters."]}
        if "OUTRO ONLY" in prompt:
            return {"paragraphs": ["Which one messed with your head most? Like, subscribe, and watch our other picks."]}
        title = "Inception" if "'Inception'" in prompt else "Arrival"
        rank = 1 if title == "Inception" else 2
        return {"paragraphs": [f"Number {rank}: {title}. A sharp, specific premise with a reason to watch tonight. "
                               "The idea is simple to follow and the payoff is worth it for anyone who loves big ideas."]}
    return ask, calls


def test_write_plans_once_edits_every_section_and_records_delivery(tmp_path):
    writer = Writer(cache_dir=tmp_path)
    writer.ask, calls = fake_model()
    writer.thumbnail_concepts = lambda *args: []
    original_ask = writer.ask

    def ask(prompt, schema, **options):
        if "thumbnail_texts" in schema["properties"]:
            return {"title": "Top 2 Mind-Bending Movies", "description": "Two sci-fi picks.",
                    "thumbnail_texts": ["Mind Benders", "Reality Breaks", "Watch These"]}
        return original_ask(prompt, schema, **options)
    writer.ask = ask
    script = writer.write("Top 2 Mind-Bending Movies", "Casual", FILMS, target_seconds=120)
    assert calls["plan"] == 1
    assert calls["edit"] == 4  # intro, two reviews, outro
    for section in script["sections"]:
        assert len(section["delivery"]) == len(section["paragraphs"])
        assert set(section["delivery"]) <= set(DELIVERIES)
    assert script["package"]["episode_plan"]["comment_question"].startswith("Which movie")


def test_editor_does_not_apply_a_factual_rejection_gate():
    writer = Writer()
    writer.ask, _ = fake_model(edit_text=["Number 1: Inception. It earned 900 million dollars in 2011 alone, a record."])
    draft = ["Number 1: Inception. A thief plants an idea inside a dream, and the hallway fight was shot on a real rotating set."]
    facts = {**film_facts(FILMS[0]), "rank": 1, "other_titles": ["Arrival"]}
    paragraphs, delivery = writer.edit("Top 2", "Casual", "the review", facts, draft, rank=1)
    assert paragraphs == ["Number 1: Inception. It earned 900 million dollars in 2011 alone, a record."]
    assert delivery == ["dramatic"]


def test_editor_output_that_passes_checks_is_used():
    writer = Writer()
    better = ["Number 1: Inception. A thief plants an idea inside a dream, and that hallway fight? A real rotating set."]
    writer.ask, _ = fake_model(edit_text=better)
    draft = ["Number 1: Inception. A thief is asked to plant an idea inside a dream, and the hallway fight was shot on a real rotating set."]
    facts = {**film_facts(FILMS[0]), "rank": 1, "other_titles": ["Arrival"]}
    paragraphs, delivery = writer.edit("Top 2", "Casual", "the review", facts, draft, rank=1)
    assert paragraphs == better
    assert delivery == ["dramatic"]


def test_plan_lines_that_name_other_titles_are_dropped():
    writer = Writer()

    def ask(prompt, schema, **options):
        return {"angle": "Reality benders", "opening_hook": "Inception and Arrival are here.", "open_loop": "Wait for it.",
                "entries": [{"rank": 2, "hook": "Better than Inception?", "angle": "Language", "standout_facts": [],
                             "why_here": "Quiet", "who_for": "Thinkers"}],
                "comment_question": "Favorite?"}
    writer.ask = ask
    plan = writer.plan("Top 2", "Casual", FILMS)
    assert plan["opening_hook"] == ""  # it named titles, which the opening must not reveal
    assert plan["entries"]["2"]["hook"] == ""  # it named another entry
    assert plan["entries"]["2"]["angle"] == "Language"


def test_titles_are_ranked_for_length_number_topic_and_no_bait():
    from movie_pipeline.script import rank_titles, title_score
    options = ["YOU WON'T BELIEVE THESE 10 INSANE ACTION MOVIES!!",
               "Top 10 Action Movies of 2026 You Shouldn't Miss",
               "Action Movies",
               "Top 10 Action Movies of 2026 That Prove Big Screen Spectacle Is Back and Better Than Ever Before Now"]
    ranked = rank_titles(options, "Top 10 Action Movies of 2026", 10)
    assert ranked[0] == "Top 10 Action Movies of 2026 You Shouldn't Miss"
    assert title_score("x" * 101, "topic", 10) == float("-inf")


def test_tags_include_each_title_and_stay_under_youtube_limit():
    from movie_pipeline.script import youtube_tags
    films = [{"title": f"Movie {i}", "year": 2020 + i % 5, "kind": "movie"} for i in range(30)]
    tags = youtube_tags(["best action movies", "#ActionMovies", "best action movies"], "Top 30 Action Movies", films)
    assert tags[0] == "best action movies" and tags.count("best action movies") == 1
    assert "ActionMovies" in tags and "Movie 0 2020" in tags
    assert len(", ".join(tags)) <= 520


def test_upload_extras_are_written(tmp_path):
    from movie_pipeline.cli import write_upload_extras
    write_upload_extras(tmp_path, {"title": "Best", "title_options": ["Best", "Second"], "tags": ["a", "b"],
                                   "pinned_comment": "Which pick surprised you?"}, FILMS)
    assert "1. Best" in (tmp_path / "title-options.txt").read_text()
    assert (tmp_path / "tags.txt").read_text().strip() == "a, b"
    assert (tmp_path / "pinned-comment.txt").read_text().startswith("Which pick surprised you?")


def test_strong_genre_openings_are_not_rejected_for_missing_the_label():
    from movie_pipeline.script import _editorial_issue
    good = ["What if time isn't a straight line? These five movies bend physics and memory until reality cracks."]
    assert _editorial_issue(good, opening=True, theme="sci-fi") is None
    vague = ["Five movies you should watch tonight. Here's why each one is worth your time."]
    assert _editorial_issue(vague, opening=True, theme="sci-fi") is None  # quality editing does not require genre keywords


def test_removed_fact_checkbox_does_not_disable_general_editorial_work(monkeypatch):
    monkeypatch.setenv("FACT_CHECK", "false")
    writer = Writer()
    better = ["Number 1: Inception. A thief has to plant an idea inside a dream."]
    writer.ask, _ = fake_model(edit_text=better)
    draft = ["Number 1: Inception. A significant and compelling account of the complex dream world."]
    result, delivery = writer.edit("Top 2", "Casual", "the review", film_facts(FILMS[0]), draft, rank=1)
    assert result == better and delivery == ["dramatic"]


def test_partial_write_warms_the_cache_for_the_full_script(tmp_path):
    warm = Writer(cache_dir=tmp_path)
    warm.ask, calls = fake_model()
    partial = warm.write("Top 2 Mind-Bending Movies", "Casual", FILMS, target_seconds=120, only={"film-1", "outro"})
    assert partial["package"] == {} and [(s["kind"], s["film"]) for s in partial["sections"]] == [("film", 1), ("outro", 0)]
    final = Writer(cache_dir=tmp_path)
    final.ask, final_calls = fake_model()
    final.thumbnail_concepts = lambda *args: []
    original = final.ask

    def ask(prompt, schema, **options):
        if "titles" in schema["properties"]:
            return {"titles": ["Top 2 Mind-Bending Movies"], "description": "Two picks.", "tags": [],
                    "thumbnail_texts": ["Mind Benders", "Reality Breaks", "Watch These"]}
        return original(prompt, schema, **options)
    prompts = []

    def recording(prompt, schema, **options):
        prompts.append(prompt)
        return ask(prompt, schema, **options)
    final.ask = recording
    final.write("Top 2 Mind-Bending Movies", "Casual", FILMS, target_seconds=120)
    # Plan, number two's review and the outro came from the cache; only the intro and number one were new.
    assert final_calls["plan"] == 0
    assert not any("'Arrival'" in p and "SINGLE REVIEW" in p for p in prompts)
    assert not any("OUTRO ONLY" in p for p in prompts)
    assert any("'Inception'" in p and "SINGLE REVIEW" in p for p in prompts)


def test_optional_bridge_validation_and_legacy_filler_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRIPT_BRIDGES", "true")
    from movie_pipeline.packaging import refresh_availability
    from movie_pipeline.script import Writer
    writer = Writer(cache_dir=tmp_path)
    films = [{"title": "Inception", "year": 2010}, {"title": "Arrival", "year": 2016}, {"title": "Dune", "year": 2021}]
    sections = [{"kind": "film", "film": i, "paragraphs": [f"Number {i+1}: {f['title']}."]} for i, f in reversed(list(enumerate(films)))]
    writer.ask = lambda prompt, schema, **kw: {"bridges": ["Up next, Arrival turns everything quieter.", "You think that was big? Number one is bigger."]}
    bridges = writer.bridges("Top 3", "", films, sections)
    assert 2 not in bridges                                   # the line naming the next title is dropped
    assert bridges[1] == "You think that was big? Number one is bigger."
    section = {"kind": "film", "film": 1, "paragraphs": ["Review.", "You can find it on Netflix.", bridges[1]],
               "availability_text": "You can find it on Netflix.", "bridge": bridges[1]}
    script = {"sections": [section]}
    refresh_availability(script, [{}, {"watch_availability": {"subscription": ["Hulu"]}}, {}])
    assert section["paragraphs"] == ["Review."]


def test_whitespace_normalization_does_not_substitute_creative_vocabulary():
    from movie_pipeline.script import plain_words
    assert plain_words("A curated list of unsettling, high-concept horror that delves into fear.") == \
        "A curated list of unsettling, high-concept horror that delves into fear."
