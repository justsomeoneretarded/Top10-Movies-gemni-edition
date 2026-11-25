"""Mixed source credits must not consume plot or survive as actor rollcalls."""
from copy import deepcopy

import pytest

from movie_pipeline.plot_context import plot_material, story_text


@pytest.mark.parametrize("source, expected", [
    ("The film stars Keanu Reeves as John Wick, a retired assassin whose dog is killed.",
     "The story follows John Wick, a retired assassin whose dog is killed."),
    ("The series stars Terry Actor as Ada, whose brother vanishes from a locked ward.",
     "The story follows Ada, whose brother vanishes from a locked ward."),
    ("The series stars Terry Actor as Dr. Ada Vale, who searches for her brother.",
     "The story follows Dr. Ada Vale, who searches for her brother."),
    ("Example is a 2025 thriller film starring Terry Actor as Ada, a doctor searching for her brother.",
     "The story follows Ada, a doctor searching for her brother."),
])
def test_named_roles_keep_supplied_story_material(source, expected):
    assert story_text(source) == expected


@pytest.mark.parametrize("role", [
    "a doctor fighting an outbreak",
    "Ada, a doctor fighting an outbreak",
])
@pytest.mark.parametrize("cast_tail", [
    "Terry Actor and Alex Actor",
    "Terry Actor, Alex Actor, and Jamie Actor",
    "Terry Actor & Alex Actor",
])
def test_trailing_cast_list_is_removed_without_losing_the_role_plot(role, cast_tail):
    source = f"The series stars Noah Wyle as {role}, alongside {cast_tail}."
    assert story_text(source) == f"The story follows {role}."


@pytest.mark.parametrize("companion", [
    "a nurse who helps her escape",
    "Ada, who helps her escape",
    "Ben to stop the outbreak",
    "Ada as she searches for an exit",
    "Ada and Ben, who help her escape",
    "Sarah Connor and John Connor to stop the machines",
    "Ada and Ben",
])
def test_story_companions_actions_and_purpose_are_retained(companion):
    premise = f"a doctor fighting an outbreak, alongside {companion}."
    assert story_text(f"The series stars Terry Actor as {premise}") == "The story follows " + premise


@pytest.mark.parametrize("source", [
    "The film stars Terry Actor as Ada.",
    "The film stars Keanu Reeves as John Wick.",
    "The film stars Terry Actor as Ada, alongside Alex Actor and Jamie Actor.",
    "The film was directed by M. Night Shyamalan.",
    "directed by j. a. bayona.",
    "Created by Jane Director.",
])
def test_credit_only_material_is_not_expanded_into_a_plot(source):
    assert story_text(source) == ""


@pytest.mark.parametrize("credit", [
    "The film was directed by M. Night Shyamalan.",
    "Directed by J. A. Bayona.",
    "directed by j. a. bayona.",
    "The film was directed by Dr. Jane Director.",
    "The film stars Robert Downey Jr.",
])
def test_initials_and_abbreviations_do_not_leave_credit_fragments(credit):
    plot = "Ada searches a locked hospital for her missing brother."
    assert story_text(credit + " " + plot) == plot


@pytest.mark.parametrize("source", [
    "Created by Umbrella, the virus spreads through the hospital.",
    "Directed by John, Ada leaves through the service tunnel.",
    "Produced by Dr. Vale to save his daughter, the cure infects the town.",
    "Written by J. Vale to lure her back, the note sends Ada to a trap.",
    "The director searches the set while her crew vanish.",
])
def test_fictional_creators_and_directors_keep_their_story_actions(source):
    assert story_text(source) == source


def test_projection_keeps_named_role_plot_and_does_not_mutate_grounded_notes():
    source = {"title": "Example", "grounded_notes": {
        "story": "The film stars Terry Actor as Ada, a doctor searching for her brother, alongside Alex Actor and Jamie Actor.",
        "directors": ["Jane Director"],
    }}
    original = deepcopy(source)
    assert plot_material(source)["grounded_notes"] == {
        "story": "The story follows Ada, a doctor searching for her brother."
    }
    assert source == original
