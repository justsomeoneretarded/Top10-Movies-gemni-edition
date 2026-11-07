import pytest

from movie_pipeline.availability import availability_line, known_service


def test_aggregator_channels_are_not_named():
    line = availability_line({"status": "found", "subscription": ["Amazon Prime Video"], "free": ["JustWatch TV"]})
    assert line == "You can find it on Prime Video."
    assert availability_line({"status": "found", "free": ["JustWatch TV"]}) == ""


@pytest.mark.parametrize("name", ["JustWatch TV", "Spectrum On Demand", "DIRECTV", "Xfinity Stream",
                                  "Some Unknown Indie Channel Amazon Channel", "Microsoft Store", "Redbox"])
def test_unrecognised_or_unwatchable_services_are_never_spoken(name):
    assert known_service(name) is None
    assert availability_line({"subscription": [name], "free": ["Tubi"]}) == "You can find it on Tubi."


@pytest.mark.parametrize("raw, plain", [("Netflix Standard with Ads", "Netflix"), ("Max Amazon Channel", "HBO Max"), ("HBO Max", "HBO Max"),
                                        ("Paramount+ with Showtime", "Paramount Plus"), ("Apple TV+", "Apple TV"),
                                        ("Shudder Amazon Channel", "Shudder"), ("Disney+", "Disney Plus"),
                                        ("Peacock Premium", "Peacock"), ("fuboTV", "Fubo")])
def test_known_service_variants_map_to_plain_names(raw, plain):
    assert known_service(raw) == plain
