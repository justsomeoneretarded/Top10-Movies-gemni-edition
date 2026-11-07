from movie_pipeline.footage import candidate_ok, identity_ok, search_query
from movie_pipeline.secondary_footage import secondary_candidate_ok

OFFICE = {"title": "The Office", "year": 2005, "kind": "tv", "cast": ["Steve Carell", "Rainn Wilson", "John Krasinski"],
          "creators": ["Greg Daniels", "Ricky Gervais"], "networks": ["NBC"]}


def video(title, description="", channel="Prime Video", verified=True, duration=120):
    return {"title": title, "description": description, "channel": channel, "channel_is_verified": verified, "duration": duration}


def test_a_same_named_remake_is_rejected():
    remake = video("The Office | Official Trailer | Prime Video", "Felicity Ward stars as Hannah Howard in the Australian version.")
    assert not identity_ok(remake, OFFICE)
    assert not candidate_ok(remake, OFFICE)
    assert not secondary_candidate_ok(remake, OFFICE)


def test_the_real_show_is_accepted():
    real = video("The Office | Official Trailer", "Steve Carell leads the Dunder Mifflin staff on NBC.", channel="The Office")
    assert identity_ok(real, OFFICE) and candidate_ok(real, OFFICE)
    by_surname = video("The Office Trailer", "Starring Carell and Krasinski.")
    assert identity_ok(by_surname, OFFICE)


def test_tv_searches_omit_network_and_titles_without_data_pass():
    assert search_query(OFFICE) == "The Office 2005 official trailer"
    assert search_query({"title": "Dune", "year": 2021, "kind": "movie", "networks": ["X"]}) == "Dune 2021 official trailer"
    assert identity_ok(video("Anything"), {"title": "Unknown", "year": 2020})
