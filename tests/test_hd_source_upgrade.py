"""Source-based quality selection and identity-safe HD promotion."""
from pathlib import Path

import pytest

from movie_pipeline import source_pool, video_settings


@pytest.fixture
def media(monkeypatch, tmp_path):
    dimensions = {}
    def add(name, width, height):
        path = tmp_path / name
        path.write_bytes(name.encode())
        dimensions[str(path)] = (width, height)
        return path
    def probe(path):
        width, height = dimensions[str(path)]
        return {"streams": [{"codec_type": "video", "width": width, "height": height}]}
    monkeypatch.setattr(video_settings, "probe", probe)
    monkeypatch.setattr(video_settings, "_SOURCE_DETAILS", {})
    return add, dimensions


def test_existing_hd_extra_promotes_without_moving_any_source_or_sidecar(media):
    add, _ = media
    low = add("tv-42-2160p.mp4", 640, 360)
    high = add("tv-42-2160p-extra-4.mp4", 1920, 1080)
    sidecar = high.with_suffix(".gemini-video.json")
    sidecar.write_text('{"source":"high-original-bytes"}')
    assert source_pool.promote_source_pool([low, high]) == [high, low]
    assert low.read_bytes() == low.name.encode()
    assert sidecar.read_text() == '{"source":"high-original-bytes"}'
    assert video_settings.choose_export_from_pools([[low, high]], 2160) == (1920, 1080)


def test_known_foreign_title_is_never_promoted_or_kept(media):
    add, _ = media
    primary = add("movie-12-2160p.mp4", 640, 360)
    foreign = add("movie-99-2160p-extra-5.mp4", 3840, 2160)
    matching = add("movie-12-2160p-extra-3.mp4", 1280, 720)
    assert source_pool.promote_source_pool([primary, foreign, matching]) == [matching, primary]


def test_unidentified_hd_cannot_displace_known_title(media):
    add, _ = media
    primary = add("movie-12-2160p.mp4", 640, 360)
    unknown = add("unverified-upload.mp4", 3840, 2160)
    assert source_pool.promote_source_pool([primary, unknown])[0] == primary


@pytest.mark.parametrize("heights,cap,expected", [
    ([360, 360], 2160, (640, 360)),
    ([720, 720], 2160, (1280, 720)),
    ([1080, 1080, 1080, 1080, 720], 2160, (1920, 1080)),
    ([2160, 2160, 2160, 1080, 1080], 2160, (2560, 1440)),
    ([1080, 1080, 1080, 1080, 360], 2160, (1920, 1080)),
    ([2160, 2160], 2160, (3840, 2160)),
    ([360, 360], 1080, (1920, 1080)),
])
def test_canvas_uses_full_pool_with_native_detail_and_majority_safeguard(media, heights, cap, expected):
    add, _ = media
    pools = [[add(f"tv-{index}-2160p.mp4", round(height * 16 / 9), height)]
             for index, height in enumerate(heights)]
    assert video_settings.choose_export_from_pools(pools, cap) == expected


def test_false_4k_filename_and_portrait_height_cannot_claim_4k(media):
    add, _ = media
    path = add("tv-8-2160p.mp4", 640, 360)
    portrait = add("tv-8-2160p-extra.mp4", 1080, 1920)
    assert video_settings.choose_export_from_pools([[path, portrait]], 2160)[1] < 1080


def test_discovery_floor_allows_hd_to_upgrade_360_without_forcing_4k_extras(media):
    add, _ = media
    sd = add("tv-4-2160p.mp4", 640, 360)
    uhd = add("tv-5-2160p.mp4", 3840, 2160)
    assert video_settings.discovery_size(sd, 2160) == (640, 360)
    assert video_settings.discovery_size(uhd, 2160) == (1920, 1080)


def test_quality_receipt_distinguishes_canvas_from_accepted_source_detail(media):
    add, _ = media
    sd = add("tv-4-2160p.mp4", 640, 360)
    hd = add("tv-4-2160p-extra.mp4", 1920, 1080)
    pool = source_pool.promote_source_pool([sd, hd])
    report = video_settings.source_quality_report([pool], (1920, 1080))
    assert report["basis"] == "accepted_pool"
    assert report["primary_sources_below_canvas"] == []
    assert report["accepted_sources_below_canvas"] == [sd.name]
    assert next(r for r in report["sources"] if r["file"] == sd.name)["upscale_factor"] == 3


def test_actual_dimensions_are_reprobed_after_source_bytes_change(media):
    add, dimensions = media
    path = add("tv-4-2160p.mp4", 640, 360)
    assert video_settings.source_detail(path)["height"] == 360
    dimensions[str(path)] = (1920, 1080)
    path.write_bytes(b"new-HD-source-with-different-size")
    assert video_settings.source_detail(path)["height"] == 1080


def test_sd_does_not_get_primary_quality_bonus_over_strong_semantic_match(media):
    add, _ = media
    sd = add("tv-4-2160p.mp4", 640, 360)
    hd = add("tv-4-2160p-extra.mp4", 1920, 1080)
    class Selector:
        features = {str(sd): object(), str(hd): object()}
        def hook_score(self, path, *args):
            return .6 if path == sd else -.1
    shots = {str(sd): [{"start": 0, "end": 8}], str(hd): [{"start": 0, "end": 8}]}
    assert source_pool._rank_sources(Selector(), [hd, sd], shots, "A character", False)[0] == sd


def test_missing_optional_quality_metadata_does_not_create_a_new_rejection(tmp_path, monkeypatch):
    bad = tmp_path / "tv-4-2160p.mp4"
    bad.write_bytes(b"unavailable-probe")
    monkeypatch.setattr(video_settings, "probe", lambda _: (_ for _ in ()).throw(OSError("probe unavailable")))
    monkeypatch.setattr(video_settings, "_SOURCE_DETAILS", {})
    assert source_pool.promote_source_pool([bad]) == [bad]
    assert video_settings.choose_export_from_pools([[bad]], 2160) == (1920, 1080)
    assert video_settings.source_quality_report([[bad]], (1920, 1080))["unknown_sources"]
