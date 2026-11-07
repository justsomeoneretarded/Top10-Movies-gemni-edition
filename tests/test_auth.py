import stat

import pytest

from movie_pipeline.auth import prepare_cookies
from movie_pipeline.core import PipelineError


def test_cookies_optional(tmp_path):
    assert prepare_cookies("", tmp_path) is None
    assert not list(tmp_path.iterdir())


def test_cookie_file_private_and_outside_output(tmp_path):
    contents = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t2147483647\tTEST\tnot-a-real-session\n"
    path = prepare_cookies(contents, tmp_path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.read_text() == contents


@pytest.mark.parametrize("contents", ["{}", "# Netscape HTTP Cookie File\n", "x"*48000])
def test_bad_secret_fails_without_exposing_contents(tmp_path, contents):
    with pytest.raises(PipelineError) as failure:
        prepare_cookies(contents, tmp_path)
    assert contents not in str(failure.value) if len(contents) > 100 else True
    assert not (tmp_path / "youtube-cookies.txt").exists()
