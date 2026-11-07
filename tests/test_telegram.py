from unittest.mock import Mock

import pytest
import requests
import yaml

from movie_pipeline.telegram import notify


def setup(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-do-not-log")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("VIDEO_TITLE", "Example countdown")


def test_missing_credentials_never_sends(monkeypatch, capsys):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    post = Mock()
    monkeypatch.setattr("movie_pipeline.telegram.requests.post", post)
    assert not notify()
    post.assert_not_called()
    assert "skipped" in capsys.readouterr().out


def test_success_sends_only_title_and_run_link(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path)
    (tmp_path/"output").mkdir()
    (tmp_path/"output/recommended-title.txt").write_text("Generated title\n")
    post = Mock(side_effect=[
        Mock(status_code=200, json=Mock(return_value={"ok": True, "result": {"message_id": 77}})),
        Mock(status_code=200, json=Mock(return_value={"ok": True})),
    ])
    monkeypatch.setattr("movie_pipeline.telegram.requests.post", post)
    assert notify()
    payload = post.call_args_list[0].kwargs["json"]
    assert payload["chat_id"] == "12345"
    assert "Generated title" in payload["text"]
    assert "https://github.com/owner/repo/actions/runs/42" in payload["text"]
    assert "files" not in post.call_args_list[0].kwargs
    assert payload["parse_mode"] == "HTML"
    assert payload["text"].count("\n"*6) == 2
    pin = post.call_args_list[1]
    assert pin.args[0].endswith("/pinChatMessage")
    assert pin.kwargs["json"]["message_id"] == 77
    assert pin.kwargs["json"]["chat_id"] == "12345"


@pytest.mark.parametrize("error", [requests.Timeout("token=test-token-do-not-log"), requests.ConnectionError("https://api.telegram.org/bottest-token-do-not-log/sendMessage")])
def test_network_failure_does_not_expose_token_or_retry(monkeypatch, tmp_path, capsys, error):
    setup(monkeypatch, tmp_path)
    post = Mock(side_effect=error)
    monkeypatch.setattr("movie_pipeline.telegram.requests.post", post)
    assert not notify()
    assert post.call_count == 1
    assert "test-token-do-not-log" not in capsys.readouterr().out


def test_failed_delivery_does_not_claim_success(monkeypatch, tmp_path, capsys):
    setup(monkeypatch, tmp_path)
    monkeypatch.setattr("movie_pipeline.telegram.requests.post", Mock(return_value=Mock(status_code=403)))
    assert not notify()
    assert "HTTP 403" in capsys.readouterr().out


def test_notification_runs_only_after_successful_artifact_upload():
    from pathlib import Path
    data = yaml.load(Path(".github/workflows/generate-video.yml").read_text(), Loader=yaml.BaseLoader)
    steps = data["jobs"]["produce"]["steps"]
    upload_index = next(i for i,s in enumerate(steps) if s.get("uses", "").startswith("actions/upload-artifact"))
    notify_index = next(i for i,s in enumerate(steps) if s.get("name") == "Telegram completion reminder")
    step = steps[notify_index]
    assert notify_index > upload_index
    assert step["if"] == "success()"
    assert step["continue-on-error"] == "true"
    assert "TELEGRAM_BOT_TOKEN" not in data["jobs"]["produce"]["env"]


def test_title_html_is_escaped(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path)
    monkeypatch.setenv("VIDEO_TITLE", "Movies <action> & thrills")
    post = Mock(return_value=Mock(status_code=200, json=Mock(return_value={"ok": True})))
    monkeypatch.setattr("movie_pipeline.telegram.requests.post", post)
    assert notify()
    text = post.call_args.kwargs["json"]["text"]
    assert "Movies &lt;action&gt; &amp; thrills" in text
    assert "<b>Title:</b>" in text


@pytest.mark.parametrize("failure", [Mock(status_code=403), requests.Timeout("test-token-do-not-log")])
def test_pinning_failure_does_not_resend_or_fail_video(monkeypatch, tmp_path, capsys, failure):
    setup(monkeypatch, tmp_path)
    post = Mock(side_effect=[Mock(status_code=200, json=Mock(return_value={"ok": True, "result": {"message_id": 77}})), failure])
    monkeypatch.setattr("movie_pipeline.telegram.requests.post", post)
    assert notify()
    assert post.call_count == 2
    assert sum(call.args[0].endswith("/sendMessage") for call in post.call_args_list) == 1
    assert "test-token-do-not-log" not in capsys.readouterr().out
