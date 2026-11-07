"""Provider integration and unchanged-reference regression checks."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from movie_pipeline import gemini
from movie_pipeline.cli import ai_configuration, warm_voice, write_ai_report
from movie_pipeline.script import Writer

SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}},
          "required": ["text"], "additionalProperties": False}


def configure_cloud(monkeypatch):
    monkeypatch.setattr(gemini, "enabled", lambda: True)
    monkeypatch.setattr(gemini, "identity", lambda: {"backend": "gemini", "text_model": "test-model"})
    monkeypatch.setattr("scripts.runner_disk.prepare", Mock())


def test_script_uses_cloud_schema_and_does_not_touch_ollama(monkeypatch):
    configure_cloud(monkeypatch)
    client = SimpleNamespace(generate_json=Mock(return_value={"text": "A complete thought."}))
    monkeypatch.setattr(gemini, "get_client", lambda: client)
    local = Mock(side_effect=AssertionError("Cloud success must not start a local model"))
    monkeypatch.setattr("movie_pipeline.script.requests.post", local)
    writer = Writer()
    assert writer.ask("Explain the actual premise.", SCHEMA) == {"text": "A complete thought."}
    assert client.generate_json.call_args.args[1] is SCHEMA
    assert "Treat every supplied source" in client.generate_json.call_args.args[0]
    writer.unload()
    local.assert_not_called()


def test_cloud_failure_fetches_existing_local_fallback_once(monkeypatch):
    configure_cloud(monkeypatch)
    client = SimpleNamespace(generate_json=Mock(side_effect=gemini.GeminiUnavailable("quota")))
    monkeypatch.setattr(gemini, "get_client", lambda: client)
    monkeypatch.setattr(gemini, "record_event", Mock())
    monkeypatch.setenv("SCRIPT_MODEL", "existing-local-model")
    response = Mock()
    response.json.return_value = {"message": {"content": '{"text":"Complete local speech."}'}}
    local = Mock(return_value=response)
    monkeypatch.setattr("movie_pipeline.script.requests.post", local)
    writer = Writer()
    configured_cache_model = writer.model
    assert writer.ask("One", SCHEMA)["text"] == "Complete local speech."
    assert writer.ask("Two", SCHEMA)["text"] == "Complete local speech."
    pulls = [call for call in local.call_args_list if call.args[0].endswith("/api/pull")]
    chats = [call for call in local.call_args_list if call.args[0].endswith("/api/chat")]
    assert len(pulls) == 1 and len(chats) == 2
    assert all(call.kwargs["json"]["model"] == "existing-local-model" for call in chats)
    assert writer.model == configured_cache_model
    writer.unload()
    assert local.call_args.args[0].endswith("/api/generate")


def test_explicit_local_writer_preserves_existing_transport(monkeypatch):
    monkeypatch.setattr(gemini, "enabled", lambda: False)
    monkeypatch.setenv("SCRIPT_MODEL", "existing-local-model")
    response = Mock()
    response.json.return_value = {"message": {"content": '{"text":"Local."}'}}
    post = Mock(return_value=response)
    monkeypatch.setattr("movie_pipeline.script.requests.post", post)
    writer = Writer()
    assert writer.model == "existing-local-model"
    assert writer.ask("Test", SCHEMA) == {"text": "Local."}
    assert len(post.call_args_list) == 1
    assert post.call_args.kwargs["json"]["format"] is SCHEMA


def test_registered_cloud_voice_skips_parallel_warm_takes(monkeypatch):
    from movie_pipeline import gemini_voice
    monkeypatch.setenv("VOICE_WARM", "true")
    monkeypatch.setattr(gemini, "enabled", lambda: True)
    monkeypatch.setattr(gemini_voice, "has_registered_voice", lambda narrator=None: True)
    # No reference/cache is present: the decision happens before constructing a narrator.
    assert warm_voice([{"paragraphs": ["A useful sentence."]}], SimpleNamespace(narrator="warguy")) == 0


def test_configuration_fingerprints_voice_without_exposing_it(monkeypatch):
    monkeypatch.setattr(gemini, "identity", lambda: {"backend": "gemini"})
    monkeypatch.setenv("GEMINI_VOICE_WARGUY", "voice_private_reference_123")
    first = ai_configuration()
    assert "voice_private_reference_123" not in json.dumps(first)
    monkeypatch.setenv("GEMINI_VOICE_WARGUY", "voice_private_reference_456")
    assert ai_configuration() != first


def test_reports_actual_backends_and_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gemini, "identity", lambda: {"backend": "gemini"})
    work = Path("work")
    work.mkdir()
    events = [{"stage": "script", "backend": "gemini", "status": "success"},
              {"stage": "voice", "backend": "chatterbox", "status": "fallback"}]
    (work / "gemini-usage.jsonl").write_text("\n".join(map(json.dumps, events)) + "\ninvalid-json\n")
    output = Path("output")
    output.mkdir()
    write_ai_report(output, [{"title": "Example", "year": 2025,
                             "research_backend": "gemini",
                             "research_sources": [{"url": "https://example.com/film"}]}])
    report = json.loads((output / "ai-production-report.json").read_text())
    assert len(report["events"]) == 2
    assert report["research"][0]["backend"] == "gemini"
    assert {item["backend"] for item in report["counts"]} == {"gemini", "chatterbox"}


def test_imported_voice_references_match_original_baseline():
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root / "BASELINE.json").read_text())
    for name in ("warguy", "kit-laser", "female"):
        relative = f"assets/voices/{name}.ogg"
        raw = (root / relative).read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        assert actual == baseline["files"][relative]["sha"]
