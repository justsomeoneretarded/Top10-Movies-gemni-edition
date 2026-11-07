"""Exercise production timing/metadata orchestration without model or media downloads."""
from __future__ import annotations

import copy
import json
import sys
import wave
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from movie_pipeline import (
    availability,
    cli,
    footage,
    memes,
    render,
    research,
    script,
    secondary_footage,
    shots,
    speech_recovery,
    voice,
)
from movie_pipeline.core import PipelineError, key, save_json, write_srt
from movie_pipeline.packaging import MIDROLLS
from movie_pipeline.video_settings import export_size



def assert_rank_introduction_sequence(events, chapters, cues, speed, count):
    """The complete announcement and its captions precede native preview and review body."""
    announcements = [event for event in events if event.get('rank_announcement')]
    previews = [event for event in events if event['type'] == 'review-preview']
    assert len(announcements) == len(previews) == count
    for position, (announcement, preview, chapter) in enumerate(zip(announcements, previews, chapters[1:-1])):
        rank = count - position
        assert announcement['type'] == 'narration'
        assert announcement['section_kind'] == 'film'
        assert announcement['text'] == f'Number {rank}: Movie {rank}.'
        assert announcement['film'] == preview['film'] == f'Movie {rank}'
        index = events.index(announcement)
        assert events[index+1] == preview
        assert preview.get('narration') is False
        assert announcement['start'] + announcement['duration'] == pytest.approx(preview['start'])
        body = events[index+2]
        assert body['type'] == 'narration' and not body.get('rank_announcement')
        assert body['film'] == announcement['film']
        assert body['text'] == f'Review number {rank} is easy to follow.'
        assert body['start'] == pytest.approx(preview['start']+preview['duration'])
        assert chapter['start'] == pytest.approx(announcement['start']/speed)
        spoken_cues = [cue for cue in cues
                       if cue['start'] >= announcement['start']/speed-.001
                       and cue['end'] <= (announcement['start']+announcement['duration'])/speed+.001]
        assert ' '.join(cue['text'] for cue in spoken_cues) == announcement['text']
        assert spoken_cues[0]['start'] == pytest.approx(announcement['start']/speed)
        assert spoken_cues[-1]['end'] <= preview['start']/speed+.001

@pytest.mark.parametrize("count,try_4k,source_height,expected_size,extra_mode,content_type,narrator", [
    (2,"true",2160,(3840,2160),"found",None,None),
    (2,"false",1080,(1920,1080),"native-scenes","Movies","Warguy"),
    # One weaker primary among 4K sources is enlarged (at most 1.5x) instead of lowering the whole video.
    (5,"true",1080,(2560,1440),"missing","Shows","Female"),
    (2,"false",2160,(1920,1080),"raises","Movies","Kit Laser"),
    (5,"true",720,(1920,1080),"found","Movies","Warguy"),
    (2,"true",2160,(3840,2160),"render-fails",None,None),
    (2,"true",2160,(3840,2160),"rewrite-fails",None,None),
    (2,"false",2160,(1920,1080),"shared-audio-retry","Movies","Warguy"),
    (2,"false",2160,(1920,1080),"cta-vad-retry","Movies","Warguy"),
    (2,"false",2160,(1920,1080),"cta-joke-fallback","Movies","Warguy"),
    (2,"false",2160,(1920,1080),"intro-partial","Movies","Warguy"),
    (2,"false",2160,(1920,1080),"initial-speaker-error","Movies","Warguy"),
    (2,"false",2160,(1920,1080),"final-audio-rejected","Movies","Warguy"),
    (2,"false",2160,(1920,1080),"final-audio-error","Movies","Warguy"),
    (2,"false",2160,(1920,1080),"title-cards","Movies","Warguy"),
    (2,"false",2160,(1920,1080),"title-card-fails","Movies","Warguy"),
    (2,"false",2160,(1920,1080),"fallback-beats","Movies","Warguy"),
])
def test_produce_keeps_cta_between_reviews_and_retimes_metadata(monkeypatch, tmp_path, capsys, count, try_4k, source_height, expected_size, extra_mode, content_type, narrator):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VIDEO_RESOLUTION", "1080")
    monkeypatch.setenv("TRY_4K", try_4k)
    monkeypatch.setenv("SCRIPT_MODEL", "qwen3:4b")
    monkeypatch.setenv("VIDEO_MINUTES", "8" if extra_mode == "rewrite-fails" else "5")
    for name, selection in (("VIDEO_TYPE", content_type), ("NARRATOR", narrator)):
        if selection is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, selection)
    expected_kind = "tv" if content_type == "Shows" else "movie"
    expected_narrator = {None:"warguy", "Warguy":"warguy", "Kit Laser":"kit-laser", "Female":"female"}[narrator]
    records = {}
    durations = {}
    text_for_audio = {}
    render_sizes = []
    written_cues = []
    embedded = []
    first = "Stop scrolling. Your next favorite movie might be here."
    second = "Two minutes of scrolling saved. Let's find your next watch."
    films = [{"title": f"Movie {i+1}", "year": 2025, "id": i+1, "kind": expected_kind,
              "cast": [f"Actor {i+1}"], "genres": ["Action"], "status": "Released"} for i in range(count)]
    reference = tmp_path / "reference.ogg"
    reference.write_bytes(b"reference")

    def make(path, seconds, data=b"test media"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        durations[str(path)] = seconds
        return path

    class FakeResearch:
        def collect(self, items, path):
            assert len(items) == count
            records['selected_kind'] = getattr(self, 'selected_kind', None)
            records['preferred_kind'] = self.preferred_kind
            save_json(path, films)
            return copy.deepcopy(films)

    def enrich(values, source, cache):
        assert len(values) == count
        return [{**film, "watch_availability": {"region": "US", "checked_at": "2026-10-20T12:00:00Z",
                 "status": "found", "subscription": ["Netflix"], "rent": ["Amazon Video"]}} for film in values]

    class FakeWriter:
        model = "qwen3:4b"

        def __init__(self, cache_dir):
            self.cache_dir = cache_dir

        def write(self, title, brief, values, **kwargs):
            records["writes"] = records.get("writes", 0)+1
            if extra_mode == "rewrite-fails" and records["writes"] > 1:
                raise RuntimeError("Optional timing rewrite unavailable")
            assert all(film['watch_availability']['region'] == 'US' for film in values)
            sections = [{"kind": "intro", "film": 0, "paragraphs": [first, second]}]
            sections += [{"kind": "film", "film": i, "paragraphs": [f"Review number {i+1} is easy to follow."]}
                         for i in reversed(range(count))]
            sections += [{"kind": "outro", "film": 0,
                          "paragraphs": ["Like, subscribe, and watch our other recommendations. Save your next movie night."]}]
            return {"sections": sections, "package": {"title": f"Top {count} Action Movies",
                    "description": "Looking for action movies to watch? Start with these picks.",
                    "thumbnail_texts": ["WATCH THESE", "NEXT MOVIE NIGHT", "ACTION PICKS"],
                    "thumbnail_concepts": [], "thumbnail_version": 2}}

        def unload(self):
            pass

        def refresh_editorial(self, title, brief, values, draft, **kwargs):
            records['editorial_refreshed'] = True
            draft['package']['editorial_test_marker'] = True
            return draft

        def refresh_intro(self, title, brief, values, draft, **kwargs):
            records.setdefault('intro_refinements', []).append(kwargs)
            # Retain these complete takes so the timing orchestration fixtures
            # remain independent of whether optional rewriting succeeds.
            return draft

    class FakeNarrator:
        def __init__(self, reference, cache):
            self.cache = cache

        def speak(self, text, **kwargs):
            records.setdefault('speech_events', []).append(('generate', text))
            records.setdefault('speech_calls', []).append({'text': text, **kwargs})
            if extra_mode == 'initial-speaker-error' and text == first and not kwargs.get('force'):
                raise RuntimeError('Initial narration generation unavailable')
            if text in {first, second}:
                seconds = 6
            elif text.startswith("Number "):
                seconds = 2.5  # a short, separately verified rank/title announcement
            elif text.startswith("Review number"):
                seconds = 120 if count == 2 else 47
            elif text.startswith(("It's listed", "You can find it")):
                seconds = 3
                if kwargs.get('attempt_offset'):
                    records['shared_regenerations'] = records.get('shared_regenerations', 0)+1
            elif text in MIDROLLS or text in records.get('cta_candidates', []):
                seconds = 6
            else:
                seconds = 20
            path = make(self.cache / (key(text)+".wav"), seconds)
            with wave.open(str(path), 'wb') as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b'\x00\x00' * 160)
            text_for_audio[str(path)] = text
            return path

        def close(self):
            pass

    def download(film, root):
        records.setdefault("footage_order", []).append(("primary", film["id"]))
        path = make(root / f"movie-{film['id']}.mp4", 150)
        save_json(path.with_suffix('.json'), {'source': f"https://example.org/trailer/{film['id']}"})
        return path

    def secondary(film, root, primary, minimum_size, *, existing=None, slot=2):
        assert len([event for event in records["footage_order"] if event[0]=="primary"]) == count
        if slot >= 3:
            previous = "secondary" if slot == 3 else f"extra-{slot-1}"
            assert len([event for event in records["footage_order"] if event[0]==previous]) == count
            assert existing is not None
        records["footage_order"].append(("secondary" if slot == 2 else f"extra-{slot}", film["id"]))
        records['extra_attempts'] = records.get('extra_attempts', 0)+1
        assert minimum_size == expected_size
        if extra_mode == 'render-fails' and slot >= 3:
            return None  # exercise failed second decoding with a missing third
        if extra_mode == 'raises':
            raise RuntimeError('Optional source unavailable')
        if extra_mode in {'missing', 'rewrite-fails', 'shared-audio-retry', 'cta-vad-retry',
                          'cta-joke-fallback', 'intro-partial', 'initial-speaker-error',
                          'final-audio-rejected', 'final-audio-error', 'title-cards',
                          'title-card-fails', 'fallback-beats'}:
            return None
        suffix = "extra" if slot == 2 else f"extra-{slot}"
        path = make(root / f"movie-{film['id']}-{suffix}.mp4", 150)
        save_json(path.with_suffix('.json'), {'source': f"https://example.org/extra/{film['id']}"})
        return path

    def captions(audio, recognizer, **kwargs):
        if str(audio) in text_for_audio:
            assert Path(audio).exists(), 'Shared speech must remain present while its coordinates are verified'
            text = text_for_audio[str(audio)]
            records.setdefault('speech_events', []).append(('recognize', text))
            if text in records.get('cta_candidates', []):
                records.setdefault('cta_transcriptions', []).append({'text': text, **kwargs})
                if extra_mode == 'cta-vad-retry' and kwargs.get('vad_filter', True):
                    return [{'start': 0, 'end': durations[str(audio)], 'text': 'An unrelated sentence.'}]
                if extra_mode == 'cta-joke-fallback' and text == records['cta_candidates'][0]:
                    return [{'start': 0, 'end': durations[str(audio)], 'text': 'Subscribe.'}]
            if extra_mode == 'intro-partial' and text == first:
                # The hypotheses together contain every word, but no complete
                # hypothesis verifies the actual beginning and end of the take.
                heard = ' '.join(text.split()[:-3] if kwargs.get('vad_filter', True) else text.split()[3:])
                return [{'start': 0, 'end': durations[str(audio)], 'text': heard}]
            if extra_mode == 'shared-audio-retry' and text.startswith(("It's listed", "You can find it")):
                records['shared_transcriptions'] = records.get('shared_transcriptions', 0)+1
                if not records.get('shared_regenerations'):
                    return [{'start': 0, 'end': durations[str(audio)], 'text': 'Missing the expected availability sentence.'}]
            return [{"start": 0, "end": durations[str(audio)], "text": text_for_audio[str(audio)]}]
        return [{"start": 30, "end": 32.9, "text": "We have to leave right now"}]

    class FakeSelector:
        def __init__(self):
            self.people = {}

        def index(self, path, choices):
            pass

        def hook_score(self, path, choices, text, action):
            return .5

        def choose(self, path, choices, text, seconds, action, **kwargs):
            return [{"start": 40, "duration": seconds, "freeze": 0}]

    def montage(selector, trailers, choices, lengths, brief, action):
        return [[{"start": 40, "duration": seconds, "freeze": 0,
                  "source": str(trailers[i % count]), "film_index": i % count}]
                for i, seconds in enumerate(lengths)]

    def speech_video(source, audio, selected, target, label, work, **kwargs):
        if extra_mode == 'render-fails' and label is None and any('-extra' in cut.get('source', '') for cut in selected):
            records['render_fallbacks'] = records.get('render_fallbacks', 0)+1
            raise PipelineError('Optional decoder failed')
        render_sizes.append(export_size())
        text = text_for_audio[str(audio)]
        announcing = text.startswith("Number ")
        assert bool(label) is announcing, 'Only the rank announcement carries the rank/title/year box'
        if announcing:
            assert selected and all(cut.get('still') for cut in selected)
            assert kwargs.get('label_seconds', 0) >= durations[str(audio)]
        records.setdefault('spoken_renders', []).append({
            'text': text, 'target': str(target), 'label': str(label) if label else None,
            'selected': copy.deepcopy(selected), **kwargs,
        })
        make(target, durations[str(audio)])

    def trailer_video(source, start, seconds, target, **kwargs):
        records.setdefault('trailer_renders', []).append({
            'source': str(source), 'start': start, 'duration': seconds,
            'target': str(target), **kwargs,
        })
        if seconds == 1.5 and start == 140:
            records.setdefault('title_card_attempts', []).append(str(source))
            assert kwargs.get('muted') is True
            assert kwargs.get('label') is None, 'The trailer title card plays without a rank overlay'
            assert kwargs.get('still', False) is False
            if extra_mode == 'title-card-fails' and len(records['title_card_attempts']) == 1:
                records['title_card_failures'] = records.get('title_card_failures', 0)+1
                raise PipelineError('An optional title-card interval could not decode')
        make(target, seconds)

    def cold_open_plan(selector, pools, shot_lookup, text, action, dialogue=None, seconds=5):
        assert seconds == 5
        records['cold_open_planned'] = True
        return [{'source': str(pools[0][0]), 'start': 40, 'duration': seconds, 'muted': True, 'film_index': 0}]

    def choose_preview(selector, sources, shot_lookup, text, seconds, action, dialogue=None):
        assert seconds == 5.0
        records.setdefault('review_previews', []).append(str(sources[0]))
        return {'source': str(sources[0]), 'start': 40, 'duration': seconds, 'muted': True}

    def choose_title_card(selector, sources, shot_lookup, title):
        if extra_mode not in {'title-cards', 'title-card-fails'}:
            return None
        film = next(film for film in films if film['title'] == title)
        assert Path(sources[0]).name == f"movie-{film['id']}.mp4"
        card = {'source': str(sources[0]), 'start': 140, 'duration': 1.5,
                'title': title, 'muted': True}
        records.setdefault('title_card_plans', []).append(card)
        return card

    def prepare_cta(title, brief, seed, work):
        records['cta_text'] = MIDROLLS[0]
        plan = {'text': MIDROLLS[0], 'path': str(tmp_path / 'reaction.gif'),
                'animated': True, 'headline': 'LIKE + SUBSCRIBE', 'theme': 'movie-night',
                'source': 'https://example.org/reaction', 'attribution': 'Test reaction asset'}
        records['cta_prepared'] = copy.deepcopy(plan)
        return plan

    def cta_variants(plan):
        plans = [plan]
        if extra_mode == 'cta-joke-fallback':
            plans.append({**plan, 'text': 'Like and subscribe. Your next movie night deserves a happy ending.'})
        records['cta_candidates'] = [candidate['text'] for candidate in plans]
        return plans

    def render_cta(plan, audio, target, work, background=None):
        assert plan['animated'] is True
        assert plan == {**records['cta_prepared'], 'text': plan['text']}
        assert text_for_audio[str(audio)] == plan['text']
        records['cta_text'] = plan['text']
        records['cta_render'] = {'plan': copy.deepcopy(plan), 'target': str(target), 'background': background}
        render_sizes.append(export_size())
        make(target, durations[str(audio)])
        return {'source': plan['path'], 'start': 0, 'duration': durations[str(audio)],
                'image': True, 'animation': True, 'still': False,
                'credit_source': plan['source'], 'attribution': plan['attribution']}

    def finish(parts, target, work, action, spec, music_theme=None, timeline=None, top_title=None, overlays=None):
        records['overlays'] = overlays
        assert music_theme == "action"
        assert timeline and top_title == films[0]['title']
        records['ending_hold'] = 3
        records['parts'] = [str(part) for part in parts]
        seconds = (sum(durations[str(part)] for part in parts)+3)/1.02
        make(target, seconds)
        return seconds, 1.02

    def capture_captions(path, cues):
        written_cues.extend(copy.deepcopy(cues))
        write_srt(path, cues)

    def embed(path, chapters, total, work):
        embedded.extend(copy.deepcopy(chapters))
        records['final_duration'] = total

    def verify_export_audio(video, events, speed, recovery, work):
        assert Path(video).is_file()
        assert speed == pytest.approx(1.02)
        assert work.is_dir()
        spoken = [event for event in events if event['type'] in {'narration', 'subscribe'}]
        assert len(spoken) >= 3
        for event in spoken:
            source = Path(event['source_audio'])
            assert source.is_file(), 'Verified speech remains available until final-export QA finishes'
            assert text_for_audio[str(source)] == event['text']
            with wave.open(str(source)) as audio:
                assert audio.getnframes() > 0
        narration = [event for event in spoken if event['type'] == 'narration']
        sampled = [narration[0], narration[len(narration)//2], narration[-1]]
        assert sampled[0]['section_kind'] == 'intro'
        assert sampled[1]['section_kind'] == 'film'
        assert sampled[-1]['section_kind'] == 'outro'
        assert len({event['start'] for event in sampled}) == 3
        records['export_audio_review'] = {
            'events': copy.deepcopy(events), 'speed': speed,
            'samples': [{'start': event['start']/speed, 'source_audio': event['source_audio']}
                        for event in sampled],
        }
        records['export_recovery_close'] = Mock(wraps=recovery.close)
        recovery.close = records['export_recovery_close']
        if extra_mode == 'final-audio-error':
            raise PipelineError('Finished audio stream is truncated before the final narration.')
        return {'accepted': extra_mode != 'final-audio-rejected',
                'status': 'rejected' if extra_mode == 'final-audio-rejected' else 'verified'}

    def thumbnails(trailers, choices, phrases, output, **kwargs):
        for i in range(1, 11):
            Image.new('RGB', (1280, 720), '#234567').save(output / f'thumbnail-{i:02}.jpg')

    def reference_for(selected_narrator, root):
        records['narrator'] = selected_narrator
        return reference

    def label_image(path, title, rank=None, year=None):
        records.setdefault('rank_labels', []).append({'title': title, 'rank': rank, 'year': year})
        path.write_bytes(b'label')

    monkeypatch.setattr(cli, 'inspect_sources', lambda paths: [{"file": p.name, "width": round((source_height if i==0 else 2160)*16/9), "height": source_height if i==0 else 2160} for i,p in enumerate(paths)])
    monkeypatch.setattr(cli, 'voice_reference', reference_for)
    monkeypatch.setattr(cli, 'duration', lambda path: durations[str(path)])
    monkeypatch.setattr(speech_recovery, 'duration', lambda path: durations[str(path)])
    monkeypatch.setattr(cli, 'require_disk', Mock())
    monkeypatch.setattr(cli, 'ffmpeg', Mock())
    monkeypatch.setattr(cli, 'validate_package', Mock())
    monkeypatch.setattr(cli, 'write_srt', capture_captions)
    monkeypatch.setattr(research, 'Research', FakeResearch)
    monkeypatch.setattr(availability, 'enrich_availability', enrich)
    monkeypatch.setattr(footage, 'download', download)
    monkeypatch.setattr(secondary_footage, 'download_secondary', secondary)
    monkeypatch.setattr(script, 'Writer', FakeWriter)
    monkeypatch.setattr(voice, 'Narrator', FakeNarrator)
    monkeypatch.setattr(voice, 'captions', captions)
    monkeypatch.setattr(shots, 'analyse', lambda *args: [{'start': 30, 'end': 90, 'motion': 8}])
    monkeypatch.setattr(shots, 'ShotSelector', FakeSelector)
    monkeypatch.setattr(shots, 'intro_montage', montage)
    monkeypatch.setattr(memes, 'prepare_intro_meme', lambda *args: None)
    monkeypatch.setattr(render, 'speech_video', speech_video)
    monkeypatch.setattr(render, 'trailer_video', trailer_video)
    monkeypatch.setattr(render, 'label_image', label_image)
    monkeypatch.setattr(render, 'subscribe_image', Mock(side_effect=AssertionError('The old static CTA card must not render')))
    monkeypatch.setattr(render, 'finish', finish)
    monkeypatch.setattr(render, 'j_cut', lambda *args, **kwargs: records.setdefault('j_cuts', []).append(args[2]))
    monkeypatch.setattr('movie_pipeline.soundtrack.narration_rate', lambda theme: 1)
    monkeypatch.setattr('movie_pipeline.sound_design.delivery_speed', lambda rate, label, **measurements: rate)
    monkeypatch.setattr('movie_pipeline.cta_animation.add_cta_animation', Mock())
    def augmented(source, target, work, *, protected_intervals=None):
        assert protected_intervals and all(end > start for start, end in protected_intervals)
        assert Path(source).exists()
        make(target, durations[str(source)])
        records['comparison'] = str(target)
        return {'same_timeline': True}
    monkeypatch.setattr('movie_pipeline.augmentation.export_augmented', augmented)
    monkeypatch.setattr(render, 'embed_chapters', embed)
    def end_screen(image, seconds, target):
        records['end_screen'] = seconds
        make(target, seconds)
    monkeypatch.setattr(render, 'end_screen_video', end_screen)
    if extra_mode == "fallback-beats":
        monkeypatch.setattr(cli, "choose_beat", lambda *args, **kwargs: {"start": 30.0, "end": 32.9})
        monkeypatch.setattr("movie_pipeline.dialogue.isolate_dialogue", lambda *args, **kwargs: None)
    if extra_mode == "native-scenes":
        def native_plan(selector,pools,lookup,films,sections,speech,dialogue,recognizer,work):
            def clip(source,start,seconds,film_index):
                return {"source":str(source),"start":start,"duration":seconds,
                        "film_index":film_index,"muted":False,"native_verified":True,
                        "cues":[{"start":start+1,"end":start+3,"text":"A complete scene sentence."}]}
            previews={};inserts={}
            for i,section in enumerate(sections):
                if section["kind"]=="film":
                    index=section["film"]
                    assert len(pools[index])==5
                    previews[i]=clip(pools[index][4],60,8,index)
                    assert section.get("rank_intro_version") == 1
                    inserts[i]={1:clip(pools[index][3],90,8,index)}
            return {"opening":[clip(pools[0][2],20,5,0)],"previews":previews,"inserts":inserts,"report":{}}
        monkeypatch.setattr("movie_pipeline.moments.plan_episode",native_plan)
    monkeypatch.setattr(render, 'thumbnails', thumbnails)
    monkeypatch.setattr("movie_pipeline.branding.render_ident", lambda target, work: (make(target, 2.3), 2.3)[1])
    monkeypatch.setitem(sys.modules, 'movie_pipeline.previews', SimpleNamespace(
        cold_open_plan=cold_open_plan, choose_preview=choose_preview, choose_title_card=choose_title_card,
        outro_recap_plan=lambda selector, pools, shots, lengths, text, action: montage(
            selector, [pool[0] for pool in pools], shots, lengths, text, action),
    ))
    monkeypatch.setitem(sys.modules, 'movie_pipeline.cta', SimpleNamespace(
        prepare_cta=prepare_cta, render_cta=render_cta, cta_variants=cta_variants,
    ))
    monkeypatch.setitem(sys.modules, 'movie_pipeline.audio_review', SimpleNamespace(
        verify_export_audio=verify_export_audio,
    ))
    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=Mock()))
    output = tmp_path / 'output'
    brief_voice = 'Use female audio.' if narrator == 'Warguy' else 'Use Warguy.'
    arguments = (f'Top {count} Action Movies', 'Make a 5-minute video. '+brief_voice,
                 '; '.join(f"{film['title']} ({film['year']})" for film in films), tmp_path / 'jobs', output)
    if extra_mode == 'intro-partial':
        with pytest.raises(speech_recovery.SpeechRecoveryError, match='complete speech for intro'):
            cli.produce(*arguments)
        checks = json.loads(next((tmp_path / 'jobs').glob('*/speech-checks/speech-verification.json')).read_text())['checks']
        opening_checks = [check for check in checks if check.get('expected') == first]
        assert len(opening_checks) == 9  # three complete hypotheses for each of three voice takes
        assert not any(check['accepted'] for check in opening_checks)
        assert not records.get('spoken_renders')
        assert not records.get('trailer_renders')
        assert not (output / 'final-video.mp4').exists()
        return
    if extra_mode in {'final-audio-rejected', 'final-audio-error'}:
        message = ('damaged export will not be delivered' if extra_mode == 'final-audio-rejected'
                   else 'Finished audio stream is truncated before the final narration')
        with pytest.raises(PipelineError, match=message):
            cli.produce(*arguments)
        assert records['export_audio_review']
        records['export_recovery_close'].assert_called_once_with()
        assert not (output / 'recommended-title.txt').exists()
        assert not (output / 'subtitles.srt').exists()
        assert 'Complete:' not in capsys.readouterr().out
        cli.validate_package.assert_not_called()
        return
    cli.produce(*arguments)
    if extra_mode == "native-scenes":
        timeline=json.loads(next((tmp_path/"jobs").glob("*/timeline.json")).read_text())
        events=timeline["events"]
        native=[event for event in events if event.get("native_verified")]
        assert len(native)==count
        assert all("extra-4" in event["source"] for event in native)
        previews=[event for event in events if event["type"]=="review-preview"]
        assert all(event["duration"]==8 for event in previews)
        assert all("extra-5" in event["shots"][0]["source"] for event in previews)
        assert len([cue for cue in written_cues if cue["text"]=="A complete scene sentence."])==1+count*2
        assert all(a["end"]<=b["start"]+.001 for a,b in pairwise(written_cues))
        assert_rank_introduction_sequence(events, embedded, written_cues, timeline["speed"], count)
        cta_index=next(i for i,e in enumerate(events) if e["type"]=="subscribe")
        assert events[cta_index-1]["type"]=="trailer-audio"
        assert events[cta_index+1].get("rank_announcement") is True
        assert records["final_duration"]==pytest.approx((sum(durations[p] for p in records["parts"])+3)/1.02)
        assert events[0]["type"]=="brand-ident"
        return

    assert records['editorial_refreshed'] is True
    assert records.get('intro_refinements', []) == ([] if extra_mode == 'initial-speaker-error'
                                                  else [{'target_words': 13}])
    records['export_recovery_close'].assert_called_once_with()
    audio_report_path = next((tmp_path / 'jobs').glob('*/audio-review.json'))
    assert json.loads(audio_report_path.read_text()) == {'accepted': True, 'status': 'verified'}
    assert records['selected_kind'] == (expected_kind if content_type else None)
    assert records['preferred_kind'] == expected_kind
    assert records['narrator'] == expected_narrator
    saved_input = json.loads(next((tmp_path / 'jobs').glob('*/input.json')).read_text())
    assert saved_input['content_type'] == (expected_kind if content_type else 'auto')
    assert saved_input['narrator'] == expected_narrator
    # Cold opening and title previews add intentional screen time to the old plan.
    added_visual_seconds = 2.3 + 5 + count*(2.5+5.0) + cli.END_SCREEN_SECONDS
    if extra_mode in {'title-cards', 'title-card-fails'}:
        added_visual_seconds += (count-(extra_mode == 'title-card-fails'))*1.5
    # All rendered parts, final hold and metadata use the same measured timebase.
    assert records['final_duration'] == pytest.approx((sum(durations[part] for part in records['parts'])+3)/1.02)
    assert records['writes'] == 1  # useful short drafts are not expanded to fill the selected runtime
    # The short joke is generated and verified before any full-length narration.
    events = records['speech_events']
    first_body_generation = events.index(('generate', first))
    assert all(('recognize', candidate) in events[:first_body_generation]
               for candidate in records['cta_candidates'])
    speech_checks = json.loads(next((tmp_path / 'jobs').glob('*/speech-checks/speech-verification.json')).read_text())['checks']
    cta_checks = [check for check in speech_checks if check.get('label') == 'subscribe break']
    assert any(check.get('accepted') and check.get('expected') == records['cta_text'] for check in cta_checks)
    if extra_mode == 'shared-audio-retry':
        assert not records.get('shared_transcriptions')
        assert not records.get('shared_regenerations')  # provider filler never enters TTS
    if extra_mode == 'cta-vad-retry':
        takes = [call for call in records['speech_calls'] if call['text'] in records['cta_candidates']]
        assert len(takes) == 1  # a no-VAD check retains the original generated audio
        assert [check['stage'] for check in cta_checks] == ['base-vad', 'base-no-vad']
        assert [check['accepted'] for check in cta_checks] == [False, True]
    if extra_mode == 'cta-joke-fallback':
        assert records['cta_text'] == records['cta_candidates'][1]
        original_takes = [call for call in records['speech_calls'] if call['text'] == records['cta_candidates'][0]]
        assert len(original_takes) == 3  # bounded retries before changing only the matching joke
        assert records['cta_render']['plan']['path'] == records['cta_prepared']['path']
    if extra_mode == 'initial-speaker-error':
        opening_takes = [call for call in records['speech_calls'] if call['text'] == first]
        assert len(opening_takes) == 2
        assert not opening_takes[0].get('force')
        assert opening_takes[1]['force'] is True
        assert opening_takes[1]['attempt_offset'] > 0
        opening_checks = [check for check in speech_checks if check.get('expected') == first]
        assert len(opening_checks) == 1
        assert opening_checks[0]['accepted'] is True
        assert opening_checks[0]['heard'] == first
        assert sum(part['text'] == first for part in records['spoken_renders']) == 1
    assert records['extra_attempts'] == count*4
    # Rounds stay strictly ordered; within a round, titles download in parallel (any order).
    order, n = records["footage_order"], len(films)
    assert order[:n] == [("primary",film["id"]) for film in films]
    assert sorted(order[n:2*n]) == sorted(("secondary",film["id"]) for film in films)
    for slot in (3,4,5):
        assert sorted(order[(slot-1)*n:slot*n]) == sorted((f"extra-{slot}",film["id"]) for film in films)
    if extra_mode == "render-fails":
        assert records['render_fallbacks'] == 1  # failed optional sources are removed before another paragraph uses them
    assert len(json.loads(next((tmp_path / 'jobs').glob('*/quality-report.json')).read_text())['sources']) == (count*5 if extra_mode == 'found' else count*2-1 if extra_mode == 'render-fails' else count)
    assert render_sizes and set(render_sizes) == {expected_size}
    quality_report = json.loads(next((tmp_path / "jobs").glob("*/quality-report.json")).read_text())
    assert quality_report["try_4k"] == (try_4k == "true")
    assert (quality_report["export_width"], quality_report["export_height"]) == expected_size
    assert records['cta_render']['plan']['animated'] is True
    assert records['cta_render']['plan']['path'].endswith('.gif')
    assert len(records['review_previews']) == count
    assert records['rank_labels'] == [
        {'title': f'Movie {i+1}', 'rank': i+1, 'year': 2025} for i in reversed(range(count))
    ]
    timeline_file = next((tmp_path / 'jobs').glob('*/timeline.json'))
    timeline = json.loads(timeline_file.read_text())
    events = timeline['events']
    assert records['export_audio_review']['events'] == events
    assert len(records['export_audio_review']['samples']) == 3
    assert events[0]['type'] == 'brand-ident'
    assert events[1]['type'] == 'cold-open'
    assert events[0]['start'] == 0
    assert sum(event['duration'] for event in events if event['type'] == 'cold-open') == pytest.approx(5)
    assert records['parts'][1] == records['trailer_renders'][0]['target']
    assert records['trailer_renders'][0]['muted'] is True
    assert min(cue['start'] for cue in written_cues) == pytest.approx(7.3/timeline['speed'])
    assert written_cues[0]['text'] == first
    previews = [event for event in events if event['type'] == 'review-preview']
    assert len(previews) == count
    assert all(event['duration'] == pytest.approx(5.0) for event in previews)
    assert all(event.get('label') is None for event in records['trailer_renders']), \
        'Native previews play after the labelled announcement, without a duplicate reveal'
    labelled = [event for event in records['spoken_renders'] if event.get('label') is not None]
    assert len(labelled) == count
    assert [event['text'] for event in labelled] == [
        f'Number {i+1}: Movie {i+1}.' for i in reversed(range(count))
    ]
    teasers = [event for event in events if event['type'] == 'teaser']
    assert len(teasers) == 0  # short lists have no countdown to tease
    if teasers:
        assert teasers[0]['duration'] == pytest.approx(3.0)
        intro_end = max(e['start'] + e['duration'] for e in events if e.get('section_kind') == 'intro')
        assert teasers[0]['start'] == pytest.approx(intro_end)  # right after the intro, before the countdown
    assert all(event['label_seconds'] >= 2.5 for event in labelled)
    ctas = [(i, event) for i, event in enumerate(events) if event.get('type') == 'subscribe']
    assert len(ctas) == 1
    index, cta = ctas[0]
    raw_duration = records['final_duration']*timeline['speed']
    assert .30 <= cta['start']/raw_duration <= .50
    assert cta['duration']/timeline['speed'] < 10
    assert events[index-1]['film'] != events[index+1]['film']
    preceding_review = next(e for e in reversed(events[:index]) if e.get('type')=='narration' and e.get('section_kind')=='film')
    assert preceding_review['text'].startswith("Review number")  # preceding review completed
    assert events[index+1].get('rank_announcement') is True
    assert events[index+2]['type'] == 'review-preview'
    assert events[index+3]['text'].startswith('Review number')  # body starts after the full preview
    cards = [(i, event) for i, event in enumerate(events) if event['type'] == 'title-card']
    if extra_mode in {'title-cards', 'title-card-fails'}:
        assert len(records['title_card_plans']) == count
        assert len(records['title_card_attempts']) == count
        assert len(cards) == count-(extra_mode == 'title-card-fails')
        if extra_mode == 'title-card-fails':
            assert records['title_card_failures'] == 1
            assert cards[0][1]['film'] == 'Movie 1'
        for card_index, card in cards:
            assert card['narration'] is False
            assert card['duration'] == pytest.approx(1.5)
            assert len(card['shots']) == 1
            plan = card['shots'][0]
            assert plan['title'] == card['film']
            film = next(film for film in films if film['title'] == card['film'])
            assert Path(plan['source']).name == f"movie-{film['id']}.mp4"
            own_review = [(i, event) for i, event in enumerate(events)
                          if event['type'] == 'narration' and event['section_kind'] == 'film'
                          and event['film'] == card['film']]
            assert own_review
            trailing=events[own_review[-1][0]+1:card_index]
            assert all(e['type']=='trailer-audio' and e['film']==card['film'] for e in trailing)
            previous = events[card_index-1]
            assert previous['start']+previous['duration'] == pytest.approx(card['start'])
            assert previous["type"] in {"narration","trailer-audio"}
            assert previous["film"] == card["film"]
            following = events[card_index+1]
            assert following['type'] == 'subscribe' or following.get('rank_announcement') or following.get('section_kind') == 'outro'
            card_start = card['start']/timeline['speed']
            card_end = (card['start']+card['duration'])/timeline['speed']
            assert all(cue['end'] <= card_start+.001 or cue['start'] >= card_end-.001
                       for cue in written_cues)
    else:
        assert cards == []
    assert [chapter['title'] for chapter in embedded] == ['Intro',
            *[f"#{i+1} Movie {i+1} (2025)" for i in reversed(range(count))], 'Outro']
    assert all(a['start'] < b['start'] for a, b in pairwise(embedded))
    assert not any('subscribe' in chapter['title'].casefold() for chapter in embedded)
    assert embedded == timeline['chapters']
    assert embedded[0]['start'] == 0
    assert_rank_introduction_sequence(events, embedded, written_cues, timeline['speed'], count)
    if extra_mode == "fallback-beats":
        native_beats = [event for event in events if event["type"] == "trailer-audio"]
        assert len(native_beats) == count
        for beat in native_beats:
            prior = events[events.index(beat)-1]
            assert prior["type"] == "narration" and not prior.get("rank_announcement")
            assert prior["film"] == beat["film"]
            assert prior["start"]+prior["duration"] == pytest.approx(beat["start"])
            assert beat["source_start"] == 30
            assert beat["duration"] == pytest.approx(2.9)
        assert len(records["j_cuts"]) == count
    cta_cues = [cue for cue in written_cues if cue['text'] in records['cta_text']]
    assert ' '.join(cue['text'] for cue in cta_cues) == records['cta_text']
    assert cta_cues[0]['start'] == pytest.approx(cta['start']/timeline['speed'])
    assert cta_cues[-1]['end']-cta_cues[0]['start'] < 10
    assert all(a['end'] <= b['start']+.001 for a,b in pairwise(written_cues))
    # Final speech is followed by the end-screen plate and the final hold (about 20 seconds).
    assert records['end_screen'] == cli.END_SCREEN_SECONDS
    assert (records['final_duration']-written_cues[-1]['end'])*timeline['speed'] == pytest.approx(cli.END_SCREEN_SECONDS+3)
    assert events[-1]['type'] == 'end-screen'
    description = (output / 'description.txt').read_text()
    assert first not in description.split('\n\n')[0]  # opens with a search hook, not narration
    assert 'featuring' in description.split('\n\n')[0]
    assert '00:00 Intro' in description
    assert ('#ActionShows' if expected_kind == 'tv' else '#ActionMovies') in description
    if extra_mode in {'found','render-fails'}:
        assert 'https://example.org/extra/' in description
        if extra_mode == 'found':
            assert any('-extra.mp4' in shot.get('source','') for event in events for shot in event.get('shots',[]))
    else:
        assert 'https://example.org/extra/' not in description
    assert ('Featured shows / series' if expected_kind == 'tv' else 'Featured movies') in description
    assert ' — Netflix' in description and 'Scott Buckley' in description
    assert 'Film information: TMDB' not in description
    assert not any('Netflix' in event.get('text', '') for event in events)
    assert all('US' not in event.get('text', '') for event in events)
    assert (output / 'subtitles.srt').exists()
    for i in range(1, 11):
        with Image.open(output / f'thumbnail-{i:02}.jpg') as thumbnail:
            assert thumbnail.size == (1280, 720)
    assert not list(timeline_file.parent.glob('part-*.mp4'))


@pytest.mark.parametrize('seconds', [8.5, 10.0, 12.0])
def test_intro_duration_is_soft_for_complete_unchanged_audio(monkeypatch, seconds):
    draft = {'sections': [{'kind': 'intro', 'paragraphs': [
        'These action movies give your next movie night a serious upgrade.',
    ]}]}
    speech = {(0, 0): Path('complete-intro.wav')}
    monkeypatch.setattr(cli, 'duration', lambda path: seconds)
    writer = SimpleNamespace(refresh_intro=Mock(return_value=draft), unload=Mock())
    speaker_factory = Mock(side_effect=AssertionError('An unchanged complete take needs no generation'))

    chosen, chosen_audio = cli.refine_intro_timing(
        writer, 'Top Action Movies', 'Keep the action specific.', [], draft, speech, speaker_factory)

    assert chosen is draft
    assert chosen_audio is speech
    assert writer.refresh_intro.call_count == (1 if seconds > 10 else 0)
    speaker_factory.assert_not_called()


@pytest.mark.parametrize('outcome', ['shorter', 'longer', 'rewrite-unavailable', 'voice-unavailable'])
def test_measured_intro_shortening_keeps_body_audio_and_complete_fallback(monkeypatch, outcome):
    draft = {'sections': [
        {'kind': 'intro', 'paragraphs': ['A complete opening that takes time.', 'Another complete opening thought.']},
        {'kind': 'film', 'paragraphs': ['Keep this review and its verified voice unchanged.']},
    ]}
    candidate = copy.deepcopy(draft)
    candidate['sections'][0]['paragraphs'] = ['Five action picks, with trouble worth watching.']
    speech = {(0, 0): Path('opening-one.wav'), (0, 1): Path('opening-two.wav'),
              (1, 0): Path('verified-review.wav')}
    replacement = Path('shorter-opening.wav')
    durations = {speech[(0, 0)]: 6, speech[(0, 1)]: 6, speech[(1, 0)]: 120,
                 replacement: 9 if outcome == 'shorter' else 13}
    monkeypatch.setattr(cli, 'duration', lambda path: durations[Path(path)])
    refresh = Mock(return_value=candidate)
    if outcome == 'rewrite-unavailable':
        refresh.side_effect = RuntimeError('Optional editorial rewrite is unavailable')
    writer = SimpleNamespace(refresh_intro=refresh, unload=Mock())
    speaker = SimpleNamespace(speak=Mock(return_value=replacement), close=Mock())
    if outcome == 'voice-unavailable':
        speaker.speak.side_effect = RuntimeError('Optional shorter voice take is unavailable')
    factory = Mock(return_value=speaker)

    chosen, chosen_audio = cli.refine_intro_timing(
        writer, 'Top Five Action Movies', 'Steady, specific action.', [], draft, speech, factory)

    refresh.assert_called_once()
    assert 8 <= refresh.call_args.kwargs['target_words'] <= 16
    assert chosen_audio[(1, 0)] == speech[(1, 0)]
    if outcome == 'shorter':
        # Nine seconds is a valid complete improvement; eight is guidance.
        assert chosen is candidate
        assert chosen_audio == {(0, 0): replacement, (1, 0): speech[(1, 0)]}
    else:
        assert chosen is draft
        assert chosen_audio is speech
    if outcome == 'rewrite-unavailable':
        factory.assert_not_called()
    else:
        speaker.close.assert_called_once_with()


def test_media_and_voice_dropdowns_isolate_cached_jobs(monkeypatch, tmp_path):
    """Identical text cannot restore another medium's research or another voice's speech."""
    monkeypatch.setenv('VIDEO_MINUTES', '5')
    monkeypatch.setenv('TRY_4K', 'false')
    reference = tmp_path / 'same-reference.ogg'
    reference.write_bytes(b'same reference bytes for every selected voice')
    monkeypatch.setattr(cli, 'voice_reference', lambda *args: reference)
    monkeypatch.setitem(sys.modules, 'movie_pipeline.audio_review', SimpleNamespace(
        verify_export_audio=Mock(side_effect=AssertionError('Research must stop before export QA')),
    ))
    monkeypatch.setitem(sys.modules, 'movie_pipeline.previews', SimpleNamespace(
        choose_preview=Mock(), choose_title_card=Mock(return_value=None),
        cold_open_plan=Mock(), outro_recap_plan=Mock(),
    ))
    selected_jobs = []

    class StopAfterCacheSelection:
        def collect(self, items, path):
            selected_jobs.append((self.selected_kind, path.parent))
            raise RuntimeError('Stop after cache selection')

    monkeypatch.setattr(research, 'Research', StopAfterCacheSelection)
    selections = [('Movies', 'Warguy'), ('Shows', 'Warguy'),
                  ('Movies', 'Kit Laser'), ('Movies', 'Female'), ('Movies', 'Warguy')]
    for content_type, narrator in selections:
        monkeypatch.setenv('VIDEO_TYPE', content_type)
        monkeypatch.setenv('NARRATOR', narrator)
        with pytest.raises(RuntimeError, match='Stop after cache selection'):
            cli.produce('Top 2 Movies', 'Use Warguy. Keep it casual.',
                        'First (2025); Second (2025)', tmp_path / 'jobs', tmp_path / 'output')

    assert [kind for kind, _ in selected_jobs] == ['movie', 'tv', 'movie', 'movie', 'movie']
    paths = [path for _, path in selected_jobs]
    assert len(set(paths[:4])) == 4
    assert paths[0] == paths[-1]
