import base64
from dataclasses import replace
from io import BytesIO
import sys
import wave
from unittest.mock import Mock

import pytest

from podcast_llm import tts_backends
from podcast_llm.config import PodcastConfig
from podcast_llm.generate import main
from podcast_llm.render import read_script, render_script


SCRIPT = """# An episode

## Script

### First chapter

**Interviewer**: First batch. [S1]

**Interviewee**: Second batch. [S2]

### Second chapter

**Interviewer**: Third batch. [S1] [S2]

## Sources

- [S1] A reference
"""


@pytest.fixture
def config(tmp_path):
    return replace(PodcastConfig.load(require_research_credentials=False),
                   google_api_key='test-google-key', elevenlabs_api_key='test-eleven-key',
                   tts_provider='gemini_3_8_flash_tts', tts_settings={},
                   output_format='wav', temp_audio_dir=str(tmp_path / 'temp'),
                   checkpoint_dir=str(tmp_path / 'checkpoints'))


def gemini_response(sample: bytes = b'\0\0'):
    buffer = BytesIO()
    with wave.open(buffer, 'wb') as audio:
        audio.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        audio.writeframes(sample * 2400)
    return Mock(**{'json.return_value': {
        'status': 'completed',
        'steps': [{'content': [{'type': 'audio', 'data': base64.b64encode(buffer.getvalue()).decode()}]}],
    }})


def test_render_resumes_inside_a_chapter_and_invalidates_only_edited_chapter(config, monkeypatch, mocker, tmp_path):
    monkeypatch.setattr(tts_backends, 'DIALOGUE_CHUNK_SIZE', 16)
    session = mocker.patch('podcast_llm.tts_backends.requests.Session').return_value.__enter__.return_value
    response = gemini_response()
    session.post.side_effect = [response, KeyboardInterrupt, response, response, response, response]
    script = tmp_path / 'episode.md'
    script.write_text(SCRIPT)
    output = tmp_path / 'episode.wav'
    output.write_bytes(b'previous audio')
    cache = tmp_path / 'checkpoints' / 'audio'

    with pytest.raises(KeyboardInterrupt):
        render_script(script, config)

    assert output.read_bytes() == b'previous audio'
    assert len(list(cache.rglob('*.zip'))) == 1
    render_script(script, config)
    assert session.post.call_count == 4
    with wave.open(str(output)) as audio:
        assert audio.getnframes() == 7200

    edited = tmp_path / 'edited.md'
    edited.write_text(SCRIPT.replace('Second batch.', 'Edited batch.'))
    render_script(edited, config)
    assert len(list(cache.rglob('*.zip'))) == 5
    sent = [call.kwargs['json']['input'][0]['content'][0]['text'] for call in session.post.call_args_list]
    assert sent == ['First batch.', 'Second batch.', 'Second batch.', 'Third batch.', 'First batch.', 'Edited batch.']


def test_resume_preserves_elevenlabs_dialogue_continuity(config, monkeypatch, mocker, tmp_path):
    monkeypatch.setattr(tts_backends, 'DIALOGUE_CHUNK_SIZE', 16)
    session = mocker.patch('podcast_llm.tts_backends.requests.Session').return_value.__enter__.return_value
    first = Mock(content=bytes(4800), headers={'content-type': 'audio/pcm', 'request-id': 'first-id'})
    second = Mock(content=bytes(4800), headers={'content-type': 'audio/pcm', 'request-id': 'second-id'})
    session.post.side_effect = [first, KeyboardInterrupt, second, second]
    script = tmp_path / 'episode.md'
    script.write_text(SCRIPT)
    active = replace(config, tts_provider='elevenlabs_v4')

    with pytest.raises(KeyboardInterrupt):
        render_script(script, active)

    render_script(script, active)
    calls = session.post.call_args_list
    assert len(calls) == 4
    assert calls[2].kwargs['json']['previous_request_ids'] == ['first-id']
    assert 'previous_request_ids' not in calls[3].kwargs['json']


def test_read_legacy_transcript_without_chapters(tmp_path):
    script = tmp_path / 'episode.md'
    script.write_text('# Episode\n\n## Outline\n\n### Section 1\n\n## Script\n\n'
                      '**Interviewer**: Hello.\nHere is my next thought.\n\n**Interviewee**: Hi.\n')

    chapter, = read_script(script)
    assert chapter.title == 'Script'
    assert chapter.turns == [
        {'speaker': 'Interviewer', 'text': 'Hello.\nHere is my next thought.'},
        {'speaker': 'Interviewee', 'text': 'Hi.'},
    ]


def test_cli_lists_chapters_without_loading_config(tmp_path, monkeypatch, mocker, capsys):
    script = tmp_path / 'episode.md'
    script.write_text(SCRIPT)
    load = mocker.patch.object(PodcastConfig, 'load')
    monkeypatch.setattr(sys, 'argv', ['podcast-llm', 'render', str(script), '--list-chapters'])

    main()

    assert capsys.readouterr().out.splitlines() == ['1. First chapter', '2. Second chapter']
    load.assert_not_called()


def test_cli_chapter_preview_is_reused_by_full_render(config, tmp_path, monkeypatch, mocker):
    monkeypatch.setattr(tts_backends, 'DIALOGUE_CHUNK_SIZE', 16)
    mocker.patch.object(PodcastConfig, 'load', return_value=config)
    session = mocker.patch('podcast_llm.tts_backends.requests.Session').return_value.__enter__.return_value
    session.post.return_value = gemini_response()
    script = tmp_path / 'episode.md'
    script.write_text(SCRIPT)
    args = ['podcast-llm', 'render', str(script), '--cache-dir', str(tmp_path / 'cache')]

    monkeypatch.setattr(sys, 'argv', [*args, '--chapter', '2'])
    main()
    assert session.post.call_count == 1
    assert session.post.call_args.kwargs['json']['input'][0]['content'][0]['text'] == 'Third batch.'
    with wave.open(str(tmp_path / 'episode.chapter-2.wav')) as audio:
        assert audio.getnframes() == 2400

    monkeypatch.setattr(sys, 'argv', args)
    main()
    assert session.post.call_count == 3
    with wave.open(str(tmp_path / 'episode.wav')) as audio:
        assert audio.getnframes() == 7200
    assert script.read_text() == SCRIPT


def test_cli_rejects_out_of_range_chapter(config, tmp_path, monkeypatch, mocker, capsys):
    mocker.patch.object(PodcastConfig, 'load', return_value=config)
    script = tmp_path / 'episode.md'
    script.write_text(SCRIPT)
    monkeypatch.setattr(sys, 'argv', ['podcast-llm', 'render', str(script), '--chapter', '3'])

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2
    assert 'Chapter 3 is out of range' in capsys.readouterr().err
    assert not (tmp_path / 'episode.chapter-3.wav').exists()


def test_cli_retake_publishes_only_after_every_batch_succeeds(config, tmp_path, monkeypatch, mocker):
    monkeypatch.setattr(tts_backends, 'DIALOGUE_CHUNK_SIZE', 16)
    mocker.patch.object(PodcastConfig, 'load', return_value=config)
    session = mocker.patch('podcast_llm.tts_backends.requests.Session').return_value.__enter__.return_value
    original, replacement = gemini_response(), gemini_response(b'\x01\x00')
    session.post.side_effect = [original, original, original, replacement, KeyboardInterrupt,
                                replacement, replacement]
    script = tmp_path / 'episode.md'
    script.write_text(SCRIPT)
    full = tmp_path / 'episode.wav'
    preview = tmp_path / 'episode.chapter-1.wav'
    args = ['podcast-llm', 'render', str(script)]
    monkeypatch.setattr(sys, 'argv', args)
    main()
    original_export = full.read_bytes()
    monkeypatch.setattr(sys, 'argv', [*args, '--chapter', '1'])
    main()
    previous_preview = preview.read_bytes()

    monkeypatch.setattr(sys, 'argv', [*args, '--chapter', '1', '--retake'])
    with pytest.raises(SystemExit) as interrupted:
        main()
    assert interrupted.value.code == 130
    assert preview.read_bytes() == previous_preview
    assert not list((tmp_path / 'checkpoints').rglob('active-*.txt'))
    monkeypatch.setattr(sys, 'argv', args)
    main()
    assert full.read_bytes() == original_export
    assert session.post.call_count == 5

    monkeypatch.setattr(sys, 'argv', [*args, '--chapter', '1', '--retake'])
    main()
    assert preview.read_bytes() != previous_preview
    monkeypatch.setattr(sys, 'argv', args)
    main()
    assert session.post.call_count == 7
    with wave.open(str(full)) as audio:
        assert audio.readframes(7200) == b'\x01\x00' * 4800 + bytes(4800)
