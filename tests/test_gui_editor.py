from dataclasses import replace
from io import BytesIO
from pathlib import Path
import wave

from fastapi import FastAPI
from fastapi.testclient import TestClient
from gradio import mount_gradio_app
import pytest

from podcast_llm import gui, text_to_speech
from podcast_llm.config import PodcastConfig
from podcast_llm.render import render_script
from podcast_llm.tts_backends import AudioChunk


TRANSCRIPT = """# JSON episode

## Script

### Introduction

**Interviewer**: Let's talk about schemas. [S1]

### Writing the Contract

**Interviewee**: Add a required list. [S1]

## Sources

- [S1] JSON Schema
"""


@pytest.fixture
def gradio_client(tmp_path, monkeypatch):
    monkeypatch.setenv('GRADIO_TEMP_DIR', str(tmp_path / 'gradio'))
    monkeypatch.setenv('GRADIO_ANALYTICS_ENABLED', 'False')
    interface = gui.build_interface()
    with TestClient(mount_gradio_app(FastAPI(), interface, path='/')) as client:
        try:
            yield client
        finally:
            interface.state_holder.delete_state('editor-test')


def call_gradio(client, endpoint, *inputs):
    response = client.post(f'/gradio_api/run/{endpoint}', json={'data': inputs, 'session_hash': 'editor-test'})
    assert response.status_code == 200, response.text
    return response.json()['data']


def test_gradio_chapter_preview_and_export_share_the_cli_cache(gradio_client, tmp_path, monkeypatch, mocker):
    script = tmp_path / 'episode.md'
    script.write_text(TRANSCRIPT)
    config = replace(PodcastConfig.load(require_research_credentials=False), tts_provider='test_editor',
                     output_format='wav', temp_audio_dir=str(tmp_path / 'temp'),
                     checkpoint_dir=str(tmp_path / 'checkpoints'))
    mocker.patch.object(PodcastConfig, 'load', return_value=config)
    monkeypatch.setattr(text_to_speech, '_TTS_BACKENDS', text_to_speech._TTS_BACKENDS.copy())
    requests = []

    def synthesize(active_config, turns):
        requests.append([turn['text'] for turn in turns])
        buffer = BytesIO()
        with wave.open(buffer, 'wb') as audio:
            audio.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
            audio.writeframes(bytes(4800))
        yield AudioChunk(buffer.getvalue(), 'wav')

    text_to_speech.register_tts_backend('test_editor', synthesize)
    render_script(script, config)
    assert requests == [["Let's talk about schemas."], ['Add a required list.']]
    requests.clear()

    upload = gradio_client.post('/gradio_api/upload', files={'files': (script.name, script.read_bytes())})
    assert upload.status_code == 200, upload.text
    uploaded_file = {'path': upload.json()[0], 'meta': {'_type': 'gradio.FileData'}}
    _, selection, body, *_ = call_gradio(gradio_client, 'open_transcript', uploaded_file, None)
    assert selection['value'] == '0'
    _, _, body, _ = call_gradio(gradio_client, 'select_chapter', None, '1', body)
    assert body == '**Interviewee**: Add a required list. [S1]'
    edited = body.replace('Add a required list.', 'List every required key.')
    _, preview = call_gradio(gradio_client, 'render_chapter', None, edited, None)
    assert requests == [['List every required key.']]
    with wave.open(BytesIO(gradio_client.get(preview['url']).content)) as audio:
        assert audio.getnframes() == 2400

    transcribe = mocker.patch('podcast_llm.audio_qa.transcribe_audio', return_value='')
    response = gradio_client.post('/gradio_api/run/check_audio', json={
        'data': [None, body, None], 'session_hash': 'editor-test',
    })
    assert response.status_code == 500
    assert 'Render the current chapter dialogue' in response.json()['error']
    transcribe.assert_not_called()

    report, = call_gradio(gradio_client, 'check_audio', None, edited, None)
    assert 'possibly missing: List every required key' in report
    _, new_preview = call_gradio(gradio_client, 'retake', None, edited, None)
    assert gradio_client.get(new_preview['url']).status_code == 200
    assert requests == [['List every required key.'], ['List every required key.']]

    # Navigation retains the change, and export needs no further speech requests.
    _, _, introduction, _ = call_gradio(gradio_client, 'select_chapter', None, '0', edited)
    _, full, transcript = call_gradio(gradio_client, 'export_episode', None, introduction, None)
    assert requests == [['List every required key.'], ['List every required key.']]
    assert gradio_client.get(transcript['url']).text == TRANSCRIPT.replace(
        'Add a required list.', 'List every required key.',
    )
    with wave.open(BytesIO(gradio_client.get(full['url']).content)) as audio:
        assert audio.getnframes() == 4800


def test_drafted_transcript_opens_in_episode_editor(gradio_client, mocker):
    def generate_transcript(**options):
        Path(options['text_output']).write_text(TRANSCRIPT)

    generate = mocker.patch.object(gui, 'generate', side_effect=generate_transcript)
    _, selection, body, *_, tab = call_gradio(
        gradio_client, 'draft_script', None, ' JSON Schema ', 'context', None,
        'https://example.com/schema\n\nignored text', 1, True, None, 'Developers', 4,
    )
    options = generate.call_args.kwargs
    assert options['topic'] == 'JSON Schema'
    assert options['mode'] == 'context'
    assert options['sources'] == ['https://example.com/schema']
    assert options['audio_output'] is None
    assert options['audience'] == 'Developers'
    assert options['target_minutes'] == 4
    assert selection['choices'][1] == ['2. Writing the Contract', '1']
    assert "Let's talk about schemas." in body
    assert tab['selected'] == 'editor'
