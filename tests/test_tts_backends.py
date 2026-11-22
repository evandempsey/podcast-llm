import base64
import wave
from dataclasses import replace
from io import BytesIO
from unittest.mock import Mock, call

import pytest

from podcast_llm import text_to_speech, tts_backends
from podcast_llm.config import PodcastConfig


TURNS = [
    {'speaker': 'Interviewer', 'text': 'Hello world'},
    {'speaker': 'Interviewee', 'text': 'Hello there'},
    {'speaker': 'Interviewer', 'text': 'A final turn'},
]


@pytest.fixture
def config(monkeypatch, tmp_path):
    monkeypatch.setattr('podcast_llm.config.config.load_dotenv', lambda: None)
    for service in ('OPENAI', 'TAVILY', 'ANTHROPIC'):
        monkeypatch.setenv(f'{service}_API_KEY', 'test-key')
    monkeypatch.delenv('GOOGLE_API_KEY', raising=False)
    monkeypatch.delenv('ELEVENLABS_API_KEY', raising=False)
    return replace(PodcastConfig.load(), temp_audio_dir=str(tmp_path / 'chunks'), output_format='wav')


@pytest.fixture
def wav_bytes():
    buffer = BytesIO()
    with wave.open(buffer, 'wb') as audio:
        audio.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        audio.writeframes(bytes(4800))
    return buffer.getvalue()


@pytest.mark.parametrize(('provider', 'renderer'), [
    ('google', 'process_line_google'),
    ('elevenlabs', 'process_line_elevenlabs'),
    ('google_multispeaker', 'process_lines_google_multispeaker'),
])
def test_existing_yaml_backends_keep_rendering_strategy(config, provider, renderer, mocker, tmp_path):
    active = replace(config, google_api_key='google-key', elevenlabs_api_key='eleven-key', tts_provider=provider)
    synthesize = mocker.patch.object(text_to_speech, renderer, return_value=b'mp3 data')
    merge = mocker.patch.object(text_to_speech, 'merge_audio_files')
    turns = TURNS + TURNS

    text_to_speech.generate_audio(active, turns, str(tmp_path / 'existing.wav'))

    if provider == 'google_multispeaker':
        assert synthesize.call_args_list == [call(active, turns[:4]), call(active, turns[4:])]
    else:
        assert synthesize.call_args_list == [call(active, turn['text'], turn['speaker']) for turn in turns]
    assert len(merge.call_args.args[0]) == synthesize.call_count


def test_eleven_v4_uses_configured_voice_ids_and_preserves_continuity(config, monkeypatch, mocker, tmp_path):
    monkeypatch.setattr(tts_backends, 'DIALOGUE_CHUNK_SIZE', 24)
    session = mocker.patch('podcast_llm.tts_backends.requests.Session').return_value.__enter__.return_value
    session.post.return_value = Mock(content=bytes(4800), headers={
        'content-type': 'audio/pcm', 'request-id': 'first-render',
    })
    config = replace(config, elevenlabs_api_key='eleven-key', tts_provider='elevenlabs_v4', tts_settings={
        'elevenlabs_v4': {'voice_mapping': {'Interviewer': 'custom-host', 'Interviewee': 'custom-guest'}},
    })
    output = tmp_path / 'eleven.wav'

    text_to_speech.generate_audio(config, TURNS, str(output))

    first, second = [call.kwargs for call in session.post.call_args_list]
    assert first['headers'] == {'xi-api-key': 'eleven-key'}
    assert first['json'] == {
        'model_id': 'eleven_v4',
        'inputs': [
            {'voice_id': 'custom-host', 'text': 'Hello world'},
            {'voice_id': 'custom-guest', 'text': 'Hello there'},
        ],
    }
    assert second['json']['previous_request_ids'] == ['first-render']
    assert second['json']['inputs'] == [{'voice_id': 'custom-host', 'text': 'A final turn'}]
    with wave.open(str(output)) as audio:
        assert audio.getnframes() == 4800


def test_gemini_dual_and_single_speaker_requests(config, monkeypatch, mocker, tmp_path, wav_bytes):
    monkeypatch.setattr(tts_backends, 'DIALOGUE_CHUNK_SIZE', 24)
    session = mocker.patch('podcast_llm.tts_backends.requests.Session').return_value.__enter__.return_value
    session.post.return_value.json.return_value = {
        'status': 'completed',
        'steps': [{'content': [{'type': 'audio', 'data': base64.b64encode(wav_bytes).decode()}]}],
    }
    config = replace(config, google_api_key='google-key', tts_provider='gemini_3_8_flash_tts', tts_settings={
        'gemini_3_8_flash_tts': {'voice_mapping': {'Interviewer': 'Kore'}},
    })
    output = tmp_path / 'gemini.wav'

    text_to_speech.generate_audio(config, TURNS, str(output))

    first, second = [call.kwargs for call in session.post.call_args_list]
    assert first['headers'] == {'x-goog-api-key': 'google-key'}
    assert first['json']['model'] == 'gemini-3.8-flash-tts'
    assert first['json']['generation_config']['speech_config'] == {'speakers': [
        {'speaker': 'Interviewer', 'voice': 'Kore'},
        {'speaker': 'Interviewee', 'voice': 'Charon'},
    ]}
    assert first['json']['input'][0]['content'][1]['annotations'] == [
        {'type': 'speech_metadata', 'speaker': 'Interviewee'},
    ]
    assert second['json']['generation_config']['speech_config'] == [{'voice': 'Kore'}]
    assert second['json']['input'][0]['content'] == [{'type': 'text', 'text': 'A final turn'}]
    with wave.open(str(output)) as audio:
        assert audio.getnframes() == 4800


def test_registered_backend_failure_preserves_output(config, monkeypatch, tmp_path, wav_bytes):
    monkeypatch.setattr(text_to_speech, '_TTS_BACKENDS', text_to_speech._TTS_BACKENDS.copy())

    def render(active_config, turns):
        assert active_config.tts_provider == 'custom'
        assert turns == TURNS
        yield tts_backends.AudioChunk(wav_bytes, 'wav')
        raise RuntimeError('request failed')

    text_to_speech.register_tts_backend('custom', render)
    config = replace(config, tts_provider='custom')
    output = tmp_path / 'existing.wav'
    output.write_bytes(b'existing')

    with pytest.raises(RuntimeError, match='request failed'):
        text_to_speech.generate_audio(config, TURNS, str(output))

    assert output.read_bytes() == b'existing'
    assert list((tmp_path / 'chunks').iterdir()) == []
