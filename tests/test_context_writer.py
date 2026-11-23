import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from podcast_llm.generate import generate


REQUEST_PATH = 'podcast_llm.context_backends.requests.post'


@pytest.fixture
def context_config(tmp_path, monkeypatch):
    monkeypatch.setattr('podcast_llm.config.config.load_dotenv', lambda: None)
    monkeypatch.setenv('GOOGLE_API_KEY', 'test-google-key')
    for service in ('OPENAI', 'TAVILY', 'ANTHROPIC'):
        monkeypatch.delenv(f'{service}_API_KEY', raising=False)
    config = tmp_path / 'config.yaml'
    config.write_text(f'checkpoint_dir: {tmp_path / "checkpoints"}\n')
    return str(config)


def make_episode(source_id='S1'):
    return {
        'title': 'A contract for JSON',
        'chapters': [{
            'title': 'Test the shape',
            'turns': [
                {'speaker': 'Interviewer', 'text': 'What does the schema check?', 'source_ids': []},
                {'speaker': 'Interviewee', 'text': 'It can check required fields.', 'source_ids': [source_id]},
            ],
        }],
    }


def gemini_response(episode):
    return Mock(**{'json.return_value': {
        'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': json.dumps(episode)}]}}],
    }})


def test_context_mode_writes_edits_and_caches_by_source_content(tmp_path, context_config, mocker):
    config_path = Path(context_config)
    with config_path.open('a') as config_file:
        config_file.write('context_writer_model: gemini-4-argon\ncontext_editor_model: gemini-4-neon\n')
    source = tmp_path / 'facts.md'
    source.write_text('Schemas can require object fields.')
    transcript = tmp_path / 'transcript.md'
    podcast = tmp_path / 'episode.mp3'
    draft = make_episode()
    edited = make_episode()
    edited['chapters'][0]['turns'][1]['text'] = 'A schema can check which fields must be present.'
    request = mocker.patch(REQUEST_PATH, side_effect=[
        gemini_response(draft), gemini_response(edited), gemini_response(draft), gemini_response(edited),
    ])
    audio = mocker.patch('podcast_llm.generate.generate_audio')
    args = {
        'topic': 'JSON Schema', 'mode': 'context', 'sources': [str(source)],
        'audience': 'New developers', 'target_minutes': 4, 'config': context_config,
        'text_output': str(transcript), 'audio_output': str(podcast),
    }

    generate(**args)
    generate(**args)

    assert request.call_count == 2
    write, edit = request.call_args_list
    assert write.args[0].endswith('/gemini-4-argon:generateContent')
    assert edit.args[0].endswith('/gemini-4-neon:generateContent')
    assert write.kwargs['json']['generationConfig']['responseMimeType'] == 'application/json'
    brief = json.loads(write.kwargs['json']['contents'][0]['parts'][0]['text'])['brief']
    assert brief['audience'] == 'New developers'
    assert brief['target_words'] == 600
    assert brief['sources'] == [{'id': 'S1', 'title': 'facts.md', 'text': source.read_text()}]
    assert json.loads(edit.kwargs['json']['contents'][0]['parts'][0]['text'])['draft'] == draft
    assert '### Test the shape' in transcript.read_text()
    assert 'A schema can check which fields must be present. [S1]' in transcript.read_text()
    assert f'[S1] facts.md · `{source}`' in transcript.read_text()
    assert audio.call_args.args[1] == [
        {'speaker': 'Interviewer', 'text': 'What does the schema check?'},
        {'speaker': 'Interviewee', 'text': 'A schema can check which fields must be present.'},
    ]
    assert audio.call_args.args[2] == str(podcast)

    source.write_text('JSON Schema distinguishes required and optional fields.')
    generate(**args)
    assert request.call_count == 4


def test_context_mode_defaults_to_a_single_model(tmp_path, context_config, mocker):
    source = tmp_path / 'facts.txt'
    source.write_text('Schemas can require object fields.')
    episode = make_episode()
    episode['chapters'][0]['turns'].insert(1, {
        'speaker': 'Interviewee', 'text': 'Oh, you mean the shape?', 'source_ids': [],
    })
    request = mocker.patch(REQUEST_PATH, return_value=gemini_response(episode))
    audio = mocker.patch('podcast_llm.generate.generate_audio')

    generate('JSON Schema', 'context', [str(source)], config=context_config,
             audio_output=str(tmp_path / 'episode.wav'))

    request.assert_called_once()
    assert request.call_args.args[0].endswith('/gemini-4-neon:generateContent')
    assert audio.call_args.args[1][1] == {'speaker': 'Interviewee', 'text': 'Oh, you mean the shape?'}


def test_context_mode_rejects_unknown_citation(tmp_path, context_config, mocker):
    source = tmp_path / 'facts.txt'
    source.write_text('Schemas can require object fields.')
    request = mocker.patch(REQUEST_PATH, return_value=gemini_response(make_episode('S2')))
    transcript = tmp_path / 'transcript.md'

    with pytest.raises(ValueError, match='unknown source IDs: S2'):
        generate('JSON Schema', 'context', [str(source)], config=context_config, text_output=str(transcript))

    request.assert_called_once()
    assert not transcript.exists()


def openai_response(episode):
    return Mock(**{'json.return_value': {'status': 'completed', 'output': [
        {'type': 'reasoning', 'summary': []},
        {'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps(episode)}]},
    ]}})


def anthropic_response(episode):
    return Mock(**{'json.return_value': {'stop_reason': 'end_turn', 'content': [
        {'type': 'thinking', 'thinking': 'Internal reasoning'},
        {'type': 'text', 'text': json.dumps(episode)},
    ]}})


@pytest.mark.parametrize(('provider', 'key_name', 'response'), [
    ('openai', 'OPENAI_API_KEY', openai_response),
    ('anthropic', 'ANTHROPIC_API_KEY', anthropic_response),
])
def test_context_routes_native_structured_output(
    tmp_path, context_config, monkeypatch, mocker, provider, key_name, response,
):
    monkeypatch.delenv('GOOGLE_API_KEY')
    monkeypatch.setenv(key_name, 'provider-key')
    Path(context_config).write_text(f'context_writer_provider: {provider}\n')
    source = tmp_path / 'source.txt'
    source.write_text('Schemas can require object fields.')
    transcript = tmp_path / 'episode.md'
    request = mocker.patch(REQUEST_PATH, return_value=response(make_episode()))

    generate('JSON Schema', 'context', [str(source)], config=context_config,
             use_checkpoints=False, text_output=str(transcript))

    request.assert_called_once()
    url, = request.call_args.args
    payload = request.call_args.kwargs['json']
    headers = request.call_args.kwargs['headers']
    if provider == 'openai':
        assert url == 'https://api.openai.com/v1/responses'
        assert headers == {'Authorization': 'Bearer provider-key'}
        assert payload['model'] == 'gpt-6.2-sol'
        assert payload['store'] is False
        assert payload['text']['format']['strict'] is True
        assert payload['text']['format']['schema']['additionalProperties'] is False
    else:
        assert url == 'https://api.anthropic.com/v1/messages'
        assert headers == {'x-api-key': 'provider-key', 'anthropic-version': '2023-06-01'}
        assert payload['model'] == 'claude-sonnet-5-5'
        schema = payload['output_config']['format']['schema']
        assert 'minLength' not in json.dumps(schema)
        assert schema['properties']['chapters']['minItems'] == 1
        assert schema['additionalProperties'] is False
    assert 'It can check required fields. [S1]' in transcript.read_text()


def test_context_can_edit_with_a_different_provider(tmp_path, context_config, monkeypatch, mocker):
    monkeypatch.delenv('GOOGLE_API_KEY')
    monkeypatch.setenv('OPENAI_API_KEY', 'writer-key')
    Path(context_config).write_text('context_writer_provider: openai\ncontext_editor_provider: anthropic\n')
    source = tmp_path / 'source.txt'
    source.write_text('Schemas can require object fields.')
    transcript = tmp_path / 'episode.md'
    draft = make_episode()
    edited = make_episode()
    edited['title'] = 'Edited episode'
    request = mocker.patch(REQUEST_PATH, side_effect=[openai_response(draft), anthropic_response(edited)])

    with pytest.raises(ValueError, match='Set ANTHROPIC_API_KEY'):
        generate('JSON Schema', 'context', [str(source)], config=context_config, use_checkpoints=False)
    request.assert_not_called()

    monkeypatch.setenv('ANTHROPIC_API_KEY', 'editor-key')
    generate('JSON Schema', 'context', [str(source)], config=context_config,
             use_checkpoints=False, text_output=str(transcript))

    write, edit = request.call_args_list
    assert write.kwargs['headers']['Authorization'] == 'Bearer writer-key'
    assert edit.kwargs['headers']['x-api-key'] == 'editor-key'
    assert json.loads(edit.kwargs['json']['messages'][0]['content'])['draft'] == draft
    assert transcript.read_text().startswith('# Edited episode')


@pytest.mark.parametrize(('provider', 'key_name', 'response', 'reason'), [
    ('openai', 'OPENAI_API_KEY', {
        'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'},
    }, 'max_output_tokens'),
    ('openai', 'OPENAI_API_KEY', {'status': 'completed', 'output': [
        {'type': 'message', 'content': [{'type': 'refusal', 'refusal': 'Refusal text'}]},
    ]}, 'refused'),
    ('anthropic', 'ANTHROPIC_API_KEY', {'stop_reason': 'max_tokens', 'content': [
        {'type': 'text', 'text': '{}'},
    ]}, 'max_tokens'),
])
def test_context_rejects_incomplete_or_refused_output(
    tmp_path, context_config, monkeypatch, mocker, provider, key_name, response, reason,
):
    monkeypatch.setenv(key_name, 'provider-key')
    Path(context_config).write_text(f'context_writer_provider: {provider}\n')
    source = tmp_path / 'source.txt'
    source.write_text('Schemas can require object fields.')
    output = tmp_path / 'transcript.md'
    mocker.patch(REQUEST_PATH, return_value=Mock(**{'json.return_value': response}))

    with pytest.raises(ValueError, match=reason):
        generate('JSON Schema', 'context', [str(source)], config=context_config,
                 use_checkpoints=False, text_output=str(output))

    assert not output.exists()
