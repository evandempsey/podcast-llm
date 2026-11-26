from dataclasses import replace
from pathlib import Path
import sys
import wave

from openai import OpenAIError
import pytest

from podcast_llm import audio_qa
from podcast_llm.config import PodcastConfig
from podcast_llm.generate import main
from podcast_llm.render import ScriptChapter


CHAPTER = ScriptChapter('Writing the Contract', [
    {'speaker': 'Host', 'text': 'Gemini uses three fields.'},
    {'speaker': 'Guest', 'text': 'Wait.'},
    {'speaker': 'Host', 'text': 'Make the ticket ID required.'},
])
TRANSCRIPT = 'Gemini uses 3 fields. Wait. Make the ticket ID required.'


def test_audio_check_flags_changed_names_numbers_and_missing_interjections():
    assert audio_qa.compare_transcript(CHAPTER, TRANSCRIPT) == []
    findings = audio_qa.compare_transcript(CHAPTER, 'Jemini uses four fields. Make the ticket ID required.')
    assert findings == [
        'Turn 1 (Host) — expected “Gemini”, heard “Jemini”',
        'Turn 1 (Host) — expected “three”, heard “four”',
        'Turn 2 (Guest) — possibly missing: Wait',
    ]
    assert audio_qa.compare_transcript(CHAPTER, TRANSCRIPT.replace('3 fields', '3,000 fields')) == [
        'Turn 1 (Host) — expected “three”, heard “3,000”',
    ]


def test_check_command_reports_and_exits_on_missing_line(tmp_path, monkeypatch, mocker, capsys):
    script = tmp_path / 'episode.md'
    script.write_text('## Script\n\n### Writing the Contract\n\n' + '\n\n'.join(
        f"**{turn['speaker']}**: {turn['text']}" for turn in CHAPTER.turns
    ) + '\n')
    recording = tmp_path / 'episode.chapter-1.wav'
    recording.write_bytes(b'audio')
    config = replace(PodcastConfig.load(require_research_credentials=False), openai_api_key='test-key')
    mocker.patch.object(PodcastConfig, 'load', return_value=config)
    transcribe = mocker.patch.object(audio_qa, 'OpenAI').return_value.__enter__.return_value.audio.transcriptions.create
    monkeypatch.setattr(sys, 'argv', ['podcast-llm', 'check', str(script), '--chapter', '1'])

    transcribe.return_value.text = TRANSCRIPT.replace('Wait. ', '')
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 1
    assert 'Turn 2 (Guest) — possibly missing: Wait' in capsys.readouterr().out
    transcribe.assert_called_once_with(file=recording, model=config.audio_qa_model)

    transcribe.return_value.text = TRANSCRIPT
    main()
    assert 'No likely omissions or changes detected.' in capsys.readouterr().out


def test_audio_transcription_splits_large_recording(tmp_path, monkeypatch, mocker):
    preview = tmp_path / 'long.wav'
    with wave.open(str(preview), 'wb') as file:
        file.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        file.writeframes(bytes(64000))
    monkeypatch.setattr(audio_qa, 'MAX_UPLOAD_BYTES', 40000)
    monkeypatch.setattr(audio_qa, 'TRANSCRIPTION_CHUNK_MS', 1000)
    config = replace(PodcastConfig.load(require_research_credentials=False), openai_api_key='test-key',
                     audio_qa_model='configured-transcriber')
    transcribe = mocker.patch.object(audio_qa, 'OpenAI').return_value.__enter__.return_value.audio.transcriptions.create
    transcribe.return_value.text = 'Chunk'

    assert audio_qa.transcribe_audio(preview, config) == 'Chunk\nChunk'
    assert transcribe.call_count == 2
    assert all(call.kwargs['model'] == 'configured-transcriber' for call in transcribe.call_args_list)


def test_audio_transcription_errors_are_reported(tmp_path, mocker):
    config = replace(PodcastConfig.load(require_research_credentials=False), openai_api_key='')
    with pytest.raises(ValueError, match='OPENAI_API_KEY'):
        audio_qa.transcribe_audio(Path('episode.wav'), config)

    recording = tmp_path / 'episode.wav'
    recording.write_bytes(b'audio')
    client = mocker.patch.object(audio_qa, 'OpenAI').return_value.__enter__.return_value
    client.audio.transcriptions.create.side_effect = OpenAIError('provider failure')
    with pytest.raises(ValueError, match='Audio transcription failed'):
        audio_qa.transcribe_audio(recording, replace(config, openai_api_key='test-key'))
