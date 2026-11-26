from dataclasses import replace
from io import BytesIO
from pathlib import Path
import wave

from podcast_llm import gui, gui_editor, text_to_speech
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


def test_gradio_chapter_preview_and_export_share_the_cli_cache(tmp_path, monkeypatch, mocker):
    script = tmp_path / 'episode.md'
    script.write_text(TRANSCRIPT)
    config = replace(PodcastConfig.load(require_research_credentials=False), tts_provider='test_editor',
                     output_format='wav', temp_audio_dir=str(tmp_path / 'temp'),
                     checkpoint_dir=str(tmp_path / 'checkpoints'))
    mocker.patch.object(gui_editor.PodcastConfig, 'load', return_value=config)
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

    state, selection, body, *_ = gui_editor.open_transcript(str(script), None)
    try:
        assert selection['value'] == '0'
        state, _, body, _ = gui_editor.select_chapter(state, '1', body)
        assert body == '**Interviewee**: Add a required list. [S1]'
        body = body.replace('Add a required list.', 'List every required key.')
        state, preview = gui_editor.render_chapter_preview(state, body, None)
        assert requests == [['List every required key.']]
        with wave.open(preview) as audio:
            assert audio.getnframes() == 2400

        # Navigation retains the change, and export needs no further speech requests.
        state, _, introduction, _ = gui_editor.select_chapter(state, '0', body)
        state, full, transcript = gui_editor.export_episode(state, introduction, None)
        assert requests == [['List every required key.']]
        assert Path(transcript).read_text() == TRANSCRIPT.replace('Add a required list.', 'List every required key.')
        with wave.open(full) as audio:
            assert audio.getnframes() == 4800
    finally:
        gui_editor.close_editor(state)


def test_drafted_transcript_opens_in_episode_editor(mocker):
    def generate_to_file(topic, mode, files, urls, qa_rounds, checkpoints, config, text_output, audio_output,
                         audience, target_minutes):
        assert topic == 'JSON Schema' and mode == 'context'
        assert audio_output == ''
        Path(text_output).write_text(TRANSCRIPT)

    mocker.patch.object(gui, 'submit_handler', side_effect=generate_to_file)
    state, selection, body, *_, tab = gui.draft_script(
        None, 'JSON Schema', 'context', ['source.md'], '', 1, True, None, 'Developers', 4,
    )
    try:
        assert selection['choices'][1] == ('2. Writing the Contract', '1')
        assert "Let's talk about schemas." in body
        assert tab['selected'] == 'editor'
    finally:
        gui_editor.close_editor(state)
