"""Chapter editing and audio callbacks for the Gradio episode editor."""

from pathlib import Path
import re
from shutil import rmtree
from tempfile import mkdtemp
from typing import NamedTuple

import gradio as gr

from podcast_llm.audio_qa import check_chapter_audio, format_audio_check
from podcast_llm.config import PodcastConfig
from podcast_llm.generate import DEFAULT_CONFIG_PATH
from podcast_llm.render import MarkdownScript, ScriptChapter, parse_chapter, read_transcript, render_chapters


class EpisodeEditor(NamedTuple):
    directory: Path
    transcript: MarkdownScript
    filename: str
    selected: int = 0
    preview: tuple[ScriptChapter, Path] | None = None


def editor_directory(state: EpisodeEditor | None) -> Path:
    return state.directory if state else Path(mkdtemp(prefix='podcast-editor-'))


def close_editor(state: EpisodeEditor | None) -> None:
    if state is not None:
        rmtree(state.directory, ignore_errors=True)


def open_transcript(file: str, state: EpisodeEditor | None):
    if not file:
        raise gr.Error('Choose a Markdown transcript to open.')
    try:
        path = Path(file)
        transcript = read_transcript(path)
        for chapter in transcript.chapters:
            parse_chapter(chapter)
    except (ValueError, OSError) as error:
        raise gr.Error(str(error)) from error

    state = EpisodeEditor(editor_directory(state), transcript, path.stem)
    choices = [(f'{index + 1}. {chapter.title}', str(index)) for index, chapter in enumerate(transcript.chapters)]
    selection = gr.update(choices=choices, value='0')
    return state, selection, transcript.chapters[0].body.strip(), None, None, None, gr.update(visible=True)


def save_chapter_text(state: EpisodeEditor, body: str) -> EpisodeEditor:
    chapters = state.transcript.chapters
    current = chapters[state.selected]
    body = (body or '').strip()
    if current.body.strip() == body:
        return state
    if re.search(r'^#{1,3}[ \t]', body, re.M):
        raise ValueError('Edit the dialogue here. Keep headings in the Markdown transcript.')
    updated = current._replace(body=f'\n{body}\n\n')
    parse_chapter(updated)
    chapters = chapters[:state.selected] + (updated,) + chapters[state.selected + 1:]
    return state._replace(transcript=state.transcript._replace(chapters=chapters))


def select_chapter(state: EpisodeEditor, selection: str | None, body: str):
    try:
        index = int(selection) if selection is not None else -1
        if not 0 <= index < len(state.transcript.chapters):
            raise ValueError('Choose a chapter from the list.')
        state = save_chapter_text(state, body)._replace(selected=index, preview=None)
    except ValueError as error:
        gr.Warning(str(error))
        return state, gr.update(value=str(state.selected)), body, gr.skip()
    chapter = state.transcript.chapters[state.selected]
    return state, gr.skip(), chapter.body.strip(), None


def render_chapter_preview(state: EpisodeEditor, body: str, config_file: str | None, *, retake: bool = False):
    try:
        state = save_chapter_text(state, body)
        config = PodcastConfig.load(config_file or DEFAULT_CONFIG_PATH, require_research_credentials=False)
        chapter = parse_chapter(state.transcript.chapters[state.selected])
        number = state.selected + 1
        directory = Path(mkdtemp(prefix='preview-', dir=state.directory))
        output = render_chapters([chapter], config, directory / f'chapter-{number}.wav',
                                 first_chapter=number, retake=retake)
    except (ValueError, OSError) as error:
        raise gr.Error(str(error)) from error
    return state._replace(preview=(chapter, output)), str(output)


def check_chapter_preview(state: EpisodeEditor, body: str, config_file: str | None):
    try:
        state = save_chapter_text(state, body)
        chapter = parse_chapter(state.transcript.chapters[state.selected])
        if state.preview is None or state.preview[0] != chapter:
            raise ValueError('Render the current chapter dialogue before checking its audio.')
        config = PodcastConfig.load(config_file or DEFAULT_CONFIG_PATH, require_research_credentials=False)
        findings = check_chapter_audio(chapter, state.preview[1], config)
    except (ValueError, OSError) as error:
        raise gr.Error(str(error)) from error
    return format_audio_check(chapter, findings)


def export_episode(state: EpisodeEditor, body: str, config_file: str | None, *, master: bool = False):
    try:
        state = save_chapter_text(state, body)
        config = PodcastConfig.load(config_file or DEFAULT_CONFIG_PATH, require_research_credentials=False)
        chapters = [parse_chapter(chapter) for chapter in state.transcript.chapters]
        directory = Path(mkdtemp(prefix='export-', dir=state.directory))
        output = render_chapters(chapters, config, directory / f'{state.filename}.{config.output_format}',
                                 master=master)
        transcript = directory / f'{state.filename}.md'
        transcript.write_text(state.transcript.as_text(), encoding='utf-8')
    except (ValueError, OSError) as error:
        raise gr.Error(str(error)) from error
    return state, str(output), str(transcript)
