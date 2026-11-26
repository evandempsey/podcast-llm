"""
Graphical user interface module for podcast generation.

This module provides a web-based GUI using Gradio for generating podcasts. It allows users
to interactively specify podcast generation parameters including:

- Topic selection
- Operation mode (research or context-based)
- Source materials (files and URLs) for context mode
- Number of Q&A rounds
- Checkpointing preferences
- Custom configuration
- Output paths for text and audio

The GUI provides a user-friendly way to access the podcast generation functionality
without needing to use the command line interface.

The module handles form submission, input validation, logging setup, and coordinates
with the core generation functionality. It uses temporary files for logging and
provides real-time feedback during the generation process.
"""


from functools import partial
import logging
import os
from pathlib import Path
import tempfile

import gradio as gr
from gradio_log import Log

from .config.logging_config import setup_logging
from .context_writer import DEFAULT_CONTEXT_AUDIENCE, DEFAULT_CONTEXT_MINUTES
from .generate import generate
from .gui_editor import (
    check_chapter_preview, close_editor, export_episode, open_transcript, render_chapter_preview, select_chapter,
)

PACKAGE_ROOT = Path(__file__).parent
DEFAULT_CONFIG_PATH = os.path.join(PACKAGE_ROOT, 'config', 'config.yaml')

temp_log_file = tempfile.NamedTemporaryFile(mode='w', delete=False).name


def submit_handler(
    topic: str,
    mode_of_operation: str,
    source_files: list[str],
    source_urls: str,
    qa_rounds: int,
    use_checkpoints: bool,
    custom_config_file: str | None,
    text_output: str,
    audio_output: str,
    audience: str = DEFAULT_CONTEXT_AUDIENCE,
    target_minutes: int = DEFAULT_CONTEXT_MINUTES,
) -> None:
    """
    Handle form submission for podcast generation.

    Processes user inputs from the GUI form and calls the generate function with appropriate parameters.
    Handles input validation, logging, and file path processing.

    Args:
        topic: The podcast topic
        mode_of_operation: Either 'research' or 'context' mode
        source_files: List of source file paths to use as context
        source_urls: Newline-separated string of URLs to use as context
        qa_rounds: Number of Q&A rounds per section
        use_checkpoints: Whether to enable checkpointing
        custom_config_file: Optional path to custom config file
        text_output: Path to save text output (optional)
        audio_output: Path to save audio output (optional)
        audience: Intended listeners in context mode
        target_minutes: Approximate spoken runtime in context mode

    Returns:
        None
    """
    setup_logging(log_level=logging.INFO, output_file=temp_log_file)
    # Print values and types of all arguments
    logging.info(f'Topic: {topic} (type: {type(topic)})')
    logging.info(f'Mode of Operation: {mode_of_operation} (type: {type(mode_of_operation)})')
    logging.info(f'Source Files: {source_files} (type: {type(source_files)})')
    logging.info(f'Source URLs: {source_urls} (type: {type(source_urls)})')
    logging.info(f'QA Rounds: {qa_rounds} (type: {type(qa_rounds)})')
    logging.info(f'Use Checkpoints: {use_checkpoints} (type: {type(use_checkpoints)})')
    logging.info(f'Custom Config File: {custom_config_file} (type: {type(custom_config_file)})')
    logging.info(f'Text Output: {text_output} (type: {type(text_output)})')
    logging.info(f'Audio Output: {audio_output} (type: {type(audio_output)})')

    text_output_file = (text_output or '').strip() or None
    audio_output_file = (audio_output or '').strip() or None

    # Split URLs by line and filter out non-URL lines
    source_urls_list = [
        url.strip() 
        for url in (source_urls or '').splitlines()
        if url.strip().startswith(('http://', 'https://'))
    ]

    # Combine source files and URLs into single sources list
    sources = (source_files or []) + source_urls_list
    sources = sources if sources else None

    generate(
        topic=topic.strip(),
        mode=mode_of_operation,
        sources=sources,
        qa_rounds=qa_rounds,
        use_checkpoints=use_checkpoints,
        audio_output=audio_output_file,
        text_output=text_output_file,
        config=custom_config_file if custom_config_file else DEFAULT_CONFIG_PATH,
        debug=False,
        log_file=temp_log_file,
        audience=audience,
        target_minutes=target_minutes if target_minutes is not None else DEFAULT_CONTEXT_MINUTES,
    )

def draft_script(state, topic, mode, source_files, source_urls, qa_rounds,
                 use_checkpoints, config_file, audience, target_minutes):
    with tempfile.TemporaryDirectory(prefix='podcast-draft-') as directory:
        transcript = Path(directory) / 'episode.md'
        submit_handler(
            topic, mode, source_files, source_urls, qa_rounds, use_checkpoints,
            config_file, str(transcript), '', audience, target_minutes,
        )
        return (*open_transcript(str(transcript), state), gr.update(selected='editor'))


def build_interface() -> gr.Blocks:
    """Build the generation form and the saved-script editor."""
    with gr.Blocks(title='Podcast LLM') as iface:
        gr.Markdown('# Podcast-LLM')
        state = gr.State(delete_callback=close_editor)
        with gr.Accordion('Settings', open=False):
            config = gr.File(label='Config file (YAML)', file_types=['.yaml', '.yml'], type='filepath')

        with gr.Tabs(selected='editor') as tabs:
            with gr.Tab('Episode editor', id='editor'):
                saved_script = gr.File(
                    label='Open a saved transcript', file_types=['.md', '.markdown'], type='filepath',
                )
                with gr.Group(visible=False) as editor:
                    chapter = gr.Dropdown(label='Chapter', interactive=True)
                    dialogue = gr.Textbox(label='Chapter dialogue', lines=14, interactive=True)
                    with gr.Row():
                        preview = gr.Button('Render chapter', variant='primary')
                        check = gr.Button('Check audio')
                        retake = gr.Button('Retake chapter')
                        export = gr.Button('Export episode')
                    audio = gr.Audio(label='Chapter preview', interactive=False, format='wav')
                    report = gr.Textbox(label='Audio check', lines=3, interactive=False)
                    with gr.Row():
                        audio_download = gr.File(label='Episode audio', interactive=False)
                        transcript_download = gr.File(label='Edited transcript', interactive=False)

            with gr.Tab('Generate', id='generate'):
                topic = gr.Textbox(label='Topic')
                mode = gr.Radio(
                    choices=['research', 'context'], label='Mode', value='research', interactive=True,
                )
                qa_rounds = gr.Number(
                    label='Q&A rounds per section', value=1, minimum=1, maximum=10, precision=0,
                )
                with gr.Row(visible=False) as context_options:
                    audience = gr.Textbox(label='Audience', value=DEFAULT_CONTEXT_AUDIENCE)
                    target_minutes = gr.Number(
                        label='Target runtime (minutes)', value=DEFAULT_CONTEXT_MINUTES, minimum=1, precision=0,
                    )
                mode.change(
                    fn=lambda selected: (
                        gr.update(visible=selected == 'research'), gr.update(visible=selected == 'context'),
                    ),
                    inputs=mode,
                    outputs=[qa_rounds, context_options],
                )
                with gr.Row():
                    source_files = gr.File(label='Source files', file_count='multiple', type='filepath')
                    source_urls = gr.TextArea(label='Source URLs')
                use_checkpoints = gr.Checkbox(label='Use checkpoints', value=True)
                draft = gr.Button('Draft script for editing', variant='primary')
                with gr.Accordion('Generate to files', open=False):
                    text_output = gr.Textbox(label='Transcript output path')
                    audio_output = gr.Textbox(label='Audio output path')
                    generate_files = gr.Button('Generate podcast')

        with gr.Accordion('System log', open=False):
            Log(temp_log_file, dark=True, xterm_font_size=12)

        editor_outputs = [state, chapter, dialogue, audio, audio_download, transcript_download, editor]
        saved_script.upload(
            open_transcript, [saved_script, state], editor_outputs, api_name='open_transcript',
        )
        chapter.input(select_chapter, [state, chapter, dialogue], [state, chapter, dialogue, audio])
        chapter_inputs = [state, dialogue, config]
        preview.click(render_chapter_preview, chapter_inputs, [state, audio], api_name='render_chapter')
        check.click(check_chapter_preview, chapter_inputs, report, api_name='check_audio')
        retake.click(partial(render_chapter_preview, retake=True), chapter_inputs, [state, audio], api_name='retake')
        audio.change(lambda: None, outputs=report, api_name=False, queue=False)
        dialogue.input(lambda: None, outputs=report, api_name=False, queue=False)
        export.click(
            export_episode, [state, dialogue, config], [state, audio_download, transcript_download],
            api_name='export_episode',
        )
        generation_inputs = [topic, mode, source_files, source_urls, qa_rounds, use_checkpoints, config]
        draft.click(
            draft_script, [state, *generation_inputs, audience, target_minutes], [*editor_outputs, tabs],
            api_name='draft_script',
        )
        generate_files.click(
            submit_handler, [*generation_inputs, text_output, audio_output, audience, target_minutes], [],
        )
    return iface


def main():
    """Launch the Gradio application."""
    setup_logging(logging.INFO, output_file=temp_log_file)
    build_interface().launch()


if __name__ == '__main__':
    main()
