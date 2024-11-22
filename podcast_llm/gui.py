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


import logging
import os
from pathlib import Path
import tempfile

import gradio as gr
from gradio_log import Log

from .config.logging_config import setup_logging
from .generate import generate

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
    audio_output: str
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

    text_output_file = text_output.strip() if text_output.strip() else None
    audio_output_file = audio_output.strip() if audio_output.strip() else None

    # Split URLs by line and filter out non-URL lines
    source_urls_list = [
        url.strip() 
        for url in source_urls.strip().split('\n') 
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
        log_file=temp_log_file
    )

def main():
    """
    Main entry point for the Gradio web interface.

    Creates and launches a Gradio interface that provides a user-friendly way to interact
    with the podcast generation system. The interface includes:
    - Topic input and conversation settings
    - Mode selection (research vs context)
    - Source file and URL inputs for context mode
    - Behavior options like checkpointing
    - Output configuration options

    The interface is organized into logical sections with clear labels and tooltips.
    All inputs are validated and passed to the generate() function.

    Returns:
        None
    """
    with gr.Blocks() as iface:
        # Title
        gr.Markdown('# Podcast-LLM', elem_classes='text-center')

        # Conversation Options Section
        gr.Markdown('## Conversation Options')
        with gr.Row():
            topic_input = gr.Textbox(label='Topic')
            qa_rounds_input = gr.Number(
                label='Number of rounds of Q&A per section',
                value=1,
                interactive=True,
                minimum=1,
                maximum=10,
                precision=0
            )

        # Mode Selection Section  
        gr.Markdown('## Mode of Operation')
        mode_of_operation = gr.Radio(
            choices=['research', 'context'],
            label='Mode',
            value='research',
            interactive=True,
            show_label=False
        )

        # Source Inputs Section
        with gr.Row(equal_height=True):
            source_files = gr.File(
                label='Source files',
                file_count='multiple',
                type='filepath'
            )
            source_urls = gr.TextArea(label='Source URLs')

        # Behavior Options Section
        gr.Markdown('## Behaviour Options')
        use_checkpoints_input = gr.Checkbox(
            label='Use Checkpoints',
            value=True
        )
        custom_config_file_input = gr.File(
            label='Config file',
            type='filepath'
        )

        # Output Options Section
        gr.Markdown('## Output Options')
        with gr.Row():
            text_output_input = gr.Textbox(label='Text output')
            audio_output_input = gr.Textbox(label='Audio output')

        # Submit Button
        submit_button = gr.Button('Generate Podcast')
        submit_button.click(
            fn=submit_handler,
            inputs=[
                topic_input,
                mode_of_operation,
                source_files,
                source_urls,
                qa_rounds_input,
                use_checkpoints_input,
                custom_config_file_input,
                text_output_input,
                audio_output_input
            ],
            outputs=[]
        )

        # Log Display
        gr.Markdown('## System Log')
        Log(temp_log_file, dark=True, xterm_font_size=12)

    iface.launch()


if __name__ == '__main__':
    main()
