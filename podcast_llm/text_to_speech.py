"""
Text-to-speech conversion module for podcast generation.

This module handles the conversion of text scripts into natural-sounding speech using
multiple TTS providers (Google Cloud TTS and ElevenLabs). It includes functionality for:

- Rate limiting API requests to stay within provider quotas
- Exponential backoff retry logic for API resilience 
- Processing individual conversation lines with appropriate voices
- Merging multiple audio segments into a complete podcast
- Managing temporary audio file storage and cleanup

The module supports different voices for interviewer/interviewee to create natural
conversational flow and allows configuration of voice settings and audio effects
through the PodcastConfig system.

Typical usage:
    config = PodcastConfig()
    convert_to_speech(
        config,
        conversation_script,
        'output.mp3',
        '.temp_audio/',
        'mp3'
    )
"""


import logging
from collections.abc import Callable, Iterable, Iterator
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import List

from elevenlabs import client as elevenlabs_client
from google.cloud import texttospeech
from google.cloud import texttospeech_v1beta1
from pydub import AudioSegment

from podcast_llm.config import PodcastConfig
from podcast_llm.tts_backends import (
    AudioChunk,
    render_elevenlabs_v4,
    render_gemini_3_8_flash_tts,
    require_api_key,
)
from podcast_llm.utils.rate_limits import (
    rate_limit_per_minute,
    retry_with_exponential_backoff
)


logger = logging.getLogger(__name__)



def clean_text_for_tts(lines: List) -> List:
    """
    Clean text lines for text-to-speech processing by removing special characters.

    Takes a list of dictionaries containing speaker and text information and removes
    characters that may interfere with text-to-speech synthesis, such as asterisks,
    underscores, and em dashes.

    Args:
        lines (List[dict]): List of dictionaries with structure:
            {
                'speaker': str,  # Speaker identifier
                'text': str      # Text to be cleaned
            }

    Returns:
        List[dict]: List of dictionaries with cleaned text and same structure as input
    """
    cleaned = []
    for l in lines:
        cleaned.append({'speaker': l['speaker'], 'text': l['text'].replace("*", "").replace("_", "").replace("—", "")})

    return cleaned



def merge_audio_files(audio_files: List, output_file: str, audio_format: str) -> None:
    """
    Merge multiple audio files into a single output file.

    Takes a list of audio files and combines them in the provided order into a single output
    file. Handles any audio format supported by pydub.

    Args:
        audio_files (list): List of paths to audio files to merge
        output_file (str): Path where merged audio file should be saved
        audio_format (str): Format of input/output audio files (e.g. 'mp3', 'wav')

    Returns:
        None

    Raises:
        Exception: If there are any errors during the merging process
    """
    logger.info("Merging audio files...")
    combined = AudioSegment.empty()
    for filename in audio_files:
        combined += AudioSegment.from_file(filename)
    combined.export(output_file, format=audio_format).close()


@retry_with_exponential_backoff(max_retries=10, base_delay=2.0)
@rate_limit_per_minute(max_requests_per_minute=20)
def process_line_google(config: PodcastConfig, text: str, speaker: str):
    """
    Process a single line of text using Google Text-to-Speech API.

    Takes a line of text and speaker identifier and generates synthesized speech using
    Google's TTS service. Uses different voices based on the speaker to create natural
    conversation flow.

    Args:
        text (str): The text content to convert to speech
        speaker (str): Speaker identifier to determine voice selection

    Returns:
        bytes: Raw audio data in bytes format containing the synthesized speech
    """
    client = texttospeech.TextToSpeechClient(client_options={'api_key': config.google_api_key})
    tts_settings = config.tts_settings['google']
    
    interviewer_voice = texttospeech.VoiceSelectionParams(
        language_code=tts_settings['language_code'],
        name=tts_settings['voice_mapping']['Interviewer'],
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    
    interviewee_voice = texttospeech.VoiceSelectionParams(
        language_code=tts_settings['language_code'],
        name=tts_settings['voice_mapping']['Interviewee'],
        ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )
    
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = interviewee_voice
    if speaker == 'Interviewer':
        voice = interviewer_voice
    
    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        effects_profile_id=tts_settings['effects_profile_id']
    )
    
    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    
    return response.audio_content


@retry_with_exponential_backoff(max_retries=10, base_delay=2.0)
@rate_limit_per_minute(max_requests_per_minute=20)
def process_line_elevenlabs(config: PodcastConfig, text: str, speaker: str):
    """
    Process a line of text into speech using ElevenLabs TTS service.

    Takes a line of text and speaker identifier and generates synthesized speech using
    ElevenLabs' TTS service. Uses different voices based on the speaker to create natural
    conversation flow.

    Args:
        config (PodcastConfig): Configuration object containing API keys and settings
        text (str): The text content to convert to speech
        speaker (str): Speaker identifier to determine voice selection

    Returns:
        bytes: Raw audio data in bytes format containing the synthesized speech
    """
    client = elevenlabs_client.ElevenLabs(api_key=config.elevenlabs_api_key)
    tts_settings = config.tts_settings['elevenlabs']

    audio = client.generate(
        text=text,
        voice=tts_settings['voice_mapping'][speaker],
        model=tts_settings['model']
    )

    # Convert audio iterator to bytes that can be written to disk
    audio_bytes = BytesIO()
    for chunk in audio:
        audio_bytes.write(chunk)
    
    return audio_bytes.getvalue()


def combine_consecutive_speaker_chunks(chunks: List[dict]) -> List[dict]:
    """
    Combine consecutive chunks from the same speaker into single chunks.
    
    Args:
        chunks (List[dict]): List of dictionaries containing conversation chunks with structure:
            {
                'speaker': str,  # Speaker identifier
                'text': str      # Text content
            }
    
    Returns:
        List[dict]: List of combined chunks where consecutive entries from the same speaker
                   are merged into single chunks
    """
    combined_chunks = []
    current_chunk = None

    for chunk in chunks:
        if current_chunk is None:
            current_chunk = chunk.copy()
        elif current_chunk['speaker'] == chunk['speaker']:
            current_chunk['text'] += ' ' + chunk['text']
        else:
            combined_chunks.append(current_chunk)
            current_chunk = chunk.copy()

    if current_chunk is not None:
        combined_chunks.append(current_chunk)

    return combined_chunks


@retry_with_exponential_backoff(max_retries=10, base_delay=2.0)
@rate_limit_per_minute(max_requests_per_minute=20)
def process_lines_google_multispeaker(config: PodcastConfig, chunks: List):
    """
    Process multiple lines of text into speech using Google's multi-speaker TTS service.

    Takes a chunk of conversation lines and generates synthesized speech using Google's
    multi-speaker TTS service. Handles several turns of conversation at once for more
    natural conversational flow.

    Args:
        config (PodcastConfig): Configuration object containing API keys and settings
        chunks (List): List of dictionaries containing conversation lines with structure:
            {
                'speaker': str,  # Speaker identifier
                'text': str      # Line content to convert to speech
            }

    Returns:
        bytes: Raw audio data in bytes format containing the synthesized speech
    """
    client = texttospeech_v1beta1.TextToSpeechClient(client_options={'api_key': config.google_api_key})
    tts_settings = config.tts_settings['google_multispeaker']

    # Combine consecutive lines from same speaker
    chunks = combine_consecutive_speaker_chunks(chunks)

    # Create multi-speaker markup
    multi_speaker_markup = texttospeech_v1beta1.MultiSpeakerMarkup()

    # Add each line as a conversation turn
    for line in chunks:
        turn = texttospeech_v1beta1.MultiSpeakerMarkup.Turn()
        turn.text = line['text']
        turn.speaker = tts_settings['voice_mapping'][line['speaker']]
        multi_speaker_markup.turns.append(turn)

    # Configure synthesis input with multi-speaker markup
    synthesis_input = texttospeech_v1beta1.SynthesisInput(
        multi_speaker_markup=multi_speaker_markup
    )

    # Configure voice parameters
    voice = texttospeech_v1beta1.VoiceSelectionParams(
        language_code=tts_settings['language_code'],
        name='en-US-Studio-MultiSpeaker'
    )

    # Configure audio output
    audio_config = texttospeech_v1beta1.AudioConfig(
        audio_encoding=texttospeech_v1beta1.AudioEncoding.MP3_64_KBPS,
        effects_profile_id=tts_settings['effects_profile_id']
    )

    # Generate speech
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    return response.audio_content


def _render_lines(config, turns, render_line) -> Iterator[AudioChunk]:
    for turn in turns:
        yield AudioChunk(render_line(config, turn['text'], turn['speaker']), 'mp3')


def _render_google(config, turns) -> Iterator[AudioChunk]:
    require_api_key(config.google_api_key, 'GOOGLE_API_KEY')
    return _render_lines(config, turns, process_line_google)


def _render_elevenlabs(config, turns) -> Iterator[AudioChunk]:
    require_api_key(config.elevenlabs_api_key, 'ELEVENLABS_API_KEY')
    return _render_lines(config, turns, process_line_elevenlabs)


def _render_google_multispeaker(config, turns) -> Iterator[AudioChunk]:
    require_api_key(config.google_api_key, 'GOOGLE_API_KEY')
    for start in range(0, len(turns), 4):
        yield AudioChunk(process_lines_google_multispeaker(config, turns[start:start + 4]), 'mp3')


TTSRenderer = Callable[[PodcastConfig, list[dict[str, str]]], Iterable[AudioChunk]]
_TTS_BACKENDS: dict[str, TTSRenderer] = {
    'google': _render_google,
    'elevenlabs': _render_elevenlabs,
    'google_multispeaker': _render_google_multispeaker,
    'elevenlabs_v4': render_elevenlabs_v4,
    'gemini_3_8_flash_tts': render_gemini_3_8_flash_tts,
}


def register_tts_backend(name: str, render: TTSRenderer) -> None:
    """Register a renderer for use by the tts_provider setting in the current process."""
    if name in _TTS_BACKENDS:
        raise ValueError(f"TTS backend {name!r} is already registered")
    _TTS_BACKENDS[name] = render


def convert_to_speech(
        config: PodcastConfig,
        conversation: list[dict[str, str]],
        output_file: str,
        temp_audio_dir: str,
        audio_format: str) -> None:
    """Render a conversation with the configured backend and assemble the audio."""
    render = _TTS_BACKENDS.get(config.tts_provider)
    if render is None:
        names = ', '.join(sorted(_TTS_BACKENDS))
        raise ValueError(f"Unknown tts_provider {config.tts_provider!r}. Available backends are {names}")

    Path(temp_audio_dir).mkdir(parents=True, exist_ok=True)
    output = Path(output_file)
    with TemporaryDirectory(prefix='tts-', dir=temp_audio_dir) as work_dir:
        audio_files = []
        for index, audio in enumerate(render(config, conversation)):
            path = Path(work_dir) / f"{index:03d}.{audio.extension}"
            path.write_bytes(audio.data)
            audio_files.append(str(path))
        if not audio_files:
            raise ValueError(f"TTS backend {config.tts_provider!r} did not produce audio")

        # Stage the export beside its destination so the final replacement is atomic.
        with NamedTemporaryFile(dir=output.parent, suffix=output.suffix, delete=False) as staged:
            export = Path(staged.name)
        try:
            merge_audio_files(audio_files, str(export), audio_format)
            export.replace(output)
        finally:
            export.unlink(missing_ok=True)


def generate_audio(config: PodcastConfig, final_script: list, output_file: str) -> str:
    """
    Generate an audio file from a podcast script using the configured backend.

    Args:
        final_script (list): List of dictionaries containing script lines with structure:
            {
                'speaker': str,  # Speaker identifier ('Interviewer' or 'Interviewee')
                'text': str      # Line content to convert to speech
            }
        output_file (str): Path where the final audio file should be saved

    Returns:
        str: Path to the generated audio file

    Raises:
        Exception: If any errors occur during TTS conversion or file operations
    """
    cleaned_script = clean_text_for_tts(final_script)

    convert_to_speech(config, cleaned_script, output_file, config.temp_audio_dir, config.output_format)

    return output_file
