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
import os
import time
from functools import wraps
from io import BytesIO
from pathlib import Path

from elevenlabs import client as elevenlabs_client
from google.cloud import texttospeech
from pydub import AudioSegment

from podcast_llm.config import PodcastConfig


logger = logging.getLogger(__name__)


def rate_limit_per_minute(max_requests_per_minute: int):
    """
    Decorator that adds per-minute rate limiting to a function.
    
    Args:
        max_requests_per_minute (int): Maximum number of requests allowed per minute
        
    Returns:
        Callable: Decorated function with rate limiting
    """
    def decorator(func):
        last_request_time = 0
        min_interval = 60.0 / max_requests_per_minute  # Time between requests in seconds
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_request_time
            current_time = time.time()
            time_since_last = current_time - last_request_time
            
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                time.sleep(sleep_time)
                
            last_request_time = time.time()
            return func(*args, **kwargs)
            
        return wrapper
    return decorator


def retry_with_exponential_backoff(max_retries: int, base_delay: float = 1.0):
    """
    Decorator that retries a function with exponential backoff when exceptions occur.
    
    Args:
        max_retries (int): Maximum number of retry attempts
        base_delay (float): Initial delay between retries in seconds. Will be exponentially increased.
        
    Returns:
        Callable: Decorated function with retry logic
        
    Example:
        @retry_with_exponential_backoff(max_retries=3, base_delay=1.0)
        def flaky_function():
            # Function that may fail intermittently
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        raise last_exception
                    
                    logger.warning(
                        f'Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}. '
                        f'Retrying in {delay:.1f}s...'
                    )
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                    
            return None  # Should never reach here
        return wrapper
    return decorator



def clean_text_for_tts(lines):
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



def merge_audio_files(audio_files: list, output_file: str, audio_format: str) -> None:
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
    try:
        combined = AudioSegment.empty()
        
        for file in audio_files:
            combined += AudioSegment.from_file(file, format=audio_format)
        
        combined.export(output_file, format=audio_format)
    except Exception as e:
        raise



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


def convert_to_speech(
        config: PodcastConfig,
        conversation: str, 
        output_file: str, 
        temp_audio_dir: str, 
        audio_format: str) -> None:
    """
    Convert a conversation script to speech audio using Google Text-to-Speech API.

    Takes a conversation script consisting of speaker/text pairs and generates audio files
    for each line using Google's TTS service. The individual audio files are then merged
    into a single output file. Uses different voices for different speakers to create a
    natural conversational feel.

    Args:
        conversation (str): List of dictionaries containing conversation lines with structure:
            {
                'speaker': str,  # Speaker identifier ('Interviewer' or 'Interviewee')
                'text': str      # Line content to convert to speech
            }
        output_file (str): Path where the final merged audio file should be saved
        temp_audio_dir (str): Directory path for temporary audio file storage
        audio_format (str): Format of the audio files (e.g. 'mp3')

    Raises:
        Exception: If any errors occur during TTS conversion or file operations
    """
    try:
        logger.info(f"Generating audio files for {len(conversation)} lines...")
        audio_files = []
        counter = 0
        for line in conversation:
            logger.info(f"Generating audio for line {counter}...")

            if config.tts_provider == 'google':
                audio = process_line_google(config, line['text'], line['speaker'])
            elif config.tts_provider == 'elevenlabs':
                audio = process_line_elevenlabs(config, line['text'], line['speaker'])

            logger.info(f"Saving audio chunk {counter}...")
            file_name = os.path.join(temp_audio_dir, f"{counter:03d}.{audio_format}")
            with open(file_name, "wb") as out:
                out.write(audio)
            audio_files.append(file_name)
            
            counter += 1

        # Merge all audio files and save the result
        merge_audio_files(audio_files, output_file, audio_format)

        # Clean up individual audio files
        for file in audio_files:
            os.remove(file)

    except Exception as e:
        raise


def generate_audio(config: PodcastConfig, final_script: list, output_file: str) -> str:
    """
    Generate audio from a podcast script using text-to-speech.

    Takes a final script consisting of speaker/text pairs and generates a single audio file
    using Google's Text-to-Speech service. The script is first cleaned and processed to be
    TTS-friendly, then converted to speech with different voices for different speakers.

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

    temp_audio_dir = Path(config.temp_audio_dir)
    temp_audio_dir.mkdir(parents=True, exist_ok=True)
    convert_to_speech(config, cleaned_script, output_file, config.temp_audio_dir, config.output_format)

    return output_file