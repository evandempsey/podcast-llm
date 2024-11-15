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
from io import BytesIO
from pathlib import Path
from typing import List

from elevenlabs import client as elevenlabs_client
from google.cloud import texttospeech
from google.cloud import texttospeech_v1beta1
from pydub import AudioSegment

from podcast_llm.config import PodcastConfig
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
    try:
        combined = AudioSegment.empty()

        for filename in audio_files:
            audio = AudioSegment.from_file(filename)

            combined += audio

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
        audio_encoding=texttospeech.AudioEncoding.MP3_64_KBPS,
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


@retry_with_exponential_backoff(max_retries=10, base_delay=2.0)
@rate_limit_per_minute(max_requests_per_minute=20)
def process_lines_google_multispeaker(config: PodcastConfig, chunks: List):
    """
    Process multiple lines of text into speech using Google's multi-speaker TTS service.

    Takes a chunk of conversation lines and generates synthesized speech using Google's
    multi-speaker TTS service. Handles up to 6 turns of conversation at once for more
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
    tts_audio_formats = {
        'elevenlabs': 'mp3',
        'google': 'mp3',
        'google_multispeaker': 'mp3'
    }

    try:
        logger.info(f"Generating audio files for {len(conversation)} lines...")
        audio_files = []
        counter = 0

        if config.tts_provider == 'google_multispeaker':
            # We will not use a line by line strategy. 
            # Instead we will process in chunks of 6.
            # Process conversation in chunks of 6 lines
            for chunk_start in range(0, len(conversation), 4):
                chunk = conversation[chunk_start:chunk_start + 4]
                logger.info(f"Processing chunk {counter} with {len(chunk)} lines...")
                
                audio = process_lines_google_multispeaker(config, chunk)
                
                file_name = os.path.join(temp_audio_dir, f"{counter:03d}.{tts_audio_formats[config.tts_provider]}")
                with open(file_name, "wb") as out:
                    out.write(audio)
                audio_files.append(file_name)
                
                counter += 1
        else:
            for line in conversation:
                logger.info(f"Generating audio for line {counter}...")

                if config.tts_provider == 'google':
                    audio = process_line_google(config, line['text'], line['speaker'])
                elif config.tts_provider == 'elevenlabs':
                    audio = process_line_elevenlabs(config, line['text'], line['speaker'])

                logger.info(f"Saving audio chunk {counter}...")
                file_name = os.path.join(temp_audio_dir, f"{counter:03d}.{tts_audio_formats[config.tts_provider]}")
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
