"""
Audio file extraction module.

This module provides functionality for extracting text content from audio files
by transcribing them using OpenAI's Whisper model. It handles various audio formats
(mp3, wav, m4a, ogg) and manages splitting long audio files into chunks that stay
within API limits.

The module includes:
- AudioSourceDocument class for handling audio file extraction
- Audio file splitting based on silence detection
- Transcription using OpenAI Whisper API
- Temporary file management for processing

Example:
    >>> from podcast_llm.extractors.audio import AudioSourceDocument
    >>> extractor = AudioSourceDocument('podcast.mp3')
    >>> extractor.extract()
    >>> print(extractor.content)
    'Transcribed text from audio file...'

The extraction process:
1. Loads the audio file using pydub
2. Splits into ~10 minute segments based on silence detection
3. Saves segments to temporary files
4. Transcribes each segment using Whisper
5. Combines transcriptions into final content

The module handles errors gracefully and cleans up temporary files after processing.
"""


import logging
from podcast_llm.extractors.base import BaseSourceDocument
from typing import Optional
import os
import math
import pydub
from pydub import AudioSegment
import openai
import tempfile


logger = logging.getLogger(__name__)


class AudioSourceDocument(BaseSourceDocument):
    """
    A document extractor for audio files.

    This class handles extracting text content from audio files (mp3, wav, m4a, ogg) 
    by splitting them into manageable segments and transcribing them using OpenAI's
    Whisper model. The audio is first split into 10-minute chunks to stay within 
    API limits.

    Attributes:
        src (str): Path to the source audio file
        src_type (str): Type of source document ('Audio File')
        title (str): Title combining source type and filename
        content (Optional[str]): Extracted text content after transcription

    Example:
        >>> extractor = AudioSourceDocument('podcast.mp3')
        >>> extractor.extract()
        >>> print(extractor.content)
        'Transcribed text from audio file...'
    """
    def __init__(self, source: str) -> None:
        """
        Initialize the PDF extractor.

        Args:
            source: Path to the PDF file to extract text from
        """
        self.src = source
        self.src_type = 'PDF File'
        self.title = f"{self.src_type}: {source}"
        self.content: Optional[str] = None

    def _split_audio(self, filename: str, temp_dir: tempfile.TemporaryDirectory) -> list:
        """
        Split an audio file into 10-minute segments to stay within API limits.

        Takes an audio file and splits it into segments of 10 minutes or less to comply
        with OpenAI Whisper API limits. The segments are saved as separate MP3 files
        in a temporary directory.

        Args:
            filename (str): Path to the input audio file
            temp_dir (tempfile.TemporaryDirectory): Directory to store temporary segment files

        Returns:
            list: List of paths to the generated audio segment files

        Example:
            >>> with tempfile.TemporaryDirectory() as temp_dir:
            >>>     segments = self._split_audio('podcast.mp3', temp_dir)
            >>>     print(len(segments))
            3
        """
        logger.info(f"Splitting audio file {filename} into segments.")
        try:
            audio = AudioSegment.from_file(filename, format="mp3")
        except pydub.exceptions.CouldntDecodeError:
            audio = AudioSegment.from_file(filename, format="mp4")

        # Duration of each segment in milliseconds (10 minutes = 600,000 ms)
        segment_duration = 10 * 60 * 1000  # 10 minutes in milliseconds

        # Calculate the number of segments needed
        num_segments = math.ceil(len(audio) / segment_duration)

        segments = []
        for i in range(num_segments):
            # Calculate the start and end of each segment
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, len(audio))  # Make sure not to exceed audio length

            # Extract the segment
            segment = audio[start_time:end_time]

            # Export the segment to a new MP3 file
            segment_filename = os.path.join(temp_dir, f"segment_{i + 1:03d}.mp3")
            segment.export(segment_filename, format="mp3")
            segments.append(segment_filename)

            logger.info(f"Exported segment {i + 1} from {start_time / 1000} to {end_time / 1000} seconds.")

        return segments

    def extract(self) -> str:
        """
        Extract text content from an audio file using OpenAI's Whisper API.

        This method takes an audio file, splits it into 10-minute segments to comply with 
        API limits, and transcribes each segment using OpenAI's Whisper speech-to-text model.
        The transcribed segments are then combined into a single text document.

        Returns:
            str: The complete transcribed text from the audio file

        Raises:
            openai.OpenAIError: If there is an error calling the Whisper API
            IOError: If there is an error reading the audio file

        Example:
            >>> extractor = AudioSourceDocument('podcast.mp3')
            >>> text = extractor.extract()
            >>> print(text[:100])
            'Welcome to today's episode where we'll be discussing...'
        """
        logger.info(f"Loading audio from file: {self.src}")

        client = openai.OpenAI()

        # Process each chunk through Whisper API
        transcribed_texts = []
        with tempfile.TemporaryDirectory() as temp_dir:
            chunks = self._split_audio(self.src, temp_dir)

            for i, chunk in enumerate(chunks):
                with open(chunk, 'rb') as audio_file:
                    logger.info(f"Transcribing chunk {i+1}...")
                    transcript = client.audio.transcriptions.create(
                        file=audio_file,
                        model="whisper-1",
                        response_format="text"
                    )
                    logger.info(f"Got transcript:\n{transcript[:200]}...")
                    transcribed_texts.append(transcript)

        logger.info(f"Transcribing complete. Combining transcripts...")
        self.content = ' '.join(transcribed_texts)
        return self.content
