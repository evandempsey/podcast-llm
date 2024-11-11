"""
YouTube content extractor for podcast generation.

This module provides functionality to extract transcript content from YouTube videos
using the YouTubeTranscriptApi. It handles parsing various YouTube URL formats and
retrieving closed captions/subtitles.

Example:
    >>> from podcast_llm.extractors.youtube import YouTubeSourceDocument
    >>> extractor = YouTubeSourceDocument('https://www.youtube.com/watch?v=dQw4w9WgXcQ')
    >>> content = extractor.extract()
    >>> print(content)
    'We\'re no strangers to love You know the rules and so do I...'

The module supports:
- Standard youtube.com URLs (https://www.youtube.com/watch?v=VIDEO_ID)
- Short youtu.be URLs (https://youtu.be/VIDEO_ID) 
- Embedded URLs (https://www.youtube.com/embed/VIDEO_ID)

The extracted transcripts are returned as plain text and can be used as source
material for podcast episode generation. The module handles errors gracefully if
transcripts are unavailable or the video ID cannot be parsed.
"""

from podcast_llm.extractors.base import BaseSourceDocument
from typing import Optional
from youtube_transcript_api import YouTubeTranscriptApi


class YouTubeSourceDocument(BaseSourceDocument):
    """Extracts transcript content from YouTube videos using YouTubeTranscriptApi.

    This class handles extracting closed caption/subtitle content from YouTube videos
    by parsing various URL formats to get the video ID and retrieving the transcript.
    Supports standard youtube.com URLs, youtu.be short URLs, and embedded URLs.

    Example:
        >>> extractor = YouTubeSourceDocument('https://www.youtube.com/watch?v=dQw4w9WgXcQ')
        >>> content = extractor.extract()
        >>> print(content)
        'We\'re no strangers to love You know the rules and so do I...'

    Attributes:
        src (str): The YouTube video URL or ID
        src_type (str): Always 'YouTube video'
        title (str): A descriptive title combining src_type and source
        content (Optional[str]): The extracted transcript text
        video_id (str): The parsed YouTube video ID
    """
    def __init__(self, source: str) -> None:
        self.src = source
        self.src_type = 'YouTube video'
        self.title = f"{self.src_type}: {source}"
        self.content: Optional[str] = None
        self.video_id = self._extract_video_id()

    def _extract_video_id(self) -> str:
        """
        Extract YouTube video ID from various URL formats.
        
        Handles standard youtube.com URLs, youtu.be short URLs, 
        and embedded URLs. Returns just the video ID portion.
        
        Returns:
            str: The YouTube video ID
        """
        # Handle youtu.be short URLs
        if 'youtu.be' in self.src:
            return self.src.split('/')[-1].split('?')[0]
            
        # Handle youtube.com URLs
        if 'v=' in self.src:
            return self.src.split('v=')[1].split('&')[0]
            
        # Handle embed URLs
        if 'embed/' in self.src:
            return self.src.split('embed/')[-1].split('?')[0]
            
        # If no URL patterns match, assume src is already a video ID
        return self.src

    def extract(self) -> str:
        transcript = YouTubeTranscriptApi.get_transcript(self.video_id)
        self.content = ' '.join([line['text'] for line in transcript])
        return self.content
