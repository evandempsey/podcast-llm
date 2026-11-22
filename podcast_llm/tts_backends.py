"""Dialogue synthesis adapters for ElevenLabs and the Gemini API."""

import base64
import textwrap
import wave
from collections.abc import Iterator
from io import BytesIO
from typing import NamedTuple

import requests

from podcast_llm.config import PodcastConfig


class AudioChunk(NamedTuple):
    data: bytes
    extension: str


# Keep exchanges together while bounding the size of each synthesis request.
DIALOGUE_CHUNK_SIZE = 2000


def require_api_key(key: str, env_var: str) -> str:
    if not key:
        raise ValueError(f"Set {env_var} to use the selected TTS backend")
    return key


def _dialogue_chunks(turns: list[dict[str, str]]) -> Iterator[list[dict[str, str]]]:
    chunk = []
    size = 0
    for turn in turns:
        text = turn['text'].strip()
        parts = [text] if len(text) <= DIALOGUE_CHUNK_SIZE else textwrap.wrap(
            text, width=DIALOGUE_CHUNK_SIZE, replace_whitespace=False, break_on_hyphens=False
        )
        for part in parts:
            if not part:
                continue
            if size + len(part) > DIALOGUE_CHUNK_SIZE:
                yield chunk
                chunk, size = [], 0
            chunk.append({'speaker': turn['speaker'], 'text': part})
            size += len(part)
    if chunk:
        yield chunk


def _voice_mapping(settings, turns, defaults):
    voices = defaults | settings.get('voice_mapping', {})
    missing = {turn['speaker'] for turn in turns if not voices.get(turn['speaker'])}
    if missing:
        raise ValueError(f"Missing TTS voices in voice_mapping for: {', '.join(sorted(missing))}")
    return voices


def render_elevenlabs_v4(config: PodcastConfig, turns: list[dict[str, str]]) -> Iterator[AudioChunk]:
    """Synthesize consecutive exchanges with Eleven v4 Text to Dialogue."""
    key = require_api_key(config.elevenlabs_api_key, 'ELEVENLABS_API_KEY')
    settings = config.tts_settings.get('elevenlabs_v4', {})
    voices = _voice_mapping(settings, turns, {
        'Interviewer': 'iP95p4xoKVk53GoZ742B',  # Chris
        'Interviewee': 'IKne3meq5aSn9XLyUdCD',  # Charlie
    })

    previous_request_id = None
    with requests.Session() as session:
        for chunk in _dialogue_chunks(turns):
            payload = {
                'model_id': 'eleven_v4',
                'inputs': [
                    {'voice_id': voices[turn['speaker']], 'text': turn['text']}
                    for turn in chunk
                ],
            }
            if settings.get('language_code'):
                payload['language_code'] = settings['language_code']
            if previous_request_id:
                payload['previous_request_ids'] = [previous_request_id]
            response = session.post(
                'https://api.elevenlabs.io/v1/text-to-dialogue',
                headers={'xi-api-key': key},
                params={'output_format': 'pcm_24000'},
                json=payload,
                timeout=(10, 120),
            )
            response.raise_for_status()
            pcm = response.content
            if not pcm or len(pcm) % 2 or not response.headers.get('content-type', '').startswith('audio/'):
                raise ValueError('ElevenLabs did not return 16-bit PCM audio')
            previous_request_id = response.headers.get('request-id')

            audio = BytesIO()
            with wave.open(audio, 'wb') as output:
                output.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
                output.writeframes(pcm)
            yield AudioChunk(audio.getvalue(), 'wav')


def render_gemini_3_8_flash_tts(config: PodcastConfig, turns: list[dict[str, str]]) -> Iterator[AudioChunk]:
    """Synthesize Gemini 3.8 Flash TTS dialogue through the Gemini Developer API."""
    key = require_api_key(config.google_api_key, 'GOOGLE_API_KEY')
    settings = config.tts_settings.get('gemini_3_8_flash_tts', {})
    voices = _voice_mapping(settings, turns, {'Interviewer': 'Aoede', 'Interviewee': 'Charon'})
    if len({turn['speaker'] for turn in turns if turn['text'].strip()}) > 2:
        raise ValueError('Gemini 3.8 Flash TTS supports up to two speakers per conversation')

    with requests.Session() as session:
        for chunk in _dialogue_chunks(turns):
            speakers = list(dict.fromkeys(turn['speaker'] for turn in chunk))
            if len(speakers) == 1:
                speech_config = [{'voice': voices[speakers[0]]}]
                content = [{'type': 'text', 'text': '\n'.join(turn['text'] for turn in chunk)}]
            else:
                speech_config = {'speakers': [
                    {'speaker': speaker, 'voice': voices[speaker]} for speaker in speakers
                ]}
                content = [{
                    'type': 'text',
                    'text': turn['text'],
                    'annotations': [{'type': 'speech_metadata', 'speaker': turn['speaker']}],
                } for turn in chunk]

            response = session.post(
                'https://generativelanguage.googleapis.com/v1beta/interactions',
                headers={'x-goog-api-key': key},
                json={
                    'model': 'gemini-3.8-flash-tts',
                    'store': False,
                    'input': [{'type': 'user_input', 'content': content}],
                    'response_format': {'type': 'audio'},
                    'generation_config': {'speech_config': speech_config},
                },
                timeout=(10, 120),
            )
            response.raise_for_status()
            result = response.json()
            if result.get('status') != 'completed':
                raise ValueError(f"Gemini speech generation ended with status {result.get('status')!r}")
            audio_parts = [
                part['data']
                for step in result.get('steps', [])
                for part in step.get('content', [])
                if part.get('type') == 'audio' and part.get('data')
            ]
            if not audio_parts:
                raise ValueError('Gemini did not return audio')
            for part in audio_parts:
                audio = base64.b64decode(part, validate=True)
                if not audio.startswith(b'RIFF'):
                    raise ValueError('Gemini did not return WAV audio')
                yield AudioChunk(audio, 'wav')
