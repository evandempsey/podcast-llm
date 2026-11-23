"""Render a saved Markdown transcript with per-chapter audio checkpoints."""

import argparse
from hashlib import sha256
import json
import logging
from pathlib import Path
import re
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import NamedTuple
from zipfile import BadZipFile, ZipFile

from podcast_llm.config import PodcastConfig, setup_logging
from podcast_llm.text_to_speech import clean_text_for_tts, export_audio_files, render_audio_chunks
from podcast_llm.tts_backends import AudioBatch, AudioChunk


logger = logging.getLogger(__name__)

SCRIPT_SECTION = re.compile(r'^##[ \t]+Script[ \t]*\n(.*?)(?=^#{1,2}[ \t]|\Z)', re.M | re.S)
CHAPTER_HEADING = re.compile(r'^###[ \t]+([^\n]+)[ \t]*$', re.M)
SPEAKER_LABEL = re.compile(r'^\*\*([^*\n]+)\*\*:[ \t]*', re.M)
SOURCE_MARKERS = re.compile(r'(?:[ \t]+\[S[0-9]+\])+$')
# Increment when synthesis defaults or the audio cache format change.
AUDIO_CACHE_VERSION = 1


class ScriptChapter(NamedTuple):
    title: str
    turns: list[dict[str, str]]


def read_script(path: Path) -> list[ScriptChapter]:
    """Read spoken turns from the Script section of an exported transcript."""
    match = SCRIPT_SECTION.search(path.read_text(encoding='utf-8-sig'))
    if not match:
        raise ValueError('The saved transcript must contain a ## Script section')

    lead, *sections = CHAPTER_HEADING.split(match[1])
    chapters = []
    if lead.strip():
        chapters.append(('Script', lead))
    chapters.extend(zip(sections[::2], sections[1::2]))
    if not chapters:
        raise ValueError('The saved transcript contains no dialogue')

    parsed = []
    for chapter, body in chapters:
        prefix, *parts = SPEAKER_LABEL.split(body)
        if prefix.strip() or not parts:
            raise ValueError(f'Chapter {chapter.strip()!r} needs **Speaker**: dialogue lines')
        turns = []
        for speaker, text in zip(parts[::2], parts[1::2]):
            spoken = SOURCE_MARKERS.sub('', text.strip()).strip()
            if not speaker.strip() or not spoken:
                raise ValueError(f'Chapter {chapter.strip()!r} has an empty speaker or turn')
            turns.append({'speaker': speaker.strip(), 'text': spoken})
        parsed.append(ScriptChapter(chapter.strip(), turns))
    return parsed


def _digest(data) -> str:
    return sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def _load_batch(path: Path) -> AudioBatch:
    with ZipFile(path) as file:
        request_id = json.loads(file.read('manifest.json'))['request_id']
        chunks = tuple(
            AudioChunk(file.read(name), name.rsplit('.', 1)[1])
            for name in file.namelist() if name != 'manifest.json'
        )
    if not chunks or any(not chunk.data for chunk in chunks):
        raise ValueError('Cached batch contains no audio')
    return AudioBatch(chunks, request_id)


def _save_batch(path: Path, batch: AudioBatch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as staged:
        temporary_path = Path(staged.name)
        try:
            with ZipFile(staged, 'w') as file:
                file.writestr('manifest.json', json.dumps({'request_id': batch.request_id}))
                for index, chunk in enumerate(batch.chunks):
                    file.writestr(f'{index:03d}.{chunk.extension}', chunk.data)
            staged.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


class ChapterCache:
    """Commit each completed provider request before moving on to the next."""

    def __init__(self, directory: Path, config: PodcastConfig, chapter: ScriptChapter):
        identity = [AUDIO_CACHE_VERSION, config.tts_provider,
                    config.tts_settings.get(config.tts_provider, {}), chapter.turns]
        self.directory = directory / _digest(identity)
        self.title = chapter.title
        self.batch_number = 0

    def __call__(self, turns, synthesize, previous_request_id) -> AudioBatch:
        self.batch_number += 1
        label = f'Chapter {self.title!r}, batch {self.batch_number}'
        key = _digest([turns, previous_request_id])
        path = self.directory / f'{self.batch_number:03d}-{key}.zip'
        try:
            batch = _load_batch(path)
        except FileNotFoundError:
            pass
        except (BadZipFile, KeyError, ValueError, IndexError):
            logger.warning('%s: ignoring unreadable cached audio', label)
        else:
            logger.info('%s: reused cached audio', label)
            return batch

        logger.info('%s: synthesizing', label)
        batch = synthesize()
        if not batch.chunks or any(not chunk.data for chunk in batch.chunks):
            raise ValueError(f'{label}: backend produced no audio')
        _save_batch(path, batch)
        logger.info('%s: saved audio checkpoint', label)
        return batch


def render_script(
    script: Path, config: PodcastConfig, output: Path | None = None, cache_dir: Path | None = None,
) -> Path:
    """Synthesize and export a saved script, reusing completed chapter batches."""
    output = output if output is not None else script.with_suffix(f'.{config.output_format}')
    if output.resolve() == script.resolve():
        raise ValueError('The audio output must be a different file from the saved script')
    audio_format = output.suffix.lstrip('.').lower()
    if audio_format not in {'mp3', 'wav'}:
        raise ValueError('Use an .mp3 or .wav audio output path')
    chapters = read_script(script)
    cache_dir = cache_dir if cache_dir is not None else Path(config.checkpoint_dir) / 'audio'
    Path(config.temp_audio_dir).mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix='tts-', dir=config.temp_audio_dir) as work_dir:
        audio_files = []
        for chapter in chapters:
            cleaned = ScriptChapter(chapter.title, clean_text_for_tts(chapter.turns))
            cache = ChapterCache(cache_dir, config, cleaned)
            start = len(audio_files)
            for audio in render_audio_chunks(config, cleaned.turns, run_batch=cache):
                path = Path(work_dir) / f'{len(audio_files):06d}.{audio.extension}'
                path.write_bytes(audio.data)
                audio_files.append(str(path))
            if len(audio_files) == start:
                raise ValueError(f'Chapter {chapter.title!r}: backend produced no audio')
        export_audio_files(audio_files, output, audio_format)
    logger.info('Wrote %s', output)
    return output


def render_cli(argv: list[str], default_config: str) -> None:
    parser = argparse.ArgumentParser(prog='podcast-llm render', description='Render a saved Markdown script')
    parser.add_argument('script', type=Path, help='Saved Markdown transcript')
    parser.add_argument('-o', '--output', '--audio-output', type=Path, help='Destination .mp3 or .wav file')
    parser.add_argument('--config', default=default_config, help='YAML configuration with TTS provider and voices')
    parser.add_argument('--cache-dir', type=Path, help='Audio checkpoint directory (default: checkpoint_dir/audio)')
    args = parser.parse_args(argv)
    setup_logging(logging.INFO)
    try:
        config = PodcastConfig.load(yaml_path=args.config, require_research_credentials=False)
        render_script(args.script, config, output=args.output, cache_dir=args.cache_dir)
    except KeyboardInterrupt:
        parser.exit(130, 'Render interrupted. Re-run the command to resume from saved audio.\n')
