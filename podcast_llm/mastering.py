"""Level chapter audio and verify the loudness of the encoded episode."""

from dataclasses import dataclass
import json
import logging
from math import isfinite
from pathlib import Path
from shutil import which
from subprocess import CalledProcessError, run
from tempfile import NamedTemporaryFile
from typing import NamedTuple

from pydub import AudioSegment

from podcast_llm.text_to_speech import merge_audio_files


logger = logging.getLogger(__name__)
MIN_FILTER_PEAK = -9.0
LOUDNESS_TOLERANCE = 1.0


class AudioLevels(NamedTuple):
    integrated: float
    true_peak: float
    loudness_range: float
    threshold: float
    offset: float


@dataclass(frozen=True)
class MasteringTargets:
    loudness: float
    true_peak: float

    def __post_init__(self):
        if not isinstance(self.loudness, (float, int)) or not -70 <= self.loudness <= -5:
            raise ValueError('mastering_loudness_lufs must be between -70 and -5 LUFS')
        if not isinstance(self.true_peak, (float, int)) or not MIN_FILTER_PEAK <= self.true_peak <= 0:
            raise ValueError('mastering_true_peak_dbtp must be between -9 and 0 dBTP')


def require_ffmpeg() -> str:
    ffmpeg = which('ffmpeg')
    if ffmpeg is None:
        raise ValueError('Audio mastering requires FFmpeg on PATH.')
    return ffmpeg


def _run_ffmpeg(ffmpeg: str, *args: str | Path) -> str:
    try:
        result = run([ffmpeg, '-hide_banner', '-nostats', '-nostdin', '-y', *args],
                     capture_output=True, text=True, check=True)
    except CalledProcessError as error:
        detail = '\n'.join(error.stderr.strip().splitlines()[-8:])
        raise ValueError(f'FFmpeg could not process the episode audio.\n{detail}') from error
    return result.stderr


def measure_audio(path: Path, ffmpeg: str, loudness: float = -16, true_peak: float = -1) -> AudioLevels:
    """Measure the decoded audio using FFmpeg's BS.1770 loudness filter."""
    log = _run_ffmpeg(ffmpeg, '-i', path, '-af',
                      f'loudnorm=I={loudness}:TP={true_peak}:LRA=50:print_format=json', '-f', 'null', '-')
    try:
        stats = json.loads(log[log.rindex('{'):log.rindex('}') + 1])
        return AudioLevels(*(float(stats[key]) for key in (
            'input_i', 'input_tp', 'input_lra', 'input_thresh', 'target_offset',
        )))
    except (KeyError, ValueError) as error:
        raise ValueError('Could not read FFmpeg loudness measurements.') from error


def _balance_chapters(chapter_files: list[list[str]], work_dir: Path, ffmpeg: str,
                      targets: MasteringTargets) -> AudioSegment:
    measured_chapters = []
    for number, files in enumerate(chapter_files, 1):
        chapter = work_dir / f'level-{number}.wav'
        merge_audio_files(files, str(chapter), 'wav')
        measured_chapters.append((chapter, measure_audio(chapter, ffmpeg)))

    # Match chapters by constant gain with headroom before final loudness processing.
    peak = min(-2.0, targets.true_peak)
    possible_levels = [levels.integrated + peak - levels.true_peak for _, levels in measured_chapters
                       if isfinite(levels.integrated)]
    if not possible_levels:
        raise ValueError('Could not measure episode loudness. The audio may be silent or too short.')
    shared_level = min(targets.loudness, *possible_levels)
    episode = AudioSegment.empty()
    for number, (chapter, levels) in enumerate(measured_chapters, 1):
        signal = AudioSegment.from_wav(chapter).set_sample_width(4)
        if isfinite(levels.integrated):
            gain = shared_level - levels.integrated
            logger.info('Mastering chapter %d: %.2f LUFS, gain %+.2f dB', number, levels.integrated, gain)
            signal = signal.apply_gain(gain)
        episode += signal
    return episode


def master_episode(chapter_files: list[list[str]], output: Path, work_dir: Path, audio_format: str,
                   targets: MasteringTargets, ffmpeg: str) -> AudioLevels:
    """Master a full episode and replace the destination after verifying the encoded file."""
    raw = work_dir / 'balanced-episode.wav'
    episode = _balance_chapters(chapter_files, work_dir, ffmpeg, targets)
    episode.export(raw, format='wav').close()
    codec = (['-c:a', 'pcm_s24le'] if audio_format == 'wav' else
             ['-c:a', 'libmp3lame', '-b:a', '128k' if episode.channels == 1 else '192k'])
    with NamedTemporaryFile(dir=output.parent, suffix=output.suffix, delete=False) as file:
        staged = Path(file.name)
    try:
        # Leave room for peak overshoot introduced by the sample rate conversion and MP3 encoder.
        peak = max(MIN_FILTER_PEAK, targets.true_peak - 0.5)
        for attempt in range(2):
            measured = measure_audio(raw, ffmpeg, targets.loudness, peak)
            filter_args = (f'loudnorm=I={targets.loudness}:TP={peak}:LRA=50:linear=true'
                           f':measured_I={measured.integrated}:measured_TP={measured.true_peak}'
                           f':measured_LRA={measured.loudness_range}:measured_thresh={measured.threshold}'
                           f':offset={measured.offset}')
            _run_ffmpeg(ffmpeg, '-i', raw, '-af', filter_args, '-ar', '48000', *codec, staged)
            final = measure_audio(staged, ffmpeg)
            if final.true_peak <= targets.true_peak:
                break
            if attempt == 0:
                peak = max(MIN_FILTER_PEAK, peak - (final.true_peak - targets.true_peak) - 0.2)
        if not isfinite(final.integrated) or not isfinite(final.true_peak):
            raise ValueError('Could not measure loudness and peaks in the mastered export.')
        if final.true_peak > targets.true_peak:
            raise ValueError(f'Mastered export measured {final.true_peak:.2f} dBTP, above the '
                             f'{targets.true_peak:.2f} dBTP ceiling.')
        if abs(final.integrated - targets.loudness) > LOUDNESS_TOLERANCE:
            logger.warning('Mastered export is outside the loudness target of %.2f ± %.1f LUFS.',
                           targets.loudness, LOUDNESS_TOLERANCE)
        staged.replace(output)
    finally:
        staged.unlink(missing_ok=True)
    logger.info('Mastered %s: %.2f LUFS integrated, %.2f dBTP true peak',
                audio_format.upper(), final.integrated, final.true_peak)
    return final
