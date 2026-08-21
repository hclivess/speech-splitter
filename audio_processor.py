"""
Audio loading, conversion and silence trimming (pydub + numpy; decoding via FFmpeg)
"""
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from pydub import AudioSegment

import childproc
from utils import MIN_TRIM_DURATION_MS, app_dir


def track_pydub_processes() -> None:
    """pydub spawns ffmpeg/ffprobe itself through subprocess.Popen. Redirect those two call sites (and only those)
    to childproc.popen so the decoders are tracked, hidden on Windows and die with the app. Idempotent."""
    import pydub.audio_segment
    import pydub.utils

    class _TrackedSubprocess:
        """Looks like the subprocess module to pydub.audio_segment, but Popen is childproc.popen"""
        Popen = staticmethod(childproc.popen)

        def __getattr__(self, name):
            return getattr(childproc.subprocess, name)

    pydub.utils.Popen = childproc.popen
    pydub.audio_segment.subprocess = _TrackedSubprocess()


def find_ffmpeg() -> Optional[str]:
    """ffmpeg next to the app, in PATH, or from an imageio-ffmpeg / static-ffmpeg install"""
    exe = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    local = os.path.join(app_dir(), exe)
    if os.path.isfile(local):
        return local
    found = shutil.which("ffmpeg")
    if found:
        return found
    return None


def configure_ffmpeg(logger: Optional[logging.Logger] = None) -> bool:
    """Point pydub at the ffmpeg we found (it only searches PATH by default)"""
    track_pydub_processes()
    path = find_ffmpeg()
    if path:
        AudioSegment.converter = path
        probe = os.path.join(os.path.dirname(path), "ffprobe.exe" if sys.platform == "win32" else "ffprobe")
        if os.path.isfile(probe):
            AudioSegment.ffprobe = probe
        return True
    if logger:
        logger.warning("ffmpeg not found - only plain WAV input will work. Install FFmpeg or put ffmpeg next to the app.")
    return False


class AudioProcessor:
    @staticmethod
    def load_and_prepare(audio_path: Path, logger: logging.Logger) -> Tuple[AudioSegment, int]:
        """Load any audio/video file, return mono 16-bit audio at the original sample rate"""
        audio = AudioSegment.from_file(str(audio_path))
        original_sr = audio.frame_rate
        if audio.channels > 1:
            audio = audio.set_channels(1)
        if audio.sample_width != 2:
            audio = audio.set_sample_width(2)
        logger.info(f"Audio loaded: {len(audio) / 1000:.1f}s, {audio.frame_rate} Hz")
        return audio, original_sr

    @staticmethod
    def to_float32(audio: AudioSegment, target_sr: Optional[int] = None) -> np.ndarray:
        """Mono float32 samples in [-1, 1], optionally resampled"""
        if target_sr and audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)
        if audio.channels > 1:
            audio = audio.set_channels(1)
        samples = np.array(audio.get_array_of_samples())
        scale = float(2 ** (8 * audio.sample_width - 1))
        return (samples.astype(np.float32) / scale).copy()

    @staticmethod
    def rms_db_windows(audio_chunk: AudioSegment, window_ms: int) -> np.ndarray:
        """RMS level in dBFS for consecutive windows"""
        samples = np.array(audio_chunk.get_array_of_samples(), dtype=np.float64)
        max_amp = float(2 ** (8 * audio_chunk.sample_width - 1))
        win = max(1, int(audio_chunk.frame_rate * window_ms / 1000))
        n = len(samples) // win
        if n == 0:
            return np.array([])
        frames = samples[: n * win].reshape(-1, win)
        rms = np.sqrt(np.mean(frames ** 2, axis=1)) / max_amp
        with np.errstate(divide="ignore"):
            return 20 * np.log10(rms + 1e-10)

    @staticmethod
    def trim_silence(audio_chunk: AudioSegment, threshold_db: float, chunk_ms: int, pad_ms: int) -> AudioSegment:
        """Cut leading/trailing silence below threshold_db, keeping pad_ms of it"""
        if len(audio_chunk) < MIN_TRIM_DURATION_MS:
            return audio_chunk
        db = AudioProcessor.rms_db_windows(audio_chunk, chunk_ms)
        if db.size == 0:
            return audio_chunk
        loud = np.where(db > threshold_db)[0]
        if loud.size == 0:
            return audio_chunk
        pad = int(pad_ms / chunk_ms)
        start = max(0, loud[0] - pad) * chunk_ms
        end = min(db.size, loud[-1] + 1 + pad) * chunk_ms
        return audio_chunk[start:end]
