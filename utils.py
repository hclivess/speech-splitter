"""
Shared helpers for the speech-splitter pipeline: data classes, logging, app paths
"""
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

MIN_TRIM_DURATION_MS = 200
MAX_CACHE_SIZE = 2000


def app_dir() -> str:
    """Folder of the executable (frozen) or of the source tree"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.argv[0]))
    return os.path.dirname(os.path.abspath(__file__))


@dataclass
class SegmentData:
    start_ms: int
    end_ms: int
    text: str
    confidence: float
    duration: float
    words: List[Dict]
    prev_word_end_ms: Optional[int] = None
    next_word_start_ms: Optional[int] = None
    classification: Optional[Dict] = None
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AudioClassification:
    music: float
    speech: float
    music_speech_mixed: float
    silence: float
    noise: float

    @classmethod
    def speech_only(cls) -> "AudioClassification":
        """Neutral result used when the classifier is unavailable - never filters anything"""
        return cls(music=0.0, speech=1.0, music_speech_mixed=0.0, silence=0.0, noise=0.0)

    def is_music(self, music_threshold: float, mixed_threshold: float) -> bool:
        return self.music > music_threshold or self.music_speech_mixed > mixed_threshold

    def is_silent(self, silence_threshold: float) -> bool:
        return self.silence > silence_threshold

    def is_noise(self, noise_threshold: float) -> bool:
        return self.noise > noise_threshold

    def to_dict(self) -> Dict[str, float]:
        return {k: round(v, 4) for k, v in asdict(self).items()}


class ColoredFormatter(logging.Formatter):
    COLORS = {"DEBUG": "\033[36m", "INFO": "\033[32m", "WARNING": "\033[33m", "ERROR": "\033[31m",
              "CRITICAL": "\033[35m"}
    RESET = "\033[0m"

    def format(self, record):
        if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            color = self.COLORS.get(record.levelname, "")
            record = logging.makeLogRecord(record.__dict__)
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(output_dir: Path, log_level: str = "INFO", console: bool = True) -> logging.Logger:
    """Logger writing to <output>/logs/<timestamp>.log (and optionally the console)"""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"speech_splitter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("SpeechSplitter")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    # written as .part while the run is live (finish_logging renames it) - a .log is always a complete log
    fh = logging.FileHandler(log_file.with_suffix(".log.part"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    if console:
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        ch.setFormatter(ColoredFormatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(ch)

    logger.info(f"Logging to {log_file}")
    return logger


def temp_path(path) -> Path:
    """Temporary sibling name for an output: write here, then os.replace() onto `path` when complete"""
    path = Path(path)
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


@contextmanager
def atomic_output(path):
    """with atomic_output(dst) as tmp: <write tmp> -> tmp is renamed onto dst only if the block succeeds,
    removed otherwise (so a Stop or a crash never leaves a partial file under the final name)"""
    path = Path(path)
    tmp = temp_path(path)
    try:
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def atomic_write_text(path, text: str, encoding: str = "utf-8", newline=None) -> None:
    with atomic_output(path) as tmp:
        with open(tmp, "w", encoding=encoding, newline=newline) as f:
            f.write(text)


def finish_logging(logger: logging.Logger) -> None:
    """Close the file handler and rename <log>.part -> <log>: a finished log has the final name"""
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
            h.close()
            part = Path(h.baseFilename)
            if part.suffix == ".part" and part.exists():
                try:
                    os.replace(part, part.with_suffix(""))
                except OSError:
                    pass


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
