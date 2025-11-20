"""
Utility classes and functions for Speech Splitter
Contains: Constants, data classes, logging configuration
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional


# =============================================================================
# CONSTANTS
# =============================================================================

MIN_PAUSE_DURATION_MS = 50
MIN_BUFFER_DURATION_MS = 20
MIN_TRIM_DURATION_MS = 200
MAX_CACHE_SIZE = 1000
CACHE_CLEANUP_RATIO = 0.25
DEFAULT_CUTOFF_SENSITIVITY = 0.35
CUTOFF_CHECK_OPTIONS = ['both', 'start', 'end', 'none']

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SegmentData:
    """Structured segment data"""
    start_ms: int
    end_ms: int
    text: str
    confidence: float
    duration: float
    words: List[Dict]
    prev_word_end_ms: Optional[int] = None
    next_word_start_ms: Optional[int] = None
    classification: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AudioClassification:
    """YAMNet classification results"""
    music: float
    speech: float
    music_speech_mixed: float
    silence: float
    
    def is_music(self, music_threshold: float, mixed_threshold: float) -> bool:
        return self.music > music_threshold or self.music_speech_mixed > mixed_threshold
    
    def is_silent(self, silence_threshold: float) -> bool:
        return self.silence > silence_threshold
    
    def is_speech(self, speech_threshold: float) -> bool:
        return self.speech > speech_threshold
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# =============================================================================
# LOGGING
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'
    
    def format(self, record):
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            color = self.COLORS.get(record.levelname, '')
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(output_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Setup logging with structured output"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"speech_splitter_{timestamp}.log"
    
    logger = logging.getLogger("SpeechSplitter")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = ColoredFormatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("Logging initialized", extra={'log_file': str(log_file)})
    return logger


def check_gpu(logger: logging.Logger) -> bool:
    """Check GPU availability with error handling"""
    try:
        import torch
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
                return True
            except Exception as e:
                logger.warning(f"GPU detected but properties unavailable: {e}")
                return True
        else:
            logger.warning("No GPU available, using CPU (slower processing)")
            return False
    except Exception as e:
        logger.error(f"GPU check failed: {e}")
        return False
