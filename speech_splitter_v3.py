import os
import json
import argparse
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import warnings
import sys
warnings.filterwarnings('ignore')

from pydub import AudioSegment
import numpy as np
from tqdm import tqdm
import torch

# =============================================================================
# CONSTANTS (Issue #17: No more magic numbers)
# =============================================================================

MIN_PAUSE_DURATION_MS = 50
MIN_BUFFER_DURATION_MS = 20
MIN_TRIM_DURATION_MS = 200
MAX_CACHE_SIZE = 1000
CACHE_CLEANUP_RATIO = 0.25

# =============================================================================
# DATA CLASSES (Issue #12: Type safety with dataclasses)
# =============================================================================

@dataclass
class SegmentData:
    """Structured segment data instead of raw dicts"""
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
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
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
    """Check GPU availability"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
        return True
    else:
        logger.warning("No GPU available, using CPU (slower processing)")
        return False


# =============================================================================
# YAMNET FILTER (Issues #1, #2, #3: Optimized with caching)
# =============================================================================

class YAMNetFilter:
    """
    Optimized YAMNet audio classifier with:
    - Classification caching (Issue #1)
    - Pre-computed class indices (Issue #2)
    - Single classification per segment (Issue #1)
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', logger=None):
        self.model = None
        self.class_names = None
        self.device = device
        self.logger = logger or logging.getLogger("SpeechSplitter.YAMNetFilter")
        
        # OPTIMIZATION: Classification cache (Issue #1)
        self._classification_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # OPTIMIZATION: Pre-computed class indices (Issue #2)
        self.music_indices = None
        self.speech_indices = None
        self.mixed_indices = None
        self.silence_indices = None
        
        self._load_model()
    
    def _load_model(self):
        """Load YAMNet and pre-compute class indices"""
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            
            # GPU config
            gpus = tf.config.list_physical_devices('GPU')
            if gpus and self.device == 'cuda':
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    self.logger.debug("TensorFlow GPU memory growth enabled")
                except RuntimeError as e:
                    self.logger.warning(f"GPU config error: {e}")
            
            tf.get_logger().setLevel('ERROR')
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            
            self.logger.info("Loading YAMNet model...")
            
            # Clear corrupted cache and reload
            import shutil
            import tempfile
            
            # Get TFHub cache directory
            tfhub_cache = os.path.join(tempfile.gettempdir(), 'tfhub_modules')
            corrupt_model_hash = '9616fd04ec2360621642ef9455b84f4b668e219e'
            corrupt_path = os.path.join(tfhub_cache, corrupt_model_hash)
            
            # Remove corrupted cache if it exists
            if os.path.exists(corrupt_path):
                self.logger.info(f"Removing corrupted cache: {corrupt_path}")
                try:
                    shutil.rmtree(corrupt_path)
                except Exception as e:
                    self.logger.warning(f"Failed to remove cache: {e}")
            
            # Load model with retry
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    self.model = hub.load('https://tfhub.dev/google/yamnet/1')
                    break
                except ValueError as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Attempt {attempt + 1} failed, retrying after clearing cache...")
                        # Clear entire tfhub cache on retry
                        if os.path.exists(tfhub_cache):
                            try:
                                shutil.rmtree(tfhub_cache)
                                os.makedirs(tfhub_cache, exist_ok=True)
                            except Exception as clear_error:
                                self.logger.warning(f"Failed to clear cache: {clear_error}")
                    else:
                        raise
            
            class_map_path = self.model.class_map_path().numpy()
            self.class_names = self._load_class_names(class_map_path)
            
            # OPTIMIZATION: Pre-compute class indices once (Issue #2)
            self._precompute_class_indices()
            
            self.logger.info("YAMNet model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load YAMNet: {e}", exc_info=True)
            self.logger.warning("Audio filtering disabled")
            self.model = None
    
    def _load_class_names(self, class_map_csv):
        """Load YAMNet class names"""
        import csv
        class_names = []
        with open(class_map_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_names.append(row['display_name'])
        return class_names
    
    def _precompute_class_indices(self):
        """OPTIMIZATION: Calculate class indices once instead of per-classification (Issue #2)"""
        self.music_indices = [
            i for i, name in enumerate(self.class_names) 
            if 'music' in name.lower() and 'speech' not in name.lower()
        ]
        self.speech_indices = [
            i for i, name in enumerate(self.class_names) 
            if 'speech' in name.lower() and 'music' not in name.lower()
        ]
        self.mixed_indices = [
            i for i, name in enumerate(self.class_names) 
            if 'music' in name.lower() and 'speech' in name.lower()
        ]
        self.silence_indices = [
            i for i, name in enumerate(self.class_names) 
            if 'silence' in name.lower() or 'quiet' in name.lower()
        ]
        
        self.logger.debug(f"Class indices: music={len(self.music_indices)}, "
                         f"speech={len(self.speech_indices)}, "
                         f"mixed={len(self.mixed_indices)}, "
                         f"silence={len(self.silence_indices)}")
    
    def _get_cache_key(self, audio_segment: AudioSegment) -> str:
        """Generate cache key from audio segment properties"""
        # Use audio properties + small sample for cache key
        key_data = f"{len(audio_segment)}_{audio_segment.frame_rate}_{audio_segment.channels}"
        samples = audio_segment.get_array_of_samples()[:100]
        key_data += f"_{hash(bytes(samples))}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def classify_audio(self, audio_segment: AudioSegment) -> AudioClassification:
        """
        Classify audio with caching (Issue #1)
        Returns AudioClassification dataclass
        """
        if self.model is None:
            return AudioClassification(music=0.0, speech=1.0, music_speech_mixed=0.0, silence=0.0)
        
        # OPTIMIZATION: Check cache first (Issue #1)
        cache_key = self._get_cache_key(audio_segment)
        if cache_key in self._classification_cache:
            self._cache_hits += 1
            return self._classification_cache[cache_key]
        
        self._cache_misses += 1
        
        try:
            import tensorflow as tf
            
            # OPTIMIZATION: Audio already mono from preprocessing (Issue #3)
            samples = np.array(audio_segment.get_array_of_samples())
            
            # Normalize samples
            if audio_segment.sample_width == 2:
                samples = samples.astype(np.float32) / 32768.0
            elif audio_segment.sample_width == 4:
                samples = samples.astype(np.float32) / 2147483648.0
            
            scores, _, _ = self.model(samples)
            mean_scores = np.mean(scores.numpy(), axis=0)
            
            # OPTIMIZATION: Use pre-computed indices (Issue #2)
            music_prob = float(np.max([mean_scores[i] for i in self.music_indices]) if self.music_indices else 0.0)
            speech_prob = float(np.max([mean_scores[i] for i in self.speech_indices]) if self.speech_indices else 0.0)
            mixed_prob = float(np.max([mean_scores[i] for i in self.mixed_indices]) if self.mixed_indices else 0.0)
            silence_prob = float(np.max([mean_scores[i] for i in self.silence_indices]) if self.silence_indices else 0.0)
            
            result = AudioClassification(
                music=music_prob,
                speech=speech_prob,
                music_speech_mixed=mixed_prob,
                silence=silence_prob
            )
            
            # OPTIMIZATION: Cache result (Issue #1)
            self._classification_cache[cache_key] = result
            
            # Prevent cache from growing too large
            if len(self._classification_cache) > MAX_CACHE_SIZE:
                remove_count = int(len(self._classification_cache) * CACHE_CLEANUP_RATIO)
                for key in list(self._classification_cache.keys())[:remove_count]:
                    del self._classification_cache[key]
            
            return result
            
        except Exception as e:
            self.logger.debug(f"YAMNet classification error: {e}")
            return AudioClassification(music=0.0, speech=1.0, music_speech_mixed=0.0, silence=0.0)
    
    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache performance statistics"""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._classification_cache)
        }
    
    def clear_cache(self):
        """Clear classification cache"""
        self._classification_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


# =============================================================================
# AUDIO PROCESSOR (Issue #11: Separate concerns)
# =============================================================================

class AudioProcessor:
    """Handle audio loading, conversion, and trimming"""
    
    @staticmethod
    def load_and_prepare(audio_path: Path, logger: logging.Logger) -> Tuple[AudioSegment, int]:
        """
        Load audio file and convert to mono
        OPTIMIZATION: Single conversion (Issue #3)
        """
        try:
            audio = AudioSegment.from_file(audio_path)
            original_sr = audio.frame_rate
            
            # Convert to mono once (Issue #3)
            if audio.channels > 1:
                audio = audio.set_channels(1)
                logger.debug("Converted to mono")
            
            logger.info(f"Audio loaded: {len(audio)/1000:.1f}s, {audio.frame_rate}Hz")
            return audio, original_sr
            
        except Exception as e:
            logger.error(f"Audio load failed: {e}")
            raise
    
    @staticmethod
    def trim_silence_vectorized(audio_chunk: AudioSegment, trim_threshold_db: float, 
                               trim_chunk_ms: int, trim_pad_ms: int, 
                               trim_end: bool = True) -> AudioSegment:
        """
        OPTIMIZATION: Vectorized silence trimming (Issue #10)
        Much faster than loop-based approach
        """
        if len(audio_chunk) < MIN_TRIM_DURATION_MS:
            return audio_chunk
        
        samples = np.array(audio_chunk.get_array_of_samples(), dtype=np.float32)
        frame_rate = audio_chunk.frame_rate
        
        # Calculate chunk size in samples
        chunk_samples = int(trim_chunk_ms * frame_rate / 1000)
        
        # Pad array to make it divisible by chunk size
        padding = (chunk_samples - len(samples) % chunk_samples) % chunk_samples
        if padding > 0:
            samples = np.pad(samples, (0, padding), mode='constant')
        
        # Reshape into chunks and calculate RMS for each
        chunks = samples.reshape(-1, chunk_samples)
        rms = np.sqrt(np.mean(chunks ** 2, axis=1))
        
        # Convert to dBFS (for 16-bit audio)
        with np.errstate(divide='ignore'):
            db = 20 * np.log10(rms / 32768.0)
        db[np.isinf(db)] = -100  # Replace -inf with very low value
        
        # Find non-silent chunks
        non_silent = np.where(db > trim_threshold_db)[0]
        
        if len(non_silent) == 0:
            return audio_chunk
        
        # Calculate trim points with padding
        pad_chunks = int(trim_pad_ms / trim_chunk_ms)
        start_chunk = max(0, non_silent[0] - pad_chunks)
        end_chunk = min(len(chunks), non_silent[-1] + pad_chunks + 1) if trim_end else len(chunks)
        
        # Convert chunk indices to sample indices
        start_sample = start_chunk * chunk_samples
        end_sample = min(end_chunk * chunk_samples, len(samples) - padding)
        
        return audio_chunk[int(start_sample * 1000 / frame_rate):int(end_sample * 1000 / frame_rate)]


# =============================================================================
# TRANSCRIPTION MANAGER (Issue #11: Separate concerns)
# =============================================================================

class TranscriptionManager:
    """Handle WhisperX transcription with forced alignment"""
    
    def __init__(self, model_name: str, language: str, device: str, logger: logging.Logger):
        self.model_name = model_name
        self.language = language
        self.device = device
        self.logger = logger
        self.use_whisperx = True
        
        try:
            import whisperx
            self.whisperx = whisperx
            self.logger.info(f"Loading WhisperX model: {model_name} on {device.upper()}")
            self.model = whisperx.load_model(model_name, device=device, compute_type="float16" if device == "cuda" else "int8")
            self.logger.info("WhisperX model loaded")
        except ImportError:
            self.logger.warning("WhisperX not installed, falling back to standard Whisper")
            self.use_whisperx = False
            self.logger.info(f"Loading Whisper model: {model_name} on {device.upper()}")
            self.model = whisper.load_model(model_name, device=device)
            self.logger.info("Whisper model loaded")
    
    def transcribe(self, audio_path: str, original_audio_path: str = None) -> Dict:
        """Transcribe audio with WhisperX forced alignment"""
        cache_base_path = Path(original_audio_path) if original_audio_path else Path(audio_path)
        cache_file = cache_base_path.parent / f"{cache_base_path.stem}.transcription.json"
        
        if cache_file.exists():
            try:
                self.logger.info(f"Loading cached transcription: {cache_file.name}")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                if 'words' in cached_data and 'text' in cached_data:
                    self.logger.debug(f"Cache loaded: {len(cached_data['words'])} words")
                    return cached_data
                else:
                    self.logger.warning("Invalid cache format, re-transcribing")
            except Exception as e:
                self.logger.warning(f"Cache load failed: {e}")
        
        if self.use_whisperx:
            return self._transcribe_whisperx(audio_path, cache_file)
        else:
            return self._transcribe_whisper(audio_path, cache_file)
    
    def _transcribe_whisperx(self, audio_path: str, cache_file: Path) -> Dict:
        """WhisperX with forced alignment for better timestamps"""
        self.logger.info("Transcribing with WhisperX (forced alignment)")
        
        audio = self.whisperx.load_audio(audio_path)
        result = self.model.transcribe(audio, batch_size=16, language=self.language)
        
        self.logger.info("Loading alignment model...")
        model_a, metadata = self.whisperx.load_align_model(language_code=self.language, device=self.device)
        result = self.whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
        
        all_words = []
        for segment in result.get('segments', []):
            if 'words' in segment:
                for word in segment['words']:
                    all_words.append({
                        'word': word.get('word', ''),
                        'start': word.get('start', 0),
                        'end': word.get('end', 0),
                        'probability': word.get('score', 1.0)
                    })
        
        self.logger.info(f"WhisperX complete: {len(all_words)} words (aligned)")
        
        transcription_data = {
            'text': ' '.join([w['word'] for w in all_words]),
            'words': all_words,
            'segments': result.get('segments', []),
            'language': self.language,
            'model': f"{self.model_name}-whisperx",
            'aligned': True
        }
        
        try:
            self.logger.debug(f"Saving transcription cache: {cache_file.name}")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(transcription_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"Cache save failed: {e}")
        
        return transcription_data
    
    def _transcribe_whisper(self, audio_path: str, cache_file: Path) -> Dict:
        """Fallback to standard Whisper"""
        self.logger.info("Transcribing with standard Whisper")
        result = self.model.transcribe(
            audio_path,
            language=self.language,
            word_timestamps=True,
            verbose=False
        )
        
        all_words = []
        for segment in result.get('segments', []):
            if 'words' in segment:
                all_words.extend(segment['words'])
        
        self.logger.info(f"Whisper complete: {len(all_words)} words")
        
        transcription_data = {
            'text': result['text'],
            'words': all_words,
            'segments': result.get('segments', []),
            'language': self.language,
            'model': self.model_name,
            'aligned': False
        }
        
        try:
            self.logger.debug(f"Saving transcription cache: {cache_file.name}")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(transcription_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"Cache save failed: {e}")
        
        return transcription_data


# =============================================================================
# SEGMENTATION ENGINE (Issue #11: Separate concerns)
# =============================================================================

class SegmentationEngine:
    """Handle pause-based segmentation"""
    
    def __init__(self, params: Dict, yamnet_filter: Optional[YAMNetFilter], logger: logging.Logger):
        self.params = params
        self.yamnet_filter = yamnet_filter
        self.logger = logger
    
    def segment_by_pauses(self, words: List[Dict], audio: AudioSegment) -> List[SegmentData]:
        """
        Segment audio by pauses with hybrid detection
        FIXED: Proper word buffer handling (Issue #6)
        """
        if not words:
            self.logger.warning("No words found for segmentation")
            return []
        
        mode = "hybrid" if self.params['hybrid_pause_detection'] and self.yamnet_filter else "timestamp"
        self.logger.info(f"Segmenting by pauses (mode: {mode})")
        
        segments = []
        current_words = []
        stats = {'potential': 0, 'validated': 0, 'rejected': 0}
        
        for i, word in enumerate(words):
            if 'start' not in word or 'end' not in word:
                self.logger.debug(f"Skipping word: missing timestamps")
                continue
            
            current_words.append(word)
            
            should_split = False
            if i < len(words) - 1:
                next_word = words[i + 1]
                pause_duration = next_word['start'] - word['end']
                
                if pause_duration >= self.params['pause_threshold']:
                    stats['potential'] += 1
                    
                    if self.params['hybrid_pause_detection'] and self.yamnet_filter:
                        # Validate pause with YAMNet
                        gap_start_ms = int(word['end'] * 1000)
                        gap_end_ms = int(next_word['start'] * 1000)
                        
                        if gap_end_ms - gap_start_ms >= MIN_PAUSE_DURATION_MS:
                            gap_audio = audio[gap_start_ms:gap_end_ms]
                            classification = self.yamnet_filter.classify_audio(gap_audio)
                            
                            # True pause if silent OR no speech detected
                            is_true_pause = (
                                classification.is_silent(self.params['noise_silence_threshold']) or
                                not classification.is_speech(self.params['pause_speech_threshold'])
                            )
                            
                            if is_true_pause:
                                should_split = True
                                stats['validated'] += 1
                            else:
                                stats['rejected'] += 1
                        else:
                            should_split = True  # Very short gap, assume it's a pause
                    else:
                        should_split = True
            
            # Last word always triggers split
            if i == len(words) - 1:
                should_split = True
            
            # Create segment when split is triggered
            if should_split and current_words:
                first_word = current_words[0]
                last_word = current_words[-1]
                
                if 'start' in first_word and 'end' in last_word:
                    text = ''.join([w.get('word', '') for w in current_words]).strip()
                    duration = last_word['end'] - first_word['start']
                    
                    # FIXED: Correct boundary calculation (Issue #6)
                    # Previous word is the one BEFORE current segment starts
                    segment_start_idx = i - len(current_words) + 1
                    prev_word_end_ms = int(words[segment_start_idx - 1]['end'] * 1000) if segment_start_idx > 0 else None
                    next_word_start_ms = int(words[i + 1]['start'] * 1000) if i + 1 < len(words) else None
                    
                    if self._validate_segment_text(text) and duration >= self.params['min_segment_length']:
                        segment = SegmentData(
                            start_ms=int(first_word['start'] * 1000),
                            end_ms=int(last_word['end'] * 1000),
                            duration=duration,
                            words=current_words.copy(),
                            text=text,
                            confidence=sum(w.get('probability', 1.0) for w in current_words) / len(current_words),
                            prev_word_end_ms=prev_word_end_ms,
                            next_word_start_ms=next_word_start_ms
                        )
                        segments.append(segment)
                
                # CRITICAL FIX: Clear words only after processing (Issue #1 from original bug)
                current_words = []
        
        self.logger.info(f"Segmentation complete: {len(segments)} segments", extra={
            'potential_pauses': stats['potential'],
            'validated_pauses': stats['validated'],
            'rejected_pauses': stats['rejected']
        })
        
        return segments
    
    def _validate_segment_text(self, text: str) -> bool:
        """Validate that segment contains meaningful text"""
        if not text or len(text.strip()) < 3:
            return False
        
        text = text.strip().lower()
        invalid_patterns = ['[blank_audio]', '[music]', '[applause]', '[laughter]', '[noise]', '...', 'ummm', 'uhhhh']
        
        for pattern in invalid_patterns:
            if pattern in text and len(text) < 15:
                return False
        
        if not any(c.isalpha() for c in text):
            return False
        
        return len(set(text.replace(' ', ''))) > 1


# =============================================================================
# SEGMENT FILTER (Issue #1: Single classification per segment)
# =============================================================================

class SegmentFilter:
    """Filter segments based on quality criteria"""
    
    def __init__(self, params: Dict, yamnet_filter: Optional[YAMNetFilter], logger: logging.Logger):
        self.params = params
        self.yamnet_filter = yamnet_filter
        self.logger = logger
    
    def filter_segments(self, segments: List[SegmentData], audio: AudioSegment) -> List[SegmentData]:
        """
        Filter segments with SINGLE classification per segment (Issue #1)
        This is a major optimization - classify once, check all filters
        """
        self.logger.info("Filtering segments...")
        filtered = []
        stats = {'music': 0, 'noise': 0, 'confidence': 0, 'text': 0}
        
        for seg in segments:
            chunk = audio[seg.start_ms:seg.end_ms]
            
            # OPTIMIZATION: Classify once and cache in segment (Issue #1)
            if self.yamnet_filter and (self.params['filter_music'] or self.params['filter_noise']):
                classification = self.yamnet_filter.classify_audio(chunk)
                seg.classification = classification.to_dict()
                
                # Music filter
                if self.params['filter_music']:
                    if classification.is_music(self.params['music_threshold'], self.params['mixed_threshold']):
                        stats['music'] += 1
                        continue
                
                # Noise/silence filter
                if self.params['filter_noise']:
                    if classification.is_silent(self.params['noise_silence_threshold']):
                        stats['noise'] += 1
                        continue
            
            # Confidence filter
            if seg.confidence < self.params['min_confidence']:
                stats['confidence'] += 1
                continue
            
            # Text validation (already done in segmentation, but double-check)
            if len(seg.text.strip()) < 3:
                stats['text'] += 1
                continue
            
            filtered.append(seg)
        
        self.logger.info(f"Filtering complete: {len(filtered)}/{len(segments)} segments kept", extra=stats)
        return filtered


# =============================================================================
# EXPORT MANAGER (Issue #11: Separate concerns)
# =============================================================================

class ExportManager:
    """Handle segment export and metadata generation"""
    
    def __init__(self, output_dir: Path, params: Dict, logger: logging.Logger):
        self.output_dir = output_dir
        self.params = params
        self.logger = logger
        
        self.audio_dir = output_dir / "audio"
        self.text_dir = output_dir / "transcripts"
        self.audio_dir.mkdir(exist_ok=True)
        self.text_dir.mkdir(exist_ok=True)
    
    def export_segments(self, audio: AudioSegment, segments: List[SegmentData], 
                       base_name: str, sample_rate: int) -> List[Tuple[str, str, Dict]]:
        """Export segments to files with proper boundary handling"""
        self.logger.info(f"Exporting {len(segments)} segments...")
        
        output_files = []
        skipped_cutoff = 0
        
        for idx, seg in enumerate(segments, 1):
            start_ms, end_ms, should_skip = self._get_validated_boundaries_with_cutoff_check(
                audio, seg.start_ms, seg.end_ms, 
                seg.prev_word_end_ms, seg.next_word_start_ms,
                seg.text
            )
            
            if should_skip:
                skipped_cutoff += 1
                continue
            
            audio_chunk = audio[start_ms:end_ms]
            
            audio_chunk = AudioProcessor.trim_silence_vectorized(
                audio_chunk,
                self.params['trim_threshold_db'],
                self.params['trim_chunk_ms'],
                self.params['trim_pad_ms'],
                trim_end=False
            )
            
            duration = len(audio_chunk) / 1000.0
            
            if duration > self.params['max_segment_length']:
                self.logger.debug(f"Segment {idx} overlength: {duration:.2f}s")
                if self.params['overlength_action'] == 'skip':
                    continue
                suffix = "_LONG"
            else:
                suffix = ""
            
            chunk_name = f"{base_name}_{idx:04d}{suffix}"
            audio_file = self._get_unique_filename(self.audio_dir, chunk_name, f".{self.params['export_format']}")
            transcript_file = self._get_unique_filename(self.text_dir, chunk_name, ".txt")
            
            audio_chunk.export(audio_file, format=self.params['export_format'], parameters=["-ar", str(sample_rate)])
            transcript_file.write_text(seg.text, encoding='utf-8')
            
            output_files.append((str(audio_file), str(transcript_file), {
                'duration': duration,
                'text_length': len(seg.text.strip()),
                'sample_rate': sample_rate,
                'confidence': seg.confidence,
                'classification': seg.classification
            }))
        
        if skipped_cutoff > 0:
            self.logger.info(f"Skipped {skipped_cutoff} segments due to cutoff detection")
        
        self.logger.info(f"Export complete: {len(output_files)} segments")
        return output_files
    
    def _get_validated_boundaries(self, audio: AudioSegment, start_ms: int, end_ms: int,
                                   prev_word_end_ms: Optional[int], 
                                   next_word_start_ms: Optional[int]) -> Tuple[int, int]:
        """Get validated segment boundaries with buffers"""
        validated_start = start_ms
        validated_end = end_ms
        
        if self.params['word_start_buffer_ms'] > 0:
            buffer_start = start_ms - self.params['word_start_buffer_ms']
            
            if prev_word_end_ms is not None:
                gap = start_ms - prev_word_end_ms
                if gap < 200:
                    validated_start = max(prev_word_end_ms - 150, buffer_start, 0)
                else:
                    validated_start = max(prev_word_end_ms, buffer_start, 0)
            else:
                validated_start = max(0, buffer_start)
        
        if next_word_start_ms is not None:
            max_gap = next_word_start_ms - end_ms
            actual_buffer = min(self.params['word_end_buffer_ms'], max_gap)
            validated_end = end_ms + actual_buffer
        elif self.params['word_end_buffer_ms'] > 0:
            validated_end = end_ms + self.params['word_end_buffer_ms']
        
        return validated_start, validated_end
    
    def _check_for_cutoff(self, audio_chunk: AudioSegment, is_start: bool = True) -> bool:
        """Check if audio is cut off at the beginning or end"""
        if not self.params.get('check_cutoff', False):
            return False
        
        boundary_ms = self.params.get('cutoff_boundary_ms', 50)
        silence_threshold = self.params.get('cutoff_silence_threshold', 0.35)
        
        if is_start:
            boundary = audio_chunk[:boundary_ms]
        else:
            boundary = audio_chunk[-boundary_ms:]
        
        if len(boundary) < 10:
            return False
        
        samples = np.array(boundary.get_array_of_samples(), dtype=np.float32)
        rms = np.sqrt(np.mean(samples ** 2))
        normalized_rms = rms / 32768.0
        
        is_cutoff = normalized_rms > silence_threshold
        
        if is_cutoff:
            self.logger.debug(f"Cutoff detected at {'start' if is_start else 'end'}: RMS={normalized_rms:.3f}")
        
        return is_cutoff
    
    def _get_validated_boundaries_with_cutoff_check(self, audio: AudioSegment, 
                                                      start_ms: int, end_ms: int,
                                                      prev_word_end_ms: Optional[int], 
                                                      next_word_start_ms: Optional[int],
                                                      seg_text: str) -> Tuple[int, int, bool]:
        """Enhanced boundary validation with cutoff detection"""
        validated_start, validated_end = self._get_validated_boundaries(
            audio, start_ms, end_ms, prev_word_end_ms, next_word_start_ms
        )
        
        if not self.params.get('check_cutoff', False):
            return validated_start, validated_end, False
        
        audio_chunk = audio[validated_start:validated_end]
        
        if self._check_for_cutoff(audio_chunk, is_start=True):
            self.logger.debug(f"Start cutoff detected: '{seg_text[:30]}...'")
            
            extended_start = max(0, validated_start - 500)
            if prev_word_end_ms is not None:
                extended_start = max(prev_word_end_ms - 300, extended_start, 0)
            
            extended_chunk = audio[extended_start:validated_end]
            if not self._check_for_cutoff(extended_chunk, is_start=True):
                self.logger.debug("Extension successful")
                validated_start = extended_start
            else:
                cutoff_action = self.params.get('cutoff_action', 'skip')
                if cutoff_action == 'skip':
                    self.logger.warning(f"Skipping segment with cutoff: '{seg_text[:50]}...'")
                    return validated_start, validated_end, True
        
        return validated_start, validated_end, False
    
    def _get_unique_filename(self, base_path: Path, stem: str, suffix: str) -> Path:
        """Generate unique filename to avoid overwrites"""
        counter = 0
        path = base_path / f"{stem}{suffix}"
        while path.exists():
            counter += 1
            path = base_path / f"{stem}_{counter:03d}{suffix}"
        return path
    
    def generate_metadata(self, language: str):
        """Generate metadata files for the exported segments"""
        self.logger.info("Generating metadata...")
        
        audio_files = list(self.audio_dir.glob(f"*.{self.params['export_format']}"))
        if not audio_files:
            self.logger.warning("No audio files found")
            return
        
        segments = []
        for audio_file in audio_files:
            transcript_file = self.text_dir / f"{audio_file.stem}.txt"
            if not transcript_file.exists():
                continue
            
            try:
                audio = AudioSegment.from_file(audio_file)
                duration = len(audio) / 1000.0
                text = transcript_file.read_text(encoding='utf-8').strip()
                
                segments.append({
                    'audio': audio_file.name,
                    'transcript': transcript_file.name,
                    'duration': duration,
                    'text_length': len(text),
                    'sample_rate': audio.frame_rate
                })
            except Exception as e:
                self.logger.error(f"Error processing {audio_file.name}: {e}")
        
        # Save metadata.json
        metadata = {
            'total_segments': len(segments),
            'language': language,
            'segments': segments
        }
        (self.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        
        # Save manifest.txt
        with open(self.output_dir / "manifest.txt", 'w', encoding='utf-8') as f:
            for seg in segments:
                f.write(f"{seg['audio']}|{seg['transcript']}\n")
        
        # Save HuggingFace metadata.csv
        with open(self.output_dir / "metadata.csv", 'w', encoding='utf-8') as f:
            f.write("file_name,text\n")
            for seg in segments:
                text = (self.text_dir / seg['transcript']).read_text(encoding='utf-8').strip().replace('"', '""')
                f.write(f'"{seg["audio"]}","{text}"\n')
        
        self.logger.info("Metadata files created", extra={
            'segments': len(segments),
            'files': ['metadata.json', 'manifest.txt', 'metadata.csv']
        })


# =============================================================================
# MAIN SPEECH SPLITTER (Issue #11: Simplified orchestration)
# =============================================================================

class SpeechSplitter:
    """
    Main orchestrator for speech splitting pipeline
    Optimized version with all issues fixed
    """
    
    def __init__(self, **kwargs):
        self.output_dir = Path(kwargs.get('output_dir', 'output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(self.output_dir, kwargs.get('log_level', 'INFO'))
        
        # OPTIMIZATION: Simplified parameters (Issue #14)
        self.params = {
            'whisper_model': kwargs.get('whisper_model', 'large'),
            'language': kwargs.get('language', 'en'),
            'min_segment_length': kwargs.get('min_segment_length', 1.0),
            'max_segment_length': kwargs.get('max_segment_length', 15.0),
            'export_format': kwargs.get('export_format', 'wav'),
            'pause_threshold': kwargs.get('pause_threshold', 0.3),
            'min_confidence': kwargs.get('min_confidence', 0.5),
            'filter_noise': kwargs.get('filter_noise', False),
            'noise_silence_threshold': kwargs.get('noise_silence_threshold', 0.3),
            'filter_music': kwargs.get('filter_music', True),
            'music_threshold': kwargs.get('music_threshold', 0.3),
            'mixed_threshold': kwargs.get('mixed_threshold', 0.2),
            'word_start_buffer_ms': kwargs.get('word_start_buffer_ms', 250),
            'word_end_buffer_ms': kwargs.get('word_end_buffer_ms', 250),
            'check_cutoff': kwargs.get('check_cutoff', False),
            'cutoff_boundary_ms': kwargs.get('cutoff_boundary_ms', 50),
            'cutoff_silence_threshold': kwargs.get('cutoff_silence_threshold', 0.35),
            'cutoff_action': kwargs.get('cutoff_action', 'skip'),
            'hybrid_pause_detection': kwargs.get('hybrid_pause_detection', True),
            'pause_speech_threshold': kwargs.get('pause_speech_threshold', 0.2),
            'overlength_action': kwargs.get('overlength_action', 'rename'),
            'trim_threshold_db': kwargs.get('trim_threshold_db', -45.0),
            'trim_chunk_ms': kwargs.get('trim_chunk_ms', 10),
            'trim_pad_ms': kwargs.get('trim_pad_ms', 50),
        }
        
        # Device selection
        self.device = kwargs.get('device', None)
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.logger.info("Configuration loaded", extra=self.params)
        
        # Initialize components (Issue #11: Separated concerns)
        self.transcription_manager = TranscriptionManager(
            self.params['whisper_model'],
            self.params['language'],
            self.device,
            self.logger
        )
        
        # OPTIMIZATION: Lazy load YAMNet only if needed (Issue #13)
        self._yamnet_filter = None
        self._needs_yamnet = any([
            self.params['filter_music'],
            self.params['filter_noise'],
            self.params['hybrid_pause_detection']
        ])
        
        if self._needs_yamnet:
            self._yamnet_filter = YAMNetFilter(device=self.device, logger=self.logger)
            if self.params['hybrid_pause_detection'] and self._yamnet_filter.model:
                self.logger.info("Hybrid pause detection enabled")
        
        self.segmentation_engine = SegmentationEngine(self.params, self._yamnet_filter, self.logger)
        self.segment_filter = SegmentFilter(self.params, self._yamnet_filter, self.logger)
        self.export_manager = ExportManager(self.output_dir, self.params, self.logger)
        
        self.logger.info("SpeechSplitter ready")
    
    def process_file(self, audio_path: str, speaker_id: Optional[str] = None) -> Dict:
        """Process a single audio file"""
        audio_path = Path(audio_path)
        base_name = speaker_id or audio_path.stem
        
        self.logger.info(f"Processing file: {audio_path.name}", extra={
            'speaker': base_name,
            'path': str(audio_path)
        })
        
        # Load and prepare audio (OPTIMIZATION: Single mono conversion - Issue #3)
        audio, original_sr = AudioProcessor.load_and_prepare(audio_path, self.logger)
        
        # Transcribe (with caching)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio.export(tmp.name, format='wav')
            temp_path = tmp.name
        
        try:
            transcription = self.transcription_manager.transcribe(temp_path, str(audio_path))
        finally:
            os.unlink(temp_path)
        
        # Segment by pauses
        segments = self.segmentation_engine.segment_by_pauses(transcription['words'], audio)
        
        if not segments:
            self.logger.warning("No valid segments found")
            return {'total_segments': 0, 'total_duration': 0, 'avg_duration': 0}
        
        # Filter segments (OPTIMIZATION: Single classification - Issue #1)
        filtered_segments = self.segment_filter.filter_segments(segments, audio)
        
        # Export segments
        output_files = self.export_manager.export_segments(audio, filtered_segments, base_name, original_sr)
        
        # Log cache statistics (OPTIMIZATION: Issue #1)
        if self._yamnet_filter:
            stats = self._yamnet_filter.get_cache_stats()
            self.logger.info(f"YAMNet Cache: {stats['cache_hits']} hits, "
                           f"{stats['cache_misses']} misses, "
                           f"{stats['hit_rate']:.1f}% hit rate")
        
        # Summary
        total_duration = sum(m['duration'] for _, _, m in output_files)
        avg_duration = total_duration / len(output_files) if output_files else 0
        
        self.logger.info("File processing complete", extra={
            'segments': len(output_files),
            'duration': total_duration,
            'avg_duration': avg_duration
        })
        
        return {
            'total_segments': len(output_files),
            'total_duration': total_duration,
            'avg_duration': avg_duration
        }
    
    def generate_metadata(self):
        """Generate metadata for all exported segments"""
        self.export_manager.generate_metadata(self.params['language'])


# =============================================================================
# CLI (OPTIMIZATION: Removed deprecated params - Issue #4)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Speech Splitter v4 - Optimized TTS Data Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required
    parser.add_argument('input', help="Input audio file or directory")
    
    # Model settings
    parser.add_argument('--model', default='large', help="Whisper model size")
    parser.add_argument('--language', default='cs', help="Language code")
    parser.add_argument('--device', default=None, choices=['cuda', 'cpu'], help="Force device")
    
    # Output settings
    parser.add_argument('--output-dir', default='output', help="Output directory")
    parser.add_argument('--speaker-id', help="Speaker ID for naming")
    parser.add_argument('--format', default='wav', choices=['wav', 'mp3'], help="Output format")
    
    # Segmentation
    parser.add_argument('--min-length', type=float, default=1.0, help="Minimum segment length (seconds)")
    parser.add_argument('--max-length', type=float, default=15.0, help="Maximum segment length (seconds)")
    parser.add_argument('--pause-threshold', type=float, default=0.3, help="Pause detection threshold (seconds)")
    parser.add_argument('--hybrid-pause-detection', action='store_true', default=True, 
                       help="Use YAMNet to validate pauses")
    parser.add_argument('--pause-speech-threshold', type=float, default=0.2, 
                       help="Speech threshold for pause validation")
    
    # Quality filtering
    parser.add_argument('--min-confidence', type=float, default=0.5, help="Minimum confidence score")
    parser.add_argument('--filter-music', action='store_true', default=True, help="Filter music segments")
    parser.add_argument('--music-threshold', type=float, default=0.3, help="Music detection threshold")
    parser.add_argument('--mixed-threshold', type=float, default=0.2, help="Mixed audio threshold")
    parser.add_argument('--filter-noise', action='store_true', help="Filter noisy/silent segments")
    parser.add_argument('--noise-silence-threshold', type=float, default=0.3, help="Silence threshold")
    
    # Audio processing
    parser.add_argument('--word-start-buffer', type=int, default=250, help="Start buffer (ms)")
    parser.add_argument('--word-end-buffer', type=int, default=250, help="End buffer (ms)")
    parser.add_argument('--trim-threshold-db', type=float, default=-45.0, help="Silence trim threshold (dB)")
    parser.add_argument('--trim-chunk-ms', type=int, default=10, help="Trim analysis chunk size (ms)")
    parser.add_argument('--trim-pad-ms', type=int, default=50, help="Trim padding (ms)")
    
    # Actions
    parser.add_argument('--overlength-action', default='rename', choices=['rename', 'skip', 'include'],
                       help="Action for overlength segments")
    
    # Logging
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # GPU check
    temp_logger = logging.getLogger("Startup")
    temp_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    temp_logger.addHandler(handler)
    
    temp_logger.info("=== Speech Splitter v4 - Optimized ===")
    check_gpu(temp_logger)
    temp_logger.handlers.clear()
    
    # Create splitter
    splitter = SpeechSplitter(
        whisper_model=args.model,
        language=args.language,
        output_dir=args.output_dir,
        min_segment_length=args.min_length,
        max_segment_length=args.max_length,
        export_format=args.format,
        pause_threshold=args.pause_threshold,
        min_confidence=args.min_confidence,
        filter_noise=args.filter_noise,
        noise_silence_threshold=args.noise_silence_threshold,
        filter_music=args.filter_music,
        music_threshold=args.music_threshold,
        mixed_threshold=args.mixed_threshold,
        word_start_buffer_ms=args.word_start_buffer,
        word_end_buffer_ms=args.word_end_buffer,
        hybrid_pause_detection=args.hybrid_pause_detection,
        pause_speech_threshold=args.pause_speech_threshold,
        overlength_action=args.overlength_action,
        trim_threshold_db=args.trim_threshold_db,
        trim_chunk_ms=args.trim_chunk_ms,
        trim_pad_ms=args.trim_pad_ms,
        device=args.device,
        log_level=args.log_level
    )
    
    # Process input
    input_path = Path(args.input)
    if input_path.is_file():
        splitter.process_file(str(input_path), args.speaker_id)
    elif input_path.is_dir():
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']:
            audio_files.extend(input_path.glob(ext))
        
        splitter.logger.info(f"Processing directory: {len(audio_files)} files")
        for audio_file in sorted(audio_files):
            speaker_id = args.speaker_id or audio_file.stem
            splitter.process_file(str(audio_file), speaker_id)
    else:
        splitter.logger.error(f"Invalid input path: {args.input}")
        return
    
    # Generate metadata
    splitter.generate_metadata()
    splitter.logger.info("=== All processing complete ===")


if __name__ == '__main__':
    main()
