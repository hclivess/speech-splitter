
# speech_splitter_v3_fixed.py
import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import warnings
import sys
warnings.filterwarnings('ignore')

import whisper
from pydub import AudioSegment
import numpy as np
from tqdm import tqdm
import torch

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
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
    
    # File handler - detailed JSON-like format
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler - colored, concise format
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
    """Check GPU availability with clean output"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
        return True
    else:
        logger.warning("No GPU available, using CPU (slower processing)")
        return False

class YAMNetFilter:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', logger=None):
        self.model = None
        self.class_names = None
        self.device = device
        self.logger = logger or logging.getLogger("SpeechSplitter.YAMNetFilter")
        self._load_model()
    
    def _load_model(self):
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
            self.model = hub.load('https://tfhub.dev/google/yamnet/1')
            
            class_map_path = self.model.class_map_path().numpy()
            self.class_names = self._load_class_names(class_map_path)
            self.logger.info("YAMNet model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load YAMNet: {e}", exc_info=True)
            self.logger.warning("Audio filtering disabled")
            self.model = None
    
    def _load_class_names(self, class_map_csv):
        import csv
        class_names = []
        with open(class_map_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_names.append(row['display_name'])
        return class_names
    
    def classify_audio(self, audio_segment: AudioSegment) -> Dict[str, float]:
        if self.model is None:
            return {'music': 0.0, 'speech': 1.0, 'music_speech_mixed': 0.0, 'silence': 0.0}
        
        try:
            import tensorflow as tf
            
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)
            
            samples = np.array(audio_segment.get_array_of_samples())
            
            if audio_segment.sample_width == 2:
                samples = samples.astype(np.float32) / 32768.0
            elif audio_segment.sample_width == 4:
                samples = samples.astype(np.float32) / 2147483648.0
            
            scores, embeddings, spectrogram = self.model(samples)
            mean_scores = np.mean(scores.numpy(), axis=0)
            
            music_classes = [i for i, name in enumerate(self.class_names) 
                           if 'music' in name.lower() and 'speech' not in name.lower()]
            speech_classes = [i for i, name in enumerate(self.class_names) 
                            if 'speech' in name.lower() and 'music' not in name.lower()]
            mixed_classes = [i for i, name in enumerate(self.class_names) 
                           if 'music' in name.lower() and 'speech' in name.lower()]
            silence_classes = [i for i, name in enumerate(self.class_names) 
                             if 'silence' in name.lower() or 'quiet' in name.lower()]
            
            music_prob = float(np.max([mean_scores[i] for i in music_classes]) if music_classes else 0.0)
            speech_prob = float(np.max([mean_scores[i] for i in speech_classes]) if speech_classes else 0.0)
            mixed_prob = float(np.max([mean_scores[i] for i in mixed_classes]) if mixed_classes else 0.0)
            silence_prob = float(np.max([mean_scores[i] for i in silence_classes]) if silence_classes else 0.0)
            
            return {
                'music': music_prob,
                'speech': speech_prob,
                'music_speech_mixed': mixed_prob,
                'silence': silence_prob
            }
            
        except Exception as e:
            self.logger.debug(f"YAMNet classification error: {e}")
            return {'music': 0.0, 'speech': 1.0, 'music_speech_mixed': 0.0, 'silence': 0.0}
    
    def is_music_or_mixed(self, audio_segment: AudioSegment, 
                          music_threshold: float = 0.3,
                          mixed_threshold: float = 0.2) -> Tuple[bool, Dict]:
        classification = self.classify_audio(audio_segment)
        is_music = classification['music'] > music_threshold
        is_mixed = classification['music_speech_mixed'] > mixed_threshold
        return (is_music or is_mixed), classification
    
    def is_silent(self, audio_segment: AudioSegment, 
                  silence_threshold: float = 0.3) -> bool:
        classification = self.classify_audio(audio_segment)
        return classification['silence'] > silence_threshold
    
    def is_true_pause(self, audio_segment: AudioSegment,
                     silence_threshold: float = 0.3,
                     speech_threshold: float = 0.2) -> bool:
        if len(audio_segment) < 50:
            return True
        
        classification = self.classify_audio(audio_segment)
        is_silence = classification['silence'] > silence_threshold
        is_not_speech = classification['speech'] < speech_threshold
        
        return is_silence or is_not_speech

class SpeechSplitter:
    def __init__(self, **kwargs):
        self.output_dir = Path(kwargs.get('output_dir', 'output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging first
        self.logger = setup_logging(self.output_dir, kwargs.get('log_level', 'INFO'))
        
        # Log initialization
        self.logger.info("SpeechSplitter initializing", extra={
            'version': '3.1',
            'device': kwargs.get('device', 'auto')
        })
        
        # Set device
        self.device = kwargs.get('device')
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Store parameters - UPDATED DEFAULTS
        self.params = {
            'whisper_model': kwargs.get('whisper_model', 'large'),
            'language': kwargs.get('language', 'cs'),
            'min_segment_length': kwargs.get('min_segment_length', 1.0),
            'max_segment_length': kwargs.get('max_segment_length', 15.0),
            'export_format': kwargs.get('export_format', 'wav'),
            'pause_threshold': kwargs.get('pause_threshold', 0.3),  # Lowered from 0.5
            'min_confidence': kwargs.get('min_confidence', 0.5),
            'filter_noise': kwargs.get('filter_noise', True),
            'noise_silence_threshold': kwargs.get('noise_silence_threshold', 0.3),
            'filter_music': kwargs.get('filter_music', True),
            'music_threshold': kwargs.get('music_threshold', 0.3),
            'mixed_threshold': kwargs.get('mixed_threshold', 0.2),
            'word_start_buffer_ms': kwargs.get('word_start_buffer_ms', 250),
            'word_end_buffer_ms': kwargs.get('word_end_buffer_ms', 1500),  # Increased from 750
            'word_end_extension_ms': kwargs.get('word_end_extension_ms', 0),  # Disabled flawed logic
            'check_cutoff': kwargs.get('check_cutoff', True),
            'cutoff_boundary_ms': kwargs.get('cutoff_boundary_ms', 100),  # Increased from 50
            'cutoff_silence_threshold': kwargs.get('cutoff_silence_threshold', 0.3),
            'cutoff_action': kwargs.get('cutoff_action', 'rename'),
            'hybrid_pause_detection': kwargs.get('hybrid_pause_detection', True),
            'pause_speech_threshold': kwargs.get('pause_speech_threshold', 0.2),
            'buffer_validation_threshold': kwargs.get('buffer_validation_threshold', 0.3),
            'overlength_action': kwargs.get('overlength_action', 'rename'),
            'trim_silence': kwargs.get('trim_silence', True),
            'trim_threshold_db': kwargs.get('trim_threshold_db', -45.0),  # Lowered from -40
            'trim_chunk_ms': kwargs.get('trim_chunk_ms', 10),
            'trim_pad_ms': kwargs.get('trim_pad_ms', 50),  # Reduced from 150
        }
        
        # Log configuration as JSON
        self.logger.info("Configuration loaded", extra=self.params)
        
        # Setup directories
        self.audio_dir = self.output_dir / "audio"
        self.text_dir = self.output_dir / "transcripts"
        self.audio_dir.mkdir(exist_ok=True)
        self.text_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Output directories: {self.output_dir}")
        
        # Load models
        self.logger.info(f"Loading Whisper model: {self.params['whisper_model']} on {self.device.upper()}")
        self.model = whisper.load_model(self.params['whisper_model'], device=self.device)
        self.logger.info("Whisper model loaded")
        
        # Load YAMNet if needed
        needs_yamnet = any([
            self.params['filter_music'],
            self.params['filter_noise'],
            self.params['check_cutoff'],
            self.params['hybrid_pause_detection']
        ])
        
        if needs_yamnet:
            self.yamnet_filter = YAMNetFilter(device=self.device, logger=self.logger)
            if self.params['hybrid_pause_detection'] and self.yamnet_filter.model:
                self.logger.info("Hybrid pause detection enabled")
        else:
            self.yamnet_filter = None
            self.logger.warning("Audio filtering disabled")
        
        self.logger.info("SpeechSplitter ready")
    
    def safe_write_transcript(self, transcript_file: Path, text: str) -> bool:
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception as e:
                self.logger.error(f"Text conversion failed: {e}")
                return False
        
        text = text.strip()
        if not text:
            self.logger.warning("Empty transcript skipped")
            return False
        
        try:
            transcript_file.write_text(text, encoding='utf-8')
            return True
        except Exception as e:
            self.logger.error(f"Transcript write failed: {e}")
            return False
    
    def validate_segment_text(self, text: str) -> bool:
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
    
    def transcribe_full_audio(self, audio_path: str, original_audio_path: str = None) -> Dict:
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
        
        self.logger.info("Transcribing audio with Whisper")
        result = self.model.transcribe(
            audio_path,
            language=self.params['language'],
            word_timestamps=True,
            verbose=False
        )
        
        all_words = []
        for segment in result.get('segments', []):
            if 'words' in segment:
                all_words.extend(segment['words'])
        
        self.logger.info(f"Transcription complete: {len(all_words)} words")
        
        transcription_data = {
            'text': result['text'],
            'words': all_words,
            'segments': result.get('segments', []),
            'language': self.params['language'],
            'model': self.params['whisper_model']
        }
        
        try:
            self.logger.debug(f"Saving transcription cache: {cache_file.name}")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(transcription_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"Cache save failed: {e}")
        
        return transcription_data
    
    def segment_by_pauses(self, words: List[Dict], audio: AudioSegment) -> List[Dict]:
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
                        gap_audio = audio[int(word['end']*1000):int(next_word['start']*1000)]
                        if self.yamnet_filter.is_true_pause(gap_audio, self.params['noise_silence_threshold'], self.params['pause_speech_threshold']):
                            should_split = True
                            stats['validated'] += 1
                        else:
                            stats['rejected'] += 1
                    else:
                        should_split = True
            
            if i == len(words) - 1:
                should_split = True
            
            if should_split and current_words:
                first_word = current_words[0]
                last_word = current_words[-1]
                
                if 'start' in first_word and 'end' in last_word:
                    text = ''.join([w.get('word', '') for w in current_words]).strip()
                    duration = last_word['end'] - first_word['start']
                    
                    # FIX: Calculate context timestamps for boundary validation
                    prev_word_end_ms = int(words[i - len(current_words)]['end'] * 1000) if i - len(current_words) >= 0 else None
                    next_word_start_ms = int(words[i + 1]['start'] * 1000) if i + 1 < len(words) else None
                    
                    if self.validate_segment_text(text) and duration >= self.params['min_segment_length']:
                        segments.append({
                            'start_ms': int(first_word['start'] * 1000),
                            'end_ms': int(last_word['end'] * 1000),
                            'duration': duration,
                            'words': current_words.copy(),
                            'text': text,
                            'confidence': sum(w.get('probability', 1.0) for w in current_words) / len(current_words),
                            'prev_word_end_ms': prev_word_end_ms,  # FIX: Store previous word end
                            'next_word_start_ms': next_word_start_ms  # FIX: Store next word start
                        })
                
                # CRITICAL FIX: Clear current_words after processing split
                current_words = []
        
        self.logger.info(f"Segmentation complete: {len(segments)} segments", extra={
            'potential_pauses': stats['potential'],
            'validated_pauses': stats['validated'],
            'rejected_pauses': stats['rejected']
        })
        
        return segments
    
    def validate_buffer_zone(self, audio: AudioSegment, start_ms: int, end_ms: int) -> bool:
        if not self.yamnet_filter or end_ms <= start_ms:
            return True
        
        try:
            buffer_audio = audio[start_ms:end_ms]
            return len(buffer_audio) < 20 or self.yamnet_filter.is_true_pause(
                buffer_audio,
                self.params['buffer_validation_threshold'],
                self.params['pause_speech_threshold']
            )
        except Exception as e:
            self.logger.debug(f"Buffer validation error: {e}")
            return False
    
    def get_validated_boundaries(self, audio: AudioSegment, start_ms: int, end_ms: int,
                                prev_word_end_ms: Optional[int], next_word_start_ms: Optional[int]) -> Tuple[int, int]:
        """FIXED: Preserve ALL silence between words until next word starts"""
        validated_start = start_ms
        validated_end = end_ms
        
        # START: Extend backward by buffer, but not into previous word
        if self.params['word_start_buffer_ms'] > 0 and prev_word_end_ms is not None:
            requested_start = max(prev_word_end_ms, start_ms - self.params['word_start_buffer_ms'])
            if requested_start < start_ms and self.validate_buffer_zone(audio, requested_start, start_ms):
                validated_start = requested_start
        
        # END: Extend forward to include ALL silence until next word (FIXED)
        if next_word_start_ms is not None:
            # Use min() to ensure we don't exceed next word's start
            requested_end = min(end_ms + self.params['word_end_buffer_ms'], next_word_start_ms)
            if requested_end > end_ms and self.validate_buffer_zone(audio, end_ms, requested_end):
                validated_end = requested_end
        
        return validated_start, validated_end
    
    def detect_cutoff(self, audio_chunk: AudioSegment, text: str) -> Tuple[bool, str]:
        if not self.params['check_cutoff'] or not self.yamnet_filter:
            return False, ""
        
        start_chunk = audio_chunk[:self.params['cutoff_boundary_ms']]
        end_chunk = audio_chunk[-self.params['cutoff_boundary_ms']:]
        
        start_silent = self.yamnet_filter.is_silent(start_chunk, self.params['cutoff_silence_threshold'])
        end_silent = self.yamnet_filter.is_silent(end_chunk, self.params['cutoff_silence_threshold'])
        
        if not start_silent and not end_silent:
            return True, "both"
        elif not start_silent:
            return True, "start"
        elif not end_silent:
            return True, "end"
        
        return False, ""
    
    def handle_special_segment(self, audio_chunk: AudioSegment, text: str, 
                             issue_type: str, base_name: str, idx: int, sample_rate: int) -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
        """Unified handler for cutoff/overlength segments"""
        if issue_type == "cutoff":
            action = self.params['cutoff_action']
            suffix = f"_CUT{action.upper()}"
        else:  # overlength
            action = self.params['overlength_action']
            suffix = "_LONG"
        
        if action == "skip":
            self.logger.debug(f"Skipping segment {idx} ({issue_type})")
            return None, None, None
        
        chunk_name = f"{base_name}_{idx:04d}{suffix}"
        audio_file = self.get_unique_filename(self.audio_dir, chunk_name, f".{self.params['export_format']}")
        transcript_file = self.get_unique_filename(self.text_dir, chunk_name, ".txt")
        
        audio_chunk.export(audio_file, format=self.params['export_format'], parameters=["-ar", str(sample_rate)])
        self.safe_write_transcript(transcript_file, text)
        
        metadata = {
            'duration': len(audio_chunk) / 1000.0,
            'text_length': len(text.strip()),
            'sample_rate': sample_rate,
            'issue': issue_type
        }
        
        return str(audio_file), str(transcript_file), metadata
    
    def get_unique_filename(self, base_path: Path, stem: str, suffix: str) -> Path:
        counter = 0
        path = base_path / f"{stem}{suffix}"
        while path.exists():
            counter += 1
            path = base_path / f"{stem}_{counter:03d}{suffix}"
        return path
    
    def trim_silence_from_audio(self, audio_chunk: AudioSegment, trim_end: bool = True) -> AudioSegment:
        """FIXED: Added trim_end parameter to control end trimming"""
        if not self.params['trim_silence'] or len(audio_chunk) < 200:
            return audio_chunk
        
        # Trim start
        start_trim = 0
        for i in range(0, len(audio_chunk), self.params['trim_chunk_ms']):
            if audio_chunk[i:i + self.params['trim_chunk_ms']].dBFS > self.params['trim_threshold_db']:
                start_trim = max(0, i - self.params['trim_pad_ms'])
                break
        
        # Trim end (only if enabled)
        end_trim = len(audio_chunk)
        if trim_end:
            for i in range(len(audio_chunk), 0, -self.params['trim_chunk_ms']):
                chunk = audio_chunk[max(0, i - self.params['trim_chunk_ms']):i]
                if chunk.dBFS > self.params['trim_threshold_db']:
                    end_trim = min(len(audio_chunk), i + self.params['trim_pad_ms'])
                    break
        
        if start_trim >= end_trim:
            return audio_chunk
        
        trimmed = audio_chunk[start_trim:end_trim]
        self.logger.debug(f"Trimmed silence: -{start_trim}ms, -{len(audio_chunk)-end_trim}ms")
        return trimmed
    
    def process_file(self, audio_path: str, speaker_id: Optional[str] = None) -> Dict:
        audio_path = Path(audio_path)
        base_name = speaker_id or audio_path.stem
        
        self.logger.info(f"Processing file: {audio_path.name}", extra={
            'speaker': base_name,
            'path': str(audio_path)
        })
        
        # Load audio
        try:
            audio = AudioSegment.from_file(audio_path)
        except Exception as e:
            self.logger.error(f"Audio load failed: {e}", extra={'file': str(audio_path)})
            return {'total_segments': 0, 'total_duration': 0, 'avg_duration': 0}
        
        # Convert to mono if needed
        original_sr = audio.frame_rate
        if audio.channels > 1:
            audio = audio.set_channels(1)
            self.logger.debug("Converted to mono")
        
        self.logger.info(f"Audio loaded: {len(audio)/1000:.1f}s, {audio.frame_rate}Hz")
        
        # Transcribe
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio.export(tmp.name, format='wav')
            temp_path = tmp.name
        
        try:
            transcription = self.transcribe_full_audio(temp_path, str(audio_path))
        finally:
            os.unlink(temp_path)
        
        # Segment
        segments = self.segment_by_pauses(transcription['words'], audio)
        
        if not segments:
            self.logger.warning("No valid segments found")
            return {'total_segments': 0, 'total_duration': 0, 'avg_duration': 0}
        
        # Filter segments
        self.logger.info("Filtering segments...")
        filtered = []
        stats = {'music': 0, 'noise': 0, 'confidence': 0, 'text': 0}
        
        for seg in segments:
            chunk = audio[seg['start_ms']:seg['end_ms']]
            
            # Music filter
            if self.yamnet_filter and self.params['filter_music']:
                is_music, _ = self.yamnet_filter.is_music_or_mixed(chunk, self.params['music_threshold'], self.params['mixed_threshold'])
                if is_music:
                    stats['music'] += 1
                    continue
            
            # Noise filter
            if self.params['filter_noise'] and self.yamnet_filter and self.yamnet_filter.is_silent(chunk, self.params['noise_silence_threshold']):
                stats['noise'] += 1
                continue
            
            # Confidence filter
            if seg['confidence'] < self.params['min_confidence']:
                stats['confidence'] += 1
                continue
            
            # Text validation
            if not self.validate_segment_text(seg['text']):
                stats['text'] += 1
                continue
            
            filtered.append(seg)
        
        self.logger.info(f"Filtering complete: {len(filtered)}/{len(segments)} segments kept", extra=stats)
        
        # Export
        output_files = self.export_segments(audio, filtered, base_name, original_sr)
        
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
    
    def export_segments(self, audio: AudioSegment, segments: List[Dict], base_name: str, sample_rate: int) -> List[Tuple[str, str, Dict]]:
        self.logger.info(f"Exporting {len(segments)} segments...")
        
        output_files = []
        for idx, seg in enumerate(segments, 1):
            if not self.validate_segment_text(seg['text']):
                continue
            
            # Get validated boundaries with context
            start_ms, end_ms = self.get_validated_boundaries(
                audio,
                seg['start_ms'],
                seg['end_ms'],
                seg.get('prev_word_end_ms'),
                seg.get('next_word_start_ms')
            )
            
            audio_chunk = audio[start_ms:end_ms]
            
            # FIX: Don't trim end silence to preserve word endings
            audio_chunk = self.trim_silence_from_audio(audio_chunk, trim_end=False)
            
            duration = len(audio_chunk) / 1000.0
            
            # Handle special cases
            if duration > self.params['max_segment_length']:
                self.logger.debug(f"Segment {idx} overlength: {duration:.2f}s")
                result = self.handle_special_segment(audio_chunk, seg['text'], "overlength", base_name, idx, sample_rate)
            else:
                has_cutoff, cutoff_type = self.detect_cutoff(audio_chunk, seg['text'])
                if has_cutoff:
                    self.logger.debug(f"Segment {idx} cutoff: {cutoff_type}")
                    result = self.handle_special_segment(audio_chunk, seg['text'], "cutoff", base_name, idx, sample_rate)
                else:
                    chunk_name = f"{base_name}_{idx:04d}"
                    audio_file = self.get_unique_filename(self.audio_dir, chunk_name, f".{self.params['export_format']}")
                    transcript_file = self.get_unique_filename(self.text_dir, chunk_name, ".txt")
                    
                    audio_chunk.export(audio_file, format=self.params['export_format'], parameters=["-ar", str(sample_rate)])
                    self.safe_write_transcript(transcript_file, seg['text'])
                    
                    result = (str(audio_file), str(transcript_file), {
                        'duration': duration,
                        'text_length': len(seg['text'].strip()),
                        'sample_rate': sample_rate,
                        'issue': None
                    })
            
            if result[0] is not None:
                output_files.append(result)
        
        self.logger.info(f"Export complete: {len(output_files)} segments")
        return output_files
    
    def regenerate_folder_metadata(self):
        self.logger.info("Regenerating folder metadata...")
        
        audio_files = list(self.audio_dir.glob(f"*.{self.params['export_format']}"))
        if not audio_files:
            self.logger.warning("No audio files found")
            return
        
        segments = []
        for audio_file in audio_files:
            transcript_file = self.text_dir / f"{audio_file.stem}.txt"
            if not transcript_file.exists():
                self.logger.debug(f"Missing transcript for {audio_file.name}")
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
        
        # Save metadata
        metadata = {
            'total_segments': len(segments),
            'language': self.params['language'],
            'segments': segments
        }
        
        (self.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        
        # Save manifest
        manifest_path = self.output_dir / "manifest.txt"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            for seg in segments:
                f.write(f"{seg['audio']}|{seg['transcript']}\n")
        
        # Save HuggingFace metadata
        hf_path = self.output_dir / "metadata.csv"
        with open(hf_path, 'w', encoding='utf-8') as f:
            f.write("file_name,text\n")
            for seg in segments:
                text = (self.text_dir / seg['transcript']).read_text(encoding='utf-8').strip().replace('"', '""')
                f.write(f'"{seg["audio"]}","{text}"\n')
        
        self.logger.info("Metadata files created", extra={
            'segments': len(segments),
            'files': ['metadata.json', 'manifest.txt', 'metadata.csv']
        })

def main():
    parser = argparse.ArgumentParser(description="Speech Splitter v3.1 - TTS Data Generator (FIXED)")
    parser.add_argument('input', help="Input audio file or directory")
    parser.add_argument('--model', default='large', help="Whisper model size")
    parser.add_argument('--language', default='cs', help="Language code")
    parser.add_argument('--output-dir', default='output', help="Output directory")
    parser.add_argument('--speaker-id', help="Speaker ID for naming")
    parser.add_argument('--min-length', type=float, default=1.0, help="Minimum segment length (seconds)")
    parser.add_argument('--max-length', type=float, default=15.0, help="Maximum segment length (seconds)")
    parser.add_argument('--format', default='wav', choices=['wav', 'mp3'], help="Output format")
    parser.add_argument('--pause-threshold', type=float, default=0.3, help="Pause detection threshold")
    parser.add_argument('--min-confidence', type=float, default=0.5, help="Minimum confidence")
    parser.add_argument('--filter-noise', action='store_true', help="Filter noisy segments")
    parser.add_argument('--noise-silence-threshold', type=float, default=0.3, help="Noise threshold")
    parser.add_argument('--filter-music', action='store_true', default=True, help="Filter music")
    parser.add_argument('--music-threshold', type=float, default=0.3, help="Music threshold")
    parser.add_argument('--mixed-threshold', type=float, default=0.2, help="Mixed threshold")
    parser.add_argument('--word-start-buffer', type=int, default=250, help="Start buffer (ms)")
    parser.add_argument('--word-end-buffer', type=int, default=1500, help="End buffer (ms) - MAX silence to keep")
    parser.add_argument('--word-end-extension', type=int, default=0, help="Disabled - use word-end-buffer instead")
    parser.add_argument('--check-cutoff', action='store_true', default=True, help="Check for cutoffs")
    parser.add_argument('--cutoff-boundary-ms', type=int, default=100, help="Cutoff boundary")
    parser.add_argument('--cutoff-silence-threshold', type=float, default=0.3, help="Cutoff threshold")
    parser.add_argument('--cutoff-action', default='rename', choices=['rename', 'skip', 'include'], help="Cutoff action")
    parser.add_argument('--hybrid-pause-detection', action='store_true', default=True, help="Hybrid pause detection")
    parser.add_argument('--pause-speech-threshold', type=float, default=0.2, help="Pause speech threshold")
    parser.add_argument('--overlength-action', default='rename', choices=['rename', 'skip', 'include'], help="Overlength action")
    parser.add_argument('--trim-silence', action='store_true', default=True, help="Trim silence")
    parser.add_argument('--trim-threshold-db', type=float, default=-45.0, help="Trim threshold")
    parser.add_argument('--trim-chunk-ms', type=int, default=10, help="Trim chunk size")
    parser.add_argument('--trim-pad-ms', type=int, default=50, help="Trim padding")
    parser.add_argument('--device', default=None, choices=['cuda', 'cpu'], help="Force device")
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help="Log level")
    
    args = parser.parse_args()
    
    # GPU check
    temp_logger = logging.getLogger("Startup")
    temp_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    temp_logger.addHandler(handler)
    
    temp_logger.info("GPU Check")
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
        word_end_extension_ms=args.word_end_extension,
        check_cutoff=args.check_cutoff,
        cutoff_boundary_ms=args.cutoff_boundary_ms,
        cutoff_silence_threshold=args.cutoff_silence_threshold,
        cutoff_action=args.cutoff_action,
        hybrid_pause_detection=args.hybrid_pause_detection,
        pause_speech_threshold=args.pause_speech_threshold,
        overlength_action=args.overlength_action,
        trim_silence=args.trim_silence,
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
        for audio_file in audio_files:
            speaker_id = args.speaker_id or audio_file.stem
            splitter.process_file(str(audio_file), speaker_id)
    else:
        splitter.logger.error(f"Invalid input path: {args.input}")
        return
    
    splitter.regenerate_folder_metadata()
    splitter.logger.info("All processing complete")

if __name__ == '__main__':
    main()
