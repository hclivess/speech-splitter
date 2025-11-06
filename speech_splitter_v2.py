#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import whisper
from pydub import AudioSegment
import numpy as np
from tqdm import tqdm
import torch


def check_gpu():
    """Check GPU availability and display info."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ GPU available: {gpu_name}")
        print(f"  GPU Memory: {gpu_memory:.2f} GB")
        return True
    else:
        print("⚠ No GPU available, will use CPU (slower)")
        return False


class YAMNetFilter:
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = None
        self.class_names = None
        self.device = device
        self._load_model()
    
    def _load_model(self):
        try:
            import tensorflow as tf
            
            # Configure TensorFlow to use GPU if available
            gpus = tf.config.list_physical_devices('GPU')
            if gpus and self.device == 'cuda':
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"✓ TensorFlow configured to use GPU for YAMNet")
                except RuntimeError as e:
                    print(f"⚠ GPU configuration error: {e}")
            
            import tensorflow_hub as hub
            
            tf.get_logger().setLevel('ERROR')
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            
            print("Loading YAMNet model...")
            self.model = hub.load('https://tfhub.dev/google/yamnet/1')
            
            class_map_path = self.model.class_map_path().numpy()
            self.class_names = self._load_class_names(class_map_path)
            
            print("✓ YAMNet model loaded")
            
        except Exception as e:
            print(f"⚠ Failed to load YAMNet: {e}")
            print("⚠ Audio filtering will be disabled")
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
            print(f"⚠ YAMNet classification error: {e}")
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
        is_not_speech = classification['speech'] < speech_threshold and classification['music'] < speech_threshold
        
        return is_silence or is_not_speech


class SpeechSplitter:
    
    def __init__(
        self,
        whisper_model: str = "large",
        language: str = "cs",
        output_dir: str = "output",
        min_segment_length: float = 1.0,
        max_segment_length: float = 15.0,
        export_format: str = "wav",
        pause_threshold: float = 0.5,
        min_confidence: float = 0.5,
        filter_noise: bool = True,
        noise_silence_threshold: float = 0.3,
        filter_music: bool = True,
        music_threshold: float = 0.3,
        mixed_threshold: float = 0.2,
        word_start_buffer_ms: int = 500,
        word_end_buffer_ms: int = 500,
        check_cutoff: bool = True,
        cutoff_boundary_ms: int = 50,
        cutoff_silence_threshold: float = 0.3,
        cutoff_action: str = "rename",
        hybrid_pause_detection: bool = True,
        pause_speech_threshold: float = 0.2,
        buffer_validation_threshold: float = 0.3,
        overlength_action: str = "rename",
        trim_silence: bool = True,
        trim_threshold_db: float = -40.0,
        trim_chunk_ms: int = 10,
        trim_pad_ms: int = 150,
        device: str = None
    ):
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.whisper_model = whisper_model
        self.language = language
        self.output_dir = Path(output_dir)
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.export_format = export_format
        self.pause_threshold = pause_threshold
        self.min_confidence = min_confidence
        self.filter_noise = filter_noise
        self.noise_silence_threshold = noise_silence_threshold
        self.filter_music = filter_music
        self.music_threshold = music_threshold
        self.mixed_threshold = mixed_threshold
        self.word_start_buffer_ms = word_start_buffer_ms
        self.word_end_buffer_ms = word_end_buffer_ms
        self.check_cutoff = check_cutoff
        self.cutoff_boundary_ms = cutoff_boundary_ms
        self.cutoff_silence_threshold = cutoff_silence_threshold
        self.cutoff_action = cutoff_action
        self.hybrid_pause_detection = hybrid_pause_detection
        self.pause_speech_threshold = pause_speech_threshold
        self.buffer_validation_threshold = buffer_validation_threshold
        self.overlength_action = overlength_action
        self.trim_silence = trim_silence
        self.trim_threshold_db = trim_threshold_db
        self.trim_chunk_ms = trim_chunk_ms
        self.trim_pad_ms = trim_pad_ms
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir = self.output_dir / "audio"
        self.text_dir = self.output_dir / "transcripts"
        self.audio_dir.mkdir(exist_ok=True)
        self.text_dir.mkdir(exist_ok=True)
        
        print(f"Loading Whisper model ({whisper_model}) on {self.device.upper()}...")
        self.model = whisper.load_model(whisper_model, device=self.device)
        
        if self.filter_music or self.filter_noise or self.check_cutoff or self.hybrid_pause_detection:
            self.yamnet_filter = YAMNetFilter(device=self.device)
            if self.hybrid_pause_detection and self.yamnet_filter.model:
                print("✓ Hybrid pause detection enabled (Whisper + YAMNet)")
        else:
            self.yamnet_filter = None
            print("⚠ Audio filtering disabled")
        
        print("✓ Models loaded\n")
    
    def safe_write_transcript(self, transcript_file: Path, text: str) -> bool:
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception as e:
                print(f"  ⚠ Could not convert text to string: {e}")
                return False
        
        text = text.strip()
        
        if not text:
            print(f"  ⚠ Empty text, skipping transcript save")
            return False
        
        try:
            with open(transcript_file, 'wb') as f:
                f.write(text.encode('utf-8'))
            return True
        except PermissionError as e:
            print(f"  ✗ Permission denied: {e}")
            return False
        except OSError as e:
            print(f"  ✗ OS error writing file: {e}")
            return False
        except Exception as e:
            print(f"  ✗ Unexpected error saving transcript: {e}")
            return False
    
    def validate_segment_text(self, text: str) -> bool:
        if not text or not text.strip():
            return False
        
        text = text.strip()
        
        if len(text) < 3:
            return False
        
        invalid_patterns = [
            '[BLANK_AUDIO]', '[MUSIC]', '[APPLAUSE]', '[LAUGHTER]', '[NOISE]',
            '...', '....', 'ummm', 'uhhhh',
        ]
        
        text_lower = text.lower()
        for pattern in invalid_patterns:
            if pattern.lower() in text_lower and len(text) < 15:
                return False
        
        if not any(c.isalpha() for c in text):
            return False
        
        if len(set(text.replace(' ', ''))) == 1:
            return False
        
        return True

    def transcribe_full_audio(self, audio_path: str, original_audio_path: str = None) -> Dict:
        cache_base_path = Path(original_audio_path) if original_audio_path else Path(audio_path)
        cache_file = cache_base_path.parent / f"{cache_base_path.stem}.transcription.json"

        if cache_file.exists():
            try:
                print(f"Loading cached transcription from {cache_file.name}...")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)

                if 'words' in cached_data and 'text' in cached_data:
                    print(f"✓ Loaded {len(cached_data['words'])} words from cache")
                    return cached_data
                else:
                    print("⚠ Cache file invalid, re-transcribing...")
            except Exception as e:
                print(f"⚠ Error loading cache: {e}, re-transcribing...")

        print("Transcribing full audio with Whisper...")

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

        print(f"✓ Transcribed {len(all_words)} words")

        transcription_data = {
            'text': result['text'],
            'words': all_words,
            'segments': result.get('segments', []),
            'language': self.language,
            'model': self.whisper_model
        }

        try:
            print(f"Saving transcription cache to {cache_file.name}...")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(transcription_data, f, ensure_ascii=False, indent=2)
            print("✓ Transcription cached")
        except Exception as e:
            print(f"⚠ Warning: Could not save transcription cache: {e}")

        return transcription_data
    
    def is_pause_validated(self, audio: AudioSegment, word_end_ms: int, next_word_start_ms: int, 
                          pause_duration: float) -> bool:
        if pause_duration < self.pause_threshold:
            return False
        
        if not self.hybrid_pause_detection or not self.yamnet_filter or not self.yamnet_filter.model:
            return True
        
        try:
            gap_audio = audio[word_end_ms:next_word_start_ms]
            
            is_pause = self.yamnet_filter.is_true_pause(
                gap_audio, 
                self.noise_silence_threshold,
                self.pause_speech_threshold
            )
            
            return is_pause
            
        except Exception as e:
            print(f"    ⚠ Error validating pause: {e}, falling back to timestamp")
            return True
    
    def validate_buffer_zone(self, audio: AudioSegment, start_ms: int, end_ms: int) -> bool:
        if not self.yamnet_filter or not self.yamnet_filter.model:
            return True
        
        if end_ms <= start_ms or start_ms < 0 or end_ms > len(audio):
            return False
        
        try:
            buffer_audio = audio[start_ms:end_ms]
            if len(buffer_audio) < 20:
                return True
            
            is_safe = self.yamnet_filter.is_true_pause(
                buffer_audio,
                self.buffer_validation_threshold,
                self.pause_speech_threshold
            )
            
            return is_safe
            
        except Exception as e:
            print(f"    ⚠ Error validating buffer: {e}, skipping buffer")
            return False
    
    def get_validated_boundaries(self, audio: AudioSegment, start_ms: int, end_ms: int,
                                prev_word_end_ms: Optional[int], next_word_start_ms: Optional[int]) -> Tuple[int, int]:
        validated_start = start_ms
        validated_end = end_ms
        
        if self.word_start_buffer_ms > 0 and prev_word_end_ms is not None:
            requested_start = max(start_ms - self.word_start_buffer_ms, prev_word_end_ms)
            
            if requested_start < start_ms:
                if self.validate_buffer_zone(audio, requested_start, start_ms):
                    validated_start = requested_start
                else:
                    print(f"    ⚠ Start buffer rejected: contains speech/noise")
        
        if self.word_end_buffer_ms > 0 and next_word_start_ms is not None:
            requested_end = min(end_ms + self.word_end_buffer_ms, next_word_start_ms)
            
            if requested_end > end_ms:
                if self.validate_buffer_zone(audio, end_ms, requested_end):
                    validated_end = requested_end
                else:
                    print(f"    ⚠ End buffer rejected: contains speech/noise")
        
        if prev_word_end_ms is None and self.word_start_buffer_ms > 0:
            requested_start = max(0, start_ms - self.word_start_buffer_ms)
            if requested_start < start_ms:
                if self.validate_buffer_zone(audio, requested_start, start_ms):
                    validated_start = requested_start
        
        if next_word_start_ms is None and self.word_end_buffer_ms > 0:
            requested_end = min(len(audio), end_ms + self.word_end_buffer_ms)
            if requested_end > end_ms:
                if self.validate_buffer_zone(audio, end_ms, requested_end):
                    validated_end = requested_end
        
        return validated_start, validated_end
    
    def segment_by_pauses(self, words: List[Dict], audio: AudioSegment) -> List[Dict]:
        if not words:
            print("⚠ No words found for segmentation")
            return []
        
        mode_str = "hybrid (Whisper + YAMNet)" if self.hybrid_pause_detection else "timestamp-only"
        print(f"Segmenting by natural pauses ({mode_str})...")
        
        segments = []
        current_segment_words = []
        total_potential_pauses = 0
        validated_pauses = 0
        rejected_pauses = 0
        
        for i, word in enumerate(words):
            if 'start' not in word or 'end' not in word:
                print(f"⚠ Skipping word with missing timestamps: {word}")
                continue
                
            current_segment_words.append(word)
            
            should_split = False
            
            if i < len(words) - 1:
                next_word = words[i + 1]
                if 'start' in next_word and 'end' in word:
                    pause_duration = next_word['start'] - word['end']
                    
                    if pause_duration >= self.pause_threshold:
                        total_potential_pauses += 1
                        
                        word_end_ms = int(word['end'] * 1000)
                        next_word_start_ms = int(next_word['start'] * 1000)
                        
                        if self.is_pause_validated(audio, word_end_ms, next_word_start_ms, pause_duration):
                            should_split = True
                            validated_pauses += 1
                        else:
                            rejected_pauses += 1
                            print(f"  ✗ Rejected pause at {word['end']:.2f}s ({pause_duration:.3f}s) - contains speech/noise")
            
            if i == len(words) - 1:
                should_split = True
            
            if should_split and current_segment_words:
                first_word = current_segment_words[0]
                last_word = current_segment_words[-1]
                
                if 'start' not in first_word or 'end' not in last_word:
                    print("⚠ Skipping segment with invalid timestamps")
                    current_segment_words = []
                    continue
                
                confidences = [w.get('probability', 1.0) for w in current_segment_words if 'probability' in w]
                if not confidences:
                    avg_confidence = 1.0
                else:
                    avg_confidence = sum(confidences) / len(confidences)
                
                segment_text = ''.join([w.get('word', '') for w in current_segment_words]).strip()
                
                segment_duration_s = last_word['end'] - first_word['start']
                
                if self.validate_segment_text(segment_text) and segment_duration_s >= self.min_segment_length:
                    first_word_idx = words.index(current_segment_words[0])
                    last_word_idx = words.index(current_segment_words[-1])
                    
                    prev_word_end_ms = None
                    if first_word_idx > 0:
                        prev_word = words[first_word_idx - 1]
                        if 'end' in prev_word:
                            prev_word_end_ms = int(prev_word['end'] * 1000)
                    
                    next_word_start_ms = None
                    if last_word_idx < len(words) - 1:
                        next_word = words[last_word_idx + 1]
                        if 'start' in next_word:
                            next_word_start_ms = int(next_word['start'] * 1000)
                    
                    segments.append({
                        'start_ms': int(first_word['start'] * 1000),
                        'end_ms': int(last_word['end'] * 1000),
                        'duration': segment_duration_s,
                        'words': current_segment_words.copy(),
                        'text': segment_text,
                        'confidence': avg_confidence,
                        'prev_word_end_ms': prev_word_end_ms,
                        'next_word_start_ms': next_word_start_ms
                    })
                    print(f"✓ Created segment: {segment_duration_s:.2f}s, '{segment_text[:50]}...'")
                else:
                    print(f"⚠ Skipping segment: {segment_duration_s:.2f}s, text valid: {self.validate_segment_text(segment_text)}")
                
                current_segment_words = []
        
        print(f"✓ Created {len(segments)} segments from word timestamps")
        if self.hybrid_pause_detection and self.yamnet_filter and self.yamnet_filter.model:
            print(f"  Pause validation stats:")
            print(f"    - Potential pauses (by timestamp): {total_potential_pauses}")
            print(f"    - Validated (true silence): {validated_pauses}")
            print(f"    - Rejected (contains audio): {rejected_pauses}")
        
        return segments
    
    def is_noisy(self, audio_chunk: AudioSegment) -> bool:
        if not self.filter_noise or not self.yamnet_filter or not self.yamnet_filter.model:
            return False
        
        return self.yamnet_filter.is_silent(audio_chunk, self.noise_silence_threshold)

    def detect_cutoff(self, audio_chunk: AudioSegment, text: str) -> Tuple[bool, str]:
        if not self.check_cutoff or not self.yamnet_filter or not self.yamnet_filter.model:
            return False, ""

        start_check_duration = min(self.cutoff_boundary_ms, len(audio_chunk))
        end_check_duration = min(self.cutoff_boundary_ms, len(audio_chunk))

        start_chunk = audio_chunk[:start_check_duration] if start_check_duration > 0 else None
        end_chunk = audio_chunk[-end_check_duration:] if end_check_duration > 0 else None

        start_has_silence = self.yamnet_filter.is_silent(start_chunk,
                                                         self.cutoff_silence_threshold) if start_chunk else False
        end_has_silence = self.yamnet_filter.is_silent(end_chunk, self.cutoff_silence_threshold) if end_chunk else False

        if not start_has_silence and not end_has_silence:
            return True, "both"
        elif not start_has_silence:
            return True, "start"
        elif not end_has_silence:
            return True, "end"
        else:
            return False, ""
    
    def handle_cutoff_segment(self, audio_chunk: AudioSegment, text: str, cutoff_type: str, 
                            base_name: str, idx: int) -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
        if self.cutoff_action == "skip":
            print(f"  ⚠ Skipping segment {idx} due to cutoff words at {cutoff_type}")
            return None, None, None
        elif self.cutoff_action == "rename":
            cutoff_indicator = ""
            if cutoff_type == "start":
                cutoff_indicator = "_CUTSTART"
            elif cutoff_type == "end":
                cutoff_indicator = "_CUTEND"
            elif cutoff_type == "both":
                cutoff_indicator = "_CUTBOTH"
            
            chunk_name = f"{base_name}_{idx:04d}{cutoff_indicator}"
        else:
            chunk_name = f"{base_name}_{idx:04d}"
        
        audio_file = self.get_unique_filename(self.audio_dir, chunk_name, f".{self.export_format}")
        transcript_file = self.get_unique_filename(self.text_dir, chunk_name, ".txt")
        
        audio_chunk.export(
            audio_file,
            format=self.export_format,
            parameters=["-ar", str(audio_chunk.frame_rate)]
        )
        
        if not self.safe_write_transcript(transcript_file, text):
            print(f"  ⚠ Warning: Transcript save failed for cutoff segment {idx}")
        
        metadata = {
            'duration': len(audio_chunk) / 1000.0,
            'text_length': len(text.strip()),
            'sample_rate': audio_chunk.frame_rate,
            'cutoff_detected': True,
            'cutoff_type': cutoff_type
        }
        
        return str(audio_file), str(transcript_file), metadata
    
    def handle_overlength_segment(self, audio_chunk: AudioSegment, text: str,
                                 base_name: str, idx: int, sample_rate: int) -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
        if self.overlength_action == "skip":
            print(f"  ⚠ Skipping segment {idx} due to overlength")
            return None, None, None
        elif self.overlength_action == "rename":
            chunk_name = f"{base_name}_{idx:04d}_LONG"
        else:
            chunk_name = f"{base_name}_{idx:04d}"
        
        audio_file = self.get_unique_filename(self.audio_dir, chunk_name, f".{self.export_format}")
        transcript_file = self.get_unique_filename(self.text_dir, chunk_name, ".txt")
        
        audio_chunk.export(
            audio_file,
            format=self.export_format,
            parameters=["-ar", str(sample_rate)]
        )
        
        if not self.safe_write_transcript(transcript_file, text):
            print(f"  ⚠ Warning: Transcript save failed for overlength segment {idx}")
        
        metadata = {
            'duration': len(audio_chunk) / 1000.0,
            'text_length': len(text.strip()),
            'sample_rate': sample_rate,
            'overlength': True
        }
        
        return str(audio_file), str(transcript_file), metadata
    
    def get_unique_filename(self, base_path: Path, stem: str, suffix: str) -> Path:
        counter = 0
        path = base_path / f"{stem}{suffix}"
        while path.exists():
            counter += 1
            path = base_path / f"{stem}_{counter:03d}{suffix}"
        return path

    def trim_silence_from_audio(self, audio_chunk: AudioSegment) -> AudioSegment:
        if not self.trim_silence or len(audio_chunk) < 200:
            return audio_chunk

        seg_len = len(audio_chunk)

        start_trim = 0
        for i in range(0, seg_len, self.trim_chunk_ms):
            chunk = audio_chunk[i:i + self.trim_chunk_ms]
            if chunk.dBFS > self.trim_threshold_db:
                start_trim = max(0, i - self.trim_pad_ms)
                break

        end_trim = seg_len
        for i in range(seg_len, 0, -self.trim_chunk_ms):
            chunk = audio_chunk[max(0, i - self.trim_chunk_ms):i]
            if chunk.dBFS > self.trim_threshold_db:
                end_trim = min(seg_len, i + self.trim_pad_ms)
                break

        if start_trim >= end_trim:
            return audio_chunk

        trimmed_audio = audio_chunk[start_trim:end_trim]
        return trimmed_audio

    def process_file(self, audio_path: str, speaker_id: Optional[str] = None) -> Dict:
        audio_path = Path(audio_path)
        base_name = speaker_id or audio_path.stem
        
        print(f"\n{'='*60}")
        print(f"Processing: {audio_path.name}")
        print(f"Speaker: {base_name}")
        print(f"{'='*60}\n")
        
        print("Loading audio...")
        try:
            audio = AudioSegment.from_file(audio_path)
        except Exception as e:
            print(f"✗ Error loading audio: {e}")
            return {
                'total_segments': 0,
                'total_duration': 0,
                'avg_duration': 0,
                'metadata_file': None,
                'manifest_file': None
            }
        
        original_sample_rate = audio.frame_rate
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        print(f"✓ Loaded: {len(audio)/1000:.1f}s, {audio.frame_rate}Hz (original), {audio.channels} channel(s)\n")
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio.export(tmp.name, format='wav')
            temp_audio_path = tmp.name
        
        try:
            transcription_result = self.transcribe_full_audio(temp_audio_path, str(audio_path))
        finally:
            os.unlink(temp_audio_path)
        
        segments = self.segment_by_pauses(transcription_result['words'], audio)
        
        if not segments:
            print("⚠ No segments created")
            return {
                'total_segments': 0,
                'total_duration': 0,
                'avg_duration': 0,
                'metadata_file': None,
                'manifest_file': None
            }
        
        print(f"\nFiltering {len(segments)} segments...")
        filtered_segments = []
        filtered_music = 0
        filtered_noise = 0
        filtered_confidence = 0
        filtered_text = 0
        
        for seg in tqdm(segments, desc="Filtering"):
            chunk = audio[seg['start_ms']:seg['end_ms']]
            
            if self.yamnet_filter and self.yamnet_filter.model and self.filter_music:
                is_filtered, classification = self.yamnet_filter.is_music_or_mixed(
                    chunk,
                    self.music_threshold,
                    self.mixed_threshold
                )
                if is_filtered:
                    filtered_music += 1
                    print(f"  ⚠ Filtered music segment: {seg['duration']:.2f}s")
                    continue
            
            if self.is_noisy(chunk):
                filtered_noise += 1
                print(f"  ⚠ Filtered noisy/silent segment")
                continue
            
            if seg['confidence'] < self.min_confidence:
                filtered_confidence += 1
                print(f"  ⚠ Filtered low confidence segment: {seg['confidence']:.3f} < {self.min_confidence:.3f}")
                continue
            
            if not self.validate_segment_text(seg['text']):
                filtered_text += 1
                print(f"  ⚠ Filtered invalid text segment: '{seg['text']}'")
                continue
            
            filtered_segments.append(seg)
        
        print(f"\n✓ Kept {len(filtered_segments)} segments (out of {len(segments)} initial segments)")
        if filtered_music > 0:
            print(f"⚠ Filtered {filtered_music} segments (music/mixed)")
        if filtered_noise > 0:
            print(f"⚠ Filtered {filtered_noise} segments (noisy/silent)")
        if filtered_confidence > 0:
            print(f"⚠ Filtered {filtered_confidence} segments (low confidence < {self.min_confidence:.2f})")
        if filtered_text > 0:
            print(f"⚠ Filtered {filtered_text} segments (invalid text)")
        
        output_files = self.export_segments(
            audio, 
            filtered_segments,
            base_name, 
            original_sample_rate
        )
        
        total_duration = sum(m['duration'] for _, _, m in output_files)
        avg_duration = total_duration / len(output_files) if output_files else 0
        
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Segments: {len(output_files)}")
        print(f"Duration: {total_duration:.2f}s")
        print(f"Average: {avg_duration:.2f}s per segment")
        print(f"Sample Rate: {original_sample_rate}Hz (preserved)")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")
        
        return {
            'total_segments': len(output_files),
            'total_duration': total_duration,
            'avg_duration': avg_duration,
            'metadata_file': None,
            'manifest_file': None
        }

    def export_segments(
            self,
            audio: AudioSegment,
            segments: List[Dict],
            base_name: str,
            sample_rate: int
    ) -> List[Tuple[str, str, Dict]]:
        print(f"Exporting {len(segments)} segments at {sample_rate}Hz...")

        output_files = []

        for idx, seg in enumerate(tqdm(segments, desc="Exporting"), 1):
            if not self.validate_segment_text(seg['text']):
                print(f"⚠ Skipping segment {idx} due to invalid text")
                continue

            chunk_name = f"{base_name}_{idx:04d}"

            start_ms = seg['start_ms']
            end_ms = seg['end_ms']
            
            validated_start, validated_end = self.get_validated_boundaries(
                audio,
                start_ms,
                end_ms,
                seg.get('prev_word_end_ms'),
                seg.get('next_word_start_ms')
            )
            
            audio_chunk = audio[validated_start:validated_end]
            
            if self.trim_silence:
                audio_chunk = self.trim_silence_from_audio(audio_chunk)
            
            segment_duration = len(audio_chunk) / 1000.0

            if segment_duration > self.max_segment_length:
                print(f"  ⚠ Overlength segment detected: {segment_duration:.2f}s > {self.max_segment_length:.2f}s")
                audio_file, transcript_file, metadata = self.handle_overlength_segment(
                    audio_chunk, seg['text'], base_name, idx, sample_rate
                )
                
                if audio_file is None:
                    continue
                
                output_files.append((audio_file, transcript_file, metadata))
                continue

            has_cutoff, cutoff_type = self.detect_cutoff(audio_chunk, seg['text'])
            
            if has_cutoff:
                print(f"  ⚠ Cutoff detected at {cutoff_type} in segment {idx}")
                audio_file, transcript_file, metadata = self.handle_cutoff_segment(
                    audio_chunk, seg['text'], cutoff_type, base_name, idx
                )
                
                if audio_file is None:
                    continue
            else:
                audio_file = self.get_unique_filename(self.audio_dir, chunk_name, f".{self.export_format}")
                audio_chunk.export(
                    audio_file,
                    format=self.export_format,
                    parameters=["-ar", str(sample_rate)]
                )

                transcript_file = self.get_unique_filename(self.text_dir, chunk_name, ".txt")
                if not self.safe_write_transcript(transcript_file, seg['text']):
                    print(f"  ⚠ Warning: Transcript save failed for segment {idx}")

                metadata = {
                    'duration': len(audio_chunk) / 1000.0,
                    'text_length': len(seg['text'].strip()),
                    'sample_rate': sample_rate,
                    'cutoff_detected': False
                }

            output_files.append((str(audio_file), str(transcript_file), metadata))

        print(f"✓ Exported {len(output_files)} segments at original sample rate ({sample_rate}Hz)")
        return output_files
    
    def regenerate_folder_metadata(self):
        print(f"\nRegenerating metadata for entire folder: {self.output_dir}")
        
        audio_files = list(self.audio_dir.glob(f'*.{self.export_format}'))
        if not audio_files:
            print("⚠ No audio files found in output directory")
            return
        
        print(f"Found {len(audio_files)} audio files")
        
        segments = []
        for audio_file in audio_files:
            transcript_file = self.text_dir / f"{audio_file.stem}.txt"
            if not transcript_file.exists():
                print(f"⚠ Transcript not found for {audio_file.name}, skipping")
                continue
            
            try:
                audio = AudioSegment.from_file(audio_file)
                duration = len(audio) / 1000.0
            except Exception as e:
                print(f"⚠ Error loading {audio_file.name}: {e}, using 0 duration")
                duration = 0.0
            
            try:
                with open(transcript_file, 'rb') as f:
                    text = f.read().decode('utf-8').strip()
            except Exception as e:
                print(f"⚠ Error reading {transcript_file.name}: {e}, using empty text")
                text = ""
            
            segments.append({
                'audio': audio_file.name,
                'transcript': transcript_file.name,
                'duration': duration,
                'text_length': len(text),
                'sample_rate': audio.frame_rate if 'audio' in locals() else 0
            })
        
        if not segments:
            print("⚠ No valid segments found for metadata generation")
            return
        
        metadata = {
            'total_segments': len(segments),
            'language': self.language,
            'segments': segments
        }
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        manifest_path = self.output_dir / "manifest.txt"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            for segment in segments:
                f.write(f"{segment['audio']}|{segment['transcript']}\n")
        
        hf_metadata_path = self.output_dir / "metadata.csv"
        with open(hf_metadata_path, 'w', encoding='utf-8') as f:
            f.write("file_name,text\n")
            for segment in segments:
                transcript_file = self.text_dir / segment['transcript']
                try:
                    with open(transcript_file, 'rb') as tf:
                        text = tf.read().decode('utf-8').strip()
                    text = text.replace('"', '""')
                    f.write(f'"{segment["audio"]}","{text}"\n')
                except Exception as e:
                    print(f"⚠ Error reading {transcript_file}: {e}")
                    f.write(f'"{segment["audio"]}",""\n')
        
        print(f"✓ Regenerated metadata for {len(segments)} segments:")
        print(f"  - {metadata_path}")
        print(f"  - {manifest_path}")
        print(f"  - {hf_metadata_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--model', default='large')
    parser.add_argument('--language', default='cs')
    parser.add_argument('--output-dir', default='output')
    parser.add_argument('--speaker-id')
    parser.add_argument('--min-length', type=float, default=1.0)
    parser.add_argument('--max-length', type=float, default=15.0)
    parser.add_argument('--format', default='wav', choices=['wav', 'mp3'])
    parser.add_argument('--pause-threshold', type=float, default=0.5)
    parser.add_argument('--min-confidence', type=float, default=0.5)
    parser.add_argument('--filter-noise', action='store_true')
    parser.add_argument('--noise-silence-threshold', type=float, default=0.3)
    parser.add_argument('--filter-music', action='store_true', default=True)
    parser.add_argument('--music-threshold', type=float, default=0.3)
    parser.add_argument('--mixed-threshold', type=float, default=0.2)
    parser.add_argument('--word-start-buffer', type=int, default=250)
    parser.add_argument('--word-end-buffer', type=int, default=750)
    parser.add_argument('--check-cutoff', action='store_true', default=True)
    parser.add_argument('--cutoff-boundary-ms', type=int, default=50)
    parser.add_argument('--cutoff-silence-threshold', type=float, default=0.3)
    parser.add_argument('--cutoff-action', default='rename', choices=['rename', 'skip', 'include'])
    parser.add_argument('--hybrid-pause-detection', action='store_true', default=True)
    parser.add_argument('--pause-speech-threshold', type=float, default=0.2)
    parser.add_argument('--buffer-validation-threshold', type=float, default=0.3)
    parser.add_argument('--overlength-action', default='rename', choices=['rename', 'skip', 'include'])
    parser.add_argument('--trim-silence', action='store_true', default=True)
    parser.add_argument('--trim-threshold-db', type=float, default=-40.0)
    parser.add_argument('--trim-chunk-ms', type=int, default=10)
    parser.add_argument('--trim-pad-ms', type=int, default=150)
    parser.add_argument('--device', default=None, choices=['cuda', 'cpu'], 
                       help='Force device (cuda/cpu), auto-detect if not specified')
    
    args = parser.parse_args()
    
    # Display GPU info at startup
    print("\n" + "="*60)
    print("GPU CHECK")
    print("="*60)
    check_gpu()
    print("="*60 + "\n")
    
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
        check_cutoff=args.check_cutoff,
        cutoff_boundary_ms=args.cutoff_boundary_ms,
        cutoff_silence_threshold=args.cutoff_silence_threshold,
        cutoff_action=args.cutoff_action,
        hybrid_pause_detection=args.hybrid_pause_detection,
        pause_speech_threshold=args.pause_speech_threshold,
        buffer_validation_threshold=args.buffer_validation_threshold,
        overlength_action=args.overlength_action,
        trim_silence=args.trim_silence,
        trim_threshold_db=args.trim_threshold_db,
        trim_chunk_ms=args.trim_chunk_ms,
        trim_pad_ms=args.trim_pad_ms,
        device=args.device
    )
    
    input_path = Path(args.input)
    if input_path.is_file():
        splitter.process_file(str(input_path), args.speaker_id)
    elif input_path.is_dir():
        audio_files = list(input_path.glob('*.wav')) + list(input_path.glob('*.mp3')) + list(input_path.glob('*.flac'))
        for audio_file in audio_files:
            speaker_id = audio_file.stem if not args.speaker_id else args.speaker_id
            splitter.process_file(str(audio_file), speaker_id)
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return
    
    splitter.regenerate_folder_metadata()


if __name__ == '__main__':
    main()