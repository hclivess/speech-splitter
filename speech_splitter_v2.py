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


class YAMNetFilter:
    
    def __init__(self):
        self.model = None
        self.class_names = None
        self._load_model()
    
    def _load_model(self):
        try:
            import tensorflow as tf
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
        """Detect if audio segment contains significant silence"""
        classification = self.classify_audio(audio_segment)
        return classification['silence'] > silence_threshold
    
    def is_true_pause(self, audio_segment: AudioSegment,
                     silence_threshold: float = 0.3,
                     speech_threshold: float = 0.2) -> bool:
        """
        Determine if an audio segment is a true pause/silence.
        Returns True if it's mostly silence and not speech/music.
        """
        if len(audio_segment) < 50:  # Too short to analyze reliably
            return True
        
        classification = self.classify_audio(audio_segment)
        
        # It's a true pause if:
        # 1. High silence score, OR
        # 2. Low speech AND low music scores
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
        pause_speech_threshold: float = 0.2
    ):
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
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir = self.output_dir / "audio"
        self.text_dir = self.output_dir / "transcripts"
        self.audio_dir.mkdir(exist_ok=True)
        self.text_dir.mkdir(exist_ok=True)
        
        print(f"Loading Whisper model ({whisper_model})...")
        self.model = whisper.load_model(whisper_model)
        
        if self.filter_music or self.filter_noise or self.check_cutoff or self.hybrid_pause_detection:
            self.yamnet_filter = YAMNetFilter()
            if self.hybrid_pause_detection and self.yamnet_filter.model:
                print("✓ Hybrid pause detection enabled (Whisper + YAMNet)")
                print("  All pauses will be validated with audio analysis")
        else:
            self.yamnet_filter = None
            print("⚠ Audio filtering disabled")
        
        print("✓ Models loaded\n")
    
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
    
    def transcribe_full_audio(self, audio_path: str) -> Dict:
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
        return {
            'text': result['text'],
            'words': all_words,
            'segments': result.get('segments', [])
        }
    
    def is_pause_validated(self, audio: AudioSegment, word_end_ms: int, next_word_start_ms: int, 
                          pause_duration: float) -> bool:
        """
        Validate ALL pauses with YAMNet audio analysis.
        First checks timestamp threshold, then validates with actual audio.
        """
        # First check: timestamp-based
        if pause_duration < self.pause_threshold:
            return False
        
        # If hybrid detection is disabled or YAMNet not available, rely on timestamps only
        if not self.hybrid_pause_detection or not self.yamnet_filter or not self.yamnet_filter.model:
            return True
        
        # Second check: validate with audio analysis
        try:
            gap_audio = audio[word_end_ms:next_word_start_ms]
            
            # Use YAMNet to verify it's actually a pause
            is_pause = self.yamnet_filter.is_true_pause(
                gap_audio, 
                self.noise_silence_threshold,
                self.pause_speech_threshold
            )
            
            return is_pause
            
        except Exception as e:
            print(f"    ⚠ Error validating pause: {e}, falling back to timestamp")
            return True  # Fall back to accepting the pause if analysis fails
    
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
                    
                    # Check if this might be a pause based on timestamp
                    if pause_duration >= self.pause_threshold:
                        total_potential_pauses += 1
                        
                        # Validate with audio analysis
                        word_end_ms = int(word['end'] * 1000)
                        next_word_start_ms = int(next_word['start'] * 1000)
                        
                        if self.is_pause_validated(audio, word_end_ms, next_word_start_ms, pause_duration):
                            should_split = True
                            validated_pauses += 1
                        else:
                            rejected_pauses += 1
                            print(f"  ✗ Rejected pause at {word['end']:.2f}s ({pause_duration:.3f}s) - contains speech/noise")
            
            # Always split at the end
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
                    segments.append({
                        'start_ms': int(first_word['start'] * 1000),
                        'end_ms': int(last_word['end'] * 1000),
                        'duration': segment_duration_s,
                        'words': current_segment_words.copy(),
                        'text': segment_text,
                        'confidence': avg_confidence
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
        """Detect if audio chunk is noisy/silent using YAMNet"""
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
        """
        Handle segments with cutoff words based on cutoff_action setting.
        Returns (audio_file_path, transcript_file_path, metadata) or (None, None, None) if skipped
        """
        if self.cutoff_action == "skip":
            print(f"  ⚠ Skipping segment {idx} due to cutoff words at {cutoff_type}")
            return None, None, None
        elif self.cutoff_action == "rename":
            # Add cutoff indicator to filename
            cutoff_indicator = ""
            if cutoff_type == "start":
                cutoff_indicator = "_CUTSTART"
            elif cutoff_type == "end":
                cutoff_indicator = "_CUTEND"
            elif cutoff_type == "both":
                cutoff_indicator = "_CUTBOTH"
            
            chunk_name = f"{base_name}_{idx:04d}{cutoff_indicator}"
        else:  # include - no special handling
            chunk_name = f"{base_name}_{idx:04d}"
        
        # Generate unique filename to avoid conflicts
        audio_file = self.get_unique_filename(self.audio_dir, chunk_name, f".{self.export_format}")
        transcript_file = self.get_unique_filename(self.text_dir, chunk_name, ".txt")
        
        # Export audio
        audio_chunk.export(
            audio_file,
            format=self.export_format,
            parameters=["-ar", str(audio_chunk.frame_rate)]
        )
        
        # Write transcript
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(text.strip())
        
        metadata = {
            'duration': len(audio_chunk) / 1000.0,
            'text_length': len(text.strip()),
            'sample_rate': audio_chunk.frame_rate,
            'cutoff_detected': True,
            'cutoff_type': cutoff_type
        }
        
        return str(audio_file), str(transcript_file), metadata
    
    def get_unique_filename(self, base_path: Path, stem: str, suffix: str) -> Path:
        """Generate a unique filename by adding a counter if file exists"""
        counter = 0
        path = base_path / f"{stem}{suffix}"
        while path.exists():
            counter += 1
            path = base_path / f"{stem}_{counter:03d}{suffix}"
        return path
    
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
            transcription_result = self.transcribe_full_audio(temp_audio_path)
        finally:
            os.unlink(temp_audio_path)
        
        print(f"Transcription result keys: {list(transcription_result.keys())}")
        print(f"Number of words: {len(transcription_result.get('words', []))}")
        if transcription_result.get('words'):
            print(f"Sample words: {transcription_result.get('words', [])[:3]}")
        
        # Pass audio to segment_by_pauses for hybrid detection
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
            'metadata_file': None,  # Will be generated for entire folder
            'manifest_file': None   # Will be generated for entire folder
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

            start_ms = int(seg['start_ms'])
            end_ms = int(seg['end_ms'])
            
            buffered_start_ms = max(start_ms - self.word_start_buffer_ms, 0)
            buffered_end_ms = min(end_ms + self.word_end_buffer_ms, len(audio))
            audio_chunk = audio[buffered_start_ms:buffered_end_ms]

            # Cutoff detection - final processing step after segment isolation
            has_cutoff, cutoff_type = self.detect_cutoff(audio_chunk, seg['text'])
            
            if has_cutoff:
                print(f"  ⚠ Cutoff detected at {cutoff_type} in segment {idx}")
                audio_file, transcript_file, metadata = self.handle_cutoff_segment(
                    audio_chunk, seg['text'], cutoff_type, base_name, idx
                )
                
                # If segment was skipped due to cutoff action, continue to next segment
                if audio_file is None:
                    continue
            else:
                # No cutoff detected, export normally
                audio_file = self.get_unique_filename(self.audio_dir, chunk_name, f".{self.export_format}")
                audio_chunk.export(
                    audio_file,
                    format=self.export_format,
                    parameters=["-ar", str(sample_rate)]
                )

                transcript_file = self.get_unique_filename(self.text_dir, chunk_name, ".txt")
                with open(transcript_file, 'w', encoding='utf-8') as f:
                    f.write(seg['text'].strip())

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
        """Regenerate metadata for the entire folder content"""
        print(f"\nRegenerating metadata for entire folder: {self.output_dir}")
        
        # Get all audio files in the audio directory
        audio_files = list(self.audio_dir.glob(f'*.{self.export_format}'))
        if not audio_files:
            print("⚠ No audio files found in output directory")
            return
        
        print(f"Found {len(audio_files)} audio files")
        
        # Build segment list from existing files
        segments = []
        for audio_file in audio_files:
            # Find corresponding transcript file
            transcript_file = self.text_dir / f"{audio_file.stem}.txt"
            if not transcript_file.exists():
                print(f"⚠ Transcript not found for {audio_file.name}, skipping")
                continue
            
            # Load audio to get duration
            try:
                audio = AudioSegment.from_file(audio_file)
                duration = len(audio) / 1000.0  # Convert to seconds
            except Exception as e:
                print(f"⚠ Error loading {audio_file.name}: {e}, using 0 duration")
                duration = 0.0
            
            # Read transcript text
            try:
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
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
        
        # Generate metadata.json
        metadata = {
            'total_segments': len(segments),
            'language': self.language,
            'segments': segments
        }
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Generate manifest.txt
        manifest_path = self.output_dir / "manifest.txt"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            for segment in segments:
                f.write(f"{segment['audio']}|{segment['transcript']}\n")
        
        # Generate metadata.csv for HuggingFace
        hf_metadata_path = self.output_dir / "metadata.csv"
        with open(hf_metadata_path, 'w', encoding='utf-8') as f:
            f.write("file_name,text\n")
            for segment in segments:
                transcript_file = self.text_dir / segment['transcript']
                try:
                    with open(transcript_file, 'r', encoding='utf-8') as tf:
                        text = tf.read().strip()
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
    parser.add_argument('--hybrid-pause-detection', action='store_true', default=True,
                       help='Enable hybrid pause detection (Whisper + YAMNet)')
    parser.add_argument('--pause-speech-threshold', type=float, default=0.2,
                       help='Speech probability threshold for pause validation (lower = stricter)')
    
    args = parser.parse_args()
    
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
        pause_speech_threshold=args.pause_speech_threshold
    )
    
    # Handle both single file and directory input
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
    
    # Regenerate metadata for the entire folder content
    splitter.regenerate_folder_metadata()


if __name__ == '__main__':
    main()