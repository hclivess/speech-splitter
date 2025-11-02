#!/usr/bin/env python3
"""
Speech Splitter - Whisper-only segmentation with perfect audio/text alignment
No VAD - relies entirely on Whisper word timestamps for accurate boundaries
"""

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
    """YAMNet-based audio classifier to filter music and music+speech"""
    
    def __init__(self):
        self.model = None
        self.class_names = None
        self._load_model()
    
    def _load_model(self):
        """Load YAMNet model from TensorFlow Hub"""
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            
            # Suppress TensorFlow warnings
            tf.get_logger().setLevel('ERROR')
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            
            print("Loading YAMNet model...")
            self.model = hub.load('https://tfhub.dev/google/yamnet/1')
            
            # Load class names
            class_map_path = self.model.class_map_path().numpy()
            self.class_names = self._load_class_names(class_map_path)
            
            print("✓ YAMNet model loaded")
            
        except Exception as e:
            print(f"⚠ Failed to load YAMNet: {e}")
            print("⚠ Music filtering will be disabled")
            self.model = None
    
    def _load_class_names(self, class_map_csv):
        """Load YAMNet class names from CSV"""
        import csv
        class_names = []
        with open(class_map_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_names.append(row['display_name'])
        return class_names
    
    def classify_audio(self, audio_segment: AudioSegment) -> Dict[str, float]:
        """
        Classify audio segment and return probabilities for relevant classes
        
        Returns:
            Dict with keys: 'music', 'speech', 'music_speech_mixed'
        """
        if self.model is None:
            return {'music': 0.0, 'speech': 1.0, 'music_speech_mixed': 0.0}
        
        try:
            import tensorflow as tf
            
            # Convert AudioSegment to numpy array
            samples = np.array(audio_segment.get_array_of_samples())
            
            # Convert to float32 in range [-1, 1]
            if audio_segment.sample_width == 2:  # 16-bit
                samples = samples.astype(np.float32) / 32768.0
            elif audio_segment.sample_width == 4:  # 32-bit
                samples = samples.astype(np.float32) / 2147483648.0
            
            # Run YAMNet
            scores, embeddings, spectrogram = self.model(samples)
            
            # Average scores across all frames
            mean_scores = np.mean(scores.numpy(), axis=0)
            
            # Find relevant class indices
            music_classes = [i for i, name in enumerate(self.class_names) 
                           if 'music' in name.lower() and 'speech' not in name.lower()]
            speech_classes = [i for i, name in enumerate(self.class_names) 
                            if 'speech' in name.lower() and 'music' not in name.lower()]
            mixed_classes = [i for i, name in enumerate(self.class_names) 
                           if 'music' in name.lower() and 'speech' in name.lower()]
            
            # Calculate probabilities
            music_prob = float(np.max([mean_scores[i] for i in music_classes]) if music_classes else 0.0)
            speech_prob = float(np.max([mean_scores[i] for i in speech_classes]) if speech_classes else 0.0)
            mixed_prob = float(np.max([mean_scores[i] for i in mixed_classes]) if mixed_classes else 0.0)
            
            return {
                'music': music_prob,
                'speech': speech_prob,
                'music_speech_mixed': mixed_prob
            }
            
        except Exception as e:
            print(f"⚠ YAMNet classification error: {e}")
            return {'music': 0.0, 'speech': 1.0, 'music_speech_mixed': 0.0}
    
    def is_music_or_mixed(self, audio_segment: AudioSegment, 
                          music_threshold: float = 0.3,
                          mixed_threshold: float = 0.2) -> Tuple[bool, Dict]:
        """
        Check if audio is music or music+speech mixed
        
        Args:
            audio_segment: Audio to classify
            music_threshold: Threshold for pure music (default 0.3)
            mixed_threshold: Threshold for music+speech mixed (default 0.2)
            
        Returns:
            (is_filtered, classification_dict)
        """
        classification = self.classify_audio(audio_segment)
        
        is_music = classification['music'] > music_threshold
        is_mixed = classification['music_speech_mixed'] > mixed_threshold
        
        return (is_music or is_mixed), classification


class SpeechSplitter:
    """
    Whisper-only Speech Splitter with perfect audio/text alignment
    
    Output Structure:
        output_dir/
            speaker1/
                audio/
                    speaker1_0001.wav
                    speaker1_0002.wav
                transcripts/
                    speaker1_0001.txt
                    speaker1_0002.txt
                speaker1_metadata.json
                speaker1_manifest.txt
    """
    def __init__(
        self,
        whisper_model: str = "large",
        language: str = "cs",
        output_dir: str = "output",
        silence_padding_ms: int = 150,
        min_segment_length: float = 1.0,
        max_segment_length: float = 15.0,
        export_format: str = "wav",
        pause_threshold: float = 0.5,
        min_confidence: float = 0.5,
        filter_noise: bool = True,
        min_rms: float = 0.01,
        filter_music: bool = True,
        music_threshold: float = 0.3,
        mixed_threshold: float = 0.2,
        word_boundary_buffer_ms: int = 350  # New parameter for word boundary buffer
    ):
        """
        Initialize Speech Splitter with Whisper-only segmentation
        
        Args:
            whisper_model: Whisper model size
            language: Language code (cs, en, de, etc.)
            output_dir: Output directory
            silence_padding_ms: Silence to add after segments
            min_segment_length: Min segment duration (seconds)
            max_segment_length: Max segment duration (seconds)
            export_format: wav or mp3
            pause_threshold: Pause duration to split segments (seconds, default 0.5)
            min_confidence: Minimum transcription confidence (0.0-1.0)
            filter_noise: Enable noise filtering
            min_rms: Minimum RMS energy level
            filter_music: Enable YAMNet music filtering
            music_threshold: YAMNet threshold for music detection
            mixed_threshold: YAMNet threshold for music+speech detection
            word_boundary_buffer_ms: Buffer to add at end of segments for complete word capture
        """
        self.whisper_model = whisper_model
        self.language = language
        self.output_dir = Path(output_dir)
        self.silence_padding_ms = silence_padding_ms
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.export_format = export_format
        self.pause_threshold = pause_threshold
        self.min_confidence = min_confidence
        self.filter_noise = filter_noise
        self.min_rms = min_rms
        self.filter_music = filter_music
        self.music_threshold = music_threshold
        self.mixed_threshold = mixed_threshold
        self.word_boundary_buffer_ms = word_boundary_buffer_ms
        
        # Create main output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Whisper
        print(f"Loading Whisper model ({whisper_model})...")
        self.model = whisper.load_model(whisper_model)
        
        # Load YAMNet if music filtering enabled
        if self.filter_music:
            self.yamnet_filter = YAMNetFilter()
        else:
            self.yamnet_filter = None
            print("⚠ Music filtering disabled")
        
        print("✓ Models loaded\n")
    
    def validate_segment_text(self, text: str) -> bool:
        """Validate if segment text is suitable for output"""
        if not text or not text.strip():
            return False
        if len(text.strip()) < 3:
            return False
        # Add more validation if needed
        return True
    
    def transcribe_full_audio(self, audio_path: str) -> Dict:
        """
        Transcribe entire audio file with Whisper and get word-level timestamps
        Returns full transcription with word details
        """
        print("Transcribing full audio with Whisper...")
        
        result = self.model.transcribe(
            audio_path,
            language=self.language,
            word_timestamps=True,
            verbose=False
        )
        
        # Extract all words with timestamps
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
    
    def segment_by_pauses(self, words: List[Dict]) -> List[Dict]:
        """
        Create segments from words based on natural pauses
        Uses word timestamps to detect pauses and group words into segments
        
        Args:
            words: List of word dicts with 'start', 'end', 'word', 'probability'
            
        Returns:
            List of segment dicts with 'start_ms', 'end_ms', 'words', 'text', 'confidence'
        """
        if not words:
            print("⚠ No words found for segmentation")
            return []
        
        print("Segmenting by natural pauses...")
        
        segments = []
        current_segment_words = []
        
        for i, word in enumerate(words):
            # Skip words with missing timestamps
            if 'start' not in word or 'end' not in word:
                print(f"⚠ Skipping word with missing timestamps: {word}")
                continue
                
            current_segment_words.append(word)
            
            # Decide if we should end the segment
            should_split = False
            
            # Check if next word has a significant pause
            if i < len(words) - 1:
                next_word = words[i + 1]
                if 'start' in next_word and 'end' in word:
                    pause_duration = next_word['start'] - word['end']
                    if pause_duration >= self.pause_threshold:
                        should_split = True
            
            # Check if segment is too long (even if no pause)
            if current_segment_words:
                first_word = current_segment_words[0]
                if 'start' in first_word:
                    segment_duration = word['end'] - first_word['start']
                    if segment_duration >= self.max_segment_length:
                        should_split = True
            
            # Last word always ends segment
            if i == len(words) - 1:
                should_split = True
            
            if should_split and current_segment_words:
                # Create segment
                first_word = current_segment_words[0]
                last_word = current_segment_words[-1]
                
                # Validate timestamps
                if 'start' not in first_word or 'end' not in last_word:
                    print("⚠ Skipping segment with invalid timestamps")
                    current_segment_words = []
                    continue
                
                # Calculate confidence
                confidences = [w.get('probability', 1.0) for w in current_segment_words if 'probability' in w]
                if not confidences:
                    avg_confidence = 1.0
                else:
                    avg_confidence = sum(confidences) / len(confidences)
                
                # Build text - words already have spaces
                segment_text = ''.join([w.get('word', '') for w in current_segment_words]).strip()
                
                segment_duration_s = last_word['end'] - first_word['start']
                
                # Validate text and length before adding segment
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
                
                # Reset for next segment
                current_segment_words = []
        
        print(f"✓ Created {len(segments)} segments from word timestamps")
        return segments
    
    def is_noisy(self, audio_chunk: AudioSegment) -> bool:
        """Check if audio chunk is too noisy/low quality using RMS"""
        if not self.filter_noise:
            return False
        
        samples = np.array(audio_chunk.get_array_of_samples())
        samples_float = samples.astype(np.float32)
        
        # Normalize to [-1, 1] range
        if audio_chunk.sample_width == 2:  # 16-bit
            samples_float = samples_float / 32768.0
        elif audio_chunk.sample_width == 4:  # 32-bit
            samples_float = samples_float / 2147483648.0
        
        # Calculate RMS
        rms = np.sqrt(np.mean(samples_float**2))
        
        is_noisy = rms < self.min_rms
        if is_noisy:
            print(f"  ⚠ Noisy segment detected: RMS={rms:.4f} < {self.min_rms:.4f}")
        
        return is_noisy
    
    def process_file(self, audio_path: str, speaker_id: Optional[str] = None) -> Dict:
        """
        Process audio file: transcribe with Whisper, segment by word timestamps, export
        
        Whisper-only approach:
        1. Transcribe entire audio with Whisper (word timestamps)
        2. Segment based on natural pauses between words
        3. YAMNet music filter (optional)
        4. Noise filter (optional)
        5. Confidence filter
        6. Export with perfect audio/text alignment
        """
        audio_path = Path(audio_path)
        base_name = speaker_id or audio_path.stem
        
        print(f"\n{'='*60}")
        print(f"Processing: {audio_path.name}")
        print(f"Speaker: {base_name}")
        print(f"{'='*60}\n")
        
        # Setup output directories
        file_output_dir = self.output_dir / base_name
        audio_dir = file_output_dir / "audio"
        text_dir = file_output_dir / "transcripts"
        
        audio_dir.mkdir(parents=True, exist_ok=True)
        text_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Load and preprocess audio
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
        
        # Convert to mono and 16kHz for Whisper
        if audio.channels > 1:
            audio = audio.set_channels(1)
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        
        print(f"✓ Loaded: {len(audio)/1000:.1f}s, {audio.frame_rate}Hz, {audio.channels} channel(s)\n")
        
        # Step 2: Transcribe entire audio with Whisper
        # Export audio to temp file for Whisper
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio.export(tmp.name, format='wav')
            temp_audio_path = tmp.name
        
        try:
            transcription_result = self.transcribe_full_audio(temp_audio_path)
        finally:
            os.unlink(temp_audio_path)
        
        # Debug information
        print(f"Transcription result keys: {list(transcription_result.keys())}")
        print(f"Number of words: {len(transcription_result.get('words', []))}")
        if transcription_result.get('words'):
            print(f"Sample words: {transcription_result.get('words', [])[:3]}")
        
        # Step 3: Segment by word timestamps
        segments = self.segment_by_pauses(transcription_result['words'])
        
        if not segments:
            print("⚠ No segments created")
            return {
                'total_segments': 0,
                'total_duration': 0,
                'avg_duration': 0,
                'metadata_file': None,
                'manifest_file': None
            }
        
        # Step 4: Filter segments
        print(f"\nFiltering {len(segments)} segments...")
        filtered_segments = []
        filtered_music = 0
        filtered_noise = 0
        filtered_confidence = 0
        filtered_text = 0
        
        for seg in tqdm(segments, desc="Filtering"):
            # Extract audio chunk for filtering
            chunk = audio[seg['start_ms']:seg['end_ms']]
            
            # Music filter
            if self.yamnet_filter and self.yamnet_filter.model:
                is_filtered, classification = self.yamnet_filter.is_music_or_mixed(
                    chunk,
                    self.music_threshold,
                    self.mixed_threshold
                )
                if is_filtered:
                    filtered_music += 1
                    print(f"  ⚠ Filtered music segment: {seg['duration']:.2f}s")
                    continue
            
            # Noise filter
            if self.is_noisy(chunk):
                filtered_noise += 1
                continue
            
            # Confidence filter
            if seg['confidence'] < self.min_confidence:
                filtered_confidence += 1
                print(f"  ⚠ Filtered low confidence segment: {seg['confidence']:.3f} < {self.min_confidence:.3f}")
                continue
            
            # Text validation (double-check)
            if not self.validate_segment_text(seg['text']):
                filtered_text += 1
                print(f"  ⚠ Filtered invalid text segment: '{seg['text']}'")
                continue
            
            filtered_segments.append(seg)
        
        print(f"\n✓ Kept {len(filtered_segments)} segments")
        if filtered_music > 0:
            print(f"⚠ Filtered {filtered_music} segments (music/mixed)")
        if filtered_noise > 0:
            print(f"⚠ Filtered {filtered_noise} segments (noisy, RMS < {self.min_rms:.3f})")
        if filtered_confidence > 0:
            print(f"⚠ Filtered {filtered_confidence} segments (low confidence < {self.min_confidence:.2f})")
        if filtered_text > 0:
            print(f"⚠ Filtered {filtered_text} segments (invalid text)")
        
        # Step 5: Export audio and text files
        output_files = self.export_segments(audio, filtered_segments, base_name, audio_dir, text_dir)
        
        # Generate metadata
        metadata = {
            'source_file': str(audio_path.name),
            'speaker_id': speaker_id,
            'language': self.language,
            'total_segments': len(output_files),
            'segments': [
                {
                    'audio': Path(audio).name,
                    'transcript': Path(transcript).name,
                    **meta
                }
                for audio, transcript, meta in output_files
            ]
        }
        
        metadata_path = file_output_dir / f"{base_name}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Generate manifest
        manifest_path = file_output_dir / f"{base_name}_manifest.txt"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            for audio_file, transcript_file, _ in output_files:
                f.write(f"{Path(audio_file).name}|{Path(transcript_file).name}\n")
        
        # Print summary
        total_duration = sum(m['duration'] for _, _, m in output_files)
        avg_duration = total_duration / len(output_files) if output_files else 0
        
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Segments: {len(output_files)}")
        print(f"Duration: {total_duration:.2f}s")
        print(f"Average: {avg_duration:.2f}s per segment")
        print(f"Output: {file_output_dir}")
        print(f"{'='*60}\n")
        
        return {
            'total_segments': len(output_files),
            'total_duration': total_duration,
            'avg_duration': avg_duration,
            'metadata_file': str(metadata_path),
            'manifest_file': str(manifest_path)
        }

    def export_segments(
            self,
            audio: AudioSegment,
            segments: List[Dict],
            base_name: str,
            audio_dir: Path,
            text_dir: Path
    ) -> List[Tuple[str, str, Dict]]:
        """Export audio chunks and transcripts to per-file directories"""
        print(f"Exporting {len(segments)} segments...")

        silence = AudioSegment.silent(duration=self.silence_padding_ms)
        output_files = []

        for idx, seg in enumerate(tqdm(segments, desc="Exporting"), 1):
            # Final text validation
            if not self.validate_segment_text(seg['text']):
                print(f"⚠ Skipping segment {idx} due to invalid text")
                continue

            chunk_name = f"{base_name}_{idx:04d}"

            # Extract audio using Whisper word timestamps with buffer for complete word capture
            start_ms = int(seg['start_ms'])
            end_ms = int(seg['end_ms'])

            # Add buffer to ensure complete word capture at the end
            buffered_end_ms = min(end_ms + self.word_boundary_buffer_ms, len(audio))
            audio_chunk = audio[start_ms:buffered_end_ms]

            # Add configured silence padding at the end
            audio_chunk = audio_chunk + silence

            # NEW: Re-transcribe the filtered audio segment to get accurate text
            final_text = seg['text'].strip()  # Default to original text

            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                temp_chunk_path = tmp.name
                # Export the exact same audio chunk that will be saved - no conversion needed
                audio_chunk.export(temp_chunk_path, format='wav')

            try:
                # Re-transcribe this specific segment to get text matching the actual audio
                retranscription = self.model.transcribe(
                    temp_chunk_path,
                    language=self.language,
                    word_timestamps=False,
                    verbose=False
                )
                retranscribed_text = retranscription['text'].strip()

                # Only use re-transcribed text if it's valid
                if self.validate_segment_text(retranscribed_text):
                    final_text = retranscribed_text
                    print(f"  ℹ Re-transcribed segment {idx}: '{seg['text'][:30]}...' → '{final_text[:30]}...'")
                else:
                    print(f"  ℹ Using original transcription for segment {idx} (re-transcription invalid)")
            except Exception as e:
                print(f"  ⚠ Re-transcription failed for segment {idx}: {e}")
            finally:
                # Clean up temp file
                os.unlink(temp_chunk_path)

            # Export audio (this is the unchanged, original audio chunk)
            audio_file = audio_dir / f"{chunk_name}.{self.export_format}"
            audio_chunk.export(
                audio_file,
                format=self.export_format,
                parameters=["-ar", "22050"]
            )

            # Export transcript with the final text (either re-transcribed or original)
            transcript_file = text_dir / f"{chunk_name}.txt"
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(final_text)

            metadata = {
                'duration': seg['duration'],
                'text_length': len(final_text)
            }

            output_files.append((str(audio_file), str(transcript_file), metadata))
            print(f"  ✓ Exported: {chunk_name} ({seg['duration']:.2f}s)")

        print(f"✓ Exported {len(output_files)} segments")
        return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Speech Splitter - Whisper-only with perfect alignment"
    )
    parser.add_argument('input', help='Input audio file')
    parser.add_argument('--model', default='large', help='Whisper model size')
    parser.add_argument('--language', default='cs', help='Language code')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--speaker-id', help='Speaker ID')
    parser.add_argument('--silence-padding', type=int, default=150, help='Silence padding (ms)')
    parser.add_argument('--min-length', type=float, default=1.0, help='Min segment length (s)')
    parser.add_argument('--max-length', type=float, default=15.0, help='Max segment length (s)')
    parser.add_argument('--format', default='wav', choices=['wav', 'mp3'], help='Output format')
    parser.add_argument('--pause-threshold', type=float, default=0.5, help='Pause threshold for segmentation (s)')
    parser.add_argument('--min-confidence', type=float, default=0.5, help='Min transcription confidence (0.0-1.0)')
    parser.add_argument('--filter-noise', action='store_true', help='Enable noise filtering')
    parser.add_argument('--min-rms', type=float, default=0.01, help='Min RMS energy (default 0.01)')
    parser.add_argument('--filter-music', action='store_true', default=True, help='Enable YAMNet music filtering')
    parser.add_argument('--music-threshold', type=float, default=0.3, help='Music detection threshold (default 0.3)')
    parser.add_argument('--mixed-threshold', type=float, default=0.2, help='Music+speech detection threshold (default 0.2)')
    parser.add_argument('--word-buffer', type=int, default=350, help='Word boundary buffer (ms, default 350)')
    
    args = parser.parse_args()
    
    splitter = SpeechSplitter(
        whisper_model=args.model,
        language=args.language,
        output_dir=args.output_dir,
        silence_padding_ms=args.silence_padding,
        min_segment_length=args.min_length,
        max_segment_length=args.max_length,
        export_format=args.format,
        pause_threshold=args.pause_threshold,
        min_confidence=args.min_confidence,
        filter_noise=args.filter_noise,
        min_rms=args.min_rms,
        filter_music=args.filter_music,
        music_threshold=args.music_threshold,
        mixed_threshold=args.mixed_threshold,
        word_boundary_buffer_ms=args.word_buffer
    )
    
    splitter.process_file(args.input, args.speaker_id)


if __name__ == '__main__':
    main()