#!/usr/bin/env python3
"""
Speech Splitter - Split audio using VAD, then transcribe chunks
Proper approach: Audio segmentation FIRST, then transcription
Now with YAMNet music filtering
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
from pydub.silence import detect_nonsilent
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
            
            # YAMNet expects mono 16kHz audio
            if audio_segment.channels > 1:
                # Already handled in preprocessing, but double-check
                pass
            
            # Resample to 16kHz if needed
            if audio_segment.frame_rate != 16000:
                # This should already be done, but as safety
                pass
            
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


class SentenceSplitter:
    """Handles sentence segmentation with proper spaCy support"""
    
    def __init__(self, language: str = "cs"):
        self.language = language
        self.nlp = None
        self._load_model()
    
    def _load_model(self):
        """Load spaCy model - proper implementation for Czech"""
        try:
            if self.language == "cs":
                # Try spacy-udpipe for Czech (proper way)
                try:
                    import spacy_udpipe
                    spacy_udpipe.download("cs")
                    self.nlp = spacy_udpipe.load("cs")
                    print(f"✓ Loaded Czech spaCy model via udpipe")
                    return
                except ImportError:
                    print("⚠ spacy-udpipe not installed, trying alternatives...")
            
            # Try standard spaCy models for other languages
            import spacy
            model_map = {
                'en': 'en_core_web_sm',
                'de': 'de_core_news_sm',
                'es': 'es_core_news_sm',
                'fr': 'fr_core_news_sm',
            }
            
            model_name = model_map.get(self.language)
            if model_name:
                try:
                    self.nlp = spacy.load(model_name)
                    print(f"✓ Loaded {model_name}")
                    return
                except OSError:
                    pass
            
            # Fallback: blank model with sentencizer
            import spacy
            self.nlp = spacy.blank(self.language)
            self.nlp.add_pipe("sentencizer")
            print(f"✓ Using basic sentencizer for {self.language}")
            
        except Exception as e:
            print(f"⚠ spaCy loading failed: {e}")
            self.nlp = None
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        if not text.strip():
            return []
        
        # Use spaCy if available
        if self.nlp:
            try:
                doc = self.nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                if sentences:
                    return sentences
            except Exception as e:
                print(f"⚠ spaCy error: {e}, using fallback")
        
        # Fallback: simple regex-based splitting
        import re
        # Split on sentence endings with proper lookahead
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÁŽŠČŘĎŤŇŮĚÉ])', text)
        return [s.strip() for s in sentences if s.strip()]


class SpeechSplitter:
    """
    Speech Splitter with VAD, YAMNet music filtering, and quality filtering
    
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
            speaker2/
                audio/
                    speaker2_0001.wav
                transcripts/
                    speaker2_0001.txt
                speaker2_metadata.json
                speaker2_manifest.txt
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
        min_silence_len: int = 700,
        silence_thresh: int = -40,
        min_confidence: float = 0.5,
        filter_noise: bool = True,
        min_rms: float = 0.01,
        filter_music: bool = True,
        music_threshold: float = 0.3,
        mixed_threshold: float = 0.2
    ):
        """
        Initialize Speech Splitter with VAD-first approach and YAMNet music filtering
        
        Args:
            whisper_model: Whisper model size
            language: Language code (cs, en, de, etc.)
            output_dir: Output directory
            silence_padding_ms: Silence to add after segments
            min_segment_length: Min segment duration (seconds)
            max_segment_length: Max segment duration (seconds)
            export_format: wav or mp3
            min_silence_len: Min silence length for splitting (ms)
            silence_thresh: Silence threshold (dBFS)
            min_confidence: Minimum transcription confidence (0.0-1.0, default 0.5)
            filter_noise: Enable noise filtering
            min_rms: Minimum RMS energy level (default 0.01)
            filter_music: Enable YAMNet music filtering (default True)
            music_threshold: YAMNet threshold for music detection (default 0.3)
            mixed_threshold: YAMNet threshold for music+speech detection (default 0.2)
        """
        self.whisper_model = whisper_model
        self.language = language
        self.output_dir = Path(output_dir)
        self.silence_padding_ms = silence_padding_ms
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.export_format = export_format
        self.min_silence_len = min_silence_len
        self.silence_thresh = silence_thresh
        self.min_confidence = min_confidence
        self.filter_noise = filter_noise
        self.min_rms = min_rms
        self.filter_music = filter_music
        self.music_threshold = music_threshold
        self.mixed_threshold = mixed_threshold
        
        # Create main output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Whisper
        print(f"Loading Whisper model ({whisper_model})...")
        self.model = whisper.load_model(whisper_model)
        
        # Load sentence splitter
        print(f"Loading sentence splitter for {language}...")
        self.sentence_splitter = SentenceSplitter(language)
        
        # Load YAMNet if music filtering enabled
        if self.filter_music:
            self.yamnet_filter = YAMNetFilter()
        else:
            self.yamnet_filter = None
            print("⚠ Music filtering disabled")
        
        print("✓ Models loaded\n")
    
    def split_audio_by_silence(self, audio: AudioSegment) -> List[Tuple[int, int]]:
        """
        Split audio using Voice Activity Detection (silence detection)
        Returns list of (start_ms, end_ms) tuples
        """
        print("Detecting speech segments using VAD...")
        
        # Detect non-silent chunks
        nonsilent_ranges = detect_nonsilent(
            audio,
            min_silence_len=self.min_silence_len,
            silence_thresh=self.silence_thresh,
            seek_step=10
        )
        
        if not nonsilent_ranges:
            # If no silence detected, treat whole audio as one segment
            return [(0, len(audio))]
        
        # Merge segments that are too short
        merged_ranges = []
        for start_ms, end_ms in nonsilent_ranges:
            duration_s = (end_ms - start_ms) / 1000.0
            
            if duration_s < self.min_segment_length:
                # Too short, merge with previous if possible
                if merged_ranges:
                    merged_ranges[-1] = (merged_ranges[-1][0], end_ms)
                else:
                    merged_ranges.append((start_ms, end_ms))
            else:
                merged_ranges.append((start_ms, end_ms))
        
        # Split segments that are too long
        final_ranges = []
        for start_ms, end_ms in merged_ranges:
            duration_s = (end_ms - start_ms) / 1000.0
            
            if duration_s > self.max_segment_length:
                # Split long segment into chunks
                chunk_duration_ms = int(self.max_segment_length * 1000)
                current = start_ms
                while current < end_ms:
                    chunk_end = min(current + chunk_duration_ms, end_ms)
                    final_ranges.append((current, chunk_end))
                    current = chunk_end
            else:
                final_ranges.append((start_ms, end_ms))
        
        print(f"✓ Detected {len(final_ranges)} speech segments")
        return final_ranges
    
    def transcribe_segment(self, audio_chunk: AudioSegment) -> Dict:
        """
        Transcribe audio segment using Whisper with word-level timestamps
        Returns dict with text, words, and confidence
        """
        # Export to temporary file for Whisper
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_chunk.export(tmp.name, format='wav')
            tmp_path = tmp.name
        
        try:
            # Transcribe with word-level timestamps
            result = self.model.transcribe(
                tmp_path,
                language=self.language,
                word_timestamps=True,
                verbose=False
            )
            
            text = result['text'].strip()
            
            # Extract word-level info if available
            words = []
            if 'segments' in result:
                for segment in result['segments']:
                    if 'words' in segment:
                        words.extend(segment['words'])
            
            # Calculate average confidence
            confidence = 1.0
            if words:
                confidences = [w.get('probability', 1.0) for w in words]
                confidence = sum(confidences) / len(confidences)
            
            return {
                'text': text,
                'words': words,
                'confidence': confidence
            }
        finally:
            os.unlink(tmp_path)
    
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
        
        return rms < self.min_rms
    
    def process_file(self, audio_path: str, speaker_id: Optional[str] = None) -> Dict:
        """
        Process audio file: split, transcribe, and export segments
        
        Optimized filtering order to maximize efficiency:
        1. VAD segmentation (splits audio by silence)
        2. Length pre-filter (instant, filters too-short segments)
        3. YAMNet music filter (medium cost, saves transcription work)
        4. Noise + Transcription + Confidence filters (expensive, done last)
        5. Export exact same audio that was transcribed
        
        Audio/Text alignment:
        - Uses 300ms buffer for both transcription AND export
        - Guarantees perfect synchronization (no audio/text mismatch)
        - Same audio chunk transcribed = same audio chunk exported
        """
        audio_path = Path(audio_path)
        base_name = speaker_id or audio_path.stem
        
        print(f"Processing: {audio_path.name}")
        print(f"Speaker ID: {base_name}\n")
        
        # Create per-file output directories
        file_output_dir = self.output_dir / base_name
        audio_dir = file_output_dir / "audio"
        text_dir = file_output_dir / "transcripts"
        audio_dir.mkdir(parents=True, exist_ok=True)
        text_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {file_output_dir}\n")
        
        # Load audio
        audio = AudioSegment.from_file(str(audio_path))
        
        # Ensure mono and 16kHz for Whisper and YAMNet
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # Step 1: Split audio by silence (VAD)
        segments = self.split_audio_by_silence(audio)
        
        # Step 2: PRE-FILTERING (cheap checks before extraction/transcription)
        print(f"\nApplying pre-filters...")
        filtered_length = 0
        prefiltered_segments = []
        
        for start_ms, end_ms in segments:
            # Length check on base segment (instant, no extraction needed)
            base_duration_s = (end_ms - start_ms) / 1000.0
            if base_duration_s < self.min_segment_length:
                filtered_length += 1
                continue
            
            prefiltered_segments.append((start_ms, end_ms))
        
        if filtered_length > 0:
            print(f"✓ Filtered {filtered_length} segments (too short < {self.min_segment_length}s)")
        segments = prefiltered_segments
        print(f"✓ Remaining after length filter: {len(segments)} segments")
        
        # Step 3: MUSIC FILTERING using YAMNet (medium cost, saves transcription)
        if self.filter_music and self.yamnet_filter and self.yamnet_filter.model:
            print(f"\nFiltering music segments using YAMNet...")
            filtered_segments = []
            filtered_music = 0
            
            for start_ms, end_ms in tqdm(segments, desc="Music filtering"):
                # Extract base segment for classification
                chunk = audio[start_ms:end_ms]
                is_filtered, classification = self.yamnet_filter.is_music_or_mixed(
                    chunk, self.music_threshold, self.mixed_threshold
                )
                
                if not is_filtered:
                    filtered_segments.append((start_ms, end_ms))
                else:
                    filtered_music += 1
            
            segments = filtered_segments
            print(f"✓ Filtered {filtered_music} music/mixed segments")
            print(f"✓ Remaining: {len(segments)} speech segments")
        
        # Step 4: TRANSCRIPTION with quality filtering (expensive, done last)
        print(f"\nTranscribing {len(segments)} segments...")
        transcriptions = []
        filtered_low_confidence = 0
        filtered_noisy = 0
        
        for i, (start_ms, end_ms) in enumerate(tqdm(segments, desc="Transcribing")):
            # Extract chunk with 300ms buffer - use SAME buffer for transcription and export
            # This ensures audio and text are perfectly aligned
            audio_end_ms = min(end_ms + 300, len(audio))
            chunk = audio[start_ms:audio_end_ms]
            duration_s = len(chunk) / 1000.0
            
            # Noise check (requires extraction, so done here)
            if self.is_noisy(chunk):
                filtered_noisy += 1
                continue
            
            try:
                result = self.transcribe_segment(chunk)
                
                # Quality filter: check confidence
                if result['confidence'] < self.min_confidence:
                    filtered_low_confidence += 1
                    continue
                
                if result['text']:
                    transcriptions.append({
                        'start_ms': start_ms,
                        'end_ms': end_ms,
                        'audio_end_ms': audio_end_ms,  # Store exact extraction point
                        'duration': duration_s,
                        'text': result['text'],
                        'words': result['words'],
                        'confidence': result['confidence']
                    })
            except Exception as e:
                print(f"⚠ Error transcribing segment {i}: {e}")
        
        print(f"\n✓ Transcribed {len(transcriptions)} segments")
        if filtered_low_confidence > 0:
            print(f"⚠ Filtered {filtered_low_confidence} segments (low confidence < {self.min_confidence:.2f})")
        if filtered_noisy > 0:
            print(f"⚠ Filtered {filtered_noisy} segments (noisy, RMS < {self.min_rms:.3f})")
        
        # Step 5: Keep segments as-is (sentence splitting causes word cutoffs due to inaccurate timestamps)
        print("\nFinalizing segments...")
        final_segments = transcriptions
        
        print(f"✓ Final: {len(final_segments)} segments\n")
        
        # Step 6: Export audio and text files
        output_files = self.export_segments(audio, final_segments, base_name, audio_dir, text_dir)
        
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
            # Skip empty text
            if not seg['text'].strip() or len(seg['text'].strip()) < 3:
                continue
            
            chunk_name = f"{base_name}_{idx:04d}"
            
            # Use the EXACT same audio boundaries that were transcribed
            # This guarantees perfect audio/text alignment
            start_ms = int(seg['start_ms'])
            audio_end_ms = int(seg['audio_end_ms'])  # Use stored extraction point
            
            audio_chunk = audio[start_ms:audio_end_ms] + silence
            
            # Export audio
            audio_file = audio_dir / f"{chunk_name}.{self.export_format}"
            audio_chunk.export(
                audio_file,
                format=self.export_format,
                parameters=["-ar", "22050"]
            )
            
            # Export transcript
            transcript_file = text_dir / f"{chunk_name}.txt"
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(seg['text'].strip())
            
            metadata = {
                'duration': seg['duration'],
                'text_length': len(seg['text'])
            }
            
            output_files.append((str(audio_file), str(transcript_file), metadata))
        
        print(f"✓ Exported {len(output_files)} segments")
        return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Speech Splitter - VAD-first approach with YAMNet music filtering"
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
    parser.add_argument('--min-silence', type=int, default=700, help='Min silence length for VAD (ms)')
    parser.add_argument('--silence-thresh', type=int, default=-40, help='Silence threshold (dBFS)')
    parser.add_argument('--min-confidence', type=float, default=0.5, help='Min transcription confidence (0.0-1.0)')
    parser.add_argument('--filter-noise', action='store_true', help='Enable noise filtering')
    parser.add_argument('--min-rms', type=float, default=0.01, help='Min RMS energy (default 0.01)')
    parser.add_argument('--filter-music', action='store_true', default=True, help='Enable YAMNet music filtering')
    parser.add_argument('--music-threshold', type=float, default=0.3, help='Music detection threshold (default 0.3)')
    parser.add_argument('--mixed-threshold', type=float, default=0.2, help='Music+speech detection threshold (default 0.2)')
    
    args = parser.parse_args()
    
    splitter = SpeechSplitter(
        whisper_model=args.model,
        language=args.language,
        output_dir=args.output_dir,
        silence_padding_ms=args.silence_padding,
        min_segment_length=args.min_length,
        max_segment_length=args.max_length,
        export_format=args.format,
        min_silence_len=args.min_silence,
        silence_thresh=args.silence_thresh,
        min_confidence=args.min_confidence,
        filter_noise=args.filter_noise,
        min_rms=args.min_rms,
        filter_music=args.filter_music,
        music_threshold=args.music_threshold,
        mixed_threshold=args.mixed_threshold
    )
    
    splitter.process_file(args.input, args.speaker_id)


if __name__ == '__main__':
    main()
