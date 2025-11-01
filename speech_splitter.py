#!/usr/bin/env python3
"""
Speech Splitter - Split audio using VAD, then transcribe chunks
Proper approach: Audio segmentation FIRST, then transcription
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
        min_rms: float = 0.01
    ):
        """
        Initialize Speech Splitter with VAD-first approach
        
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
        
        # Create output directories
        self.audio_dir = self.output_dir / "audio"
        self.text_dir = self.output_dir / "transcripts"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Whisper
        print(f"Loading Whisper model ({whisper_model})...")
        self.model = whisper.load_model(whisper_model)
        
        # Load sentence splitter
        print(f"Loading sentence splitter for {language}...")
        self.sentence_splitter = SentenceSplitter(language)
        
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
    
    def is_noisy(self, audio_segment: AudioSegment) -> bool:
        """Check if audio segment is too noisy"""
        if not self.filter_noise:
            return False
        
        # Calculate RMS energy
        samples = np.array(audio_segment.get_array_of_samples())
        if audio_segment.sample_width == 2:
            samples = samples.astype(np.float32) / 32768.0
        elif audio_segment.sample_width == 4:
            samples = samples.astype(np.float32) / 2147483648.0
        
        rms = np.sqrt(np.mean(samples**2))
        
        return rms < self.min_rms
    
    def transcribe_segment(self, audio_segment: AudioSegment) -> Dict:
        """Transcribe a single audio segment with word timestamps and confidence"""
        # Convert to numpy array for Whisper
        samples = np.array(audio_segment.get_array_of_samples())
        
        # Normalize to float32
        if audio_segment.sample_width == 2:  # 16-bit
            samples = samples.astype(np.float32) / 32768.0
        elif audio_segment.sample_width == 4:  # 32-bit
            samples = samples.astype(np.float32) / 2147483648.0
        
        # Transcribe with word timestamps
        result = self.model.transcribe(
            samples,
            language=self.language,
            task='transcribe',
            fp16=False,
            word_timestamps=True
        )
        
        # Extract word-level timing and calculate average confidence
        words = []
        confidences = []
        for segment in result.get('segments', []):
            # Segment-level confidence from no_speech_prob
            seg_confidence = 1.0 - segment.get('no_speech_prob', 0.0)
            confidences.append(seg_confidence)
            
            for word in segment.get('words', []):
                words.append({
                    'word': word['word'].strip(),
                    'start': word['start'],
                    'end': word['end'],
                    'probability': word.get('probability', 1.0)
                })
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        return {
            'text': result['text'].strip(),
            'words': words,
            'confidence': avg_confidence
        }
    
    def process_file(self, audio_path: str, speaker_id: str = None) -> Dict:
        """
        Process audio file: Split by VAD, transcribe chunks, align sentences
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        base_name = f"{speaker_id}_{audio_path.stem}" if speaker_id else audio_path.stem
        
        print(f"\n{'='*60}")
        print(f"Processing: {audio_path.name}")
        print(f"{'='*60}\n")
        
        # Load audio
        print("Loading audio...")
        audio = AudioSegment.from_file(str(audio_path))
        
        # Ensure mono and 16kHz for Whisper
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # Step 1: Split audio by silence (VAD)
        segments = self.split_audio_by_silence(audio)
        
        # Step 2: Transcribe each segment with quality filtering
        print(f"\nTranscribing {len(segments)} segments...")
        transcriptions = []
        filtered_low_confidence = 0
        filtered_noisy = 0
        
        for i, (start_ms, end_ms) in enumerate(tqdm(segments, desc="Transcribing")):
            # Add 500ms buffer to END for transcription (helps Whisper complete words)
            transcribe_end_ms = min(end_ms + 500, len(audio))
            chunk = audio[start_ms:transcribe_end_ms]
            duration_s = len(chunk) / 1000.0
            
            # Skip very short segments
            if duration_s < self.min_segment_length:
                continue
            
            # Check for noise
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
        
        # Step 3: Keep segments as-is (sentence splitting causes word cutoffs due to inaccurate timestamps)
        print("\nFinalizing segments...")
        final_segments = transcriptions
        
        print(f"✓ Final: {len(final_segments)} segments\n")
        
        # Step 4: Export audio and text files
        output_files = self.export_segments(audio, final_segments, base_name)
        
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
        
        metadata_path = self.output_dir / f"{base_name}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Generate manifest
        manifest_path = self.output_dir / f"{base_name}_manifest.txt"
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
        print(f"Output: {self.output_dir}")
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
        base_name: str
    ) -> List[Tuple[str, str, Dict]]:
        """Export audio chunks and transcripts"""
        print(f"Exporting {len(segments)} segments...")
        
        silence = AudioSegment.silent(duration=self.silence_padding_ms)
        output_files = []
        
        for idx, seg in enumerate(tqdm(segments, desc="Exporting"), 1):
            # Skip empty text
            if not seg['text'].strip() or len(seg['text'].strip()) < 3:
                continue
            
            chunk_name = f"{base_name}_{idx:04d}"
            
            # Extract audio with extra buffer at end to prevent cutoffs
            start_ms = int(seg['start_ms'])
            end_ms = int(seg['end_ms'])
            
            # Add 300ms buffer to end for audio extraction only
            # (prevents words being cut short due to timing inaccuracies)
            audio_end_ms = min(end_ms + 300, len(audio))
            
            audio_chunk = audio[start_ms:audio_end_ms] + silence
            
            # Export audio
            audio_file = self.audio_dir / f"{chunk_name}.{self.export_format}"
            audio_chunk.export(
                audio_file,
                format=self.export_format,
                parameters=["-ar", "22050"]
            )
            
            # Export transcript
            transcript_file = self.text_dir / f"{chunk_name}.txt"
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
        description="Speech Splitter - VAD-first approach with proper transcription"
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
        min_rms=args.min_rms
    )
    
    splitter.process_file(args.input, args.speaker_id)


if __name__ == '__main__':
    main()
