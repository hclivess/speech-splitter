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
            print("⚠ Music filtering will be disabled")
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
            return {'music': 0.0, 'speech': 1.0, 'music_speech_mixed': 0.0}
        
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
        classification = self.classify_audio(audio_segment)
        
        is_music = classification['music'] > music_threshold
        is_mixed = classification['music_speech_mixed'] > mixed_threshold
        
        return (is_music or is_mixed), classification


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
        min_rms: float = 0.01,
        filter_music: bool = True,
        music_threshold: float = 0.3,
        mixed_threshold: float = 0.2,
        word_start_buffer_ms: int = 500,
        word_end_buffer_ms: int = 500
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
        self.min_rms = min_rms
        self.filter_music = filter_music
        self.music_threshold = music_threshold
        self.mixed_threshold = mixed_threshold
        self.word_start_buffer_ms = word_start_buffer_ms
        self.word_end_buffer_ms = word_end_buffer_ms
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading Whisper model ({whisper_model})...")
        self.model = whisper.load_model(whisper_model)
        
        if self.filter_music:
            self.yamnet_filter = YAMNetFilter()
        else:
            self.yamnet_filter = None
            print("⚠ Music filtering disabled")
        
        print("✓ Models loaded\n")
    
    def validate_segment_text(self, text: str) -> bool:
        if not text or not text.strip():
            return False
        
        text = text.strip()
        
        if len(text) < 3:
            return False
        
        invalid_patterns = [
            '[BLANK_AUDIO]',
            '[MUSIC]',
            '[APPLAUSE]',
            '[LAUGHTER]',
            '[NOISE]',
            '...',
            '....',
            'ummm',
            'uhhhh',
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
    
    def segment_by_pauses(self, words: List[Dict]) -> List[Dict]:
        if not words:
            print("⚠ No words found for segmentation")
            return []
        
        print("Segmenting by natural pauses...")
        
        segments = []
        current_segment_words = []
        
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
                        should_split = True
            
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
        return segments
    
    def is_noisy(self, audio_chunk: AudioSegment) -> bool:
        if not self.filter_noise:
            return False
        
        samples = np.array(audio_chunk.get_array_of_samples())
        samples_float = samples.astype(np.float32)
        
        if audio_chunk.sample_width == 2:
            samples_float = samples_float / 32768.0
        elif audio_chunk.sample_width == 4:
            samples_float = samples_float / 2147483648.0
        
        rms = np.sqrt(np.mean(samples_float**2))
        
        is_noisy = rms < self.min_rms
        if is_noisy:
            print(f"  ⚠ Noisy segment detected: RMS={rms:.4f} < {self.min_rms:.4f}")
        
        return is_noisy
    
    def process_file(self, audio_path: str, speaker_id: Optional[str] = None) -> Dict:
        audio_path = Path(audio_path)
        base_name = speaker_id or audio_path.stem
        
        print(f"\n{'='*60}")
        print(f"Processing: {audio_path.name}")
        print(f"Speaker: {base_name}")
        print(f"{'='*60}\n")
        
        file_output_dir = self.output_dir / base_name
        audio_dir = file_output_dir / "audio"
        text_dir = file_output_dir / "transcripts"
        
        if file_output_dir.exists():
            if audio_dir.exists() and list(audio_dir.glob(f'*.{self.export_format}')):
                print(f"⭐ Skipping: Output folder already exists with processed files")
                print(f"   Location: {file_output_dir}")
                print(f"{'='*60}\n")
                
                existing_audio = list(audio_dir.glob(f'*.{self.export_format}'))
                metadata_path = file_output_dir / f"{base_name}_metadata.json"
                
                if metadata_path.exists():
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        return {
                            'total_segments': metadata.get('total_segments', len(existing_audio)),
                            'total_duration': sum(seg.get('duration', 0) for seg in metadata.get('segments', [])),
                            'avg_duration': 0,
                            'metadata_file': str(metadata_path),
                            'manifest_file': str(file_output_dir / f"{base_name}_manifest.txt")
                        }
                
                return {
                    'total_segments': len(existing_audio),
                    'total_duration': 0,
                    'avg_duration': 0,
                    'metadata_file': None,
                    'manifest_file': None
                }
        
        audio_dir.mkdir(parents=True, exist_ok=True)
        text_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        if audio.channels > 1:
            audio = audio.set_channels(1)
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        
        print(f"✓ Loaded: {len(audio)/1000:.1f}s, {audio.frame_rate}Hz, {audio.channels} channel(s)\n")
        
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
        
        print(f"\nFiltering {len(segments)} segments...")
        filtered_segments = []
        filtered_music = 0
        filtered_noise = 0
        filtered_confidence = 0
        filtered_text = 0
        
        for seg in tqdm(segments, desc="Filtering"):
            chunk = audio[seg['start_ms']:seg['end_ms']]
            
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
            
            if self.is_noisy(chunk):
                filtered_noise += 1
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
        
        print(f"\n✓ Kept {len(filtered_segments)} segments")
        if filtered_music > 0:
            print(f"⚠ Filtered {filtered_music} segments (music/mixed)")
        if filtered_noise > 0:
            print(f"⚠ Filtered {filtered_noise} segments (noisy, RMS < {self.min_rms:.3f})")
        if filtered_confidence > 0:
            print(f"⚠ Filtered {filtered_confidence} segments (low confidence < {self.min_confidence:.2f})")
        if filtered_text > 0:
            print(f"⚠ Filtered {filtered_text} segments (invalid text)")
        
        output_files = self.export_segments(audio, filtered_segments, base_name, audio_dir, text_dir)
        
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
        
        manifest_path = file_output_dir / f"{base_name}_manifest.txt"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            for audio_file, transcript_file, _ in output_files:
                f.write(f"{Path(audio_file).name}|{Path(transcript_file).name}\n")
        
        hf_metadata_path = file_output_dir / "metadata.csv"
        with open(hf_metadata_path, 'w', encoding='utf-8') as f:
            f.write("file_name,text\n")
            for audio_file, transcript_file, meta in output_files:
                with open(transcript_file, 'r', encoding='utf-8') as tf:
                    text = tf.read().strip()
                text = text.replace('"', '""')
                audio_filename = Path(audio_file).name
                f.write(f'"{audio_filename}","{text}"\n')
        
        print(f"✓ Generated HuggingFace metadata.csv")
        
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
        print(f"Exporting {len(segments)} segments...")

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

            audio_file = audio_dir / f"{chunk_name}.{self.export_format}"
            audio_chunk.export(
                audio_file,
                format=self.export_format,
                parameters=["-ar", "16000"]
            )

            transcript_file = text_dir / f"{chunk_name}.txt"
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(seg['text'].strip())

            metadata = {
                'duration': len(audio_chunk) / 1000.0,
                'text_length': len(seg['text'].strip()),
                'sample_rate': 16000
            }

            output_files.append((str(audio_file), str(transcript_file), metadata))
            print(f"  ✓ Exported: {chunk_name} ({metadata['duration']:.2f}s)")

        print(f"✓ Exported {len(output_files)} segments")
        return output_files


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
    parser.add_argument('--min-rms', type=float, default=0.01)
    parser.add_argument('--filter-music', action='store_true', default=True)
    parser.add_argument('--music-threshold', type=float, default=0.3)
    parser.add_argument('--mixed-threshold', type=float, default=0.2)
    parser.add_argument('--word-start-buffer', type=int, default=250)
    parser.add_argument('--word-end-buffer', type=int, default=750)
    
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
        min_rms=args.min_rms,
        filter_music=args.filter_music,
        music_threshold=args.music_threshold,
        mixed_threshold=args.mixed_threshold,
        word_start_buffer_ms=args.word_start_buffer,
        word_end_buffer_ms=args.word_end_buffer
    )
    
    splitter.process_file(args.input, args.speaker_id)


if __name__ == '__main__':
    main()