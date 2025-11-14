"""
Export manager for segment export and metadata generation
Handles: File export, cutoff detection, metadata creation
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from pydub import AudioSegment

from utils import SegmentData
from audio_processor import AudioProcessor


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
        """Export segments to files with proper boundary handling and cutoff detection"""

        self.logger.info("EXPORTING SEGMENTS")
        
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
            
            # Trim silence
            audio_chunk = AudioProcessor.trim_silence_vectorized(
                audio_chunk,
                self.params['trim_threshold_db'],
                self.params['trim_chunk_ms'],
                self.params['trim_pad_ms'],
                trim_end=False
            )
            
            duration = len(audio_chunk) / 1000.0
            
            # Handle overlength segments
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
            
            # Export files
            audio_chunk.export(
                audio_file,
                format=self.params['export_format'],
                bitrate='256k' if self.params['export_format'] == 'mp3' else None
            )
            
            transcript_file.write_text(seg.text, encoding='utf-8')
            
            metadata = {
                'duration': duration,
                'confidence': seg.confidence,
                'classification': seg.classification
            }
            
            output_files.append((str(audio_file), str(transcript_file), metadata))
        
        if skipped_cutoff > 0:
            self.logger.warning(f"⚠ Cutoff detection: {skipped_cutoff} segments skipped")
        
        self.logger.info(f"✓ Exported {len(output_files)} segments")
        
        return output_files
    
    def _check_for_cutoff(self, audio_chunk: AudioSegment, is_start: bool = True) -> bool:
        """
        Check if audio is cut off at boundary using RMS analysis.
        Returns True if cutoff detected.
        """
        if not self.params.get('check_cutoff', False):
            return False
        
        boundary_ms = self.params.get('cutoff_boundary_ms', 50)
        threshold = self.params.get('cutoff_silence_threshold', 0.35)
        
        if len(audio_chunk) < boundary_ms:
            return False
        
        try:
            # Extract boundary region
            if is_start:
                boundary = audio_chunk[:boundary_ms]
            else:
                boundary = audio_chunk[-boundary_ms:]
            
            # Calculate RMS
            samples = np.array(boundary.get_array_of_samples())
            rms = np.sqrt(np.mean(samples**2))
            
            # Normalize RMS
            max_amplitude = 2**(boundary.sample_width * 8 - 1)
            normalized_rms = rms / max_amplitude
            
            # If RMS is above threshold, it's likely cut off
            return normalized_rms > threshold
            
        except Exception as e:
            self.logger.debug(f"Cutoff check error: {e}")
            return False
    
    def _get_validated_boundaries_with_cutoff_check(
        self,
        audio: AudioSegment,
        start_ms: int,
        end_ms: int,
        prev_word_end_ms: Optional[int],
        next_word_start_ms: Optional[int],
        seg_text: str
    ) -> Tuple[int, int, bool]:
        """
        Validate and adjust segment boundaries with cutoff detection.
        Returns: (start_ms, end_ms, should_skip)
        """
        # Apply word buffers
        validated_start = max(0, start_ms - self.params['word_start_buffer_ms'])
        validated_end = min(len(audio), end_ms + self.params['word_end_buffer_ms'])
        
        # Prevent overlap with previous word
        if prev_word_end_ms is not None:
            validated_start = max(validated_start, prev_word_end_ms)
        
        # Prevent overlap with next word
        if next_word_start_ms is not None:
            validated_end = min(validated_end, next_word_start_ms)
        
        # Extract audio chunk for cutoff check
        audio_chunk = audio[validated_start:validated_end]
        cutoff_action = self.params.get('cutoff_action', 'skip')
        
        # Check START cutoff
        if self._check_for_cutoff(audio_chunk, is_start=True):
            self.logger.debug(f"Start cutoff detected: '{seg_text[:30]}...'")
            
            if cutoff_action == 'extend':
                # Try to extend backwards
                extension_ms = 500
                
                # Never extend before previous word end
                if prev_word_end_ms is not None:
                    max_extension = validated_start - prev_word_end_ms
                    extension_ms = min(extension_ms, max_extension)
                
                extended_start = max(0, validated_start - extension_ms)
                
                extended_chunk = audio[extended_start:validated_end]
                if not self._check_for_cutoff(extended_chunk, is_start=True):
                    self.logger.debug(f"Start extension successful: {validated_start}ms -> {extended_start}ms")
                    validated_start = extended_start
                else:
                    self.logger.warning(f"Start extension failed, skipping: '{seg_text[:50]}...'")
                    return validated_start, validated_end, True
            elif cutoff_action == 'skip':
                self.logger.warning(f"Skipping segment with start cutoff: '{seg_text[:50]}...'")
                return validated_start, validated_end, True
        
        # Check END cutoff
        if self._check_for_cutoff(audio_chunk, is_start=False):
            self.logger.debug(f"End cutoff detected: '...{seg_text[-30:]}'")
            
            if cutoff_action == 'extend':
                # Try to extend forwards
                extension_ms = 500
                audio_length_ms = len(audio)
                
                # Never extend beyond next word start
                if next_word_start_ms is not None:
                    max_extension = next_word_start_ms - validated_end
                    extension_ms = min(extension_ms, max_extension)
                
                extended_end = min(audio_length_ms, validated_end + extension_ms)
                
                extended_chunk = audio[validated_start:extended_end]
                if not self._check_for_cutoff(extended_chunk, is_start=False):
                    self.logger.debug(f"End extension successful: {validated_end}ms -> {extended_end}ms")
                    validated_end = extended_end
                else:
                    self.logger.warning(f"End extension failed, skipping: '...{seg_text[-50:]}'")
                    return validated_start, validated_end, True
            elif cutoff_action == 'skip':
                self.logger.warning(f"Skipping segment with end cutoff: '...{seg_text[-50:]}'")
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
