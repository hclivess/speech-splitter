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
            # Get validated boundaries (without cutoff check yet)
            start_ms, end_ms = self._get_validated_boundaries(
                audio, seg.start_ms, seg.end_ms, 
                seg.prev_word_end_ms, seg.next_word_start_ms
            )
            
            # Extract audio chunk
            audio_chunk = audio[start_ms:end_ms]
            
            # Trim silence (NOW INCLUDING END TRIMMING)
            trimmed_chunk = AudioProcessor.trim_silence_vectorized(
                audio_chunk,
                self.params['trim_threshold_db'],
                self.params['trim_chunk_ms'],
                self.params['trim_pad_ms'],
                trim_end=True  # FIX: Enable end trimming
            )
            
            # NOW check for cutoffs on the TRIMMED audio
            if self.params.get('check_cutoff', False):
                has_start_cutoff = self._check_for_cutoff(trimmed_chunk, is_start=True)
                has_end_cutoff = self._check_for_cutoff(trimmed_chunk, is_start=False)
                
                cutoff_action = self.params.get('cutoff_action', 'skip')
                
                # Handle start cutoff
                if has_start_cutoff:
                    self.logger.debug(f"Start cutoff detected in trimmed audio: '{seg.text[:30]}...'")
                    
                    if cutoff_action == 'extend':
                        # Try extending the original segment before trimming
                        extension_ms = 500
                        if seg.prev_word_end_ms is not None:
                            max_extension = start_ms - seg.prev_word_end_ms
                            extension_ms = min(extension_ms, max_extension)
                        
                        extended_start = max(0, start_ms - extension_ms)
                        extended_chunk = audio[extended_start:end_ms]
                        extended_trimmed = AudioProcessor.trim_silence_vectorized(
                            extended_chunk,
                            self.params['trim_threshold_db'],
                            self.params['trim_chunk_ms'],
                            self.params['trim_pad_ms'],
                            trim_end=True
                        )
                        
                        if not self._check_for_cutoff(extended_trimmed, is_start=True):
                            self.logger.debug(f"Start extension successful")
                            trimmed_chunk = extended_trimmed
                            start_ms = extended_start
                        else:
                            self.logger.warning(f"Start extension failed, skipping: '{seg.text[:50]}...'")
                            skipped_cutoff += 1
                            continue
                    elif cutoff_action == 'skip':
                        self.logger.warning(f"Skipping segment with start cutoff: '{seg.text[:50]}...'")
                        skipped_cutoff += 1
                        continue
                
                # Handle end cutoff
                if has_end_cutoff:
                    self.logger.debug(f"End cutoff detected in trimmed audio: '...{seg.text[-30:]}'")
                    
                    if cutoff_action == 'extend':
                        # Try extending the original segment before trimming
                        extension_ms = 500
                        audio_length_ms = len(audio)
                        if seg.next_word_start_ms is not None:
                            max_extension = seg.next_word_start_ms - end_ms
                            extension_ms = min(extension_ms, max_extension)
                        
                        extended_end = min(audio_length_ms, end_ms + extension_ms)
                        extended_chunk = audio[start_ms:extended_end]
                        extended_trimmed = AudioProcessor.trim_silence_vectorized(
                            extended_chunk,
                            self.params['trim_threshold_db'],
                            self.params['trim_chunk_ms'],
                            self.params['trim_pad_ms'],
                            trim_end=True
                        )
                        
                        if not self._check_for_cutoff(extended_trimmed, is_start=False):
                            self.logger.debug(f"End extension successful")
                            trimmed_chunk = extended_trimmed
                            end_ms = extended_end
                        else:
                            self.logger.warning(f"End extension failed, skipping: '...{seg.text[-50:]}'")
                            skipped_cutoff += 1
                            continue
                    elif cutoff_action == 'skip':
                        self.logger.warning(f"Skipping segment with end cutoff: '...{seg.text[-50:]}'")
                        skipped_cutoff += 1
                        continue
            
            duration = len(trimmed_chunk) / 1000.0
            
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
            trimmed_chunk.export(
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
            self.logger.warning(f"⚠️  Cutoff detection: {skipped_cutoff} segments skipped")
        
        self.logger.info(f"✓ Exported {len(output_files)} segments")
        
        return output_files
    
    def _check_for_cutoff(self, audio_chunk: AudioSegment, is_start: bool = True) -> bool:
        """
        Check if audio is cut off at boundary using RMS analysis.
        IMPROVED: Checks multiple windows and uses stricter thresholds.
        Returns True if cutoff detected.
        """
        if not self.params.get('check_cutoff', False):
            return False
        
        boundary_ms = self.params.get('cutoff_boundary_ms', 50)
        threshold = self.params.get('cutoff_silence_threshold', 0.35)
        
        # Need at least 2x boundary for proper detection
        if len(audio_chunk) < boundary_ms * 2:
            return False
        
        try:
            # Extract boundary region - CHECK MULTIPLE WINDOWS
            window_count = 3
            window_ms = boundary_ms // window_count
            cutoff_detected_count = 0
            
            for i in range(window_count):
                if is_start:
                    window_start = i * window_ms
                    window_end = window_start + window_ms
                    boundary = audio_chunk[window_start:window_end]
                else:
                    window_start = len(audio_chunk) - ((i + 1) * window_ms)
                    window_end = window_start + window_ms
                    boundary = audio_chunk[window_start:window_end]
                
                if len(boundary) < window_ms // 2:
                    continue
                
                # Calculate RMS
                samples = np.array(boundary.get_array_of_samples())
                rms = np.sqrt(np.mean(samples**2))
                
                # Normalize RMS
                max_amplitude = 2**(boundary.sample_width * 8 - 1)
                normalized_rms = rms / max_amplitude
                
                # If RMS is above threshold, it's likely cut off
                if normalized_rms > threshold:
                    cutoff_detected_count += 1
            
            # If majority of windows show cutoff, consider it cut off
            return cutoff_detected_count >= (window_count // 2 + 1)
            
        except Exception as e:
            self.logger.debug(f"Cutoff check error: {e}")
            return False
    
    def _get_validated_boundaries(
        self,
        audio: AudioSegment,
        start_ms: int,
        end_ms: int,
        prev_word_end_ms: Optional[int],
        next_word_start_ms: Optional[int]
    ) -> Tuple[int, int]:
        """
        Validate and adjust segment boundaries (without cutoff check).
        Returns: (start_ms, end_ms)
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
        
        return validated_start, validated_end
    
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
