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
        
        # Cutoff detection stats
        self.cutoff_stats = {
            'start_cutoffs_detected': 0,
            'end_cutoffs_detected': 0,
            'extensions_successful': 0,
            'extensions_failed': 0
        }
    
    def export_segments(self, audio: AudioSegment, segments: List[SegmentData], 
                       base_name: str, sample_rate: int) -> List[Tuple[str, str, Dict]]:
        """Export segments to files with proper boundary handling and cutoff detection"""

        self.logger.info("EXPORTING SEGMENTS")
        
        output_files = []
        skipped_cutoff = 0
        export_counter = 1  # Track actual export numbering
        
        for idx, seg in enumerate(segments, 1):
            # Get validated boundaries (without cutoff check yet)
            start_ms, end_ms = self._get_validated_boundaries(
                audio, seg.start_ms, seg.end_ms, 
                seg.prev_word_end_ms, seg.next_word_start_ms
            )
            
            # Extract audio chunk
            audio_chunk = audio[start_ms:end_ms]
            
            # Trim silence (with end trimming enabled)
            trimmed_chunk = AudioProcessor.trim_silence_vectorized(
                audio_chunk,
                self.params['trim_threshold_db'],
                self.params['trim_chunk_ms'],
                self.params['trim_pad_ms'],
                trim_end=True
            )
            
            # Check if trimming removed too much (might indicate the segment was mostly silence)
            trim_ratio = len(trimmed_chunk) / len(audio_chunk) if len(audio_chunk) > 0 else 1.0
            if trim_ratio < 0.3:
                self.logger.info(f"Segment {idx} heavily trimmed ({trim_ratio:.1%}), might be mostly silence")
            
            # NOW check for cutoffs on the TRIMMED audio
            if self.params.get('check_cutoff', False):
                cutoff_info_start = self._check_for_cutoff_detailed(trimmed_chunk, is_start=True)
                cutoff_info_end = self._check_for_cutoff_detailed(trimmed_chunk, is_start=False)
                
                has_start_cutoff = cutoff_info_start['has_cutoff']
                has_end_cutoff = cutoff_info_end['has_cutoff']
                
                cutoff_action = self.params.get('cutoff_action', 'skip')
                
                # Handle start cutoff
                if has_start_cutoff:
                    self.cutoff_stats['start_cutoffs_detected'] += 1
                    self.logger.warning(
                        f"START CUTOFF #{idx}: '{seg.text[:40]}...' "
                        f"[RMS: {cutoff_info_start['max_rms']:.3f}, "
                        f"Windows: {cutoff_info_start['windows_over_threshold']}/{cutoff_info_start['total_windows']}]"
                    )
                    
                    if cutoff_action == 'extend':
                        # Try extending with progressively larger extensions
                        extended_chunk = self._try_extend_segment(
                            audio, start_ms, end_ms, seg, 
                            is_start=True, max_extension_ms=1000
                        )
                        
                        if extended_chunk is not None:
                            self.cutoff_stats['extensions_successful'] += 1
                            self.logger.info(f"  ✓ Start extension successful")
                            trimmed_chunk = extended_chunk
                        else:
                            self.cutoff_stats['extensions_failed'] += 1
                            self.logger.error(f"  ✗ Start extension failed, SKIPPING segment")
                            skipped_cutoff += 1
                            continue
                    elif cutoff_action == 'skip':
                        self.logger.error(f"  ✗ SKIPPING segment with start cutoff")
                        skipped_cutoff += 1
                        continue
                
                # Handle end cutoff
                if has_end_cutoff:
                    self.cutoff_stats['end_cutoffs_detected'] += 1
                    self.logger.warning(
                        f"END CUTOFF #{idx}: '...{seg.text[-40:]}' "
                        f"[RMS: {cutoff_info_end['max_rms']:.3f}, "
                        f"Windows: {cutoff_info_end['windows_over_threshold']}/{cutoff_info_end['total_windows']}]"
                    )
                    
                    if cutoff_action == 'extend':
                        # Try extending with progressively larger extensions
                        extended_chunk = self._try_extend_segment(
                            audio, start_ms, end_ms, seg, 
                            is_start=False, max_extension_ms=1000
                        )
                        
                        if extended_chunk is not None:
                            self.cutoff_stats['extensions_successful'] += 1
                            self.logger.info(f"  ✓ End extension successful")
                            trimmed_chunk = extended_chunk
                        else:
                            self.cutoff_stats['extensions_failed'] += 1
                            self.logger.error(f"  ✗ End extension failed, SKIPPING segment")
                            skipped_cutoff += 1
                            continue
                    elif cutoff_action == 'skip':
                        self.logger.error(f"  ✗ SKIPPING segment with end cutoff")
                        skipped_cutoff += 1
                        continue
            
            duration = len(trimmed_chunk) / 1000.0
            
            # Handle overlength segments
            suffix = ""
            if duration > self.params['max_segment_length']:
                self.logger.info(f"Segment {idx} overlength: {duration:.2f}s")
                if self.params['overlength_action'] == 'skip':
                    continue
                suffix = "_LONG"
            
            # Use export_counter for sequential numbering
            chunk_name = f"{base_name}_{export_counter:04d}{suffix}"
            audio_file = self.audio_dir / f"{chunk_name}.{self.params['export_format']}"
            transcript_file = self.text_dir / f"{chunk_name}.txt"
            
            # Ensure files don't exist (shouldn't happen with sequential counter)
            if audio_file.exists() or transcript_file.exists():
                self.logger.warning(f"File collision detected for {chunk_name}, using fallback naming")
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
            export_counter += 1  # Increment only on successful export
        
        # Log cutoff statistics
        if self.params.get('check_cutoff', False):
            self.logger.info("="*70)
            self.logger.info("CUTOFF DETECTION STATISTICS:")
            self.logger.info(f"  Start cutoffs detected: {self.cutoff_stats['start_cutoffs_detected']}")
            self.logger.info(f"  End cutoffs detected: {self.cutoff_stats['end_cutoffs_detected']}")
            self.logger.info(f"  Extensions successful: {self.cutoff_stats['extensions_successful']}")
            self.logger.info(f"  Extensions failed: {self.cutoff_stats['extensions_failed']}")
            self.logger.info(f"  Total segments skipped: {skipped_cutoff}")
            self.logger.info("="*70)
        
        if skipped_cutoff > 0:
            self.logger.warning(f"⚠️  Cutoff detection: {skipped_cutoff} segments skipped")
        
        self.logger.info(f"✓ Exported {len(output_files)} segments")
        
        return output_files
    
    def _try_extend_segment(
        self, 
        audio: AudioSegment, 
        start_ms: int, 
        end_ms: int, 
        seg: SegmentData,
        is_start: bool,
        max_extension_ms: int = 1000
    ) -> Optional[AudioSegment]:
        """
        Try to extend segment with progressively larger extensions until cutoff is resolved.
        Returns extended and trimmed audio, or None if extension fails.
        """
        # Try extensions: 200ms, 500ms, 800ms, 1000ms
        extension_steps = [200, 500, 800, max_extension_ms]
        
        for extension_ms in extension_steps:
            if is_start:
                # Extend backwards
                max_extension = start_ms - (seg.prev_word_end_ms or 0)
                actual_extension = min(extension_ms, max_extension)
                new_start = max(0, start_ms - actual_extension)
                
                extended_chunk = audio[new_start:end_ms]
            else:
                # Extend forwards
                audio_length_ms = len(audio)
                max_extension = (seg.next_word_start_ms or audio_length_ms) - end_ms
                actual_extension = min(extension_ms, max_extension)
                new_end = min(audio_length_ms, end_ms + actual_extension)
                
                extended_chunk = audio[start_ms:new_end]
            
            # Trim the extended chunk
            extended_trimmed = AudioProcessor.trim_silence_vectorized(
                extended_chunk,
                self.params['trim_threshold_db'],
                self.params['trim_chunk_ms'],
                self.params['trim_pad_ms'],
                trim_end=True
            )
            
            # Check if extension resolved the cutoff
            cutoff_info = self._check_for_cutoff_detailed(extended_trimmed, is_start=is_start)
            
            if not cutoff_info['has_cutoff']:
                self.logger.info(f"    Extension {actual_extension}ms resolved cutoff")
                return extended_trimmed
            else:
                self.logger.info(
                    f"    Extension {actual_extension}ms still has cutoff "
                    f"(RMS: {cutoff_info['max_rms']:.3f})"
                )
        
        return None
    
    def _check_for_cutoff_detailed(self, audio_chunk: AudioSegment, is_start: bool = True) -> Dict:
        """
        Enhanced cutoff detection with detailed diagnostics.
        Returns dict with: has_cutoff, max_rms, windows_over_threshold, total_windows
        """
        if not self.params.get('check_cutoff', False):
            return {'has_cutoff': False, 'max_rms': 0.0, 'windows_over_threshold': 0, 'total_windows': 0}
        
        boundary_ms = self.params.get('cutoff_boundary_ms', 50)
        threshold = self.params.get('cutoff_silence_threshold', 0.35)
        
        # Need sufficient audio for analysis
        if len(audio_chunk) < boundary_ms:
            return {'has_cutoff': False, 'max_rms': 0.0, 'windows_over_threshold': 0, 'total_windows': 0}
        
        try:
            # Analyze multiple small windows within the boundary
            window_count = 5  # Increased from 3 for finer analysis
            window_ms = boundary_ms // window_count
            
            if window_ms < 10:  # Minimum window size
                window_ms = 10
                window_count = boundary_ms // window_ms
            
            rms_values = []
            windows_over_threshold = 0
            
            for i in range(window_count):
                if is_start:
                    window_start = i * window_ms
                    window_end = min(window_start + window_ms, len(audio_chunk))
                else:
                    window_end = len(audio_chunk) - (i * window_ms)
                    window_start = max(0, window_end - window_ms)
                
                if window_end - window_start < 5:  # Skip if window too small
                    continue
                
                boundary = audio_chunk[window_start:window_end]
                
                # Calculate RMS
                samples = np.array(boundary.get_array_of_samples())
                rms = np.sqrt(np.mean(samples**2))
                
                # Normalize RMS
                max_amplitude = 2**(boundary.sample_width * 8 - 1)
                normalized_rms = rms / max_amplitude
                
                rms_values.append(normalized_rms)
                
                if normalized_rms > threshold:
                    windows_over_threshold += 1
            
            if not rms_values:
                return {'has_cutoff': False, 'max_rms': 0.0, 'windows_over_threshold': 0, 'total_windows': 0}
            
            max_rms = max(rms_values)
            total_windows = len(rms_values)
            
            # Cutoff detected if:
            # 1. At least 40% of windows are over threshold, OR
            # 2. Any single window has RMS > threshold * 1.5 (very loud)
            has_cutoff = (
                (windows_over_threshold >= total_windows * 0.4) or
                (max_rms > threshold * 1.5)
            )
            
            return {
                'has_cutoff': has_cutoff,
                'max_rms': max_rms,
                'windows_over_threshold': windows_over_threshold,
                'total_windows': total_windows
            }
            
        except Exception as e:
            self.logger.info(f"Cutoff check error: {e}")
            return {'has_cutoff': False, 'max_rms': 0.0, 'windows_over_threshold': 0, 'total_windows': 0}
    
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
        
        audio_files = sorted(self.audio_dir.glob(f"*.{self.params['export_format']}"))
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
