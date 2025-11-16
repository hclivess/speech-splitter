"""
Export manager for segment export and metadata generation
Handles: File export, cutoff detection, metadata creation

IMPROVED VERSION:
- More aggressive end-trimming to remove fragments
- Better fragment detection after silence
- Enhanced cutoff detection for artifacts
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
            'extensions_failed': 0,
            'fragments_removed': 0  # NEW: Track fragment removal
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
            
            # IMPROVED: More aggressive trimming with fragment detection
            trimmed_chunk = self._trim_and_remove_fragments(audio_chunk)
            
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
            if self.params.get('cutoff_action') == 'extend':
                self.logger.info(f"  Extensions successful: {self.cutoff_stats['extensions_successful']}")
                self.logger.info(f"  Extensions failed: {self.cutoff_stats['extensions_failed']}")
            self.logger.info(f"  Segments skipped due to cutoffs: {skipped_cutoff}")
            self.logger.info("="*70)
        
        if self.cutoff_stats['fragments_removed'] > 0:
            self.logger.info(f"  Fragments removed from segment ends: {self.cutoff_stats['fragments_removed']}")
        
        self.logger.info(f"Exported {len(output_files)} segments")
        return output_files
    
    def _trim_and_remove_fragments(self, audio_chunk: AudioSegment) -> AudioSegment:
        """
        IMPROVED: Aggressive trimming with fragment detection
        
        This function:
        1. Trims silence from both start and end
        2. Detects and removes isolated fragments after silence
        3. Returns clean audio
        """
        # Step 1: Standard silence trimming (both ends)
        trimmed = AudioProcessor.trim_silence_vectorized(
            audio_chunk,
            self.params['trim_threshold_db'],
            self.params['trim_chunk_ms'],
            self.params['trim_pad_ms'],
            trim_end=True  # Always trim the end
        )
        
        # Step 2: Fragment detection and removal at the end
        fragment_removed = self._detect_and_remove_end_fragment(trimmed)
        
        if fragment_removed is not None and len(fragment_removed) < len(trimmed):
            self.cutoff_stats['fragments_removed'] += 1
            return fragment_removed
        
        return trimmed
    
    def _detect_and_remove_end_fragment(self, audio_chunk: AudioSegment, 
                                       silence_threshold_db: float = -40.0,
                                       min_silence_duration_ms: int = 200,
                                       max_fragment_duration_ms: int = 300) -> Optional[AudioSegment]:
        """
        Detect and remove word fragments that appear after a silence gap at the end.
        
        Algorithm:
        1. Scan from the end backward
        2. Find any speech (above threshold)
        3. Continue backward to find silence gap
        4. If silence gap is significant (>200ms) and fragment is small (<300ms), remove it
        
        Returns: Trimmed audio without fragment, or None if no fragment detected
        """
        if len(audio_chunk) < min_silence_duration_ms + 100:
            return None
        
        try:
            # Convert to numpy for analysis
            samples = np.array(audio_chunk.get_array_of_samples())
            frame_rate = audio_chunk.frame_rate
            
            # Calculate RMS in small windows (10ms chunks)
            window_ms = 10
            window_samples = int(frame_rate * window_ms / 1000)
            
            if window_samples == 0:
                return None
            
            # Calculate RMS for each window
            num_windows = len(samples) // window_samples
            if num_windows < 10:
                return None
            
            rms_values = []
            for i in range(num_windows):
                start = i * window_samples
                end = start + window_samples
                window = samples[start:end]
                rms = np.sqrt(np.mean(window**2))
                
                # Convert to dB
                max_amplitude = 2**(audio_chunk.sample_width * 8 - 1)
                with np.errstate(divide='ignore'):
                    db = 20 * np.log10(rms / max_amplitude + 1e-10)
                rms_values.append(db)
            
            # Scan from end backward
            is_speech = [db > silence_threshold_db for db in rms_values]
            
            # Find last speech region
            last_speech_idx = None
            for i in range(len(is_speech) - 1, -1, -1):
                if is_speech[i]:
                    last_speech_idx = i
                    break
            
            if last_speech_idx is None or last_speech_idx < 10:
                return None
            
            # Find the start of this speech region (go backward from last_speech_idx)
            fragment_start_idx = last_speech_idx
            for i in range(last_speech_idx, -1, -1):
                if is_speech[i]:
                    fragment_start_idx = i
                else:
                    break
            
            # Calculate fragment duration
            fragment_duration_ms = (last_speech_idx - fragment_start_idx + 1) * window_ms
            
            # Find silence gap before fragment
            silence_end_idx = fragment_start_idx
            silence_start_idx = None
            
            for i in range(fragment_start_idx - 1, -1, -1):
                if not is_speech[i]:
                    silence_start_idx = i
                else:
                    break
            
            if silence_start_idx is None:
                return None
            
            silence_duration_ms = (silence_end_idx - silence_start_idx) * window_ms
            
            # Decision: Remove if significant silence before small fragment
            if (silence_duration_ms >= min_silence_duration_ms and 
                fragment_duration_ms <= max_fragment_duration_ms):
                
                # Cut at the start of the silence gap
                cut_point_ms = silence_start_idx * window_ms
                
                if cut_point_ms > 0 and cut_point_ms < len(audio_chunk):
                    return audio_chunk[:cut_point_ms]
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Fragment detection error: {e}")
            return None
    
    def _try_extend_segment(self, audio: AudioSegment, start_ms: int, end_ms: int, 
                           seg: SegmentData, is_start: bool, max_extension_ms: int = 1000) -> Optional[AudioSegment]:
        """Try to extend segment boundaries to resolve cutoffs"""
        extensions = [100, 250, 500, 750, 1000]
        
        for extension in extensions:
            if extension > max_extension_ms:
                break
            
            # Calculate new boundaries
            if is_start:
                new_start = max(0, start_ms - extension)
                new_end = end_ms
                actual_extension = start_ms - new_start
            else:
                new_start = start_ms
                new_end = min(len(audio), end_ms + extension)
                actual_extension = new_end - end_ms
            
            if actual_extension < 10:  # No meaningful extension possible
                continue
            
            # Extract extended chunk
            extended_chunk = audio[new_start:new_end]
            
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
        
        # IMPROVED: Don't extend to next word (prevents including fragments)
        # The segmentation already ends at the last word, so we just add a small buffer
        
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
            f.write("file_name,transcription\n")
            for seg in segments:
                text = (self.text_dir / seg['transcript']).read_text(encoding='utf-8').strip().replace('"', '""')
                f.write(f'"{seg["audio"]}","{text}"\n')
        
        self.logger.info("Metadata files created", extra={
            'segments': len(segments),
            'files': ['metadata.json', 'manifest.txt', 'metadata.csv']
        })
