"""
Audio processing utilities
Handles: Audio loading, conversion, silence trimming
"""

import logging
import numpy as np
from pathlib import Path
from typing import Tuple
from pydub import AudioSegment

from utils import MIN_TRIM_DURATION_MS


class AudioProcessor:
    """Handle audio loading, conversion, and trimming"""
    
    @staticmethod
    def load_and_prepare(audio_path: Path, logger: logging.Logger) -> Tuple[AudioSegment, int]:
        """Load audio file and convert to mono"""
        try:
            audio = AudioSegment.from_file(audio_path)
            original_sr = audio.frame_rate
            
            # Convert to mono once
            if audio.channels > 1:
                audio = audio.set_channels(1)
                logger.debug("Converted to mono")
            
            logger.info(f"Audio loaded: {len(audio)/1000:.1f}s, {audio.frame_rate}Hz")
            return audio, original_sr
            
        except Exception as e:
            logger.error(f"Audio load failed: {e}")
            raise
    
    @staticmethod
    def trim_silence_vectorized(audio_chunk: AudioSegment, trim_threshold_db: float, 
                               trim_chunk_ms: int, trim_pad_ms: int, 
                               trim_end: bool = True) -> AudioSegment:
        """Vectorized silence trimming"""
        if len(audio_chunk) < MIN_TRIM_DURATION_MS:
            return audio_chunk
        
        try:
            # Convert to numpy array
            samples = np.array(audio_chunk.get_array_of_samples())
            
            # Calculate RMS in chunks
            chunk_size = int(audio_chunk.frame_rate * trim_chunk_ms / 1000)
            num_chunks = len(samples) // chunk_size
            
            if num_chunks == 0:
                return audio_chunk
            
            # Reshape and calculate RMS
            samples_trimmed = samples[:num_chunks * chunk_size].reshape(-1, chunk_size)
            rms_values = np.sqrt(np.mean(samples_trimmed**2, axis=1))
            
            # Convert to dB
            with np.errstate(divide='ignore'):
                db_values = 20 * np.log10(rms_values + 1e-10)
            
            # Find non-silent regions
            non_silent = db_values > trim_threshold_db
            
            if not np.any(non_silent):
                return audio_chunk
            
            # Find trim points
            non_silent_indices = np.where(non_silent)[0]
            start_chunk = non_silent_indices[0]
            end_chunk = non_silent_indices[-1] + 1
            
            # Apply padding
            pad_chunks = int(trim_pad_ms / trim_chunk_ms)
            start_chunk = max(0, start_chunk - pad_chunks)
            end_chunk = min(num_chunks, end_chunk + pad_chunks)
            
            # Convert back to milliseconds
            start_ms = int(start_chunk * trim_chunk_ms)
            end_ms = int(end_chunk * trim_chunk_ms)
            
            # Trim start
            trimmed = audio_chunk[start_ms:]
            
            # Optionally trim end
            if trim_end and end_ms < len(audio_chunk):
                trimmed = trimmed[:end_ms - start_ms]
            
            return trimmed
            
        except Exception:
            return audio_chunk
