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
    def enhance_speech_deepfilternet(audio_chunk: AudioSegment, logger: logging.Logger,
                                     device: str = 'cpu') -> AudioSegment:
        """Apply DeepFilterNet speech enhancement - CPU ONLY"""
        try:
            import torch
            import numpy as np
            import sys
            import os

            # FORCE CPU MODE - ignore device parameter
            if device != 'cpu':
                logger.warning("DeepFilterNet device parameter is ignored - CPU mode only")
            device = 'cpu'

            # Use a local reference to the function object for caching model/state
            df_func = AudioProcessor.enhance_speech_deepfilternet

            # Check if torch already has CUDA initialized
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                logger.warning("CUDA was previously initialized. DeepFilterNet may have device issues.")

            # Set CUDA_VISIBLE_DEVICES to hide CUDA
            old_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

            # Initialize model (lazy load once)
            if not hasattr(df_func, 'model'):
                logger.info(f"Loading DeepFilterNet model on CPU...")

                # CRITICAL: Import and patch df.utils and df.config FIRST
                import df.utils as df_utils
                import df.config as df_config

                # Patch get_device()
                def forced_cpu_device():
                    return torch.device('cpu')

                df_utils._original_get_device = getattr(df_utils, 'get_device', None)
                df_utils.get_device = forced_cpu_device
                logger.debug("Patched df.utils.get_device() to force CPU mode")

                # Patch config
                class MockConfig:
                    def __call__(self, key, default=None, section=None):
                        if key == "DEVICE" or key == "device":
                            return "cpu"
                        return default if default is not None else ""

                    def load(self, *args, **kwargs):
                        pass

                # --- FIX: Safely get original config using getattr ---
                # This prevents the "AttributeError: 'Config' object has no attribute 'config'"
                original_config = getattr(df_config, 'config', None)
                df_config.config = MockConfig()
                logger.debug("Patched df.config to return CPU device")

                # NOW force reload of other df modules so they pick up our patches
                modules_to_reload = [
                    'df.enhance', 'df.deepfilternet3', 'df.modules',
                    'df.model', 'df.checkpoint'
                ]
                for mod in modules_to_reload:
                    if mod in sys.modules:
                        del sys.modules[mod]

                # Now import df.enhance - it will use our patched functions
                import df.enhance as df_enhance_module

                # Load model, state, and df parameters
                model, state, df = df_enhance_module.init_df()

                # Force CPU for all components
                model = model.cpu()

                # Ensure all model parameters are on CPU
                for param in model.parameters():
                    if param.device.type != 'cpu':
                        param.data = param.data.cpu()

                # Store references
                df_func.model = model
                df_func.state = state
                df_func.df = df
                df_func.enhance_module = df_enhance_module
                df_func.original_config = original_config
                logger.info("✓ DeepFilterNet loaded (CPU only)")

            # Restore CUDA visibility (for rest of the program)
            if old_cuda_visible is None:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_visible

            # Convert AudioSegment samples to numpy float32
            samples = np.array(audio_chunk.get_array_of_samples(), dtype=np.float32)
            samples = samples / 32768.0

            # Enhance - with CUDA hidden again
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            try:
                with torch.no_grad():
                    # Keep input on CPU
                    audio_tensor = torch.from_numpy(samples).unsqueeze(0)

                    enhanced = df_func.enhance_module.enhance(
                        df_func.model,
                        df_func.state,
                        audio_tensor
                    )
            finally:
                if old_cuda_visible is None:
                    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                else:
                    os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_visible

            # Move result to CPU, convert to numpy, and flatten back to 1D array
            enhanced = enhanced.cpu().numpy().flatten()

            # Convert back to int16 PCM
            enhanced = (enhanced * 32768).astype(np.int16)

            return AudioSegment(
                enhanced.tobytes(),
                frame_rate=audio_chunk.frame_rate,
                sample_width=2,
                channels=1
            )

        except Exception as e:
            logger.warning(f"DeepFilterNet enhancement failed: {e}")
            logger.error("=" * 70)
            logger.error("DEEPFILTERNET ERROR")
            logger.error("If you see device mismatch errors, ensure no other process")
            logger.error("is holding onto the GPU memory.")
            logger.error("=" * 70)
            import traceback
            traceback.print_exc()
            return audio_chunk

    @staticmethod
    def load_and_prepare(audio_path: Path, logger: logging.Logger) -> Tuple[AudioSegment, int]:
        """Load audio file and convert to mono"""
        try:
            audio = AudioSegment.from_file(audio_path)
            original_sr = audio.frame_rate
            
            # Convert to mono once
            if audio.channels > 1:
                audio = audio.set_channels(1)
                logger.info("Converted to mono")
            
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
