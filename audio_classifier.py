"""
Audio classification using YAMNet (optional)
Provides music/speech/silence detection for filtering
"""

import os
import csv
import logging
import hashlib
import numpy as np
from typing import Dict, Optional
from pydub import AudioSegment

from utils import AudioClassification, MAX_CACHE_SIZE, CACHE_CLEANUP_RATIO


class YAMNetFilter:
    """
    Optimized YAMNet audio classifier with:
    - Classification caching
    - Pre-computed class indices
    - Single classification per segment
    - Optional fallback mode
    """
    
    def __init__(self, device='cuda', logger=None):
        self.model = None
        self.class_names = None
        self.device = device
        self.logger = logger or logging.getLogger("SpeechSplitter.YAMNetFilter")
        
        # Classification cache
        self._classification_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Pre-computed class indices
        self.music_indices = None
        self.speech_indices = None
        self.mixed_indices = None
        self.silence_indices = None
        
        self.available = self._load_model()
    
    def _load_model(self) -> bool:
        """Load YAMNet and pre-compute class indices"""
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            import shutil
            import tempfile
            import time
            
            # GPU config
            gpus = tf.config.list_physical_devices('GPU')
            if gpus and self.device == 'cuda':
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    self.logger.info("TensorFlow GPU memory growth enabled")
                except RuntimeError as e:
                    self.logger.warning(f"GPU config error: {e}")
            
            tf.get_logger().setLevel('ERROR')
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            
            self.logger.info("Loading YAMNet model...")
            
            # Clear corrupted cache if exists
            tfhub_cache = os.path.join(tempfile.gettempdir(), 'tfhub_modules')
            if os.path.exists(tfhub_cache):
                try:
                    shutil.rmtree(tfhub_cache)
                    os.makedirs(tfhub_cache, exist_ok=True)
                except Exception as e:
                    self.logger.warning(f"Failed to clear cache: {e}")
            
            # Load model with retry
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    self.model = hub.load('https://tfhub.dev/google/yamnet/1')
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                        time.sleep(2)
                    else:
                        raise
            
            class_map_path = self.model.class_map_path().numpy()
            self.class_names = self._load_class_names(class_map_path)
            
            # Pre-compute class indices once
            self._precompute_class_indices()
            
            self.logger.info("✓ YAMNet model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.warning(f"YAMNet loading failed: {e}")
            self.logger.warning("Audio filtering disabled - using timestamp-only mode")
            self.model = None
            return False
    
    def _load_class_names(self, class_map_csv):
        """Load YAMNet class names"""
        class_names = []
        with open(class_map_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_names.append(row['display_name'])
        return class_names

    def _precompute_class_indices(self):
        """Calculate class indices once instead of per-classification"""
        self.music_indices = [
            i for i, name in enumerate(self.class_names)
            if 'music' in name.lower() and 'speech' not in name.lower()
        ]
        self.speech_indices = [
            i for i, name in enumerate(self.class_names)
            if 'speech' in name.lower() and 'music' not in name.lower()
        ]
        self.mixed_indices = [
            i for i, name in enumerate(self.class_names)
            if 'music' in name.lower() and 'speech' in name.lower()
        ]
        self.silence_indices = [
            i for i, name in enumerate(self.class_names)
            if 'silence' in name.lower() or 'quiet' in name.lower()
        ]
        # NEW: Noise indices - various noise categories
        self.noise_indices = [
            i for i, name in enumerate(self.class_names)
            if any(term in name.lower() for term in
                   ['noise', 'hum', 'buzz', 'static', 'white noise', 'pink noise', 'brown noise'])
        ]

        self.logger.info(f"Class indices: music={len(self.music_indices)}, "
                         f"speech={len(self.speech_indices)}, "
                         f"mixed={len(self.mixed_indices)}, "
                         f"silence={len(self.silence_indices)}, "
                         f"noise={len(self.noise_indices)}")
    
    def _get_cache_key(self, audio_segment: AudioSegment) -> str:
        """Generate stable cache key from audio segment properties"""
        samples = np.array(audio_segment.get_array_of_samples()[:1000], dtype=np.int16)
        key_data = f"{len(audio_segment)}_{audio_segment.frame_rate}_{audio_segment.channels}_{samples.tobytes()}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def classify_audio(self, audio_segment: AudioSegment) -> AudioClassification:
        """Classify audio with caching - returns safe defaults if model unavailable"""
        if not self.available or self.model is None:
            # Return safe defaults that won't filter out speech
            return AudioClassification(music=0.0, speech=1.0, music_speech_mixed=0.0, silence=0.0, noise=0.0)
        
        # Check cache first
        cache_key = self._get_cache_key(audio_segment)
        if cache_key in self._classification_cache:
            self._cache_hits += 1
            return self._classification_cache[cache_key]
        
        self._cache_misses += 1
        
        try:
            import tensorflow as tf
            
            samples = np.array(audio_segment.get_array_of_samples())
            
            # Normalize samples
            if audio_segment.sample_width == 2:
                samples = samples.astype(np.float32) / 32768.0
            elif audio_segment.sample_width == 4:
                samples = samples.astype(np.float32) / 2147483648.0
            
            scores, _, _ = self.model(samples)
            mean_scores = np.mean(scores.numpy(), axis=0)
            
            # Use pre-computed indices
            music_prob = float(np.max([mean_scores[i] for i in self.music_indices]) if self.music_indices else 0.0)
            speech_prob = float(np.max([mean_scores[i] for i in self.speech_indices]) if self.speech_indices else 0.0)
            mixed_prob = float(np.max([mean_scores[i] for i in self.mixed_indices]) if self.mixed_indices else 0.0)
            silence_prob = float(np.max([mean_scores[i] for i in self.silence_indices]) if self.silence_indices else 0.0)
            noise_prob = float(np.max([mean_scores[i] for i in self.noise_indices]) if self.noise_indices else 0.0)

            result = AudioClassification(
                music=music_prob,
                speech=speech_prob,
                music_speech_mixed=mixed_prob,
                silence=silence_prob,
                noise=noise_prob
            )
            
            # Cache result
            self._classification_cache[cache_key] = result
            
            # Prevent cache from growing too large
            if len(self._classification_cache) > MAX_CACHE_SIZE:
                remove_count = int(len(self._classification_cache) * CACHE_CLEANUP_RATIO)
                for key in list(self._classification_cache.keys())[:remove_count]:
                    del self._classification_cache[key]
            
            return result
            
        except Exception as e:
            self.logger.info(f"YAMNet classification error: {e}")
            # Return safe defaults on error
            return AudioClassification(music=0.0, speech=1.0, music_speech_mixed=0.0, silence=0.0, noise=0.0)
    
    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache performance statistics"""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._classification_cache)
        }
    
    def clear_cache(self):
        """Clear classification cache"""
        self._classification_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
