"""
Audio classification with YAMNet (AudioSet, 521 classes) running on onnxruntime.
Used to reject music / noise / silence and to validate segment boundaries.
The model (~16 MB, Apache-2.0 conversion of google/yamnet) is fetched from the Hugging Face
hub on first use and cached like the Whisper models.
"""
import csv
import hashlib
import logging
import os
from typing import Dict, List, Optional

import numpy as np
from pydub import AudioSegment

from audio_processor import AudioProcessor
from config import YAMNET_FILES, YAMNET_REPO
from utils import MAX_CACHE_SIZE, AudioClassification


def is_yamnet_cached(model_dir: str = "") -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        for fn in YAMNET_FILES:
            res = try_to_load_from_cache(YAMNET_REPO, fn, cache_dir=model_dir or None)
            if not (isinstance(res, str) and os.path.isfile(res)):
                return False
        return True
    except Exception:
        return False


def download_yamnet(model_dir: str = "") -> List[str]:
    from huggingface_hub import hf_hub_download
    return [hf_hub_download(YAMNET_REPO, fn, cache_dir=model_dir or None) for fn in YAMNET_FILES]


class YAMNetFilter:
    SAMPLE_RATE = 16000

    def __init__(self, logger: Optional[logging.Logger] = None, model_dir: str = ""):
        self.logger = logger or logging.getLogger("SpeechSplitter.YAMNet")
        self.model_dir = model_dir
        self.session = None
        self.input_name = ""
        self.class_names: List[str] = []
        self._cache: Dict[str, AudioClassification] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.available = self._load()

    def _load(self) -> bool:
        try:
            import onnxruntime as ort
            model_path, class_map = download_yamnet(self.model_dir)
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            with open(class_map, encoding="utf-8") as f:
                self.class_names = [row["display_name"] for row in csv.DictReader(f)]
            self._precompute_indices()
            self.logger.info(f"YAMNet ready ({len(self.class_names)} classes)")
            return True
        except Exception as exc:
            self.logger.warning(f"YAMNet unavailable ({exc}) - audio filtering disabled")
            return False

    def _precompute_indices(self):
        low = [n.lower() for n in self.class_names]
        self.music_idx = [i for i, n in enumerate(low) if "music" in n and "speech" not in n]
        self.speech_idx = [i for i, n in enumerate(low) if "speech" in n and "music" not in n]
        self.mixed_idx = [i for i, n in enumerate(low) if "music" in n and "speech" in n]
        self.silence_idx = [i for i, n in enumerate(low) if "silence" in n or "quiet" in n]
        noise_terms = ("noise", "hum", "buzz", "static", "hiss", "wind", "rain", "crackle", "rumble")
        self.noise_idx = [i for i, n in enumerate(low) if any(t in n for t in noise_terms)]
        # "Singing" and friends count as music for TTS purposes
        self.music_idx += [i for i, n in enumerate(low)
                           if any(t in n for t in ("singing", "choir", "yodeling", "chant", "rapping"))
                           and i not in self.music_idx]

    @staticmethod
    def _key(samples: np.ndarray) -> str:
        h = hashlib.blake2b(digest_size=12)
        h.update(samples.shape[0].to_bytes(4, "little"))
        h.update(samples[: 4000].tobytes())
        h.update(samples[-4000:].tobytes())
        return h.hexdigest()

    def classify_samples(self, samples: np.ndarray) -> AudioClassification:
        """samples: float32 mono 16 kHz"""
        if not self.available:
            return AudioClassification.speech_only()
        if samples.size < self.SAMPLE_RATE // 2:     # pad to >= 0.5 s so YAMNet emits a frame
            samples = np.pad(samples, (0, self.SAMPLE_RATE // 2 - samples.size))
        key = self._key(samples)
        hit = self._cache.get(key)
        if hit is not None:
            self.cache_hits += 1
            return hit
        self.cache_misses += 1
        try:
            scores = self.session.run(None, {self.input_name: samples.astype(np.float32)})[0]
            mean = scores.mean(axis=0)

            def top(idx):
                return float(mean[idx].max()) if idx else 0.0
            result = AudioClassification(music=top(self.music_idx), speech=top(self.speech_idx),
                                         music_speech_mixed=top(self.mixed_idx), silence=top(self.silence_idx),
                                         noise=top(self.noise_idx))
        except Exception as exc:
            self.logger.debug(f"YAMNet inference error: {exc}")
            return AudioClassification.speech_only()
        if len(self._cache) >= MAX_CACHE_SIZE:
            for k in list(self._cache)[: MAX_CACHE_SIZE // 4]:
                del self._cache[k]
        self._cache[key] = result
        return result

    def classify_audio(self, chunk: AudioSegment) -> AudioClassification:
        if not self.available:
            return AudioClassification.speech_only()
        return self.classify_samples(AudioProcessor.to_float32(chunk, self.SAMPLE_RATE))

    def cache_stats(self) -> Dict[str, float]:
        total = self.cache_hits + self.cache_misses
        return {"hits": self.cache_hits, "misses": self.cache_misses,
                "hit_rate": (self.cache_hits / total * 100) if total else 0.0}
