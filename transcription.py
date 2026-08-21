"""
Transcription with word timestamps using faster-whisper (CTranslate2).
Results are cached next to the source file as <name>.transcription.json and reused when
the same model + language is requested again.
"""
import json
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

from audio_processor import AudioProcessor
from cuda_utils import setup_cuda
from utils import atomic_write_text

CACHE_VERSION = 3


def cuda_available(cuda_lib_dir: str = "") -> bool:
    try:
        setup_cuda(cuda_lib_dir)
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def resolve_device(device: str, cuda_lib_dir: str = "") -> str:
    if device == "auto":
        return "cuda" if cuda_available(cuda_lib_dir) else "cpu"
    if device == "cuda":
        setup_cuda(cuda_lib_dir)
    return device


def model_repo_id(model: str) -> str:
    try:
        from faster_whisper.utils import _MODELS
        return _MODELS.get(model, model)
    except Exception:
        return model


def is_model_cached(model: str, model_dir: str = "") -> bool:
    if os.path.isdir(model) and os.path.isfile(os.path.join(model, "model.bin")):
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
        res = try_to_load_from_cache(model_repo_id(model), "model.bin", cache_dir=model_dir or None)
        return isinstance(res, str) and os.path.isfile(res)
    except Exception:
        return False


def download_model(model: str, model_dir: str = "") -> str:
    from faster_whisper import download_model as _dl
    return _dl(model, cache_dir=model_dir or None)


class TranscriptionManager:
    def __init__(self, params: Dict, logger: logging.Logger,
                 status: Optional[Callable[[str], None]] = None):
        self.params = params
        self.logger = logger
        self.status = status or (lambda m: None)
        self.model_name = params["whisper_model"]
        lang = params.get("language", "auto")
        self.language = None if lang in ("", "auto") else lang
        self.device = resolve_device(params.get("device", "auto"), params.get("cuda_lib_dir", ""))
        compute = params.get("compute_type", "default")
        if compute == "default":
            compute = "float16" if self.device == "cuda" else "int8"
        self.compute_type = compute
        self.model = None

    def load(self):
        if self.model is not None:
            return
        from faster_whisper import WhisperModel
        self.status(f"Loading model {self.model_name} ({self.device}, {self.compute_type})")
        self.logger.info(f"Loading faster-whisper model {self.model_name} on {self.device} [{self.compute_type}]")
        kwargs = {"device": self.device, "compute_type": self.compute_type,
                  "download_root": self.params.get("model_dir") or None}
        if self.device == "cpu" and int(self.params.get("cpu_threads") or 0) > 0:
            kwargs["cpu_threads"] = int(self.params["cpu_threads"])
        try:
            self.model = WhisperModel(self.model_name, **kwargs)
        except Exception as exc:
            if self.device == "cuda":
                self.logger.warning(f"CUDA model load failed ({exc}); falling back to CPU")
                self.device, self.compute_type = "cpu", "int8"
                kwargs.update(device="cpu", compute_type="int8")
                self.model = WhisperModel(self.model_name, **kwargs)
            else:
                raise
        self.logger.info("Model loaded")

    def _cache_path(self, source: Path) -> Path:
        return source.with_suffix(".transcription.json")

    def _cache_valid(self, data: Dict) -> bool:
        return (data.get("cache_version") == CACHE_VERSION and data.get("model") == self.model_name
                and (self.language is None or data.get("language") == self.language) and bool(data.get("words")))

    def transcribe(self, audio, source_path: Path, progress: Optional[Callable[[float, float], None]] = None,
                   should_stop: Optional[Callable[[], bool]] = None) -> Dict:
        """audio: pydub AudioSegment (mono). Returns {'text', 'words', 'segments', 'language', ...}"""
        cache = self._cache_path(source_path)
        if self.params.get("use_transcription_cache", True) and cache.exists():
            try:
                data = json.loads(cache.read_text(encoding="utf-8"))
                if self._cache_valid(data):
                    self.logger.info(f"Using cached transcription {cache.name}")
                    return data
                self.logger.info("Cached transcription is from a different model/language - re-transcribing")
            except Exception as exc:
                self.logger.warning(f"Cache unreadable ({exc}) - re-transcribing")

        self.load()
        samples = AudioProcessor.to_float32(audio, 16000)
        total = len(samples) / 16000.0
        self.status("Transcribing")
        segments_iter, info = self.model.transcribe(
            samples,
            language=self.language,
            beam_size=int(self.params.get("beam_size", 5)),
            word_timestamps=True,
            vad_filter=bool(self.params.get("vad_filter", True)),
            vad_parameters={"min_silence_duration_ms": 500},
            initial_prompt=self.params.get("initial_prompt") or None,
            condition_on_previous_text=True,
        )
        detected = info.language
        if self.language is None:
            self.logger.info(f"Detected language: {detected} (p={info.language_probability:.2f})")

        words: List[Dict] = []
        segs: List[Dict] = []
        for seg in segments_iter:
            if should_stop and should_stop():
                raise InterruptedError("stopped")
            segs.append({"start": seg.start, "end": seg.end, "text": seg.text.strip(),
                         "no_speech_prob": seg.no_speech_prob, "avg_logprob": seg.avg_logprob})
            if seg.words:
                for w in seg.words:
                    words.append({"word": w.word.strip(), "start": float(w.start), "end": float(w.end),
                                  "probability": float(w.probability)})
            else:
                toks = seg.text.split()
                if toks:
                    step = (seg.end - seg.start) / len(toks)
                    for i, t in enumerate(toks):
                        words.append({"word": t, "start": seg.start + i * step, "end": seg.start + (i + 1) * step,
                                      "probability": 1.0})
            if progress:
                progress(min(seg.end, total), total)

        data = {
            "cache_version": CACHE_VERSION,
            "model": self.model_name,
            "language": detected,
            "text": " ".join(w["word"] for w in words),
            "words": words,
            "segments": segs,
        }
        self.logger.info(f"Transcription complete: {len(segs)} segments, {len(words)} words")
        if self.params.get("use_transcription_cache", True):
            try:
                atomic_write_text(cache, json.dumps(data, ensure_ascii=False, indent=1))
            except Exception as exc:
                self.logger.warning(f"Could not save transcription cache: {exc}")
        return data
