#!/usr/bin/env python3
"""
speech-splitter - turn audio into a TTS training dataset.

Pipeline:  load audio -> faster-whisper (word timestamps) -> sentence segmentation
           -> YAMNet quality filter -> export clips + transcripts + metadata.csv

Run `python speech_splitter.py --help` for the CLI, or `python speech_splitter_gui.py` for the GUI.
"""
import argparse
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import AUDIO_EXTENSIONS, DEFAULT_SETTINGS, LANGUAGES  # noqa: E402
from utils import setup_logging  # noqa: E402


class SpeechSplitter:
    """Orchestrates one run (one set of settings, one output folder, any number of files)"""

    def __init__(self, settings: Dict, logger: Optional[logging.Logger] = None,
                 status: Optional[Callable[[str], None]] = None,
                 progress: Optional[Callable[[str, float, float], None]] = None,
                 should_stop: Optional[Callable[[], bool]] = None):
        self.params = dict(DEFAULT_SETTINGS)
        self.params.update({k: v for k, v in settings.items() if v is not None})
        self.output_dir = Path(self.params["output_dir"] or "output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or setup_logging(self.output_dir, self.params["log_level"])
        self.status = status or (lambda m: None)
        self.progress = progress or (lambda phase, done, total: None)
        self.should_stop = should_stop or (lambda: False)

        from audio_processor import configure_ffmpeg
        from transcription import TranscriptionManager
        from audio_classifier import YAMNetFilter
        from segmentation import SegmentationEngine
        from segment_filter import SegmentFilter
        from export_manager import ExportManager

        configure_ffmpeg(self.logger)
        self.transcriber = TranscriptionManager(self.params, self.logger, self.status)
        self.yamnet = None
        if self.params["filter_music"] or self.params["filter_noise"] or self.params["filter_silence"]:
            self.status("Loading YAMNet classifier")
            self.yamnet = YAMNetFilter(self.logger, self.params.get("model_dir", ""))
        self.segmenter = SegmentationEngine(self.params, self.logger)
        self.filter = SegmentFilter(self.params, self.yamnet, self.logger)
        self.exporter = ExportManager(self.output_dir, self.params, self.logger)
        self.detected_language = None

    @property
    def device(self) -> str:
        return self.transcriber.device

    def process_file(self, audio_path, speaker_id: Optional[str] = None) -> Dict:
        from audio_processor import AudioProcessor
        audio_path = Path(audio_path)
        base = speaker_id or audio_path.stem
        self.logger.info(f"=== {audio_path.name} (speaker id '{base}')")

        self.status("Loading audio")
        audio, sr = AudioProcessor.load_and_prepare(audio_path, self.logger)

        transcription = self.transcriber.transcribe(
            audio, audio_path, progress=lambda d, t: self.progress("transcribe", d, t), should_stop=self.should_stop)
        self.detected_language = transcription.get("language")

        self.status("Segmenting")
        segments = self.segmenter.segment(transcription["words"])
        if not segments:
            self.logger.warning("No usable segments")
            return {"segments": 0, "duration": 0.0, "rejected": 0}

        self.status("Filtering")
        kept = self.filter.filter_segments(segments, audio, should_stop=self.should_stop,
                                           progress=lambda d, t: self.progress("filter", d, t))

        self.status("Exporting")
        exported = self.exporter.export_segments(audio, kept, base, should_stop=self.should_stop,
                                                 progress=lambda d, t: self.progress("export", d, t))
        if self.yamnet and self.yamnet.available:
            st = self.yamnet.cache_stats()
            self.logger.debug(f"YAMNet cache: {st['hits']} hits / {st['misses']} misses")
        total = sum(e["duration"] for e in exported)
        self.logger.info(f"{audio_path.name}: {len(exported)} clips, {total / 60:.1f} min of speech")
        return {"segments": len(exported), "duration": total, "rejected": len(segments) - len(exported)}

    def generate_metadata(self) -> Dict:
        lang = self.params["language"]
        if lang in ("", "auto"):
            lang = self.detected_language or "auto"
        return self.exporter.generate_metadata(lang)


def collect_inputs(paths: List[str]) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files += sorted(f for f in path.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS)
        elif path.is_file():
            files.append(path)
    return files


def build_parser() -> argparse.ArgumentParser:
    d = DEFAULT_SETTINGS
    ap = argparse.ArgumentParser(description="speech-splitter - turn audio into a TTS training dataset",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("input", nargs="+", help="audio/video files or folders")
    ap.add_argument("-o", "--output-dir", default=d["output_dir"])
    ap.add_argument("--speaker-id", default=None, help="name prefix for clips (default: source file name)")
    ap.add_argument("--format", dest="export_format", choices=["wav", "flac", "mp3"], default=d["export_format"])
    ap.add_argument("--sample-rate", dest="export_sample_rate", type=int, default=0, help="0 = keep original")

    m = ap.add_argument_group("model")
    m.add_argument("--model", dest="whisper_model", default=d["whisper_model"],
                   help="faster-whisper model size or Hugging Face repo id")
    m.add_argument("--language", default=d["language"], help="language code or 'auto' (%s ...)"
                   % ", ".join(list(LANGUAGES)[1:8]))
    m.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    m.add_argument("--compute-type", default="default", choices=["default", "int8", "int8_float16", "float16", "float32"])
    m.add_argument("--beam-size", type=int, default=d["beam_size"])
    m.add_argument("--no-vad", dest="vad_filter", action="store_false", help="disable Silero VAD pre-filter")
    m.add_argument("--initial-prompt", default="", help="vocabulary / style hint for Whisper")
    m.add_argument("--model-dir", default="", help="model cache folder (default: Hugging Face cache)")
    m.add_argument("--cuda-lib-dir", default="", help="folder with cuBLAS 12 / cuDNN 9 libraries")
    m.add_argument("--no-cache", dest="use_transcription_cache", action="store_false",
                   help="ignore / don't write <file>.transcription.json")

    s = ap.add_argument_group("segmentation")
    s.add_argument("--min-length", dest="min_segment_length", type=float, default=d["min_segment_length"])
    s.add_argument("--max-length", dest="max_segment_length", type=float, default=d["max_segment_length"])
    s.add_argument("--no-merge-short", dest="merge_short", action="store_false",
                   help="drop short sentences instead of merging them with the next one")
    s.add_argument("--merge-max-gap", type=float, default=d["merge_max_gap"], help="max pause (s) to merge across")
    s.add_argument("--overlength-action", choices=["split", "rename", "skip", "include"], default=d["overlength_action"])
    s.add_argument("--word-start-buffer", dest="word_start_buffer_ms", type=int, default=d["word_start_buffer_ms"])
    s.add_argument("--word-end-buffer", dest="word_end_buffer_ms", type=int, default=d["word_end_buffer_ms"])

    q = ap.add_argument_group("quality")
    q.add_argument("--min-confidence", type=float, default=d["min_confidence"])
    q.add_argument("--no-filter-music", dest="filter_music", action="store_false")
    q.add_argument("--music-threshold", type=float, default=d["music_threshold"])
    q.add_argument("--mixed-threshold", type=float, default=d["mixed_threshold"])
    q.add_argument("--filter-noise", action="store_true")
    q.add_argument("--noise-threshold", type=float, default=d["noise_threshold"])
    q.add_argument("--filter-silence", action="store_true")
    q.add_argument("--silence-threshold", type=float, default=d["silence_threshold"])
    q.add_argument("--skip-edge-segments", action="store_true", help="drop N segments at the start and end of each file")
    q.add_argument("--skip-segments-count", type=int, default=d["skip_segments_count"])

    c = ap.add_argument_group("cutoff detection")
    c.add_argument("--check-cutoff", action="store_true")
    c.add_argument("--cutoff-check-mode", choices=["both", "start", "end"], default=d["cutoff_check_mode"])
    c.add_argument("--cutoff-sensitivity", type=float, default=d["cutoff_sensitivity"],
                   help="0-1, lower = more sensitive")
    c.add_argument("--cutoff-action", choices=["skip", "rename", "include"], default=d["cutoff_action"])

    t = ap.add_argument_group("trimming")
    t.add_argument("--no-trim", dest="trim_silence", action="store_false")
    t.add_argument("--trim-threshold-db", type=float, default=d["trim_threshold_db"])
    t.add_argument("--trim-chunk-ms", type=int, default=d["trim_chunk_ms"])
    t.add_argument("--trim-pad-ms", type=int, default=d["trim_pad_ms"])
    t.add_argument("--keep-fragments", dest="remove_fragments", action="store_false",
                   help="don't remove short bursts after a pause at clip ends")

    ap.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    settings = {k: v for k, v in vars(args).items() if k != "input"}
    files = collect_inputs(args.input)
    if not files:
        print("No input files found", file=sys.stderr)
        return 1
    splitter = SpeechSplitter(settings, status=lambda m: print(f"-- {m}", flush=True))
    print(f"Device: {splitter.device}   files: {len(files)}", flush=True)
    failed = 0
    for f in files:
        try:
            splitter.process_file(f, args.speaker_id)
        except KeyboardInterrupt:
            print("\nInterrupted")
            break
        except Exception as exc:  # noqa: BLE001
            failed += 1
            splitter.logger.error(f"{f.name}: {exc}")
    meta = splitter.generate_metadata()
    print(f"Done: {meta['total_segments']} clips, {meta['total_duration'] / 60:.1f} min -> {splitter.output_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
