"""
Quality filtering of segments: music, noise, silence (YAMNet), transcription confidence, edge segments
"""
import logging
from typing import Dict, List, Optional

from pydub import AudioSegment

from utils import AudioClassification, SegmentData


class SegmentFilter:
    def __init__(self, params: Dict, yamnet, logger: logging.Logger):
        self.params = params
        self.yamnet = yamnet if (yamnet is not None and yamnet.available) else None
        self.logger = logger

    def _needs_classifier(self) -> bool:
        return bool(self.params["filter_music"] or self.params["filter_noise"] or self.params["filter_silence"])

    def filter_segments(self, segments: List[SegmentData], audio: AudioSegment,
                        should_stop=None, progress=None) -> List[SegmentData]:
        p = self.params
        stats = {"music": 0, "noise": 0, "silence": 0, "confidence": 0, "edge": 0}
        examples: Dict[str, List[str]] = {k: [] for k in stats}

        edge = set()
        if p.get("skip_edge_segments") and len(segments) > 2 * int(p.get("skip_segments_count", 1)):
            n = int(p["skip_segments_count"])
            edge = set(range(n)) | set(range(len(segments) - n, len(segments)))

        use_classifier = self.yamnet is not None and self._needs_classifier()
        if self._needs_classifier() and self.yamnet is None:
            self.logger.warning("Audio filters requested but YAMNet is unavailable - skipping music/noise/silence checks")

        kept: List[SegmentData] = []
        for idx, seg in enumerate(segments):
            if should_stop and should_stop():
                raise InterruptedError("stopped")
            if progress:
                progress(idx + 1, len(segments))

            def reject(reason: str, detail: str = ""):
                stats[reason] += 1
                if len(examples[reason]) < 3:
                    examples[reason].append(f"'{seg.text[:50]}' {detail}".strip())

            if idx in edge:
                reject("edge", "(start)" if idx < len(segments) / 2 else "(end)")
                continue

            if use_classifier:
                c = self.yamnet.classify_audio(audio[seg.start_ms:seg.end_ms])
                seg.classification = c.to_dict()
                if p["filter_music"] and c.is_music(p["music_threshold"], p["mixed_threshold"]):
                    reject("music", f"(music={c.music:.2f}, mixed={c.music_speech_mixed:.2f})")
                    continue
                if p["filter_noise"] and c.is_noise(p["noise_threshold"]):
                    reject("noise", f"(noise={c.noise:.2f})")
                    continue
                if p["filter_silence"] and c.is_silent(p["silence_threshold"]):
                    reject("silence", f"(silence={c.silence:.2f})")
                    continue
            else:
                seg.classification = None

            if seg.confidence < p["min_confidence"]:
                reject("confidence", f"(conf={seg.confidence:.2f})")
                continue
            kept.append(seg)

        total = len(segments) or 1
        self.logger.info(f"Filtering: {len(kept)}/{len(segments)} segments kept ({len(kept) / total * 100:.0f}%)")
        labels = {"edge": "edge segments", "music": "music", "noise": "noise", "silence": "silence",
                  "confidence": "low confidence"}
        for key, label in labels.items():
            if stats[key]:
                self.logger.info(f"  rejected {stats[key]} for {label}")
                for ex in examples[key]:
                    self.logger.debug(f"     {ex}")
        return kept
