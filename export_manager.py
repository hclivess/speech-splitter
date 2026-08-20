"""
Segment export: boundary buffers, cutoff detection, silence trimming, fragment removal,
file writing and dataset metadata (metadata.csv for Hugging Face AudioFolder, metadata.json, manifest.txt)
"""
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydub import AudioSegment

from audio_processor import AudioProcessor
from utils import SegmentData


class ExportManager:
    def __init__(self, output_dir: Path, params: Dict, logger: logging.Logger):
        self.output_dir = Path(output_dir)
        self.params = params
        self.logger = logger
        self.audio_dir = self.output_dir / "audio"
        self.text_dir = self.output_dir / "transcripts"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)
        self.records: Dict[str, Dict] = {}       # audio file name -> info (this run)
        self.stats = {"start_cutoffs": 0, "end_cutoffs": 0, "fragments_removed": 0, "skipped_cutoff": 0,
                      "skipped_long": 0}

    # ------------------------------------------------------------------ export
    def export_segments(self, audio: AudioSegment, segments: List[SegmentData], base_name: str,
                        should_stop=None, progress=None) -> List[Dict]:
        p = self.params
        fmt = p["export_format"]
        check_cutoff = bool(p.get("check_cutoff"))
        mode = p.get("cutoff_check_mode", "both")
        check_start, check_end = mode in ("both", "start"), mode in ("both", "end")
        counter = self._next_index(base_name, fmt)
        exported: List[Dict] = []

        for idx, seg in enumerate(segments, 1):
            if should_stop and should_stop():
                raise InterruptedError("stopped")
            if progress:
                progress(idx, len(segments))

            start_ms, end_ms = self._boundaries(audio, seg)
            chunk = audio[start_ms:end_ms]
            flags: List[str] = []

            if check_cutoff:
                sens = float(p.get("cutoff_sensitivity", 0.35))
                if check_start and self._edge_is_loud(chunk, True, sens):
                    self.stats["start_cutoffs"] += 1
                    flags.append("CUTSTART")
                if check_end and self._edge_is_loud(chunk, False, sens):
                    self.stats["end_cutoffs"] += 1
                    flags.append("CUTEND")
                if flags:
                    action = p.get("cutoff_action", "skip")
                    self.logger.debug(f"cutoff {'+'.join(flags)} in #{idx}: '{seg.text[:40]}'")
                    if action == "skip":
                        self.stats["skipped_cutoff"] += 1
                        continue
                    if action == "include":
                        flags = []

            if p.get("trim_silence", True):
                chunk = AudioProcessor.trim_silence(chunk, float(p["trim_threshold_db"]), int(p["trim_chunk_ms"]),
                                                    int(p["trim_pad_ms"]))
            if p.get("remove_fragments", True):
                cut = self._remove_end_fragment(chunk)
                if cut is not None:
                    self.stats["fragments_removed"] += 1
                    chunk = cut

            duration = len(chunk) / 1000.0
            if duration > p["max_segment_length"]:
                action = p.get("overlength_action", "split")
                if action == "skip":
                    self.stats["skipped_long"] += 1
                    continue
                if action in ("rename", "split"):     # split leftovers that still exceed max get marked
                    flags.append("LONG")

            target_sr = int(p.get("export_sample_rate") or 0)
            if target_sr and chunk.frame_rate != target_sr:
                chunk = chunk.set_frame_rate(target_sr)

            suffix = "".join(f"_{f}" for f in flags)
            name = f"{base_name}_{counter:04d}{suffix}"
            audio_file = self.audio_dir / f"{name}.{fmt}"
            text_file = self.text_dir / f"{name}.txt"
            while audio_file.exists() or text_file.exists():
                counter += 1
                name = f"{base_name}_{counter:04d}{suffix}"
                audio_file = self.audio_dir / f"{name}.{fmt}"
                text_file = self.text_dir / f"{name}.txt"

            export_kwargs = {"format": fmt}
            if fmt == "mp3":
                export_kwargs["bitrate"] = "256k"
            chunk.export(str(audio_file), **export_kwargs)
            text_file.write_text(seg.text, encoding="utf-8")

            info = {"file_name": f"audio/{audio_file.name}", "transcript": f"transcripts/{text_file.name}",
                    "text": seg.text, "duration": round(duration, 3), "sample_rate": chunk.frame_rate,
                    "confidence": round(seg.confidence, 4), "source": base_name, "flags": flags,
                    "classification": seg.classification}
            self.records[audio_file.name] = info
            exported.append(info)
            counter += 1

        s = self.stats
        self.logger.info(f"Exported {len(exported)} segments from {base_name}")
        if check_cutoff:
            self.logger.info(f"  cutoffs: start={s['start_cutoffs']} end={s['end_cutoffs']} skipped={s['skipped_cutoff']}")
        if s["fragments_removed"]:
            self.logger.info(f"  trailing fragments removed: {s['fragments_removed']}")
        if s["skipped_long"]:
            self.logger.info(f"  overlength segments skipped: {s['skipped_long']}")
        return exported

    def _next_index(self, base_name: str, fmt: str) -> int:
        """Continue numbering after files already in the output folder for this speaker"""
        best = 0
        for f in self.audio_dir.glob(f"{base_name}_[0-9][0-9][0-9][0-9]*.{fmt}"):
            tail = f.stem[len(base_name) + 1:][:4]
            if tail.isdigit():
                best = max(best, int(tail))
        return best + 1

    def _boundaries(self, audio: AudioSegment, seg: SegmentData) -> Tuple[int, int]:
        """Extend into the pauses around the segment, never into the neighbouring words"""
        start = max(0, seg.start_ms - int(self.params["word_start_buffer_ms"]))
        end = min(len(audio), seg.end_ms + int(self.params["word_end_buffer_ms"]))
        if seg.prev_word_end_ms is not None:
            start = max(start, min(seg.prev_word_end_ms, seg.start_ms))
        if seg.next_word_start_ms is not None:
            end = min(end, max(seg.next_word_start_ms, seg.end_ms))
        return start, end

    # --------------------------------------------------------- cutoff / trim
    @staticmethod
    def _edge_is_loud(chunk: AudioSegment, at_start: bool, sensitivity: float) -> bool:
        """Speech energy right at the boundary means the word was cut"""
        if len(chunk) < 50:
            return False
        samples = np.abs(np.array(chunk.get_array_of_samples(), dtype=np.float64))
        max_amp = float(2 ** (8 * chunk.sample_width - 1))
        n = max(50, min(int(chunk.frame_rate * 0.01), len(samples) // 10))
        window = samples[:n] if at_start else samples[-n:]
        if window.size > 10:
            k = max(2, min(5, window.size // 5))
            window = np.convolve(window, np.ones(k) / k, mode="same")
        return float(window.max() / max_amp) > max(0.0, min(1.0, sensitivity))

    def _remove_end_fragment(self, chunk: AudioSegment, threshold_db: float = -40.0,
                             min_silence_ms: int = 200, max_fragment_ms: int = 300) -> Optional[AudioSegment]:
        """Drop a short burst of sound that follows a clear pause at the very end (a clipped next word)"""
        if len(chunk) < min_silence_ms + 100:
            return None
        win = 10
        db = AudioProcessor.rms_db_windows(chunk, win)
        if db.size < 10:
            return None
        loud = db > threshold_db
        idx = np.where(loud)[0]
        if idx.size == 0 or idx[-1] < 10:
            return None
        last = idx[-1]
        frag_start = last
        while frag_start > 0 and loud[frag_start - 1]:
            frag_start -= 1
        sil_start = frag_start
        while sil_start > 0 and not loud[sil_start - 1]:
            sil_start -= 1
        if sil_start == frag_start:
            return None
        fragment_ms = (last - frag_start + 1) * win
        silence_ms = (frag_start - sil_start) * win
        if silence_ms >= min_silence_ms and fragment_ms <= max_fragment_ms and sil_start > 0:
            return chunk[: sil_start * win]
        return None

    # ---------------------------------------------------------------- metadata
    def generate_metadata(self, language: str):
        """Rebuild metadata for everything in the output folder (cumulative across runs)"""
        fmt = self.params["export_format"]
        rows: List[Dict] = []
        for audio_file in sorted(self.audio_dir.glob(f"*.{fmt}")):
            text_file = self.text_dir / f"{audio_file.stem}.txt"
            if not text_file.exists():
                continue
            info = self.records.get(audio_file.name)
            if info is None:
                try:
                    a = AudioSegment.from_file(str(audio_file))
                    info = {"file_name": f"audio/{audio_file.name}", "transcript": f"transcripts/{text_file.name}",
                            "text": text_file.read_text(encoding="utf-8").strip(),
                            "duration": round(len(a) / 1000.0, 3), "sample_rate": a.frame_rate}
                except Exception as exc:
                    self.logger.error(f"Cannot read {audio_file.name}: {exc}")
                    continue
            rows.append(info)

        total = sum(r["duration"] for r in rows)
        meta = {"total_segments": len(rows), "total_duration": round(total, 2), "language": language,
                "segments": rows}
        (self.output_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")
        with open(self.output_dir / "manifest.txt", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(f"{r['file_name']}|{r['transcript']}\n")
        # Hugging Face AudioFolder: file_name relative to metadata.csv + transcription column
        with open(self.output_dir / "metadata.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file_name", "transcription", "duration"])
            for r in rows:
                w.writerow([r["file_name"], r["text"], r["duration"]])
        self.logger.info(f"Metadata written: {len(rows)} segments, {total / 60:.1f} min "
                         f"(metadata.csv, metadata.json, manifest.txt)")
        return meta
