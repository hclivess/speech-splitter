"""
Sentence-based segmentation of word-level transcripts.

- splits at sentence punctuation (. ! ? … and their CJK forms) when preceded by a letter
  (so "4." or "3.5" never split), in any script
- merges sentences shorter than min_segment_length with the following sentence when the
  pause between them is short, instead of throwing them away
- sentences longer than max_segment_length are split at the longest pause (preferring
  commas) when overlength_action == "split"
- segments end at the last word's END, never at the next word's start
"""
import logging
import re
from typing import Dict, List, Optional

from utils import SegmentData

SENTENCE_END = re.compile(r"[^\W\d_][\"'»)\]]*[.!?…。！？]+[\"'»)\]]*$", re.UNICODE)
CLAUSE_END = re.compile(r"[,;:，；、—–-]$")
INVALID_TOKENS = ("[blank_audio]", "[music]", "[applause]", "[laughter]", "[noise]", "(music)", "♪")


class SegmentationEngine:
    def __init__(self, params: Dict, logger: logging.Logger):
        self.params = params
        self.logger = logger

    # ------------------------------------------------------------------ public
    def segment(self, words: List[Dict]) -> List[SegmentData]:
        words = [w for w in words if "start" in w and "end" in w and w.get("word", "").strip()]
        if not words:
            self.logger.warning("No words to segment")
            return []

        sentences = self._split_sentences(words)
        self.logger.info(f"Sentences found: {len(sentences)}")

        if self.params.get("merge_short", True):
            sentences = self._merge_short(sentences)

        if self.params.get("overlength_action", "split") == "split":
            sentences = self._split_long(sentences)

        segments: List[SegmentData] = []
        for group in sentences:
            seg = self._make_segment(group, words)
            if seg is None:
                continue
            if seg.duration < self.params["min_segment_length"]:
                continue
            if not self._valid_text(seg.text):
                continue
            segments.append(seg)
        self.logger.info(f"Segments created: {len(segments)}")
        return segments

    # ----------------------------------------------------------------- helpers
    def _split_sentences(self, words: List[Dict]) -> List[List[Dict]]:
        groups, current = [], []
        for i, w in enumerate(words):
            current.append(w)
            if SENTENCE_END.search(w["word"].strip()) or i == len(words) - 1:
                groups.append(current)
                current = []
        return groups

    @staticmethod
    def _dur(group: List[Dict]) -> float:
        return group[-1]["end"] - group[0]["start"]

    def _merge_short(self, groups: List[List[Dict]]) -> List[List[Dict]]:
        min_len = self.params["min_segment_length"]
        max_len = self.params["max_segment_length"]
        max_gap = float(self.params.get("merge_max_gap", 1.0))
        out: List[List[Dict]] = []
        i = 0
        while i < len(groups):
            cur = groups[i]
            while (self._dur(cur) < min_len and i + 1 < len(groups)
                   and groups[i + 1][0]["start"] - cur[-1]["end"] <= max_gap
                   and groups[i + 1][-1]["end"] - cur[0]["start"] <= max_len):
                cur = cur + groups[i + 1]
                i += 1
            # a short trailing sentence can also join the previous one
            if self._dur(cur) < min_len and out and cur[0]["start"] - out[-1][-1]["end"] <= max_gap \
                    and cur[-1]["end"] - out[-1][0]["start"] <= max_len:
                out[-1] = out[-1] + cur
            else:
                out.append(cur)
            i += 1
        return out

    def _split_long(self, groups: List[List[Dict]]) -> List[List[Dict]]:
        max_len = self.params["max_segment_length"]
        min_len = self.params["min_segment_length"]
        out: List[List[Dict]] = []
        stack = list(reversed(groups))
        while stack:
            g = stack.pop()
            if self._dur(g) <= max_len or len(g) < 4:
                out.append(g)
                continue
            cut = self._best_cut(g, min_len)
            if cut is None:
                out.append(g)
                continue
            stack.append(g[cut:])
            stack.append(g[:cut])
        return out

    def _best_cut(self, g: List[Dict], min_len: float) -> Optional[int]:
        """Index to cut before: longest pause, with a bonus for clause punctuation; keeps both halves >= min_len"""
        best, best_score = None, -1.0
        for i in range(1, len(g)):
            left = g[i - 1]["end"] - g[0]["start"]
            right = g[-1]["end"] - g[i]["start"]
            if left < min_len or right < min_len:
                continue
            gap = max(0.0, g[i]["start"] - g[i - 1]["end"])
            score = gap + (0.4 if CLAUSE_END.search(g[i - 1]["word"].strip()) else 0.0)
            # prefer balanced halves a little
            score -= 0.02 * abs(left - right)
            if score > best_score:
                best, best_score = i, score
        return best

    def _make_segment(self, group: List[Dict], words: List[Dict]) -> Optional[SegmentData]:
        first, last = group[0], group[-1]
        text = " ".join(w["word"].strip() for w in group).strip()
        text = re.sub(r"\s+", " ", text)
        idx_first = words.index(first)
        idx_last = words.index(last)
        prev_end = int(words[idx_first - 1]["end"] * 1000) if idx_first > 0 else None
        next_start = int(words[idx_last + 1]["start"] * 1000) if idx_last + 1 < len(words) else None
        conf = sum(float(w.get("probability", 1.0)) for w in group) / len(group)
        return SegmentData(
            start_ms=int(first["start"] * 1000), end_ms=int(last["end"] * 1000), text=text,
            confidence=conf, duration=last["end"] - first["start"], words=[dict(w) for w in group],
            prev_word_end_ms=prev_end, next_word_start_ms=next_start)

    @staticmethod
    def _valid_text(text: str) -> bool:
        t = text.strip().lower()
        if len(t) < 3 or not any(c.isalpha() for c in t):
            return False
        if any(tok in t for tok in INVALID_TOKENS) and len(t) < 20:
            return False
        return len(set(t.replace(" ", ""))) > 1
