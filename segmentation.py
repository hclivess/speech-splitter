"""
Segmentation engine for splitting audio by sentence boundaries
Uses punctuation (., !, ?) to detect natural speech breaks

IMPROVED VERSION:
- Does NOT include pause gaps to next words (prevents fragment inclusion)
- Uses word-end boundaries instead of next-word-start boundaries
- More conservative segmentation to avoid artifacts
"""

import re
import logging
from typing import List, Dict, Optional
from pydub import AudioSegment

from utils import SegmentData


class SegmentationEngine:
    """Handle segmentation by sentence boundaries (fullstops, !, ?)"""

    def __init__(self, params: Dict, logger: logging.Logger):
        self.params = params
        self.logger = logger

        # Sentence ending patterns
        self.sentence_endings = re.compile(r'[.!?]+\s*$')

    def segment_by_sentences(self, words: List[Dict], audio: AudioSegment) -> List[SegmentData]:
        """
        Segment audio by sentence boundaries (fullstops, !, ?)
        
        IMPROVED: Segments end at the last word's END time, not at the next word's START time.
        This prevents including fragments that appear after silence gaps.
        """
        if not words:
            self.logger.warning("No words found for segmentation")
            return []

        self.logger.info("Segmenting by sentence boundaries (., !, ?)")

        segments = []
        current_words = []
        stats = {'sentences_found': 0, 'segments_created': 0}

        for i, word in enumerate(words):
            if 'start' not in word or 'end' not in word:
                self.logger.info(f"Skipping word: missing timestamps")
                continue

            current_words.append(word)

            should_split = False
            word_text = word.get('word', '').strip()

            # Check if word ends with sentence punctuation
            if self.sentence_endings.search(word_text):
                should_split = True
                stats['sentences_found'] += 1

            # Last word always triggers split (even without punctuation)
            if i == len(words) - 1:
                should_split = True

            # Create segment when split is triggered
            if should_split and current_words:
                first_word = current_words[0]
                last_word = current_words[-1]

                if 'start' in first_word and 'end' in last_word:
                    text = ' '.join([w.get('word', '') for w in current_words]).strip()
                    duration = last_word['end'] - first_word['start']

                    # Calculate segment boundaries
                    segment_start_idx = i - len(current_words) + 1
                    prev_word_end_ms = int(
                        words[segment_start_idx - 1]['end'] * 1000) if segment_start_idx > 0 else None
                    next_word_start_ms = int(words[i + 1]['start'] * 1000) if i + 1 < len(words) else None

                    # IMPROVED: End at last word's END, not next word's START
                    # This prevents including fragments after silence gaps
                    end_ms = int(last_word['end'] * 1000)

                    if self._validate_segment_text(text) and duration >= self.params['min_segment_length']:
                        segment = SegmentData(
                            start_ms=int(first_word['start'] * 1000),
                            end_ms=end_ms,
                            duration=duration,
                            words=current_words.copy(),
                            text=text,
                            confidence=sum(w.get('probability', 1.0) for w in current_words) / len(current_words),
                            prev_word_end_ms=prev_word_end_ms,
                            next_word_start_ms=next_word_start_ms  # Keep for reference but don't use for end_ms
                        )
                        segments.append(segment)
                        stats['segments_created'] += 1

                # Clear words for next segment
                current_words = []

        self.logger.info("SEGMENTATION RESULTS")
        self.logger.info(f"Sentence boundaries found: {stats['sentences_found']}")
        self.logger.info(f"Total segments created: {stats['segments_created']}")

        return segments

    def _validate_segment_text(self, text: str) -> bool:
        """Validate that segment contains meaningful text"""
        if not text or len(text.strip()) < 3:
            return False

        text = text.strip().lower()
        invalid_patterns = ['[blank_audio]', '[music]', '[applause]', '[laughter]', '[noise]', '...', 'ummm', 'uhhhh']

        for pattern in invalid_patterns:
            if pattern in text and len(text) < 15:
                return False

        if not any(c.isalpha() for c in text):
            return False

        return len(set(text.replace(' ', ''))) > 1
