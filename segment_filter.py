"""
Segment filtering based on quality criteria
Filters: Music, noise/silence, low confidence, invalid text
"""

import logging
from typing import List, Dict, Optional
from pydub import AudioSegment

from utils import SegmentData, AudioClassification


class SegmentFilter:
    """Filter segments based on quality criteria"""
    
    def __init__(self, params: Dict, yamnet_filter: Optional, logger: logging.Logger):
        self.params = params
        self.yamnet_filter = yamnet_filter
        self.logger = logger
        self.filtering_available = yamnet_filter is not None and yamnet_filter.available

    def filter_segments(self, segments: List[SegmentData], audio: AudioSegment,
                        skip_edge_segments: bool = False, skip_segments_count: int = 1) -> List[SegmentData]:
        """Filter segments with single classification per segment"""

        self.logger.info("FILTERING SEGMENTS")
        if skip_edge_segments:
            self.logger.info(f"  Edge segment removal ENABLED (skip {skip_segments_count} from each end)")

        filtered = []
        stats = {
            'music': 0,
            'noise': 0,
            'confidence': 0,
            'text': 0,
            'edge_skipped': 0,
            'total': len(segments)
        }

        # Track examples for each rejection reason
        examples = {
            'music': [],
            'noise': [],
            'confidence': [],
            'text': [],
            'edge_skipped': []
        }

        # Mark edge segments
        edge_indices = set()
        if skip_edge_segments and len(segments) > 2 * skip_segments_count:
            # Skip N segments from start
            for i in range(skip_segments_count):
                edge_indices.add(i)
            # Skip N segments from end
            for i in range(len(segments) - skip_segments_count, len(segments)):
                edge_indices.add(i)

        for idx, seg in enumerate(segments):
            # Check if this is an edge segment
            if idx in edge_indices:
                stats['edge_skipped'] += 1
                position = 'START' if idx < skip_segments_count else 'END'
                if len(examples['edge_skipped']) < 4:  # Show up to 4 examples
                    examples['edge_skipped'].append({
                        'text': seg.text[:50],
                        'position': position,
                        'index': idx
                    })
                continue

            chunk = audio[seg.start_ms:seg.end_ms]
            rejection_reason = None

            # Classify once and cache in segment
            if self.yamnet_filter and self.filtering_available and (
                    self.params['filter_music'] or self.params['filter_noise']):
                classification = self.yamnet_filter.classify_audio(chunk)
                seg.classification = classification.to_dict()

                # Music filter
                if self.params['filter_music']:
                    if classification.is_music(self.params['music_threshold'], self.params['mixed_threshold']):
                        stats['music'] += 1
                        rejection_reason = 'music'
                        if len(examples['music']) < 3:
                            examples['music'].append({
                                'text': seg.text[:50],
                                'music': classification.music,
                                'mixed': classification.music_speech_mixed
                            })
                        continue

                if self.params['filter_noise']:
                    if classification.is_noise(self.params['noise_threshold']):
                        stats['noise'] += 1
                        rejection_reason = 'noise'
                        if len(examples['noise']) < 3:
                            examples['noise'].append({
                                'text': seg.text[:50],
                                'noise': classification.noise
                            })
                        continue

                # Silence filter
                if self.params['filter_noise']:
                    if classification.is_silent(self.params['noise_silence_threshold']):
                        stats['noise'] += 1
                        rejection_reason = 'noise'
                        if len(examples['noise']) < 3:
                            examples['noise'].append({
                                'text': seg.text[:50],
                                'silence': classification.silence
                            })
                        continue
            else:
                # No filtering available, set default classification
                seg.classification = AudioClassification(0.0, 1.0, 0.0, 0.0).to_dict()

            # Confidence filter
            if seg.confidence < self.params['min_confidence']:
                stats['confidence'] += 1
                rejection_reason = 'confidence'
                if len(examples['confidence']) < 3:
                    examples['confidence'].append({
                        'text': seg.text[:50],
                        'confidence': seg.confidence
                    })
                continue

            # Text validation
            if len(seg.text.strip()) < 3:
                stats['text'] += 1
                rejection_reason = 'text'
                if len(examples['text']) < 3:
                    examples['text'].append({
                        'text': seg.text
                    })
                continue

            filtered.append(seg)

        # Detailed logging
        self.logger.info(f"Input segments: {stats['total']}")
        self.logger.info(f"Accepted: {len(filtered)} ({len(filtered) / stats['total'] * 100:.1f}%)")
        self.logger.info("")

        if stats['edge_skipped'] > 0:
            self.logger.info(f"❌ EDGE SEGMENTS: {stats['edge_skipped']} segments removed "
                             f"({skip_segments_count} from each end)")
            for ex in examples['edge_skipped']:
                self.logger.info(f"   [{ex['position']}] #{ex['index']} '{ex['text']}...'")

        if stats['music'] > 0:
            self.logger.warning(f"❌ MUSIC: {stats['music']} segments rejected")
            for ex in examples['music']:
                self.logger.warning(f"   '{ex['text']}...' (music={ex['music']:.2f}, mixed={ex['mixed']:.2f})")

        if stats['noise'] > 0:
            self.logger.warning(f"❌ NOISE/SILENCE: {stats['noise']} segments rejected")
            for ex in examples['noise']:
                if 'noise' in ex:
                    self.logger.warning(f"   '{ex['text']}...' (noise={ex['noise']:.2f})")
                else:
                    self.logger.warning(f"   '{ex['text']}...' (silence={ex['silence']:.2f})")

        if stats['confidence'] > 0:
            self.logger.warning(f"❌ LOW CONFIDENCE: {stats['confidence']} segments rejected")
            for ex in examples['confidence']:
                self.logger.warning(f"   '{ex['text']}...' (conf={ex['confidence']:.2f})")

        if stats['text'] > 0:
            self.logger.warning(f"❌ INVALID TEXT: {stats['text']} segments rejected")
            for ex in examples['text']:
                self.logger.warning(f"   '{ex['text']}'")

        return filtered
