"""
Speech Splitter v4 - Main Orchestrator
Sentence-based audio segmentation for TTS data generation

This is the main entry point that coordinates all modules:
- Audio processing
- WhisperX transcription
- Sentence-based segmentation
- Quality filtering
- Export with cutoff detection
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Dict, Optional

import torch

warnings.filterwarnings('ignore')

# Import all modules
from utils import setup_logging, check_gpu
from audio_processor import AudioProcessor
from transcription import TranscriptionManager
from audio_classifier import YAMNetFilter
from segmentation import SegmentationEngine
from segment_filter import SegmentFilter
from export_manager import ExportManager


class SpeechSplitter:
    """Main orchestrator for speech splitting pipeline"""
    
    def __init__(self, **kwargs):
        self.output_dir = Path(kwargs.get('output_dir', 'output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(self.output_dir, kwargs.get('log_level', 'INFO'))
        
        # Parameters
        self.params = {
            'whisper_model': kwargs.get('whisper_model', 'large'),
            'language': kwargs.get('language', 'en'),
            'batch_size': kwargs.get('batch_size', 6),
            'min_segment_length': kwargs.get('min_segment_length', 1.0),
            'max_segment_length': kwargs.get('max_segment_length', 15.0),
            'export_format': kwargs.get('export_format', 'wav'),
            'min_confidence': kwargs.get('min_confidence', 0.5),
            'filter_noise': kwargs.get('filter_noise', False),
            'noise_silence_threshold': kwargs.get('noise_silence_threshold', 0.3),
            'noise_threshold': kwargs.get('noise_threshold', 0.3),  # NEW
            'filter_silence': kwargs.get('filter_silence', False),  # NEW - rename from filter_noise
            'filter_music': kwargs.get('filter_music', True),
            'music_threshold': kwargs.get('music_threshold', 0.3),
            'mixed_threshold': kwargs.get('mixed_threshold', 0.2),
            'word_start_buffer_ms': kwargs.get('word_start_buffer_ms', 250),
            'word_end_buffer_ms': kwargs.get('word_end_buffer_ms', 250),
            'check_cutoff': kwargs.get('check_cutoff', False),
            'cutoff_check_mode': kwargs.get('cutoff_check_mode', 'both'),
            'cutoff_sensitivity': kwargs.get('cutoff_sensitivity', 0.35),
            'cutoff_boundary_ms': kwargs.get('cutoff_boundary_ms', 50),
            'cutoff_silence_threshold': kwargs.get('cutoff_silence_threshold', 0.35),
            'cutoff_action': kwargs.get('cutoff_action', 'skip'),
            'overlength_action': kwargs.get('overlength_action', 'rename'),
            'trim_threshold_db': kwargs.get('trim_threshold_db', -45.0),
            'trim_chunk_ms': kwargs.get('trim_chunk_ms', 10),
            'trim_pad_ms': kwargs.get('trim_pad_ms', 50),
        }
        
        # Device selection
        self.device = kwargs.get('device', None)
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.logger.info("Configuration loaded", extra=self.params)
        
        # Initialize transcription manager (WhisperX)
        self.transcription_manager = TranscriptionManager(
            self.params['whisper_model'],
            self.params['language'],
            self.device,
            self.logger,
            self.params['batch_size']
        )
        
        # Initialize YAMNet filter if needed (optional)
        self._yamnet_filter = None
        self._needs_yamnet = any([
            self.params['filter_music'],
            self.params['filter_noise']
        ])
        
        if self._needs_yamnet:
            self.logger.info("Initializing YAMNet filter (optional)...")
            self._yamnet_filter = YAMNetFilter(device=self.device, logger=self.logger)
            if self._yamnet_filter.available:
                self.logger.info("✓ YAMNet filter ready")
            else:
                self.logger.warning("⚠ YAMNet not available - using timestamp-only mode")
                self._yamnet_filter = None
        else:
            self.logger.info("YAMNet filtering disabled")
        
        # Initialize other components
        self.segmentation_engine = SegmentationEngine(self.params, self.logger)
        self.segment_filter = SegmentFilter(self.params, self._yamnet_filter, self.logger)
        self.export_manager = ExportManager(self.output_dir, self.params, self.logger)
        
        self.logger.info("SpeechSplitter ready")
    
    def process_file(self, audio_path: str, speaker_id: Optional[str] = None) -> Dict:
        """Process a single audio file"""
        audio_path = Path(audio_path)
        base_name = speaker_id or audio_path.stem
        
        self.logger.info(f"Processing file: {audio_path.name}", extra={
            'speaker': base_name,
            'path': str(audio_path)
        })
        
        # Load and prepare audio
        audio, original_sr = AudioProcessor.load_and_prepare(audio_path, self.logger)
        
        # Transcribe with WhisperX
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio.export(tmp.name, format='wav')
            temp_path = tmp.name
        
        try:
            transcription = self.transcription_manager.transcribe(temp_path, str(audio_path))
        finally:
            os.unlink(temp_path)
        
        # Segment by sentences
        segments = self.segmentation_engine.segment_by_sentences(transcription['words'], audio)
        
        if not segments:
            self.logger.warning("No valid segments found")
            return {'total_segments': 0, 'total_duration': 0, 'avg_duration': 0}
        
        # Filter segments
        filtered_segments = self.segment_filter.filter_segments(segments, audio)
        
        # Export segments
        output_files = self.export_manager.export_segments(audio, filtered_segments, base_name, original_sr)
        
        # Log cache statistics
        if self._yamnet_filter and self._yamnet_filter.available:
            stats = self._yamnet_filter.get_cache_stats()
            self.logger.info(f"YAMNet Cache: {stats['cache_hits']} hits, "
                           f"{stats['cache_misses']} misses, "
                           f"{stats['hit_rate']:.1f}% hit rate")
        
        # Summary
        total_duration = sum(m['duration'] for _, _, m in output_files)
        avg_duration = total_duration / len(output_files) if output_files else 0
        
        self.logger.info("File processing complete", extra={
            'segments': len(output_files),
            'duration': total_duration,
            'avg_duration': avg_duration
        })
        
        return {
            'total_segments': len(output_files),
            'total_duration': total_duration,
            'avg_duration': avg_duration
        }
    
    def generate_metadata(self):
        """Generate metadata for all exported segments"""
        self.export_manager.generate_metadata(self.params['language'])


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Speech Splitter v4 - Sentence-Based TTS Data Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required
    parser.add_argument('input', help="Input audio file or directory")

    # Cutoff detection
    parser.add_argument('--check-cutoff', action='store_true', help="Enable cutoff detection")
    parser.add_argument('--cutoff-check-mode', default='both',
                        choices=['both', 'start', 'end', 'none'],
                        help="Which boundaries to check: both, start, end, or none")
    parser.add_argument('--cutoff-sensitivity', type=float, default=0.35,
                        help="Cutoff detection sensitivity (0.0-1.0, lower = more sensitive)")
    parser.add_argument('--cutoff-action', default='skip',
                        choices=['include', 'skip', 'rename'],
                        help="Action for cutoff segments")
    # Model settings
    parser.add_argument('--model', default='large', help="Whisper model size")
    parser.add_argument('--language', default='cs', help="Language code")
    parser.add_argument('--batch-size', type=int, default=6, help="Transcription batch size")
    parser.add_argument('--device', default=None, choices=['cuda', 'cpu'], help="Force device")
    
    # Output settings
    parser.add_argument('--output-dir', default='output', help="Output directory")
    parser.add_argument('--speaker-id', help="Speaker ID for naming")
    parser.add_argument('--format', default='wav', choices=['wav', 'mp3'], help="Output format")
    
    # Segmentation
    parser.add_argument('--min-length', type=float, default=1.0, help="Minimum segment length (seconds)")
    parser.add_argument('--max-length', type=float, default=15.0, help="Maximum segment length (seconds)")
    
    # Quality filtering
    parser.add_argument('--filter-music', action='store_true', default=True, help="Filter music segments")
    parser.add_argument('--music-threshold', type=float, default=0.3, help="Music detection threshold")
    parser.add_argument('--mixed-threshold', type=float, default=0.2, help="Mixed audio threshold")
    parser.add_argument('--filter-noise', action='store_true', help="Filter background noise")
    parser.add_argument('--noise-threshold', type=float, default=0.3, help="Noise detection threshold")
    parser.add_argument('--filter-silence', action='store_true', help="Filter silent/quiet segments")
    parser.add_argument('--noise-silence-threshold', type=float, default=0.3, help="Silence threshold")

    # Audio processing
    parser.add_argument('--word-start-buffer', type=int, default=250, help="Start buffer (ms)")
    parser.add_argument('--word-end-buffer', type=int, default=250, help="End buffer (ms)")
    parser.add_argument('--trim-threshold-db', type=float, default=-45.0, help="Silence trim threshold (dB)")
    parser.add_argument('--trim-chunk-ms', type=int, default=10, help="Trim analysis chunk size (ms)")
    parser.add_argument('--trim-pad-ms', type=int, default=50, help="Trim padding (ms)")
    
    # Cutoff detection
    parser.add_argument('--check-cutoff', action='store_true', help="Enable cutoff detection")
    parser.add_argument('--cutoff-boundary-ms', type=int, default=75, help="Cutoff detection boundary (ms)")
    parser.add_argument('--cutoff-silence-threshold', type=float, default=0.15, help="Cutoff RMS threshold (lower = more sensitive)")
    parser.add_argument('--cutoff-action', default='skip', choices=['include', 'skip', 'rename'],
                       help="Action for cutoff segments: include=export normally, skip=don't export, rename=add _CUTOFF suffix")
    
    # Actions
    parser.add_argument('--overlength-action', default='rename', choices=['rename', 'skip', 'include'],
                       help="Action for overlength segments")
    
    # Logging
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # GPU check
    import logging
    temp_logger = logging.getLogger("Startup")
    temp_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    temp_logger.addHandler(handler)
    
    temp_logger.info("=== Speech Splitter v4 - Sentence-Based ===")
    check_gpu(temp_logger)
    temp_logger.handlers.clear()
    
    # Create splitter
    try:
        splitter = SpeechSplitter(
            filter_noise=args.filter_noise,
            noise_threshold=args.noise_threshold,
            filter_silence=args.filter_silence,
            noise_silence_threshold=args.noise_silence_threshold,
            whisper_model=args.model,
            language=args.language,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            min_segment_length=args.min_length,
            max_segment_length=args.max_length,
            export_format=args.format,
            min_confidence=args.min_confidence,
            filter_music=args.filter_music,
            music_threshold=args.music_threshold,
            mixed_threshold=args.mixed_threshold,
            word_start_buffer_ms=args.word_start_buffer,
            word_end_buffer_ms=args.word_end_buffer,
            check_cutoff=args.check_cutoff,
            cutoff_check_mode=args.cutoff_check_mode,
            cutoff_sensitivity=args.cutoff_sensitivity,
            cutoff_boundary_ms=args.cutoff_boundary_ms,
            cutoff_silence_threshold=args.cutoff_silence_threshold,
            cutoff_action=args.cutoff_action,
            overlength_action=args.overlength_action,
            trim_threshold_db=args.trim_threshold_db,
            trim_chunk_ms=args.trim_chunk_ms,
            trim_pad_ms=args.trim_pad_ms,
            device=args.device,
            log_level=args.log_level
        )
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        print("\nPlease install WhisperX:")
        print("pip install git+https://github.com/m-bain/whisperx.git")
        return 1
    
    # Process input
    input_path = Path(args.input)
    if input_path.is_file():
        splitter.process_file(str(input_path), args.speaker_id)
    elif input_path.is_dir():
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']:
            audio_files.extend(input_path.glob(ext))
        
        splitter.logger.info(f"Processing directory: {len(audio_files)} files")
        for audio_file in sorted(audio_files):
            speaker_id = args.speaker_id or audio_file.stem
            splitter.process_file(str(audio_file), speaker_id)
    else:
        splitter.logger.error(f"Invalid input path: {args.input}")
        return 1
    
    # Generate metadata
    splitter.generate_metadata()
    splitter.logger.info("=== All processing complete ===")
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
