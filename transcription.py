"""
Transcription management using WhisperX
Handles: Model loading, transcription, alignment, caching
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional


class TranscriptionManager:
    """Manage WhisperX transcription with caching"""
    
    def __init__(self, model_name: str, language: str, device: str, logger: logging.Logger, batch_size: int = 6):
        self.model_name = model_name
        self.language = language
        self.device = device
        self.logger = logger
        self.batch_size = batch_size
        
        # Load WhisperX
        try:
            import whisperx
            self.whisperx = whisperx
        except ImportError:
            raise RuntimeError("WhisperX not found. Install: pip install git+https://github.com/m-bain/whisperx.git")
        
        # Load model
        self.logger.info(f"Loading WhisperX model: {model_name}")
        self.model = whisperx.load_model(
            model_name,
            device=device,
            compute_type='float16' if device == 'cuda' else 'int8',
            language=language
        )
        self.logger.info("✓ WhisperX model loaded")
    
    def transcribe(self, audio_path: str, original_path: str) -> Dict:
        """Transcribe audio with caching - saves as <filename>.transcription.json alongside source"""
        # Cache alongside source file with clear name
        source_cache = Path(original_path).with_suffix('.transcription.json')
        
        # Check for cache
        if source_cache.exists():
            self.logger.info(f"Loading cached transcription: {source_cache.name}")
            try:
                with open(source_cache, encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Cache read failed: {e}, re-transcribing")
        
        self.logger.info("Transcribing audio...")
        
        # Transcribe
        audio = self.whisperx.load_audio(audio_path)
        result = self.model.transcribe(audio, batch_size=self.batch_size)
        
        self.logger.info(f"✓ Transcription complete: {len(result.get('segments', []))} segments")
        
        # Align (with fallback)
        try:
            self.logger.info("Loading alignment model...")
            model_a, metadata = self.whisperx.load_align_model(
                language_code=self.language, 
                device=self.device
            )
            result = self.whisperx.align(
                result["segments"], 
                model_a, 
                metadata, 
                audio, 
                self.device, 
                return_char_alignments=False
            )
            self.logger.info("✓ Alignment completed")
        except Exception as e:
            self.logger.warning(f"Alignment failed (continuing without it): {e}")
            self.logger.warning("Speech splitter will work but timestamps may be less accurate")
        
        # Extract words
        all_words = []
        for segment in result.get('segments', []):
            if 'words' in segment:
                for word in segment['words']:
                    all_words.append({
                        'word': word.get('word', ''),
                        'start': word.get('start', 0),
                        'end': word.get('end', 0),
                        'probability': word.get('score', 1.0)
                    })
            else:
                # Fallback: use segment timestamps
                segment_words = segment.get('text', '').split()
                if segment_words:
                    seg_start = segment.get('start', 0)
                    seg_end = segment.get('end', 0)
                    seg_duration = seg_end - seg_start
                    word_duration = seg_duration / len(segment_words) if len(segment_words) > 0 else 0
                    
                    for i, word_text in enumerate(segment_words):
                        all_words.append({
                            'word': word_text,
                            'start': seg_start + i * word_duration,
                            'end': seg_start + (i + 1) * word_duration,
                            'probability': 1.0
                        })
        
        self.logger.info(f"WhisperX complete: {len(all_words)} words")
        
        # Prepare transcription data
        transcription_data = {
            'text': ' '.join([w['word'] for w in all_words]),
            'words': all_words,
            'segments': result.get('segments', []),
            'language': self.language,
            'model': f"{self.model_name}-whisperx",
            'aligned': 'words' in result.get('segments', [{}])[0] if result.get('segments') else False
        }
        
        # Save cache alongside source file
        try:
            source_cache = Path(original_path).with_suffix('.transcription.json')
            self.logger.info(f"Saving transcription: {source_cache.name}")
            with open(source_cache, 'w', encoding='utf-8') as f:
                json.dump(transcription_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"Cache save failed: {e}")
        
        return transcription_data
