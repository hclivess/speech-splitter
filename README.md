# Speech Splitter with YAMNet Music Filtering

## Downsides: 
- Whisper transcription does not always get proper names right

![image](aavsnx.jpg)

Advanced audio segmentation tool for TTS training data preparation with intelligent music detection and filtering.

## Features

- **Whisper Transcription** - Accurate speech-to-text with word-level timestamps
- **YAMNet Music Filtering** - Neural network-based music detection and removal
- **Quality Filtering** - Confidence scoring and noise threshold filtering
- **Perfect Audio/Text Sync** - Guaranteed alignment between audio segments and transcripts
- **Batch Processing** - Process multiple files at once
- **GUI & CLI** - Choose your preferred interface

## What Gets Filtered

YAMNet automatically detects and removes:
- ✅ Pure music segments (instrumental, vocal music)
- ✅ Speech with background music
- ✅ Songs with lyrics
- ✅ Musical interludes
- ✅ Low quality/noisy segments
- ✅ Low confidence transcriptions

## Installation

**Important:** Use a virtual environment to avoid compatibility issues between TensorFlow, YAMNet, and other dependencies.

### Step 1: Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
call .venv\Scripts\activate
```

**Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- OpenAI Whisper
- YAMNet (TensorFlow + TensorFlow Hub)
- spaCy
- pydub
- PySide6 (GUI)
- Other dependencies

### Step 3: Install ffmpeg

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to PATH

## Output Structure

Each input file creates its own organized directory:

```
output/
  podcast_episode1/
    audio/
      podcast_episode1_0001.wav
      podcast_episode1_0002.wav
      ...
    transcripts/
      podcast_episode1_0001.txt
      podcast_episode1_0002.txt
      ...
    podcast_episode1_metadata.json
    podcast_episode1_manifest.txt
  
  interview_recording/
    audio/
      interview_recording_0001.wav
    transcripts/
      interview_recording_0001.txt
    interview_recording_metadata.json
    interview_recording_manifest.txt
```

## Usage

### GUI Application

```bash
python speech_splitter_gui.py
```

**GUI Features:**
- Drag and drop file selection
- Real-time processing logs
- Configurable settings with tabs:
  - **Basic**: Model, language, output settings
  - **Segments**: Length and padding controls
  - **VAD**: Voice activity detection thresholds
  - **Quality**: Confidence and noise filtering
  - **Music Filter**: YAMNet settings and thresholds
- Settings auto-save and restore

**Music Filter Tab:**
- Enable/disable YAMNet filtering
- Adjust music detection threshold (default: 0.3)
- Adjust mixed content threshold (default: 0.2)
- Lower values = more aggressive filtering
- Higher values = less aggressive filtering

### Command Line

**Basic usage:**
```bash
python speech_splitter.py input.mp3
```

**With options:**
```bash
python speech_splitter.py input.mp3 \
  --model large \
  --language en \
  --output-dir output \
  --speaker-id "john_smith" \
  --filter-music \
  --music-threshold 0.3 \
  --mixed-threshold 0.2 \
  --min-confidence 0.5 \
  --filter-noise \
  --min-rms 0.01
```

**All options:**
```
--model              Whisper model: tiny, base, small, medium, large, large-v2 (default: large)
--language           Language code: en, cs, de, es, fr, it, pl, ru (default: cs)
--output-dir         Output directory (default: output)
--speaker-id         Speaker identifier (default: filename)
--format             Output format: wav, mp3 (default: wav)

Segmentation:
--silence-padding    Silence padding in ms (default: 150)
--min-length         Min segment length in seconds (default: 1.0)
--max-length         Max segment length in seconds (default: 15.0)
--min-silence        Min silence length for VAD in ms (default: 700)
--silence-thresh     Silence threshold in dBFS (default: -40)

Quality Filtering:
--min-confidence     Min transcription confidence 0.0-1.0 (default: 0.5)
--filter-noise       Enable noise filtering
--min-rms            Min RMS energy level (default: 0.01)

Music Filtering:
--filter-music       Enable YAMNet music filtering (default: True)
--music-threshold    Music detection threshold (default: 0.3)
--mixed-threshold    Music+speech detection threshold (default: 0.2)
```

## Processing Pipeline

The tool uses an optimized filtering order for maximum efficiency:

```
1. Load Audio
   ↓
2. Whisper Transcription (full audio with word timestamps)
   ↓
3. Pause-based Segmentation (split by natural pauses using word timestamps)
   ↓
4. Length Pre-Filter (instant, removes too-short segments)
   ↓
5. YAMNet Music Filter (medium cost, saves transcription work)
   ↓
6. Transcription + Noise Check + Confidence Filter (expensive, done last)
   ↓
7. Export (exact same audio that was transcribed, with re-transcription for accuracy)
```

**Why this order?**
- Cheap filters first (length check takes <1ms)
- Music filtering before transcription (saves 30-50% processing time)
- Expensive transcription only on segments that passed all filters
- Perfect audio/text sync (same 300ms buffer for transcription and export)

## Configuration Examples

### Default Settings (Recommended)
Good balance for most audio:
```bash
python speech_splitter.py podcast.mp3
```

### Aggressive Filtering
For noisy audio with lots of music:
```bash
python speech_splitter.py recording.mp3 \
  --music-threshold 0.2 \
  --mixed-threshold 0.15 \
  --min-confidence 0.6 \
  --filter-noise \
  --min-rms 0.015
```

### Conservative Filtering
For clean audio, only filter obvious music:
```bash
python speech_splitter.py lecture.mp3 \
  --music-threshold 0.5 \
  --mixed-threshold 0.4 \
  --min-confidence 0.4
```

### Fast Processing
Skip music filtering for speed:
```bash
python speech_splitter.py audio.mp3 \
  --model base \
  --no-filter-music
```

### Batch Processing
Process multiple files:
```bash
for file in audio/*.mp3; do
  python speech_splitter.py "$file" --speaker-id "$(basename "$file" .mp3)"
done
```

For issues or questions, check the troubleshooting section above or report issues on the repository.
