# Speech Splitter with YAMNet Music Filtering

Advanced audio segmentation tool for TTS training data preparation with intelligent music detection and filtering.

## Features

- **Whisper Transcription** - Accurate speech-to-text with word-level timestamps
- **VAD (Voice Activity Detection)** - Intelligent silence-based segmentation
- **YAMNet Music Filtering** - Neural network-based music detection and removal
- **Quality Filtering** - Confidence scoring and noise threshold filtering
- **Optimized Pipeline** - Efficient filter ordering (cheap filters first, expensive last)
- **Perfect Audio/Text Sync** - Guaranteed alignment between audio segments and transcripts
- **Organized Output** - Each input file gets its own subdirectory
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
2. VAD Segmentation (split by silence)
   ↓
3. Length Pre-Filter (instant, removes too-short segments)
   ↓
4. YAMNet Music Filter (medium cost, saves transcription work)
   ↓
5. Transcription + Noise Check + Confidence Filter (expensive, done last)
   ↓
6. Export (exact same audio that was transcribed)
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

## Threshold Tuning Guide

### Music Detection Threshold

**Lower (0.1-0.2)**: Aggressive
- Catches subtle music
- May filter some speech with music-like prosody
- Use for: Audiobooks, podcasts with music beds

**Medium (0.3)**: Balanced ✓ RECOMMENDED
- Good general-purpose setting
- Filters obvious music
- Keeps most clean speech

**Higher (0.4-0.9)**: Conservative
- Only filters clear music
- May let some mixed content through
- Use for: Clean recordings, interviews

### Mixed Content Threshold

**Lower (0.1-0.15)**: Aggressive
- Filters speech with even quiet background music
- Strictest quality control
- Use for: High-quality TTS training

**Medium (0.2)**: Balanced ✓ RECOMMENDED
- Filters noticeable background music
- Good for most use cases

**Higher (0.3-0.5)**: Conservative
- Only filters obvious music backgrounds
- More permissive

### Confidence Threshold

- **0.3-0.4**: Permissive (keeps most segments)
- **0.5**: Balanced ✓ RECOMMENDED
- **0.6-0.8**: Strict (only high-quality transcriptions)

## Troubleshooting

### YAMNet fails to load
```
ERROR: Could not load YAMNet
```
**Solution:**
- Install TensorFlow: `pip install tensorflow tensorflow-hub`
- First run downloads model (~27MB), needs internet
- Check: `python -c "import tensorflow; print(tensorflow.__version__)"`

### Too much filtered (false positives)
**Symptoms:** Good speech segments removed

**Solution:**
- Increase music threshold: `--music-threshold 0.4`
- Increase mixed threshold: `--mixed-threshold 0.3`
- Lower confidence threshold: `--min-confidence 0.4`

### Music not filtered (false negatives)
**Symptoms:** Music segments in output

**Solution:**
- Decrease thresholds: `--music-threshold 0.2 --mixed-threshold 0.15`
- Check YAMNet is enabled: `--filter-music`
- Test segment: `python test_yamnet.py segment.mp3`

### Slow processing
**Solutions:**
- Use smaller Whisper model: `--model base` or `--model small`
- Disable music filtering: `--no-filter-music`
- Install TensorFlow with GPU support
- Process files in parallel (separate terminal windows)

### Audio/text mismatch
**Should not happen** - fixed in current version
- Same 300ms buffer used for transcription and export
- Exact audio chunk transcribed = exact audio chunk exported
- If you see this, report as a bug

### Words cut off at end
**Solution:**
- Already using 300ms buffer
- Check `--silence-padding` (default 150ms)
- May need to increase: `--silence-padding 300`

## Performance Tips

1. **GPU Acceleration**: Install TensorFlow with CUDA support for 5-10x speedup
2. **Model Size**: Use `base` or `small` for 3-5x faster processing with slight accuracy trade-off
3. **Batch Processing**: Process multiple files sequentially (GUI) or parallel (CLI)
4. **Pre-filtering**: Music filtering before transcription saves 30-50% time on music-heavy audio
5. **Disk Speed**: Use SSD for output directory

## Output Files

Each processed file generates:

**audio/** - Individual audio segments
- Format: WAV (22.05kHz) or MP3
- Naming: `{speaker_id}_{0001}.wav`
- Padding: Configured silence at end

**transcripts/** - Text transcriptions
- Format: UTF-8 plain text
- Naming: `{speaker_id}_{0001}.txt`
- Content: Transcribed speech only

**{speaker_id}_metadata.json** - Detailed segment information
```json
{
  "source_file": "podcast.mp3",
  "speaker_id": "podcast",
  "language": "en",
  "total_segments": 42,
  "segments": [
    {
      "audio": "podcast_0001.wav",
      "transcript": "podcast_0001.txt",
      "duration": 3.5,
      "text_length": 85
    }
  ]
}
```

**{speaker_id}_manifest.txt** - Audio-transcript pairs
```
podcast_0001.wav|podcast_0001.txt
podcast_0002.wav|podcast_0002.txt
```

## Dependencies

**Core:**
- openai-whisper - Speech transcription
- pydub - Audio manipulation
- tensorflow - YAMNet neural network
- tensorflow-hub - Pre-trained model loading
- numpy - Numerical operations
- tqdm - Progress bars

**GUI (optional):**
- PySide6 - Qt-based interface

**Language Support (optional):**
- spacy - Advanced sentence segmentation
- spacy-udpipe - Czech language support

## Known Limitations

1. **First Run**: Downloads YAMNet model (~27MB), requires internet
2. **GPU Memory**: TensorFlow uses more memory with GPU (but faster)
3. **Short Segments**: Very short segments (<1s) may not classify accurately
4. **Edge Cases**: Complex music/speech mixing may not filter perfectly

## Testing

Test YAMNet classification on any file:
```bash
python test_yamnet.py sample.mp3
```

Shows:
- Music probability
- Speech probability
- Mixed content probability
- Filtering decisions at different thresholds

## Credits

- **YAMNet**: Google Research (Apache 2.0 License)
- **Whisper**: OpenAI (MIT License)
- **TensorFlow**: Google Brain

## License

See individual component licenses above.

---

For issues or questions, check the troubleshooting section above or report issues on the repository.
