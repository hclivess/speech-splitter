"""
speech-splitter - configuration and constants
"""

APP_NAME = "speech-splitter"
APP_VERSION = "3.0.1"
WINDOW_MIN_WIDTH = 900
WINDOW_MIN_HEIGHT = 600

AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif",
                    ".mp4", ".mkv", ".webm", ".mov", ".avi"]

MODEL_SIZES = [
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large-v2", "large-v3", "large-v3-turbo", "distil-large-v3",
]
DEFAULT_MODEL = "large-v3-turbo"

DEVICES = ["auto", "cpu", "cuda"]
COMPUTE_TYPES = ["default", "int8", "int8_float16", "float16", "float32"]

# code -> display name ("auto" = let Whisper detect)
LANGUAGES = {
    "auto": "Auto-detect",
    "en": "English", "cs": "Czech", "sk": "Slovak", "de": "German", "es": "Spanish", "fr": "French",
    "it": "Italian", "pt": "Portuguese", "pl": "Polish", "nl": "Dutch", "ru": "Russian", "uk": "Ukrainian",
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "ar": "Arabic", "hi": "Hindi", "tr": "Turkish",
    "sv": "Swedish", "da": "Danish", "no": "Norwegian", "fi": "Finnish", "hu": "Hungarian", "ro": "Romanian",
    "el": "Greek", "bg": "Bulgarian", "hr": "Croatian", "sl": "Slovenian", "sr": "Serbian", "lt": "Lithuanian",
    "lv": "Latvian", "et": "Estonian", "he": "Hebrew", "fa": "Persian", "vi": "Vietnamese", "th": "Thai",
    "id": "Indonesian", "ms": "Malay", "ca": "Catalan", "eu": "Basque", "gl": "Galician", "ta": "Tamil",
    "te": "Telugu", "ml": "Malayalam", "ur": "Urdu", "tl": "Tagalog", "ka": "Georgian",
}

EXPORT_FORMATS = ["wav", "flac", "mp3"]
CUTOFF_ACTIONS = {"skip": "Skip (don't export)", "rename": "Export with _CUTOFF suffix", "include": "Export normally"}
OVERLENGTH_ACTIONS = {"split": "Split at the longest pause", "rename": "Export with _LONG suffix",
                      "skip": "Skip (don't export)", "include": "Export normally"}
CUTOFF_MODES = ["both", "start", "end"]

# YAMNet (AudioSet classifier) as ONNX - Apache-2.0 conversion of google/yamnet, runs on onnxruntime
YAMNET_REPO = "audiomagic/yamnet-onnx"
YAMNET_FILES = ("yamnet.onnx", "yamnet_class_map.csv")

DEFAULT_SETTINGS = {
    # Model
    "whisper_model": DEFAULT_MODEL,
    "language": "auto",
    "device": "auto",
    "compute_type": "default",
    "beam_size": 5,
    "vad_filter": True,
    "initial_prompt": "",
    "model_dir": "",
    "cuda_lib_dir": "",
    "cpu_threads": 0,
    # Segmentation
    "min_segment_length": 1.0,
    "max_segment_length": 15.0,
    "merge_short": True,
    "merge_max_gap": 1.0,
    "word_start_buffer_ms": 250,
    "word_end_buffer_ms": 250,
    "overlength_action": "split",
    # Quality
    "min_confidence": 0.5,
    "filter_music": True,
    "music_threshold": 0.3,
    "mixed_threshold": 0.2,
    "filter_noise": False,
    "noise_threshold": 0.3,
    "filter_silence": False,
    "silence_threshold": 0.3,
    "skip_edge_segments": False,
    "skip_segments_count": 1,
    # Cutoff detection
    "check_cutoff": False,
    "cutoff_check_mode": "both",
    "cutoff_sensitivity": 0.35,
    "cutoff_action": "skip",
    # Export
    "output_dir": "output",
    "speaker_id": "",
    "export_format": "wav",
    "export_sample_rate": 0,        # 0 = keep original
    "trim_silence": True,
    "trim_threshold_db": -45.0,
    "trim_chunk_ms": 10,
    "trim_pad_ms": 50,
    "remove_fragments": True,
    "use_transcription_cache": True,
    "log_level": "INFO",
}
