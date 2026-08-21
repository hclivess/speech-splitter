#!/usr/bin/env python3
"""
speech-splitter - GUI (PySide6)

Queue + progress + log on the left, settings tabs on the right, presets in the menu.
Set SPEECH_SPLITTER_SELFTEST=<audio file> to run a headless smoke test (tiny.en, CPU) and exit.
"""
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtCore import QSettings, Qt, QThread, QTimer, Signal  # noqa: E402
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QIcon, QTextCursor  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit,
    QVBoxLayout, QWidget,
)

import childproc  # noqa: E402
from naturalsort import path_key  # noqa: E402
from config import (APP_NAME, APP_VERSION, AUDIO_EXTENSIONS, COMPUTE_TYPES, CUTOFF_ACTIONS, CUTOFF_MODES,  # noqa: E402
                    DEFAULT_SETTINGS, DEVICES, EXPORT_FORMATS, LANGUAGES, MODEL_SIZES, OVERLENGTH_ACTIONS,
                    WINDOW_MIN_HEIGHT, WINDOW_MIN_WIDTH)
from utils import app_dir, atomic_write_text, finish_logging, format_duration  # noqa: E402

_BTN = """
    QPushButton { font-size: 14px; font-weight: bold; padding: 8px; background-color: %s; color: %s; border-radius: 5px; }
    QPushButton:hover { background-color: %s; }
    QPushButton:disabled { background-color: #cccccc; color: #666; }
"""
_BAR = """
    QProgressBar { border: 1px solid #c8c8c8; border-radius: 4px; background: #f0f0f0;
                   text-align: center; height: 18px; font-weight: bold; color: #333; }
    QProgressBar::chunk { background-color: %s; border-radius: 3px; }
"""
_STATUS_COLORS = {"pending": "#ffffff", "running": "#fff3cd", "done": "#d4edda", "failed": "#f8d7da",
                  "stopped": "#e2e3e5"}

PRESETS = {
    "fast": {"name": "Fast (small model, no audio filter)",
             "settings": {"whisper_model": "small", "beam_size": 1, "filter_music": False, "filter_noise": False,
                          "filter_silence": False, "check_cutoff": False}},
    "balanced": {"name": "Balanced (large-v3-turbo, music filter)",
                 "settings": {"whisper_model": "large-v3-turbo", "beam_size": 5, "filter_music": True,
                              "music_threshold": 0.3, "mixed_threshold": 0.2, "filter_noise": False,
                              "filter_silence": False, "check_cutoff": False, "min_confidence": 0.5}},
    "strict": {"name": "Strict (large-v3, all filters, cutoff check)",
               "settings": {"whisper_model": "large-v3", "beam_size": 5, "filter_music": True,
                            "music_threshold": 0.25, "mixed_threshold": 0.15, "filter_noise": True,
                            "noise_threshold": 0.3, "filter_silence": True, "silence_threshold": 0.3,
                            "check_cutoff": True, "cutoff_action": "skip", "min_confidence": 0.65,
                            "skip_edge_segments": True, "skip_segments_count": 1}},
}


# =============================================================================== worker
class QtLogHandler(logging.Handler):
    def __init__(self, emit_fn):
        super().__init__()
        self.emit_fn = emit_fn

    def emit(self, record):
        try:
            self.emit_fn(record.levelname, self.format(record))
        except Exception:
            pass


class Worker(QThread):
    log = Signal(str, str)                      # level, message
    status = Signal(str)                        # phase text
    file_started = Signal(int)
    file_progress = Signal(str, float, float)   # phase, done, total
    file_done = Signal(int, str, dict)          # index, state, stats
    finished_all = Signal(int, int, dict)       # ok, total, metadata

    def __init__(self, files: List[str], settings: Dict):
        super().__init__()
        self.files = files
        self.settings = settings
        self._stop = False

    def request_stop(self):
        self._stop = True

    def run(self):
        from utils import setup_logging
        from speech_splitter import SpeechSplitter
        out = Path(self.settings["output_dir"] or "output")
        try:
            out.mkdir(parents=True, exist_ok=True)
            logger = setup_logging(out, self.settings.get("log_level", "INFO"), console=False)
            handler = QtLogHandler(lambda lvl, msg: self.log.emit(lvl, msg))
            handler.setFormatter(logging.Formatter("%(message)s"))
            handler.setLevel(logging.DEBUG if self.settings.get("log_level") == "DEBUG" else logging.INFO)
            logger.addHandler(handler)
            self.status.emit("Loading models")
            splitter = SpeechSplitter(self.settings, logger=logger, status=self.status.emit,
                                      progress=lambda ph, d, t: self.file_progress.emit(ph, d, t),
                                      should_stop=lambda: self._stop)
            self.log.emit("INFO", f"Device: {splitter.device}")
        except Exception as exc:  # noqa: BLE001
            self.log.emit("ERROR", f"Initialisation failed: {exc}")
            self.log.emit("DEBUG", traceback.format_exc())
            self.finished_all.emit(0, len(self.files), {})
            return

        ok = 0
        for i, f in enumerate(self.files):
            if self._stop:
                self.file_done.emit(i, "stopped", {})
                continue
            self.file_started.emit(i)
            try:
                stats = splitter.process_file(f, self.settings.get("speaker_id") or None)
                ok += 1
                self.file_done.emit(i, "done", stats)
            except InterruptedError:
                self.file_done.emit(i, "stopped", {})
            except Exception as exc:  # noqa: BLE001
                self.log.emit("ERROR", f"{Path(f).name}: {exc}")
                self.log.emit("DEBUG", traceback.format_exc())
                self.file_done.emit(i, "failed", {})
        meta = {}
        try:
            self.status.emit("Writing metadata")
            meta = splitter.generate_metadata()
        except Exception as exc:  # noqa: BLE001
            self.log.emit("ERROR", f"Metadata generation failed: {exc}")
        finish_logging(logger)
        self.finished_all.emit(ok, len(self.files), meta)


class DownloadThread(QThread):
    progress = Signal(str, int, int)
    done = Signal(object, str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.stop = False

    def run(self):
        try:
            self.done.emit(self.fn(self), "")
        except Exception as exc:  # noqa: BLE001
            self.done.emit(None, str(exc))


# =============================================================================== widgets
class QueueTable(QTableWidget):
    files_dropped = Signal(list)

    def __init__(self):
        super().__init__(0, 3)
        self.setHorizontalHeaderLabels(["File", "Status", "Clips"])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAcceptDrops(True)
        self.setStyleSheet("QTableWidget { border: 2px solid #aaa; border-radius: 5px; background: #f9f9f9; }")

    def dragEnterEvent(self, e: QDragEnterEvent):
        e.accept() if e.mimeData().hasUrls() else e.ignore()

    def dragMoveEvent(self, e):
        e.accept() if e.mimeData().hasUrls() else e.ignore()

    def dropEvent(self, e: QDropEvent):
        if e.mimeData().hasUrls():
            e.accept()
            self.files_dropped.emit([u.toLocalFile() for u in e.mimeData().urls()])


# =============================================================================== main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.resize(1180, 760)
        self.qsettings = QSettings(APP_NAME, "Settings")
        self.presets_dir = os.path.join(app_dir(), "presets")
        os.makedirs(self.presets_dir, exist_ok=True)
        self.files: List[str] = []
        self._dupes_seen = 0
        self.worker: Optional[Worker] = None
        self.controls: Dict[str, QWidget] = {}
        self._dl_thread: Optional[DownloadThread] = None
        self._run_start = 0.0
        self._clips = 0
        self._seconds = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._build_ui()
        self._build_menu()
        self.apply_settings(dict(DEFAULT_SETTINGS))
        self._load_persisted()
        self._refresh_gpu_status()
        self._refresh_model_labels()
        self._check_ffmpeg()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._left_panel())
        split.addWidget(self._right_panel())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([640, 520])
        root.addWidget(split, 1)
        self.statusBar().showMessage("Ready")

    def _left_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)

        lay.addWidget(QLabel("<b>Queue</b> — drop audio / video files or folders here"))
        self.table = QueueTable()
        self.table.files_dropped.connect(self.add_paths)
        lay.addWidget(self.table, 3)
        row = QHBoxLayout()
        for text, fn in (("Add files", self.add_files), ("Add folder", self.add_folder),
                         ("Remove", self.remove_selected), ("Clear", self.clear_queue)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch()
        open_out = QPushButton("Open output folder")
        open_out.clicked.connect(self.open_output)
        row.addWidget(open_out)
        lay.addLayout(row)

        prog = QGroupBox("Progress")
        pl = QVBoxLayout(prog)
        top = QHBoxLayout()
        self.phase_label = QLabel("Idle")
        self.phase_label.setStyleSheet("font-weight: bold;")
        self.counter_label = QLabel("")
        self.counter_label.setStyleSheet("color: #666;")
        top.addWidget(self.phase_label)
        top.addStretch()
        top.addWidget(self.counter_label)
        pl.addLayout(top)
        self.file_bar = QProgressBar()
        self.file_bar.setStyleSheet(_BAR % "#4a90d9")
        self.file_bar.setFormat("%p%")
        pl.addWidget(self.file_bar)
        self.total_bar = QProgressBar()
        self.total_bar.setStyleSheet(_BAR % "#28a745")
        self.total_bar.setFormat("%v / %m files")
        pl.addWidget(self.total_bar)
        stats = QHBoxLayout()
        self.stat_clips = self._stat(stats, "clips")
        self.stat_minutes = self._stat(stats, "speech")
        self.stat_elapsed = self._stat(stats, "elapsed")
        self.stat_device = self._stat(stats, "device")
        pl.addLayout(stats)
        lay.addWidget(prog)

        lay.addWidget(QLabel("<b>Log</b>"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("QTextEdit { font-family: monospace; font-size: 12px; background: #fcfcfc; }")
        lay.addWidget(self.log, 2)

        btns = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start")
        self.start_btn.setStyleSheet(_BTN % ("#28a745", "white", "#218838"))
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setStyleSheet(_BTN % ("#dc3545", "white", "#c82333"))
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        btns.addWidget(self.start_btn, 2)
        btns.addWidget(self.stop_btn, 1)
        lay.addLayout(btns)
        return w

    @staticmethod
    def _stat(layout, caption) -> QLabel:
        box = QVBoxLayout()
        c = QLabel(caption)
        c.setStyleSheet("color: #888; font-size: 10px;")
        v = QLabel("--")
        v.setStyleSheet("font-weight: bold; font-size: 13px;")
        box.addWidget(c)
        box.addWidget(v)
        layout.addLayout(box)
        return v

    def _right_panel(self) -> QWidget:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._wrap(self._model_tab()), "Model")
        self.tabs.addTab(self._wrap(self._segmentation_tab()), "Segmentation")
        self.tabs.addTab(self._wrap(self._quality_tab()), "Quality")
        self.tabs.addTab(self._wrap(self._export_tab()), "Export")
        for w in self.controls.values():
            if isinstance(w, QComboBox):
                w.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                w.setMinimumContentsLength(8)
            if isinstance(w, (QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox)):
                w.setMinimumWidth(0)
                w.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        return self.tabs

    @staticmethod
    def _wrap(widget: QWidget) -> QScrollArea:
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setWidget(widget)
        sa.setFrameShape(QScrollArea.Shape.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return sa

    def _path_row(self, line: QLineEdit, browse_fn) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(line, 1)
        b = QPushButton("…")
        b.setFixedWidth(32)
        b.clicked.connect(browse_fn)
        row.addWidget(b)
        return row

    def _browse_dir(self, line: QLineEdit, title: str):
        d = QFileDialog.getExistingDirectory(self, title, line.text() or os.path.expanduser("~"))
        if d:
            line.setText(d)
            line.editingFinished.emit()

    @staticmethod
    def _hint(text: str) -> QLabel:
        lab = QLabel(text)
        lab.setWordWrap(True)
        lab.setStyleSheet("color: #666; font-size: 11px;")
        return lab

    def _model_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        g = QGroupBox("Whisper model (faster-whisper)")
        f = QFormLayout(g)
        c = self.controls
        c["whisper_model"] = QComboBox()
        c["whisper_model"].setEditable(True)
        c["whisper_model"].addItems(MODEL_SIZES)
        c["whisper_model"].setToolTip("Model size, or any faster-whisper repo id from Hugging Face")
        c["whisper_model"].currentTextChanged.connect(self._refresh_model_labels)
        f.addRow("Model:", c["whisper_model"])
        c["language"] = QComboBox()
        for code, name in LANGUAGES.items():
            c["language"].addItem(f"{name} ({code})" if code != "auto" else name, code)
        f.addRow("Language:", c["language"])
        c["device"] = QComboBox()
        c["device"].addItems(DEVICES)
        f.addRow("Device:", c["device"])
        c["compute_type"] = QComboBox()
        c["compute_type"].addItems(COMPUTE_TYPES)
        c["compute_type"].setToolTip("default = float16 on GPU, int8 on CPU")
        f.addRow("Compute type:", c["compute_type"])
        c["beam_size"] = QSpinBox()
        c["beam_size"].setRange(1, 10)
        f.addRow("Beam size:", c["beam_size"])
        c["cpu_threads"] = QSpinBox()
        c["cpu_threads"].setRange(0, 64)
        c["cpu_threads"].setSpecialValueText("auto")
        f.addRow("CPU threads:", c["cpu_threads"])
        c["vad_filter"] = QCheckBox("Silero VAD pre-filter (skip silences, fewer hallucinations)")
        f.addRow("", c["vad_filter"])
        c["initial_prompt"] = QLineEdit()
        c["initial_prompt"].setPlaceholderText("optional: names, jargon, punctuation style hint")
        f.addRow("Initial prompt:", c["initial_prompt"])
        c["model_dir"] = QLineEdit()
        c["model_dir"].setPlaceholderText("default: Hugging Face cache")
        c["model_dir"].editingFinished.connect(self._refresh_model_labels)
        f.addRow("Model folder:", self._path_row(c["model_dir"], lambda: self._browse_dir(c["model_dir"], "Model folder")))
        c["use_transcription_cache"] = QCheckBox("Cache transcription next to the source file")
        f.addRow("", c["use_transcription_cache"])
        dl = QHBoxLayout()
        self.model_label = self._hint("")
        self.model_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.model_btn = QPushButton("Download / check model")
        self.model_btn.clicked.connect(self._download_model)
        dl.addWidget(self.model_label, 1)
        dl.addWidget(self.model_btn)
        f.addRow(dl)
        lay.addWidget(g)

        g2 = QGroupBox("Audio classifier (YAMNet, used by the Quality filters)")
        f2 = QFormLayout(g2)
        dl2 = QHBoxLayout()
        self.yamnet_label = self._hint("")
        self.yamnet_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.yamnet_btn = QPushButton("Download / check YAMNet")
        self.yamnet_btn.clicked.connect(self._download_yamnet)
        dl2.addWidget(self.yamnet_label, 1)
        dl2.addWidget(self.yamnet_btn)
        f2.addRow(dl2)
        lay.addWidget(g2)

        g3 = QGroupBox("GPU acceleration (NVIDIA CUDA)")
        f3 = QVBoxLayout(g3)
        self.gpu_label = self._hint("")
        self.gpu_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        f3.addWidget(self.gpu_label)
        c["cuda_lib_dir"] = QLineEdit()
        c["cuda_lib_dir"].setPlaceholderText("auto (pip packages, CUDA_PATH, 'cuda' folder next to the app)")
        c["cuda_lib_dir"].editingFinished.connect(self._refresh_gpu_status)
        r = QFormLayout()
        r.addRow("CUDA libraries folder:", self._path_row(c["cuda_lib_dir"],
                                                        lambda: self._browse_dir(c["cuda_lib_dir"], "CUDA libraries folder")))
        f3.addLayout(r)
        br = QHBoxLayout()
        self.cuda_dl_label = self._hint("")
        self.cuda_dl_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.cuda_btn = QPushButton("Download CUDA libraries")
        self.cuda_btn.setToolTip("cuBLAS 12 + cuDNN 9 from PyPI (official NVIDIA wheels, ~1 GB) into a 'cuda' folder next to the app")
        self.cuda_btn.clicked.connect(self._download_cuda)
        recheck = QPushButton("Re-check GPU")
        recheck.clicked.connect(self._refresh_gpu_status)
        br.addWidget(self.cuda_dl_label, 1)
        br.addWidget(self.cuda_btn)
        br.addWidget(recheck)
        f3.addLayout(br)
        lay.addWidget(g3)
        lay.addStretch()
        return w

    def _segmentation_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        c = self.controls
        g = QGroupBox("Sentence segments")
        f = QFormLayout(g)
        c["min_segment_length"] = QDoubleSpinBox()
        c["min_segment_length"].setRange(0.2, 30.0)
        c["min_segment_length"].setSuffix(" s")
        c["min_segment_length"].setSingleStep(0.5)
        f.addRow("Min length:", c["min_segment_length"])
        c["max_segment_length"] = QDoubleSpinBox()
        c["max_segment_length"].setRange(1.0, 120.0)
        c["max_segment_length"].setSuffix(" s")
        c["max_segment_length"].setSingleStep(1.0)
        f.addRow("Max length:", c["max_segment_length"])
        c["merge_short"] = QCheckBox("Merge short sentences with the next one")
        f.addRow("", c["merge_short"])
        c["merge_max_gap"] = QDoubleSpinBox()
        c["merge_max_gap"].setRange(0.0, 5.0)
        c["merge_max_gap"].setSuffix(" s")
        c["merge_max_gap"].setSingleStep(0.1)
        c["merge_max_gap"].setToolTip("Only merge when the pause between the sentences is at most this long")
        f.addRow("Merge max pause:", c["merge_max_gap"])
        c["overlength_action"] = QComboBox()
        for k, v in OVERLENGTH_ACTIONS.items():
            c["overlength_action"].addItem(v, k)
        f.addRow("Longer than max:", c["overlength_action"])
        f.addRow(self._hint("Segments are split at sentence punctuation (. ! ?) in any language; numbers like "
                            "\"3.5\" never split. Clips end at the last word, never at the next word's start."))
        lay.addWidget(g)

        g2 = QGroupBox("Boundary buffers")
        f2 = QFormLayout(g2)
        c["word_start_buffer_ms"] = QSpinBox()
        c["word_start_buffer_ms"].setRange(0, 2000)
        c["word_start_buffer_ms"].setSuffix(" ms")
        c["word_start_buffer_ms"].setSingleStep(50)
        f2.addRow("Before first word:", c["word_start_buffer_ms"])
        c["word_end_buffer_ms"] = QSpinBox()
        c["word_end_buffer_ms"].setRange(0, 2000)
        c["word_end_buffer_ms"].setSuffix(" ms")
        c["word_end_buffer_ms"].setSingleStep(50)
        f2.addRow("After last word:", c["word_end_buffer_ms"])
        f2.addRow(self._hint("Buffers extend into the surrounding pause but are clamped to the neighbouring words, "
                             "so they never cut into other speech."))
        lay.addWidget(g2)

        g3 = QGroupBox("Cutoff detection")
        f3 = QFormLayout(g3)
        c["check_cutoff"] = QCheckBox("Detect clipped first / last words")
        f3.addRow("", c["check_cutoff"])
        c["cutoff_check_mode"] = QComboBox()
        c["cutoff_check_mode"].addItems(CUTOFF_MODES)
        f3.addRow("Check:", c["cutoff_check_mode"])
        c["cutoff_sensitivity"] = QDoubleSpinBox()
        c["cutoff_sensitivity"].setRange(0.0, 1.0)
        c["cutoff_sensitivity"].setSingleStep(0.05)
        c["cutoff_sensitivity"].setToolTip("Peak level at the very edge that counts as speech. Lower = more sensitive")
        f3.addRow("Sensitivity:", c["cutoff_sensitivity"])
        c["cutoff_action"] = QComboBox()
        for k, v in CUTOFF_ACTIONS.items():
            c["cutoff_action"].addItem(v, k)
        f3.addRow("Action:", c["cutoff_action"])
        lay.addWidget(g3)
        lay.addStretch()
        return w

    def _quality_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        c = self.controls
        g = QGroupBox("Transcription confidence")
        f = QFormLayout(g)
        c["min_confidence"] = QDoubleSpinBox()
        c["min_confidence"].setRange(0.0, 1.0)
        c["min_confidence"].setSingleStep(0.05)
        c["min_confidence"].setToolTip("Average word probability reported by Whisper; clips below are rejected")
        f.addRow("Min confidence:", c["min_confidence"])
        lay.addWidget(g)

        g2 = QGroupBox("Audio content filters (YAMNet)")
        f2 = QFormLayout(g2)
        c["filter_music"] = QCheckBox("Reject music, singing, speech over music")
        f2.addRow("", c["filter_music"])
        c["music_threshold"] = QDoubleSpinBox()
        c["music_threshold"].setRange(0.0, 1.0)
        c["music_threshold"].setSingleStep(0.05)
        f2.addRow("Music threshold:", c["music_threshold"])
        c["mixed_threshold"] = QDoubleSpinBox()
        c["mixed_threshold"].setRange(0.0, 1.0)
        c["mixed_threshold"].setSingleStep(0.05)
        c["mixed_threshold"].setToolTip("Score for 'speech with background music' classes")
        f2.addRow("Mixed threshold:", c["mixed_threshold"])
        c["filter_noise"] = QCheckBox("Reject noisy clips (hum, hiss, wind…)")
        f2.addRow("", c["filter_noise"])
        c["noise_threshold"] = QDoubleSpinBox()
        c["noise_threshold"].setRange(0.0, 1.0)
        c["noise_threshold"].setSingleStep(0.05)
        f2.addRow("Noise threshold:", c["noise_threshold"])
        c["filter_silence"] = QCheckBox("Reject mostly silent clips")
        f2.addRow("", c["filter_silence"])
        c["silence_threshold"] = QDoubleSpinBox()
        c["silence_threshold"].setRange(0.0, 1.0)
        c["silence_threshold"].setSingleStep(0.05)
        f2.addRow("Silence threshold:", c["silence_threshold"])
        f2.addRow(self._hint("Lower thresholds reject more. The classifier downloads once (~16 MB) and runs on the CPU."))
        lay.addWidget(g2)

        g3 = QGroupBox("Edge segments")
        f3 = QFormLayout(g3)
        c["skip_edge_segments"] = QCheckBox("Drop first / last segments of each file")
        f3.addRow("", c["skip_edge_segments"])
        c["skip_segments_count"] = QSpinBox()
        c["skip_segments_count"].setRange(1, 20)
        f3.addRow("Segments per end:", c["skip_segments_count"])
        lay.addWidget(g3)
        lay.addStretch()
        return w

    def _export_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        c = self.controls
        g = QGroupBox("Output")
        f = QFormLayout(g)
        c["output_dir"] = QLineEdit()
        f.addRow("Output folder:", self._path_row(c["output_dir"], lambda: self._browse_dir(c["output_dir"], "Output folder")))
        c["speaker_id"] = QLineEdit()
        c["speaker_id"].setPlaceholderText("empty = source file name")
        c["speaker_id"].setToolTip("Prefix for clip names; numbering continues across runs")
        f.addRow("Speaker id:", c["speaker_id"])
        c["export_format"] = QComboBox()
        c["export_format"].addItems(EXPORT_FORMATS)
        f.addRow("Format:", c["export_format"])
        c["export_sample_rate"] = QComboBox()
        for sr in (0, 16000, 22050, 24000, 44100, 48000):
            c["export_sample_rate"].addItem("keep original" if sr == 0 else f"{sr} Hz", sr)
        f.addRow("Sample rate:", c["export_sample_rate"])
        f.addRow(self._hint("Writes audio/<name>_0001.wav, transcripts/<name>_0001.txt, metadata.csv "
                            "(Hugging Face AudioFolder: file_name, transcription), metadata.json and manifest.txt."))
        lay.addWidget(g)

        g2 = QGroupBox("Silence trimming")
        f2 = QFormLayout(g2)
        c["trim_silence"] = QCheckBox("Trim leading / trailing silence")
        f2.addRow("", c["trim_silence"])
        c["trim_threshold_db"] = QDoubleSpinBox()
        c["trim_threshold_db"].setRange(-80.0, -10.0)
        c["trim_threshold_db"].setSuffix(" dB")
        f2.addRow("Threshold:", c["trim_threshold_db"])
        c["trim_chunk_ms"] = QSpinBox()
        c["trim_chunk_ms"].setRange(5, 50)
        c["trim_chunk_ms"].setSuffix(" ms")
        f2.addRow("Analysis window:", c["trim_chunk_ms"])
        c["trim_pad_ms"] = QSpinBox()
        c["trim_pad_ms"].setRange(0, 500)
        c["trim_pad_ms"].setSuffix(" ms")
        f2.addRow("Keep padding:", c["trim_pad_ms"])
        c["remove_fragments"] = QCheckBox("Remove a stray burst after a pause at the end")
        f2.addRow("", c["remove_fragments"])
        lay.addWidget(g2)

        g3 = QGroupBox("Logging")
        f3 = QFormLayout(g3)
        c["log_level"] = QComboBox()
        c["log_level"].addItems(["INFO", "DEBUG"])
        f3.addRow("Log level:", c["log_level"])
        lay.addWidget(g3)
        lay.addStretch()
        return w

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("&File")
        fm.addAction("Add files…", self.add_files, "Ctrl+O")
        fm.addAction("Add folder…", self.add_folder, "Ctrl+Shift+O")
        fm.addAction("Open output folder", self.open_output)
        fm.addSeparator()
        fm.addAction("Exit", self.close, "Ctrl+Q")

        self.presets_menu = mb.addMenu("&Presets")
        self._rebuild_presets_menu()

        hm = mb.addMenu("&Help")
        hm.addAction("About", self._about)

    def _rebuild_presets_menu(self):
        m = self.presets_menu
        m.clear()
        for key, p in PRESETS.items():
            m.addAction(p["name"], lambda k=key: self._apply_preset(k))
        m.addSeparator()
        user = sorted((f[:-5] for f in os.listdir(self.presets_dir) if f.endswith(".json") and f != "defaults.json"),
                      key=path_key)
        if user:
            for name in user:
                m.addAction(name, lambda n=name: self._load_preset_file(os.path.join(self.presets_dir, n + ".json")))
            m.addSeparator()
        m.addAction("Save preset…", self._save_preset)
        m.addAction("Delete preset…", self._delete_preset)
        m.addAction("Import preset…", self._import_preset)
        m.addAction("Export preset…", self._export_preset)
        m.addSeparator()
        m.addAction("Save current settings as defaults", self._save_defaults)
        m.addAction("Reset to factory defaults", lambda: self.apply_settings(dict(DEFAULT_SETTINGS)))

    # ------------------------------------------------------------- settings io
    def get_settings(self) -> Dict:
        s = {}
        for key, w in self.controls.items():
            if isinstance(w, QCheckBox):
                s[key] = w.isChecked()
            elif isinstance(w, QComboBox):
                data = w.currentData()
                s[key] = data if data is not None and not w.isEditable() else w.currentText().strip()
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                s[key] = w.value()
            elif isinstance(w, QLineEdit):
                s[key] = w.text().strip()
        return s

    def apply_settings(self, s: Dict):
        for key, w in self.controls.items():
            if key not in s:
                continue
            v = s[key]
            try:
                if isinstance(w, QCheckBox):
                    w.setChecked(bool(v))
                elif isinstance(w, QComboBox):
                    idx = w.findData(v)
                    if idx < 0:
                        idx = w.findText(str(v))
                    if idx >= 0:
                        w.setCurrentIndex(idx)
                    elif w.isEditable():
                        w.setCurrentText(str(v))
                elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                    w.setValue(v)
                elif isinstance(w, QLineEdit):
                    w.setText("" if v is None else str(v))
            except Exception:
                pass
        self._refresh_model_labels()

    def _load_persisted(self):
        defaults = os.path.join(self.presets_dir, "defaults.json")
        if os.path.isfile(defaults):
            try:
                with open(defaults, encoding="utf-8") as fh:
                    self.apply_settings(json.load(fh))
            except Exception:
                pass
        raw = self.qsettings.value("settings")
        if raw:
            try:
                self.apply_settings(json.loads(raw))
            except Exception:
                pass
        geo = self.qsettings.value("geometry")
        if geo:
            self.restoreGeometry(geo)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            if QMessageBox.question(self, "Quit", "Processing is running. Stop and quit?") != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.worker.request_stop()
            self.worker.wait(5000)
        self.qsettings.setValue("settings", json.dumps(self.get_settings()))
        self.qsettings.setValue("geometry", self.saveGeometry())
        event.accept()

    # ---------------------------------------------------------------- presets
    def _apply_preset(self, key):
        self.apply_settings(PRESETS[key]["settings"])
        self.statusBar().showMessage(f"Preset applied: {PRESETS[key]['name']}")

    def _write_json(self, path: str, data: Dict) -> bool:
        try:
            atomic_write_text(path, json.dumps(data, indent=2))
            return True
        except Exception as exc:
            QMessageBox.warning(self, "Preset", f"Could not write {path}:\n{exc}")
            return False

    def _save_preset(self):
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        if ok and name.strip():
            if self._write_json(os.path.join(self.presets_dir, name.strip() + ".json"), self.get_settings()):
                self._rebuild_presets_menu()

    def _delete_preset(self):
        user = sorted((f[:-5] for f in os.listdir(self.presets_dir) if f.endswith(".json") and f != "defaults.json"),
                      key=path_key)
        if not user:
            QMessageBox.information(self, "Presets", "No saved presets.")
            return
        name, ok = QInputDialog.getItem(self, "Delete preset", "Preset:", user, 0, False)
        if ok and name:
            os.remove(os.path.join(self.presets_dir, name + ".json"))
            self._rebuild_presets_menu()

    def _load_preset_file(self, path):
        try:
            with open(path, encoding="utf-8") as fh:
                self.apply_settings(json.load(fh))
            self.statusBar().showMessage(f"Preset loaded: {os.path.basename(path)}")
        except Exception as exc:
            QMessageBox.warning(self, "Preset", f"Could not read preset:\n{exc}")

    def _import_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import preset", "", "JSON (*.json)")
        if path:
            self._load_preset_file(path)

    def _export_preset(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export preset", "preset.json", "JSON (*.json)")
        if path:
            self._write_json(path, self.get_settings())

    def _save_defaults(self):
        if self._write_json(os.path.join(self.presets_dir, "defaults.json"), self.get_settings()):
            self.statusBar().showMessage("Current settings saved as defaults")

    # ------------------------------------------------------------- downloads
    def _refresh_gpu_status(self, *_):
        from cuda_utils import cuda_status
        st = cuda_status(self.controls["cuda_lib_dir"].text().strip())
        self.gpu_label.setText(str(st["text"]))
        self.gpu_label.setStyleSheet("color: %s; font-size: 11px;" % ("#155724" if st["ready"] else "#856404"))

    def _refresh_model_labels(self, *_):
        if not hasattr(self, "model_label"):
            return
        from transcription import is_model_cached, model_repo_id
        from audio_classifier import is_yamnet_cached
        model = self.controls["whisper_model"].currentText().strip()
        mdir = self.controls["model_dir"].text().strip()
        if model:
            cached = is_model_cached(model, mdir)
            self.model_label.setText(f"{model_repo_id(model)} — {'downloaded' if cached else 'not downloaded yet'}")
        self.yamnet_label.setText("YAMNet — " + ("downloaded" if is_yamnet_cached(mdir) else "not downloaded yet"))

    def _run_download(self, button: QPushButton, label: QLabel, fn, done_text):
        if self._dl_thread and self._dl_thread.isRunning():
            QMessageBox.information(self, "Download", "Another download is still running.")
            return
        button.setEnabled(False)
        old = button.text()
        button.setText("Downloading…")
        t = DownloadThread(fn)
        t.progress.connect(lambda m, d, tot: label.setText(f"{m} — {d / 1048576:.0f} / {tot / 1048576:.0f} MB" if tot else m))

        def done(result, err):
            button.setEnabled(True)
            button.setText(old)
            if err:
                label.setText("Download failed")
                QMessageBox.warning(self, "Download failed", err)
            else:
                label.setText(done_text(result))
            self._refresh_model_labels()
            self._refresh_gpu_status()
        t.done.connect(done)
        self._dl_thread = t
        t.start()

    def _download_model(self):
        from transcription import download_model
        model = self.controls["whisper_model"].currentText().strip()
        mdir = self.controls["model_dir"].text().strip()
        self.model_label.setText(f"Downloading {model}…")
        self._run_download(self.model_btn, self.model_label, lambda th: download_model(model, mdir),
                           lambda r: f"Model ready: {r}")

    def _download_yamnet(self):
        from audio_classifier import download_yamnet
        mdir = self.controls["model_dir"].text().strip()
        self.yamnet_label.setText("Downloading YAMNet…")
        self._run_download(self.yamnet_btn, self.yamnet_label, lambda th: download_yamnet(mdir),
                           lambda r: "YAMNet ready")

    def _download_cuda(self):
        from cuda_utils import default_cuda_dir, download_cuda_libraries
        if sys.platform == "darwin":
            QMessageBox.information(self, "CUDA", "CUDA is not available on macOS.")
            return
        dest = self.controls["cuda_lib_dir"].text().strip() or default_cuda_dir()
        if QMessageBox.question(self, "Download CUDA libraries",
                                f"Download cuBLAS 12 and cuDNN 9 (official NVIDIA wheels from PyPI, about 1 GB) into:\n"
                                f"{dest}\n\nContinue?") != QMessageBox.StandardButton.Yes:
            return

        def fn(th):
            return download_cuda_libraries(dest, progress=lambda m, d, t: th.progress.emit(m, d, t),
                                           should_stop=lambda: th.stop)

        def done_text(files):
            if not self.controls["cuda_lib_dir"].text().strip():
                self.controls["cuda_lib_dir"].setText(dest)
            return f"{len(files)} libraries installed in {dest}"
        self._run_download(self.cuda_btn, self.cuda_dl_label, fn, done_text)

    def _check_ffmpeg(self):
        from audio_processor import find_ffmpeg, track_pydub_processes
        track_pydub_processes()
        if not find_ffmpeg():
            self._append_log("WARNING", "ffmpeg not found: only plain WAV input will work. Install FFmpeg "
                                        "(or put ffmpeg next to the app) for mp3 / m4a / flac / video input.")

    # ------------------------------------------------------------------ queue
    def add_paths(self, paths: List[str]):
        added = 0
        self._dupes_seen = 0
        for p in paths:
            path = Path(p)
            if path.is_dir():
                found = [f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS]
                for f in sorted(found, key=path_key):
                    added += self._add_one(str(f))
            elif path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                added += self._add_one(str(path))
        dupes = self._dupes_seen
        self._dupes_seen = 0
        parts = []
        if added:
            parts.append(f"{added} file(s) added")
        if dupes:
            parts.append(f"{dupes} already in queue")
        if parts:
            self.statusBar().showMessage(", ".join(parts) + f" — {len(self.files)} in queue")

    @staticmethod
    def _key(path: str) -> str:
        return os.path.normcase(os.path.abspath(path))

    def _add_one(self, path: str) -> int:
        key = self._key(path)
        if any(self._key(f) == key for f in self.files):
            self._dupes_seen += 1
            return 0
        self.files.append(path)
        r = self.table.rowCount()
        self.table.insertRow(r)
        item = QTableWidgetItem(os.path.basename(path))
        item.setToolTip(path)
        self.table.setItem(r, 0, item)
        self.table.setItem(r, 1, QTableWidgetItem("pending"))
        self.table.setItem(r, 2, QTableWidgetItem(""))
        self._color_row(r, "pending")
        return 1

    def _color_row(self, row: int, state: str):
        for col in range(3):
            it = self.table.item(row, col)
            if it:
                it.setBackground(QColor(_STATUS_COLORS.get(state, "#ffffff")))

    def add_files(self):
        exts = " ".join(f"*{e}" for e in AUDIO_EXTENSIONS)
        files, _ = QFileDialog.getOpenFileNames(self, "Add audio files", "", f"Audio / video ({exts});;All files (*)")
        self.add_paths(files)

    def add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Add folder")
        if d:
            self.add_paths([d])

    def remove_selected(self):
        if self.worker and self.worker.isRunning():
            return
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
            del self.files[r]

    def clear_queue(self):
        if self.worker and self.worker.isRunning():
            return
        self.files.clear()
        self.table.setRowCount(0)

    def open_output(self):
        out = os.path.abspath(self.controls["output_dir"].text().strip() or "output")
        os.makedirs(out, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(out)  # noqa
            elif sys.platform == "darwin":
                # deliberately not via childproc: the file manager is the user's, it must outlive us
                subprocess.Popen(["open", out])
            else:
                subprocess.Popen(["xdg-open", out])
        except Exception as exc:
            QMessageBox.information(self, "Output folder", f"{out}\n\n({exc})")

    # -------------------------------------------------------------- processing
    def _validate(self, s: Dict) -> List[str]:
        problems = []
        if s["min_segment_length"] >= s["max_segment_length"]:
            problems.append("Min segment length must be smaller than max segment length.")
        if s["filter_music"] and s["music_threshold"] < s["mixed_threshold"]:
            problems.append("Music threshold is below the mixed threshold (mixed is usually the stricter one).")
        model = s["whisper_model"]
        if model.endswith(".en") and s["language"] not in ("en", "auto"):
            problems.append(f"{model} is an English-only model but language is '{s['language']}'.")
        if s["device"] == "cuda":
            from transcription import cuda_available
            if not cuda_available(s.get("cuda_lib_dir", "")):
                problems.append("Device is set to CUDA but no usable GPU / CUDA libraries were found.")
        return problems

    def start(self):
        if not self.files:
            QMessageBox.information(self, "Queue", "Add some audio files first.")
            return
        s = self.get_settings()
        problems = self._validate(s)
        if problems:
            if QMessageBox.question(self, "Check settings", "\n\n".join(problems) + "\n\nStart anyway?") \
                    != QMessageBox.StandardButton.Yes:
                return
        for r in range(self.table.rowCount()):
            self.table.item(r, 1).setText("pending")
            self.table.item(r, 2).setText("")
            self._color_row(r, "pending")
        self.log.clear()
        self._clips = 0
        self._seconds = 0.0
        self._run_start = time.time()
        self.total_bar.setRange(0, len(self.files))
        self.total_bar.setValue(0)
        self.file_bar.setRange(0, 100)
        self.file_bar.setValue(0)
        self.stat_clips.setText("0")
        self.stat_minutes.setText("0.0 min")
        self.stat_device.setText("--")
        self._timer.start(1000)
        self.worker = Worker(list(self.files), s)
        w = self.worker
        w.log.connect(self._append_log)
        w.status.connect(self._on_status)
        w.file_started.connect(self._on_file_started)
        w.file_progress.connect(self._on_file_progress)
        w.file_done.connect(self._on_file_done)
        w.finished_all.connect(self._on_finished)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.tabs.setEnabled(False)
        w.start()

    def stop(self):
        if self.worker:
            self.worker.request_stop()
            self.phase_label.setText("Stopping…")
            self.stop_btn.setEnabled(False)

    def _tick(self):
        self.stat_elapsed.setText(format_duration(time.time() - self._run_start))

    def _on_status(self, text: str):
        self.phase_label.setText(text)
        self.statusBar().showMessage(text)
        if text.startswith("Loading"):
            self.file_bar.setRange(0, 0)       # busy indicator while models load

    def _on_file_started(self, i: int):
        self.table.item(i, 1).setText("running")
        self._color_row(i, "running")
        self.table.scrollToItem(self.table.item(i, 0))
        self.counter_label.setText(f"file {i + 1} / {len(self.files)} — {os.path.basename(self.files[i])}")
        self.file_bar.setRange(0, 100)
        self.file_bar.setValue(0)

    def _on_file_progress(self, phase: str, done: float, total: float):
        # transcribe = 0-70 %, filter = 70-90 %, export = 90-100 %
        spans = {"transcribe": (0, 70), "filter": (70, 90), "export": (90, 100)}
        lo, hi = spans.get(phase, (0, 100))
        frac = (done / total) if total else 0.0
        if self.file_bar.maximum() == 0:
            self.file_bar.setRange(0, 100)
        self.file_bar.setValue(int(lo + (hi - lo) * min(1.0, frac)))

    def _on_file_done(self, i: int, state: str, stats: Dict):
        self.table.item(i, 1).setText(state)
        self._color_row(i, state)
        if state == "done":
            self.table.item(i, 2).setText(str(stats.get("segments", 0)))
            self._clips += stats.get("segments", 0)
            self._seconds += stats.get("duration", 0.0)
            self.stat_clips.setText(str(self._clips))
            self.stat_minutes.setText(f"{self._seconds / 60:.1f} min")
            self.file_bar.setRange(0, 100)
            self.file_bar.setValue(100)
        self.total_bar.setValue(self.total_bar.value() + 1)

    def _on_finished(self, ok: int, total: int, meta: Dict):
        self._timer.stop()
        self._tick()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.tabs.setEnabled(True)
        if self.file_bar.maximum() == 0:
            self.file_bar.setRange(0, 100)
        self.phase_label.setText("Finished" if ok == total else "Finished with problems")
        self.counter_label.setText("")
        if meta:
            out = os.path.abspath(self.controls["output_dir"].text() or "output")
            self._append_log("INFO", f"Dataset: {meta.get('total_segments', 0)} clips, "
                                     f"{meta.get('total_duration', 0) / 60:.1f} min → {out}")
        self.statusBar().showMessage(f"Done: {ok}/{total} files")
        if self.worker:
            self.worker.wait(10000)
            self.worker.deleteLater()
        self.worker = None

    def _append_log(self, level: str, msg: str):
        color = {"DEBUG": "#888", "INFO": "#2c3e50", "WARNING": "#b8860b", "ERROR": "#c0392b",
                 "CRITICAL": "#c0392b"}.get(level, "#2c3e50")
        if msg.startswith("Device: "):
            self.stat_device.setText(msg[8:])
        safe = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.log.append(f'<span style="color:{color};">{safe}</span>')
        self.log.moveCursor(QTextCursor.MoveOperation.End)

    def _about(self):
        QMessageBox.about(self, f"About {APP_NAME}",
                          f"<b>{APP_NAME} {APP_VERSION}</b><br>Turn audio into a TTS training dataset.<br><br>"
                          "faster-whisper (CTranslate2) for word-level transcription, YAMNet (onnxruntime) for "
                          "music / noise rejection, FFmpeg for decoding.<br><br>"
                          "<a href='https://github.com/hclivess/speech-splitter'>github.com/hclivess/speech-splitter</a>")


# =============================================================================== entry
def main():
    app = QApplication(sys.argv)
    childproc.install_qt_hook(app)          # every ffmpeg/ffprobe/nvidia-smi we start dies with the app
    app.setStyle("Fusion")
    icon = os.path.join(app_dir(), "icon.ico")
    if os.path.exists(icon):
        app.setWindowIcon(QIcon(icon))
    win = MainWindow()
    win.show()
    selftest = os.environ.get("SPEECH_SPLITTER_SELFTEST")
    if selftest:
        _selftest(win, selftest)
    sys.exit(app.exec())


def _selftest(win: MainWindow, path: str):
    """Headless smoke test for CI: tiny.en on the CPU, music filter on (exercises YAMNet), exit 0 on success"""
    out = os.path.join(os.path.dirname(os.path.abspath(path)), "selftest_out")
    win.apply_settings({"whisper_model": "tiny.en", "language": "en", "device": "cpu", "beam_size": 1,
                        "filter_music": True, "output_dir": out, "use_transcription_cache": False})
    win.add_paths([path])

    def finished(ok, total, meta):
        print(f"selftest: finished {ok}/{total}, {meta.get('total_segments', 0)} clips", flush=True)
        code = 0 if ok == total and meta.get("total_segments", 0) > 0 else 1
        QTimer.singleShot(0, lambda: (win.worker and win.worker.wait(10000), QApplication.exit(code)))

    def go():
        win.start()
        win.worker.log.connect(lambda lvl, m: print(f"selftest[{lvl}]: {m}", flush=True))
        win.worker.status.connect(lambda m: print(f"selftest: {m}", flush=True))
        win.worker.finished_all.connect(finished)
    QTimer.singleShot(300, go)


if __name__ == "__main__":
    main()
