import sys
import os
from pathlib import Path
from typing import List
import json
import mimetypes

# Fix for torch._pytree compatibility issue - MUST BE BEFORE PySide6 imports
import warnings
warnings.filterwarnings('ignore')

try:
    import torch.utils._pytree as pytree
    if not hasattr(pytree, 'register_pytree_node'):
        pytree.register_pytree_node = lambda *args, **kwargs: None
except (ImportError, AttributeError):
    pass

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QTextEdit, QProgressBar, QGroupBox, QListWidget, QMessageBox,
    QCheckBox, QTabWidget, QListWidgetItem, QStatusBar, QMenuBar, QMenu,
    QGridLayout, QFrame, QSplitter, QScrollArea
)
from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtGui import QFont, QIcon, QDragEnterEvent, QDropEvent, QColor, QTextCursor
from PySide6.QtMultimedia import QMediaFormat


class ProcessingThread(QThread):
    progress = Signal(int)
    status = Signal(str)
    log = Signal(str, str)  # message, level
    finished = Signal(dict)
    error = Signal(str)
    
    def __init__(self, settings: dict, files: List[str]):
        super().__init__()
        self.settings = settings
        self.files = files
        self._is_running = True
    
    def stop(self):
        self._is_running = False
    
    def run(self):
        try:
            from speech_splitter_v3 import SpeechSplitter
            
            self.status.emit("Initializing models...")
            self.log.emit(f"Loading Whisper model: <b>{self.settings['model']}</b>", "info")
            
            splitter = SpeechSplitter(**self.settings)
            
            self.log.emit("✓ Models loaded successfully", "success")
            
            total_files = len(self.files)
            results = []
            
            for idx, file_path in enumerate(self.files):
                if not self._is_running:
                    self.log.emit("\n⚠ Processing cancelled by user", "warning")
                    break
                
                self.status.emit(f"Processing file {idx + 1}/{total_files}")
                self.log.emit(f"<hr><b>File:</b> {Path(file_path).name}", "info")
                
                try:
                    speaker_id = self.settings.get('speaker_id') or Path(file_path).stem
                    stats = splitter.process_file(file_path, speaker_id=speaker_id)
                    
                    results.append({
                        'file': file_path,
                        'status': 'success',
                        'segments': stats['total_segments'],
                        'duration': stats['total_duration']
                    })
                    
                    self.log.emit(f"✓ Complete: {stats['total_segments']} segments, {stats['total_duration']:.1f}s", "success")
                    
                except Exception as e:
                    self.log.emit(f"✗ Error: {str(e)}", "error")
                    results.append({'file': file_path, 'status': 'failed', 'error': str(e)})
                
                progress = int((idx + 1) / total_files * 100)
                self.progress.emit(progress)
            
            # Summary
            success = sum(1 for r in results if r['status'] == 'success')
            total_segs = sum(r.get('segments', 0) for r in results)
            
            self.log.emit("<hr><h3>Processing Summary</h3>", "info")
            self.log.emit(f"<b>Success:</b> {success}/{total_files} files", "info")
            self.log.emit(f"<b>Total segments:</b> {total_segs}", "info")
            self.log.emit(f"<b>Output directory:</b> {self.settings['output_dir']}", "info")
            
            if self._is_running and results:
                self.status.emit("Generating metadata files...")
                try:
                    splitter.generate_metadata()
                    self.log.emit("✓ Metadata files created", "success")
                except Exception as e:
                    self.log.emit(f"⚠ Metadata generation warning: {e}", "warning")
            
            self.finished.emit({'results': results, 'success': success, 'total_segments': total_segs})
            
        except Exception as e:
            self.error.emit(str(e))


class SpeechSplitterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processing_thread = None
        self.selected_files = []
        self.init_ui()
        self.load_settings()
    
    def init_ui(self):
        self.setWindowTitle("Speech Splitter v4 - TTS Data Generator")
        self.setMinimumSize(1200, 800)
        
        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        
        save_action = file_menu.addAction("Save Settings")
        save_action.triggered.connect(self.save_settings)
        save_action.setShortcut("Ctrl+S")
        
        load_action = file_menu.addAction("Load Settings")
        load_action.triggered.connect(self.load_settings_dialog)
        load_action.setShortcut("Ctrl+O")
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut("Ctrl+Q")
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Main content splitter
        content_splitter = QSplitter(Qt.Horizontal)
        
        # Settings panel
        settings_widget = self.create_settings_panel()
        content_splitter.addWidget(settings_widget)
        
        # Files and log panel
        right_panel = self.create_right_panel()
        content_splitter.addWidget(right_panel)
        
        content_splitter.setStretchFactor(0, 1)
        content_splitter.setStretchFactor(1, 2)
        main_layout.addWidget(content_splitter)
        
        # Progress bar and control buttons
        controls = QHBoxLayout()
        
        self.start_button = QPushButton("▶ Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        self.start_button.setStyleSheet("background: #27ae60; color: white; font-size: 14px; padding: 10px;")
        controls.addWidget(self.start_button)
        
        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("background: #e74c3c; color: white; font-size: 14px; padding: 10px;")
        controls.addWidget(self.stop_button)
        
        main_layout.addLayout(controls)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Set initial splitter sizes
        content_splitter.setSizes([400, 800])
    
    def create_header(self):
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)
        
        title = QLabel("Speech Splitter v4")
        title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        title.setStyleSheet("color: #2c3e50;")
        
        subtitle = QLabel("AI-Powered TTS Dataset Generator (Optimized)")
        subtitle.setFont(QFont("Segoe UI", 12))
        subtitle.setStyleSheet("color: #7f8c8d;")
        
        vbox = QVBoxLayout()
        vbox.addWidget(title)
        vbox.addWidget(subtitle)
        layout.addLayout(vbox)
        
        layout.addStretch()
        
        # GPU indicator
        gpu_label = QLabel("🎮 GPU: Checking...")
        gpu_label.setFont(QFont("Segoe UI", 10))
        gpu_label.setStyleSheet("padding: 5px 10px; border-radius: 5px; background: #ecf0f1;")
        
        # Check GPU in background
        def check_gpu():
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_label.setText(f"🎮 GPU: {torch.cuda.get_device_name(0)[:20]}...")
                    gpu_label.setStyleSheet("padding: 5px 10px; border-radius: 5px; background: #d4edda; color: #155724;")
                else:
                    gpu_label.setText("💻 CPU Mode")
                    gpu_label.setStyleSheet("padding: 5px 10px; border-radius: 5px; background: #fff3cd; color: #856404;")
            except Exception as e:
                gpu_label.setText(f"⚠ GPU Check Failed: {str(e)[:30]}")
                gpu_label.setStyleSheet("padding: 5px 10px; border-radius: 5px; background: #f8d7da; color: #721c24;")
        
        import threading
        threading.Thread(target=check_gpu, daemon=True).start()
        
        layout.addWidget(gpu_label)
        
        return header
    
    def create_settings_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        tabs = QTabWidget()
        tabs.addTab(self.create_basic_tab(), "Basic")
        tabs.addTab(self.create_quality_tab(), "Quality")
        tabs.addTab(self.create_advanced_tab(), "Advanced")
        tabs.addTab(self.create_output_tab(), "Output")
        
        layout.addWidget(tabs)
        
        # Settings actions
        btn_layout = QHBoxLayout()
        
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_settings)
        reset_btn.setStyleSheet("background: #95a5a6; color: white;")
        btn_layout.addWidget(reset_btn)
        
        validate_btn = QPushButton("Validate Settings")
        validate_btn.clicked.connect(self.validate_settings)
        validate_btn.setStyleSheet("background: #3498db; color: white;")
        btn_layout.addWidget(validate_btn)
        
        layout.addLayout(btn_layout)
        
        return panel
    
    def create_basic_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(15)
        
        # Model settings
        model_group = QGroupBox("Whisper Model")
        model_layout = QVBoxLayout()
        model_layout.setSpacing(8)
        
        model_layout.addWidget(QLabel("Model Size:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'])
        self.model_combo.setCurrentText('large')
        self.model_combo.setMinimumHeight(32)
        self.model_combo.setToolTip("Larger models are more accurate but slower")
        model_layout.addWidget(self.model_combo)
        
        model_layout.addWidget(QLabel("Language:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(['cs', 'en', 'de', 'es', 'fr', 'it', 'pl', 'ru'])
        self.language_combo.setMinimumHeight(32)
        self.language_combo.setToolTip("Must match your audio language")
        model_layout.addWidget(self.language_combo)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Segment settings
        seg_group = QGroupBox("Segmentation")
        seg_layout = QVBoxLayout()
        seg_layout.setSpacing(8)
        
        seg_layout.addWidget(QLabel("Min Length (s):"))
        self.min_length_spin = QDoubleSpinBox()
        self.min_length_spin.setRange(0.5, 10.0)
        self.min_length_spin.setValue(1.0)
        self.min_length_spin.setSingleStep(0.5)
        self.min_length_spin.setMinimumHeight(32)
        self.min_length_spin.setToolTip("Shorter segments are filtered out")
        seg_layout.addWidget(self.min_length_spin)
        
        seg_layout.addWidget(QLabel("Max Length (s):"))
        self.max_length_spin = QDoubleSpinBox()
        self.max_length_spin.setRange(5.0, 30.0)
        self.max_length_spin.setValue(15.0)
        self.max_length_spin.setSingleStep(1.0)
        self.max_length_spin.setMinimumHeight(32)
        self.max_length_spin.setToolTip("Longer segments are marked as _LONG")
        seg_layout.addWidget(self.max_length_spin)
        
        seg_layout.addWidget(QLabel("Pause Threshold (s):"))
        self.pause_threshold_spin = QDoubleSpinBox()
        self.pause_threshold_spin.setRange(0.1, 2.0)
        self.pause_threshold_spin.setValue(0.5)
        self.pause_threshold_spin.setSingleStep(0.1)
        self.pause_threshold_spin.setMinimumHeight(32)
        self.pause_threshold_spin.setToolTip("Lower = more splits, Higher = fewer splits")
        seg_layout.addWidget(self.pause_threshold_spin)
        
        seg_group.setLayout(seg_layout)
        layout.addWidget(seg_group)
        
        layout.addStretch()
        
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        return tab
    
    def create_quality_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(15)
        
        # Confidence
        conf_group = QGroupBox("Transcription Quality")
        conf_layout = QVBoxLayout()
        conf_layout.setSpacing(8)
        
        conf_layout.addWidget(QLabel("Min Confidence:"))
        self.min_confidence_spin = QDoubleSpinBox()
        self.min_confidence_spin.setRange(0.0, 1.0)
        self.min_confidence_spin.setValue(0.5)
        self.min_confidence_spin.setSingleStep(0.05)
        self.min_confidence_spin.setMinimumHeight(32)
        self.min_confidence_spin.setToolTip("Higher = stricter quality control")
        conf_layout.addWidget(self.min_confidence_spin)
        
        conf_group.setLayout(conf_layout)
        layout.addWidget(conf_group)
        
        # Noise filtering
        noise_group = QGroupBox("Noise Filtering")
        noise_layout = QVBoxLayout()
        noise_layout.setSpacing(8)
        
        self.filter_noise_check = QCheckBox("Enable Noise Filtering")
        self.filter_noise_check.setChecked(True)
        self.filter_noise_check.setMinimumHeight(28)
        self.filter_noise_check.setToolTip("Filter silent or noisy segments")
        noise_layout.addWidget(self.filter_noise_check)
        
        noise_layout.addWidget(QLabel("Silence Threshold:"))
        self.noise_silence_threshold_spin = QDoubleSpinBox()
        self.noise_silence_threshold_spin.setRange(0.1, 0.9)
        self.noise_silence_threshold_spin.setValue(0.3)
        self.noise_silence_threshold_spin.setSingleStep(0.05)
        self.noise_silence_threshold_spin.setMinimumHeight(32)
        noise_layout.addWidget(self.noise_silence_threshold_spin)
        
        noise_group.setLayout(noise_layout)
        layout.addWidget(noise_group)
        
        # Music filtering
        music_group = QGroupBox("Music Filtering")
        music_layout = QVBoxLayout()
        music_layout.setSpacing(8)
        
        self.filter_music_check = QCheckBox("Enable Music Filtering")
        self.filter_music_check.setChecked(True)
        self.filter_music_check.setMinimumHeight(28)
        music_layout.addWidget(self.filter_music_check)
        
        music_layout.addWidget(QLabel("Music Threshold:"))
        self.music_threshold_spin = QDoubleSpinBox()
        self.music_threshold_spin.setRange(0.1, 0.9)
        self.music_threshold_spin.setValue(0.3)
        self.music_threshold_spin.setMinimumHeight(32)
        music_layout.addWidget(self.music_threshold_spin)
        
        music_layout.addWidget(QLabel("Mixed Threshold:"))
        self.mixed_threshold_spin = QDoubleSpinBox()
        self.mixed_threshold_spin.setRange(0.1, 0.9)
        self.mixed_threshold_spin.setValue(0.2)
        self.mixed_threshold_spin.setMinimumHeight(32)
        music_layout.addWidget(self.mixed_threshold_spin)
        
        music_group.setLayout(music_layout)
        layout.addWidget(music_group)
        
        layout.addStretch()
        
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        return tab
    
    def create_advanced_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(15)
        
        # Buffer settings
        buffer_group = QGroupBox("Audio Buffers")
        buffer_layout = QVBoxLayout()
        buffer_layout.setSpacing(8)
        
        buffer_layout.addWidget(QLabel("Start Buffer (ms):"))
        self.word_start_buffer_spin = QSpinBox()
        self.word_start_buffer_spin.setRange(0, 1500)
        self.word_start_buffer_spin.setValue(750)
        self.word_start_buffer_spin.setMinimumHeight(32)
        self.word_start_buffer_spin.setToolTip("Padding before first word - increase to 750+ to fix cutoff issues")
        buffer_layout.addWidget(self.word_start_buffer_spin)
        
        buffer_layout.addWidget(QLabel("End Buffer (ms):"))
        self.word_end_buffer_spin = QSpinBox()
        self.word_end_buffer_spin.setRange(0, 1000)
        self.word_end_buffer_spin.setValue(300)
        self.word_end_buffer_spin.setMinimumHeight(32)
        self.word_end_buffer_spin.setToolTip("Padding after last word")
        buffer_layout.addWidget(self.word_end_buffer_spin)
        
        buffer_group.setLayout(buffer_layout)
        layout.addWidget(buffer_group)
        
        # Cutoff detection
        cutoff_group = QGroupBox("Cutoff Detection")
        cutoff_layout = QVBoxLayout()
        cutoff_layout.setSpacing(8)
        
        self.check_cutoff_check = QCheckBox("Enable Cutoff Detection")
        self.check_cutoff_check.setChecked(False)
        self.check_cutoff_check.setMinimumHeight(28)
        self.check_cutoff_check.setToolTip("Detect and handle segments cut off mid-word")
        cutoff_layout.addWidget(self.check_cutoff_check)
        
        cutoff_layout.addWidget(QLabel("Boundary Check (ms):"))
        self.cutoff_boundary_spin = QSpinBox()
        self.cutoff_boundary_spin.setRange(30, 100)
        self.cutoff_boundary_spin.setValue(50)
        self.cutoff_boundary_spin.setMinimumHeight(32)
        self.cutoff_boundary_spin.setToolTip("Size of boundary region to analyze")
        cutoff_layout.addWidget(self.cutoff_boundary_spin)
        
        cutoff_layout.addWidget(QLabel("Silence Threshold:"))
        self.cutoff_silence_threshold_spin = QDoubleSpinBox()
        self.cutoff_silence_threshold_spin.setRange(0.1, 0.9)
        self.cutoff_silence_threshold_spin.setValue(0.35)
        self.cutoff_silence_threshold_spin.setSingleStep(0.05)
        self.cutoff_silence_threshold_spin.setMinimumHeight(32)
        self.cutoff_silence_threshold_spin.setToolTip("Energy threshold for cutoff detection")
        cutoff_layout.addWidget(self.cutoff_silence_threshold_spin)
        
        cutoff_layout.addWidget(QLabel("Cutoff Action:"))
        self.cutoff_action_combo = QComboBox()
        self.cutoff_action_combo.addItems(['skip', 'extend', 'include'])
        self.cutoff_action_combo.setCurrentText('skip')
        self.cutoff_action_combo.setMinimumHeight(32)
        self.cutoff_action_combo.setToolTip("What to do when cutoff detected")
        cutoff_layout.addWidget(self.cutoff_action_combo)
        
        cutoff_group.setLayout(cutoff_layout)
        layout.addWidget(cutoff_group)
        
        # Pause detection
        pause_group = QGroupBox("Pause Detection")
        pause_layout = QVBoxLayout()
        pause_layout.setSpacing(8)
        
        self.hybrid_pause_check = QCheckBox("Hybrid Pause Detection")
        self.hybrid_pause_check.setChecked(True)
        self.hybrid_pause_check.setMinimumHeight(28)
        self.hybrid_pause_check.setToolTip("Use Whisper + YAMNet for better accuracy")
        pause_layout.addWidget(self.hybrid_pause_check)
        
        pause_layout.addWidget(QLabel("Speech Threshold:"))
        self.pause_speech_threshold_spin = QDoubleSpinBox()
        self.pause_speech_threshold_spin.setRange(0.0, 0.5)
        self.pause_speech_threshold_spin.setValue(0.2)
        self.pause_speech_threshold_spin.setMinimumHeight(32)
        pause_layout.addWidget(self.pause_speech_threshold_spin)
        
        pause_group.setLayout(pause_layout)
        layout.addWidget(pause_group)
        
        # Overlength action
        overlength_group = QGroupBox("Overlength Handling")
        overlength_layout = QVBoxLayout()
        overlength_layout.setSpacing(8)
        
        overlength_layout.addWidget(QLabel("Action for segments > max_length:"))
        self.overlength_action_combo = QComboBox()
        self.overlength_action_combo.addItems(['rename', 'skip', 'split'])
        self.overlength_action_combo.setCurrentText('rename')
        self.overlength_action_combo.setMinimumHeight(32)
        self.overlength_action_combo.setToolTip("How to handle segments longer than max_length")
        overlength_layout.addWidget(self.overlength_action_combo)
        
        overlength_group.setLayout(overlength_layout)
        layout.addWidget(overlength_group)
        
        layout.addStretch()
        
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        return tab
    
    def create_output_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(15)
        
        # Output settings
        out_group = QGroupBox("Output")
        out_layout = QVBoxLayout()
        out_layout.setSpacing(8)
        
        out_layout.addWidget(QLabel("Output Directory:"))
        output_layout = QHBoxLayout()
        self.output_input = QLineEdit("output")
        self.output_input.setMinimumHeight(32)
        self.output_input.setToolTip("Where segments will be saved")
        output_layout.addWidget(self.output_input)
        browse_btn = QPushButton("Browse")
        browse_btn.setMinimumHeight(32)
        browse_btn.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(browse_btn)
        out_layout.addLayout(output_layout)
        
        out_layout.addWidget(QLabel("Speaker ID:"))
        self.speaker_input = QLineEdit()
        self.speaker_input.setMinimumHeight(32)
        self.speaker_input.setToolTip("Optional: override filename prefix")
        out_layout.addWidget(self.speaker_input)
        
        out_layout.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(['wav', 'mp3'])
        self.format_combo.setMinimumHeight(32)
        self.format_combo.setToolTip("WAV: lossless, MP3: compressed")
        out_layout.addWidget(self.format_combo)
        
        out_group.setLayout(out_layout)
        layout.addWidget(out_group)
        
        # VAD settings
        vad_group = QGroupBox("Silence Trimming")
        vad_layout = QVBoxLayout()
        vad_layout.setSpacing(8)
        
        self.trim_silence_check = QCheckBox("Enable VAD Trimming")
        self.trim_silence_check.setChecked(True)
        self.trim_silence_check.setMinimumHeight(28)
        vad_layout.addWidget(self.trim_silence_check)
        
        vad_layout.addWidget(QLabel("Threshold (dB):"))
        self.trim_threshold_spin = QDoubleSpinBox()
        self.trim_threshold_spin.setRange(-60.0, -20.0)
        self.trim_threshold_spin.setValue(-40.0)
        self.trim_threshold_spin.setSingleStep(1.0)
        self.trim_threshold_spin.setMinimumHeight(32)
        vad_layout.addWidget(self.trim_threshold_spin)
        
        vad_layout.addWidget(QLabel("Chunk Size (ms):"))
        self.trim_chunk_spin = QSpinBox()
        self.trim_chunk_spin.setRange(1, 100)
        self.trim_chunk_spin.setValue(10)
        self.trim_chunk_spin.setMinimumHeight(32)
        self.trim_chunk_spin.setToolTip("Size of audio chunks for analysis")
        vad_layout.addWidget(self.trim_chunk_spin)
        
        vad_layout.addWidget(QLabel("Padding (ms):"))
        self.trim_pad_spin = QSpinBox()
        self.trim_pad_spin.setRange(0, 500)
        self.trim_pad_spin.setValue(150)
        self.trim_pad_spin.setMinimumHeight(32)
        vad_layout.addWidget(self.trim_pad_spin)
        
        vad_group.setLayout(vad_layout)
        layout.addWidget(vad_group)
        
        layout.addStretch()
        
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        return tab
    
    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Files group
        files_group = QGroupBox("Audio Files")
        files_layout = QVBoxLayout()
        
        self.file_list = QListWidget()
        self.file_list.setAcceptDrops(True)
        self.file_list.dragEnterEvent = self.dragEnterEvent
        self.file_list.dragMoveEvent = self.dragEnterEvent
        self.file_list.dropEvent = self.dropEvent
        files_layout.addWidget(self.file_list)
        
        file_buttons = QHBoxLayout()
        
        add_files_btn = QPushButton("📁 Add Files")
        add_files_btn.clicked.connect(self.add_files)
        file_buttons.addWidget(add_files_btn)
        
        add_folder_btn = QPushButton("📂 Add Folder")
        add_folder_btn.clicked.connect(self.add_folder)
        file_buttons.addWidget(add_folder_btn)
        
        clear_btn = QPushButton("🗑 Clear")
        clear_btn.clicked.connect(self.clear_files)
        clear_btn.setStyleSheet("background: #e74c3c; color: white;")
        file_buttons.addWidget(clear_btn)
        
        files_layout.addLayout(file_buttons)
        files_group.setLayout(files_layout)
        layout.addWidget(files_group, 1)
        
        # Log group
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier New", 9))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555555;
            }
        """)
        log_layout.addWidget(self.log_text)
        
        # Log level selector
        log_controls = QHBoxLayout()
        log_controls.addWidget(QLabel("Log Level:"))
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(['INFO', 'DEBUG', 'WARNING', 'ERROR'])
        self.log_level_combo.setCurrentText('INFO')
        log_controls.addWidget(self.log_level_combo)
        log_controls.addStretch()
        
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.log_text.clear)
        log_controls.addWidget(clear_log_btn)
        
        log_layout.addLayout(log_controls)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group, 2)
        
        return panel
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if os.path.isfile(file_path):
                mime_type, _ = mimetypes.guess_type(file_path)
                if mime_type and mime_type.startswith('audio'):
                    self.add_file(file_path)
            elif os.path.isdir(file_path):
                self.add_folder_files(file_path)
    
    def add_file(self, file_path: str):
        if file_path not in self.selected_files:
            self.selected_files.append(file_path)
        
            item = QListWidgetItem(Path(file_path).name)
        
            # Fast duration check using tinytag (no fallback)
            try:
                from tinytag import TinyTag
                tag = TinyTag.get(file_path)
                if tag.duration:
                    item.setText(f"{Path(file_path).name} ({tag.duration:.1f}s)")
            except Exception:
                # If reading fails, just show filename without duration
                pass
        
        self.file_list.addItem(item)
    
    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Audio Files",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.m4a *.ogg);;All Files (*)"
        )
        for file in files:
            self.add_file(file)
    
    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.add_folder_files(folder)
    
    def add_folder_files(self, folder: str):
        audio_exts = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        for file in Path(folder).rglob('*'):
            if file.suffix.lower() in audio_exts:
                self.add_file(str(file))
    
    def clear_files(self):
        self.selected_files.clear()
        self.file_list.clear()
    
    def browse_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.output_input.setText(folder)
    
    def append_log(self, message: str, level: str = "info"):
        """Add colored log message"""
        colors = {
            "info": "#ffffff",
            "success": "#28a745",
            "warning": "#ffc107",
            "error": "#dc3545",
            "debug": "#6c757d"
        }
        
        color = colors.get(level, "#ffffff")
        formatted = f'<span style="color: {color};">{message}</span>'
        self.log_text.append(formatted)
        
        # Auto-scroll
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def start_processing(self):
        if not self.selected_files:
            QMessageBox.warning(self, "No Files", "Please add audio files first")
            return
        
        # Validate settings
        if not self.validate_settings():
            return
        
        settings = self.get_settings()
        
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_bar.showMessage("Processing...")
        
        # Log start
        self.append_log(f"<h3>Starting processing of {len(self.selected_files)} files</h3>", "info")
        self.append_log(f"<b>Output directory:</b> {settings['output_dir']}", "info")
        self.append_log(f"<b>Model:</b> {settings['model']} | <b>Language:</b> {settings['language']}", "info")
        
        self.processing_thread = ProcessingThread(settings, self.selected_files)
        self.processing_thread.progress.connect(self.update_progress)
        self.processing_thread.status.connect(self.status_bar.showMessage)
        self.processing_thread.log.connect(self.append_log)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.error.connect(self.processing_error)
        self.processing_thread.start()
    
    def stop_processing(self):
        if self.processing_thread:
            self.processing_thread.stop()
            self.stop_button.setEnabled(False)
            self.status_bar.showMessage("Stopping...")
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def processing_finished(self, result):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_bar.showMessage("Processing complete", 5000)
        
        QMessageBox.information(
            self,
            "Complete",
            f"Processing finished!\n\n"
            f"✓ Success: {result['success']}/{len(self.selected_files)} files\n"
            f"📊 Total segments: {result['total_segments']}\n\n"
            f"Output saved to:\n{self.output_input.text()}"
        )
    
    def processing_error(self, error):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_bar.showMessage("Error occurred", 5000)
        self.append_log(f"<b>Error:</b> {error}", "error")
        QMessageBox.critical(self, "Error", f"Processing failed:\n{error}")
    
    def get_settings(self):
        return {
            'model': self.model_combo.currentText(),
            'language': self.language_combo.currentText(),
            'speaker_id': self.speaker_input.text() or None,
            'output_dir': self.output_input.text(),
            'min_segment_length': self.min_length_spin.value(),
            'max_segment_length': self.max_length_spin.value(),
            'pause_threshold': self.pause_threshold_spin.value(),
            'export_format': self.format_combo.currentText(),
            'min_confidence': self.min_confidence_spin.value(),
            'filter_noise': self.filter_noise_check.isChecked(),
            'noise_silence_threshold': self.noise_silence_threshold_spin.value(),
            'filter_music': self.filter_music_check.isChecked(),
            'music_threshold': self.music_threshold_spin.value(),
            'mixed_threshold': self.mixed_threshold_spin.value(),
            'word_start_buffer_ms': self.word_start_buffer_spin.value(),
            'word_end_buffer_ms': self.word_end_buffer_spin.value(),
            'check_cutoff': self.check_cutoff_check.isChecked(),
            'cutoff_boundary_ms': self.cutoff_boundary_spin.value(),
            'cutoff_silence_threshold': self.cutoff_silence_threshold_spin.value(),
            'cutoff_action': self.cutoff_action_combo.currentText(),
            'hybrid_pause_detection': self.hybrid_pause_check.isChecked(),
            'pause_speech_threshold': self.pause_speech_threshold_spin.value(),
            'overlength_action': self.overlength_action_combo.currentText(),
            'trim_threshold_db': self.trim_threshold_spin.value(),
            'trim_chunk_ms': self.trim_chunk_spin.value(),
            'trim_pad_ms': self.trim_pad_spin.value(),
            'log_level': self.log_level_combo.currentText()
        }
    
    def validate_settings(self):
        """Validate settings and show warnings"""
        settings = self.get_settings()
        warnings = []
        
        if settings['min_segment_length'] >= settings['max_segment_length']:
            warnings.append("Min length should be less than max length")
        
        if settings['word_start_buffer_ms'] + settings['word_end_buffer_ms'] > 2000:
            warnings.append("Total buffer size > 2s may cause overlapping segments")
        
        if settings['filter_music'] and settings['music_threshold'] < settings['mixed_threshold']:
            warnings.append("Music threshold should be >= mixed threshold")
        
        if warnings:
            msg = "Settings warnings:\n\n" + "\n".join(f"⚠ {w}" for w in warnings)
            msg += "\n\nContinue anyway?"
            reply = QMessageBox.question(self, "Settings Validation", msg, QMessageBox.Yes | QMessageBox.No)
            return reply == QMessageBox.Yes
        
        return True
    
    def reset_settings(self):
        """Reset all settings to defaults"""
        reply = QMessageBox.question(self, "Reset Settings", "Reset all settings to defaults?", QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        
        # Basic tab
        self.model_combo.setCurrentText('large')
        self.language_combo.setCurrentText('cs')
        self.min_length_spin.setValue(1.0)
        self.max_length_spin.setValue(15.0)
        self.pause_threshold_spin.setValue(0.5)
        
        # Quality tab
        self.min_confidence_spin.setValue(0.5)
        self.filter_noise_check.setChecked(True)
        self.noise_silence_threshold_spin.setValue(0.3)
        self.filter_music_check.setChecked(True)
        self.music_threshold_spin.setValue(0.3)
        self.mixed_threshold_spin.setValue(0.2)
        
        # Advanced tab
        self.word_start_buffer_spin.setValue(750)
        self.word_end_buffer_spin.setValue(300)
        self.check_cutoff_check.setChecked(False)
        self.cutoff_boundary_spin.setValue(50)
        self.cutoff_silence_threshold_spin.setValue(0.35)
        self.cutoff_action_combo.setCurrentText('skip')
        self.hybrid_pause_check.setChecked(True)
        self.pause_speech_threshold_spin.setValue(0.2)
        self.overlength_action_combo.setCurrentText('rename')
        
        # Output tab
        self.output_input.setText('output')
        self.speaker_input.clear()
        self.format_combo.setCurrentText('wav')
        self.trim_silence_check.setChecked(True)
        self.trim_threshold_spin.setValue(-40.0)
        self.trim_chunk_spin.setValue(10)
        self.trim_pad_spin.setValue(150)
        
        self.append_log("Settings reset to defaults", "info")
    
    def load_settings(self):
        """Load settings from default location"""
        settings_file = Path.home() / '.speech_splitter_v3_settings.json'
        if settings_file.exists():
            try:
                with open(settings_file) as f:
                    settings = json.load(f)
                self.apply_settings(settings)
                self.append_log("Settings loaded", "info")
            except Exception as e:
                self.append_log(f"Failed to load settings: {e}", "warning")
    
    def load_settings_dialog(self):
        """Load settings from user-selected file"""
        file, _ = QFileDialog.getOpenFileName(self, "Load Settings", "", "JSON Files (*.json)")
        if file:
            try:
                with open(file) as f:
                    settings = json.load(f)
                self.apply_settings(settings)
                self.append_log(f"Settings loaded from {file}", "info")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load settings:\n{e}")
    
    def apply_settings(self, settings: dict):
        """Apply settings dict to UI"""
        self.model_combo.setCurrentText(settings.get('model', 'large'))
        self.language_combo.setCurrentText(settings.get('language', 'cs'))
        self.output_input.setText(settings.get('output_dir', 'output'))
        self.speaker_input.setText(settings.get('speaker_id', ''))
        self.min_length_spin.setValue(settings.get('min_segment_length', 1.0))
        self.max_length_spin.setValue(settings.get('max_segment_length', 15.0))
        self.pause_threshold_spin.setValue(settings.get('pause_threshold', 0.5))
        self.format_combo.setCurrentText(settings.get('export_format', 'wav'))
        self.min_confidence_spin.setValue(settings.get('min_confidence', 0.5))
        self.filter_noise_check.setChecked(settings.get('filter_noise', True))
        self.noise_silence_threshold_spin.setValue(settings.get('noise_silence_threshold', 0.3))
        self.filter_music_check.setChecked(settings.get('filter_music', True))
        self.music_threshold_spin.setValue(settings.get('music_threshold', 0.3))
        self.mixed_threshold_spin.setValue(settings.get('mixed_threshold', 0.2))
        self.word_start_buffer_spin.setValue(settings.get('word_start_buffer_ms', 750))
        self.word_end_buffer_spin.setValue(settings.get('word_end_buffer_ms', 300))
        self.check_cutoff_check.setChecked(settings.get('check_cutoff', False))
        self.cutoff_boundary_spin.setValue(settings.get('cutoff_boundary_ms', 50))
        self.cutoff_silence_threshold_spin.setValue(settings.get('cutoff_silence_threshold', 0.35))
        self.cutoff_action_combo.setCurrentText(settings.get('cutoff_action', 'skip'))
        self.hybrid_pause_check.setChecked(settings.get('hybrid_pause_detection', True))
        self.pause_speech_threshold_spin.setValue(settings.get('pause_speech_threshold', 0.2))
        self.overlength_action_combo.setCurrentText(settings.get('overlength_action', 'rename'))
        self.trim_silence_check.setChecked(settings.get('trim_silence', True))
        self.trim_threshold_spin.setValue(settings.get('trim_threshold_db', -40.0))
        self.trim_chunk_spin.setValue(settings.get('trim_chunk_ms', 10))
        self.trim_pad_spin.setValue(settings.get('trim_pad_ms', 150))
    
    def save_settings(self):
        """Save settings to default location"""
        settings = self.get_settings()
        settings_file = Path.home() / '.speech_splitter_v3_settings.json'
        
        try:
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            QMessageBox.information(self, "Saved", f"Settings saved to:\n{settings_file}")
            self.append_log("Settings saved", "success")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings:\n{e}")


def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Load stylesheet
    stylesheet = """
    QMainWindow {
        background-color: #f5f5f5;
    }
    QGroupBox {
        font-weight: bold;
        border: 1px solid #d0d0d0;
        border-radius: 5px;
        margin-top: 10px;
        padding-top: 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
    }
    QPushButton {
        padding: 8px 15px;
        border-radius: 5px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #e0e0e0;
    }
    QProgressBar {
        height: 20px;
        border-radius: 5px;
        text-align: center;
    }
    QProgressBar::chunk {
        background-color: #3498db;
    }
    """
    app.setStyleSheet(stylesheet)
    
    window = SpeechSplitterGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
