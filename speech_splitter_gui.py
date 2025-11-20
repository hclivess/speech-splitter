"""
Speech Splitter v4 - GUI
Modern graphical interface for sentence-based TTS data generation
"""

import sys
import os
from pathlib import Path
from typing import List
import json

# Fix for torch._pytree compatibility issue
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
    QCheckBox, QTabWidget, QListWidgetItem, QStatusBar, QMenuBar,
    QGridLayout, QSplitter, QScrollArea
)
from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtGui import QFont, QDragEnterEvent, QDropEvent, QColor, QTextCursor


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
            # Import the modular speech_splitter
            sys.path.insert(0, str(Path(__file__).parent))
            from speech_splitter import SpeechSplitter
            
            self.status.emit("Initializing models...")
            self.log.emit(f"Loading Whisper model: <b>{self.settings['whisper_model']}</b>", "info")
            
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
        self.check_cutoff_check = None
        self.cutoff_sensitivity_spin = None
        self.cutoff_check_mode_combo = None
        self.processing_thread = None
        self.selected_files = []
        self.init_ui()
        self.load_settings()
    
    def init_ui(self):
        self.setWindowTitle("Speech Splitter v4 - Sentence-Based TTS Generator")
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
        layout = QVBoxLayout(header)
        
        title = QLabel("Speech Splitter v4")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setStyleSheet("color: #2c3e50; margin: 10px;")
        layout.addWidget(title)
        
        subtitle = QLabel("Sentence-Based TTS Data Generator • Powered by WhisperX")
        subtitle.setStyleSheet("color: #7f8c8d; font-size: 12px; margin-left: 10px; margin-bottom: 10px;")
        layout.addWidget(subtitle)
        
        return header
    
    def create_settings_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        container = QWidget()
        layout = QVBoxLayout(container)
        
        # Tabs for settings
        tabs = QTabWidget()
        
        tabs.addTab(self.create_basic_tab(), "Basic")
        tabs.addTab(self.create_quality_tab(), "Quality")
        tabs.addTab(self.create_advanced_tab(), "Advanced")
        tabs.addTab(self.create_output_tab(), "Output")
        
        layout.addWidget(tabs)
        
        # Reset button
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_settings)
        layout.addWidget(reset_btn)
        
        layout.addStretch()
        scroll.setWidget(container)
        return scroll
    
    def create_basic_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Model settings
        model_group = QGroupBox("Model Settings")
        model_layout = QGridLayout()
        
        model_layout.addWidget(QLabel("Whisper Model:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3', 'turbo'])
        self.model_combo.setCurrentText('large-v2')
        model_layout.addWidget(self.model_combo, 0, 1)
        
        model_layout.addWidget(QLabel("Language:"), 1, 0)
        self.language_combo = QComboBox()
        self.language_combo.addItems(['en', 'cs', 'de', 'es', 'fr', 'it', 'ja', 'ko', 'pl', 'pt', 'ru', 'zh'])
        self.language_combo.setCurrentText('cs')
        self.language_combo.setEditable(True)
        model_layout.addWidget(self.language_combo, 1, 1)
        
        model_layout.addWidget(QLabel("Batch Size:"), 2, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(6)
        self.batch_size_spin.setToolTip("Higher = faster but more VRAM. Lower if you get OOM errors.")
        model_layout.addWidget(self.batch_size_spin, 2, 1)
        
        model_layout.addWidget(QLabel("Device:"), 3, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(['auto', 'cuda', 'cpu'])
        self.device_combo.setCurrentText('auto')
        self.device_combo.setToolTip("auto = use GPU if available, cuda = force GPU, cpu = force CPU")
        model_layout.addWidget(self.device_combo, 3, 1)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Segmentation settings (updated for sentence-based)
        seg_group = QGroupBox("Segmentation Settings")
        seg_layout = QGridLayout()
        
        seg_layout.addWidget(QLabel("Min Segment Length (s):"), 0, 0)
        self.min_length_spin = QDoubleSpinBox()
        self.min_length_spin.setRange(0.1, 30.0)
        self.min_length_spin.setValue(1.0)
        self.min_length_spin.setSingleStep(0.5)
        seg_layout.addWidget(self.min_length_spin, 0, 1)
        
        seg_layout.addWidget(QLabel("Max Segment Length (s):"), 1, 0)
        self.max_length_spin = QDoubleSpinBox()
        self.max_length_spin.setRange(1.0, 60.0)
        self.max_length_spin.setValue(15.0)
        self.max_length_spin.setSingleStep(1.0)
        seg_layout.addWidget(self.max_length_spin, 1, 1)
        
        info_label = QLabel("ℹ️ Segments are split by sentence punctuation (., !, ?)")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #3498db; font-style: italic; padding: 5px;")
        seg_layout.addWidget(info_label, 2, 0, 1, 2)
        
        seg_group.setLayout(seg_layout)
        layout.addWidget(seg_group)
        
        layout.addStretch()
        return widget
    
    def create_quality_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Confidence filter
        conf_group = QGroupBox("Transcription Quality")
        conf_layout = QGridLayout()
        
        conf_layout.addWidget(QLabel("Min Confidence:"), 0, 0)
        self.min_confidence_spin = QDoubleSpinBox()
        self.min_confidence_spin.setRange(0.0, 1.0)
        self.min_confidence_spin.setValue(0.5)
        self.min_confidence_spin.setSingleStep(0.05)
        conf_layout.addWidget(self.min_confidence_spin, 0, 1)
        
        conf_group.setLayout(conf_layout)
        layout.addWidget(conf_group)
        
        # Audio filtering
        audio_group = QGroupBox("Audio Filtering (Requires YAMNet)")
        audio_layout = QVBoxLayout()
        
        # Music filter
        music_layout = QGridLayout()
        self.filter_music_check = QCheckBox("Filter Music")
        self.filter_music_check.setChecked(True)
        music_layout.addWidget(self.filter_music_check, 0, 0, 1, 2)
        
        music_layout.addWidget(QLabel("  Music Threshold:"), 1, 0)
        self.music_threshold_spin = QDoubleSpinBox()
        self.music_threshold_spin.setRange(0.0, 1.0)
        self.music_threshold_spin.setValue(0.3)
        self.music_threshold_spin.setSingleStep(0.05)
        music_layout.addWidget(self.music_threshold_spin, 1, 1)
        
        music_layout.addWidget(QLabel("  Mixed Threshold:"), 2, 0)
        self.mixed_threshold_spin = QDoubleSpinBox()
        self.mixed_threshold_spin.setRange(0.0, 1.0)
        self.mixed_threshold_spin.setValue(0.2)
        self.mixed_threshold_spin.setSingleStep(0.05)
        music_layout.addWidget(self.mixed_threshold_spin, 2, 1)
        
        audio_layout.addLayout(music_layout)
        
        # Noise/silence filter
        noise_layout = QGridLayout()
        self.filter_noise_check = QCheckBox("Filter Noise/Silence")
        self.filter_noise_check.setChecked(False)
        noise_layout.addWidget(self.filter_noise_check, 0, 0, 1, 2)
        
        noise_layout.addWidget(QLabel("  Silence Threshold:"), 1, 0)
        self.noise_silence_threshold_spin = QDoubleSpinBox()
        self.noise_silence_threshold_spin.setRange(0.0, 1.0)
        self.noise_silence_threshold_spin.setValue(0.3)
        self.noise_silence_threshold_spin.setSingleStep(0.05)
        noise_layout.addWidget(self.noise_silence_threshold_spin, 1, 1)
        
        audio_layout.addLayout(noise_layout)
        audio_group.setLayout(audio_layout)
        layout.addWidget(audio_group)
        
        layout.addStretch()
        return widget

    def create_advanced_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Word buffers
        buffer_group = QGroupBox("Word Boundaries")
        buffer_layout = QGridLayout()

        buffer_layout.addWidget(QLabel("Start Buffer (ms):"), 0, 0)
        self.word_start_buffer_spin = QSpinBox()
        self.word_start_buffer_spin.setRange(0, 2000)
        self.word_start_buffer_spin.setValue(250)
        self.word_start_buffer_spin.setSingleStep(50)
        buffer_layout.addWidget(self.word_start_buffer_spin, 0, 1)

        buffer_layout.addWidget(QLabel("End Buffer (ms):"), 1, 0)
        self.word_end_buffer_spin = QSpinBox()
        self.word_end_buffer_spin.setRange(0, 2000)
        self.word_end_buffer_spin.setValue(250)
        self.word_end_buffer_spin.setSingleStep(50)
        buffer_layout.addWidget(self.word_end_buffer_spin, 1, 1)

        buffer_group.setLayout(buffer_layout)
        layout.addWidget(buffer_group)

        # Cutoff detection - UPDATED
        cutoff_group = QGroupBox("Cutoff Detection")
        cutoff_layout = QGridLayout()

        self.check_cutoff_check = QCheckBox("Enable Cutoff Detection")
        self.check_cutoff_check.setChecked(False)
        #self.cutoff_check_mode_combo.setCurrentText('both')
        #self.cutoff_sensitivity_spin.setValue(0.35)
        cutoff_layout.addWidget(self.check_cutoff_check, 0, 0, 1, 2)

        # NEW: Check mode selector
        cutoff_layout.addWidget(QLabel("  Check:"), 1, 0)
        self.cutoff_check_mode_combo = QComboBox()
        self.cutoff_check_mode_combo.addItems(['both', 'start', 'end', 'none'])
        self.cutoff_check_mode_combo.setCurrentText('both')
        self.cutoff_check_mode_combo.setToolTip("Which boundaries to check for cutoffs")
        cutoff_layout.addWidget(self.cutoff_check_mode_combo, 1, 1)

        # NEW: Sensitivity slider
        cutoff_layout.addWidget(QLabel("  Sensitivity:"), 2, 0)
        self.cutoff_sensitivity_spin = QDoubleSpinBox()
        self.cutoff_sensitivity_spin.setRange(0.0, 1.0)
        self.cutoff_sensitivity_spin.setValue(0.35)
        self.cutoff_sensitivity_spin.setSingleStep(0.05)
        self.cutoff_sensitivity_spin.setToolTip(
            "Lower = more sensitive (detects more cutoffs)\nHigher = less sensitive")
        cutoff_layout.addWidget(self.cutoff_sensitivity_spin, 2, 1)

        cutoff_layout.addWidget(QLabel("  Action:"), 3, 0)
        self.cutoff_action_combo = QComboBox()
        self.cutoff_action_combo.addItems(['include', 'skip', 'rename'])
        self.cutoff_action_combo.setCurrentText('skip')
        self.cutoff_action_combo.setToolTip(
            "include = export normally\nskip = don't export\nrename = add _CUTOFF suffix")
        cutoff_layout.addWidget(self.cutoff_action_combo, 3, 1)

        cutoff_group.setLayout(cutoff_layout)
        layout.addWidget(cutoff_group)

        # Overlength action
        overlength_group = QGroupBox("Overlength Segments")
        overlength_layout = QGridLayout()

        overlength_layout.addWidget(QLabel("Action:"), 0, 0)
        self.overlength_action_combo = QComboBox()
        self.overlength_action_combo.addItems(['rename', 'skip', 'include'])
        self.overlength_action_combo.setCurrentText('rename')
        overlength_layout.addWidget(self.overlength_action_combo, 0, 1)

        overlength_group.setLayout(overlength_layout)
        layout.addWidget(overlength_group)

        layout.addStretch()
        return widget
    
    def create_output_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Output settings
        output_group = QGroupBox("Output Settings")
        output_layout = QGridLayout()
        
        output_layout.addWidget(QLabel("Output Directory:"), 0, 0)
        self.output_input = QLineEdit('output')
        output_layout.addWidget(self.output_input, 0, 1)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(browse_btn, 0, 2)
        
        output_layout.addWidget(QLabel("Speaker ID (optional):"), 1, 0)
        self.speaker_input = QLineEdit()
        self.speaker_input.setPlaceholderText("Leave empty to use filename")
        output_layout.addWidget(self.speaker_input, 1, 1, 1, 2)
        
        output_layout.addWidget(QLabel("Format:"), 2, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(['wav', 'mp3'])
        output_layout.addWidget(self.format_combo, 2, 1)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Silence trimming
        trim_group = QGroupBox("Silence Trimming")
        trim_layout = QGridLayout()
        
        self.trim_silence_check = QCheckBox("Enable Silence Trimming")
        self.trim_silence_check.setChecked(True)
        trim_layout.addWidget(self.trim_silence_check, 0, 0, 1, 2)
        
        trim_layout.addWidget(QLabel("  Threshold (dB):"), 1, 0)
        self.trim_threshold_spin = QDoubleSpinBox()
        self.trim_threshold_spin.setRange(-60.0, -20.0)
        self.trim_threshold_spin.setValue(-45.0)
        self.trim_threshold_spin.setSingleStep(1.0)
        trim_layout.addWidget(self.trim_threshold_spin, 1, 1)
        
        trim_layout.addWidget(QLabel("  Chunk Size (ms):"), 2, 0)
        self.trim_chunk_spin = QSpinBox()
        self.trim_chunk_spin.setRange(5, 50)
        self.trim_chunk_spin.setValue(10)
        trim_layout.addWidget(self.trim_chunk_spin, 2, 1)
        
        trim_layout.addWidget(QLabel("  Padding (ms):"), 3, 0)
        self.trim_pad_spin = QSpinBox()
        self.trim_pad_spin.setRange(0, 500)
        self.trim_pad_spin.setValue(50)
        trim_layout.addWidget(self.trim_pad_spin, 3, 1)
        
        trim_group.setLayout(trim_layout)
        layout.addWidget(trim_group)
        
        layout.addStretch()
        return widget
    
    def create_right_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Files section
        files_label = QLabel("<b>Audio Files</b>")
        layout.addWidget(files_label)
        
        self.file_list = QListWidget()
        self.file_list.setAcceptDrops(True)
        self.file_list.setDragEnabled(True)
        self.file_list.dragEnterEvent = self.drag_enter_event
        self.file_list.dropEvent = self.drop_event
        layout.addWidget(self.file_list)
        
        # File buttons
        file_buttons = QHBoxLayout()
        add_file_btn = QPushButton("Add Files")
        add_file_btn.clicked.connect(self.add_files)
        file_buttons.addWidget(add_file_btn)
        
        add_folder_btn = QPushButton("Add Folder")
        add_folder_btn.clicked.connect(self.add_folder)
        file_buttons.addWidget(add_folder_btn)
        
        clear_btn = QPushButton("Clear List")
        clear_btn.clicked.connect(self.clear_files)
        file_buttons.addWidget(clear_btn)
        
        layout.addLayout(file_buttons)
        
        # Log section
        log_label = QLabel("<b>Processing Log</b>")
        layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        return widget
    
    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.output_input.setText(folder)
    
    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Files", "",
            "Audio Files (*.wav *.mp3 *.flac *.m4a *.ogg);;All Files (*)"
        )
        for file in files:
            if file not in self.selected_files:
                self.selected_files.append(file)
                self.file_list.addItem(Path(file).name)

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with Audio Files")
        if folder:
            folder_path = Path(folder)
            for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']:
                for file in folder_path.rglob(ext):  # <-- Use rglob instead of glob
                    file_str = str(file)
                    if file_str not in self.selected_files:
                        self.selected_files.append(file_str)
                        self.file_list.addItem(file.name)
    
    def clear_files(self):
        self.selected_files.clear()
        self.file_list.clear()
    
    def drag_enter_event(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def drop_event(self, event: QDropEvent):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            path = Path(file_path)
            
            if path.is_file() and path.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
                if str(path) not in self.selected_files:
                    self.selected_files.append(str(path))
                    self.file_list.addItem(path.name)
            elif path.is_dir():
                for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']:
                    for file in path.rglob(ext):
                        file_str = str(file)
                        if file_str not in self.selected_files:
                            self.selected_files.append(file_str)
                            self.file_list.addItem(file.name)
    
    def append_log(self, message: str, level: str = "info"):
        colors = {
            'info': '#2c3e50',
            'success': '#27ae60',
            'warning': '#f39c12',
            'error': '#e74c3c'
        }
        color = colors.get(level, colors['info'])
        self.log_text.append(f'<span style="color: {color};">{message}</span>')
        self.log_text.moveCursor(QTextCursor.End)
    
    def start_processing(self):
        if not self.selected_files:
            QMessageBox.warning(self, "No Files", "Please add audio files to process")
            return
        
        if not self.validate_settings():
            return
        
        settings = self.get_settings()
        
        self.log_text.clear()
        self.append_log("<h3>Starting processing...</h3>", "info")
        
        self.processing_thread = ProcessingThread(settings, self.selected_files)
        self.processing_thread.progress.connect(self.progress_bar.setValue)
        self.processing_thread.status.connect(self.status_bar.showMessage)
        self.processing_thread.log.connect(self.append_log)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.error.connect(self.processing_error)
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        self.processing_thread.start()
    
    def stop_processing(self):
        if self.processing_thread:
            self.processing_thread.stop()
            self.append_log("Stopping processing...", "warning")
    
    def processing_finished(self, results: dict):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_bar.showMessage("Processing complete")
        
        QMessageBox.information(
            self,
            "Complete",
            f"Processing finished!\n\n"
            f"Successful: {results['success']} files\n"
            f"Total segments: {results['total_segments']}"
        )
    
    def processing_error(self, error_msg: str):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_bar.showMessage("Error occurred")
        
        self.append_log(f"<b>Error:</b> {error_msg}", "error")
        QMessageBox.critical(self, "Error", f"Processing failed:\n\n{error_msg}")
    
    def get_settings(self):
        device = self.device_combo.currentText()
        return {
            'whisper_model': self.model_combo.currentText(),
            'language': self.language_combo.currentText(),
            'batch_size': self.batch_size_spin.value(),
            'device': None if device == 'auto' else device,
            'output_dir': self.output_input.text(),
            'speaker_id': self.speaker_input.text() or None,
            'min_segment_length': self.min_length_spin.value(),
            'max_segment_length': self.max_length_spin.value(),
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
            'cutoff_check_mode': self.cutoff_check_mode_combo.currentText(),
            'cutoff_sensitivity': self.cutoff_sensitivity_spin.value(),
            'cutoff_action': self.cutoff_action_combo.currentText(),
            'overlength_action': self.overlength_action_combo.currentText(),
            'trim_threshold_db': self.trim_threshold_spin.value(),
            'trim_chunk_ms': self.trim_chunk_spin.value(),
            'trim_pad_ms': self.trim_pad_spin.value(),
            'log_level': 'INFO'
        }
    
    def validate_settings(self):
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
        reply = QMessageBox.question(self, "Reset Settings", "Reset all settings to defaults?", QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        
        # Basic
        self.model_combo.setCurrentText('large')
        self.language_combo.setCurrentText('cs')
        self.batch_size_spin.setValue(6)
        self.device_combo.setCurrentText('auto')
        self.min_length_spin.setValue(1.0)
        self.max_length_spin.setValue(15.0)
        
        # Quality
        self.min_confidence_spin.setValue(0.5)
        self.filter_noise_check.setChecked(False)
        self.noise_silence_threshold_spin.setValue(0.3)
        self.filter_music_check.setChecked(True)
        self.music_threshold_spin.setValue(0.3)
        self.mixed_threshold_spin.setValue(0.2)
        
        # Advanced
        self.word_start_buffer_spin.setValue(250)
        self.word_end_buffer_spin.setValue(250)
        self.check_cutoff_check.setChecked(False)
        self.cutoff_action_combo.setCurrentText('skip')
        self.overlength_action_combo.setCurrentText('rename')
        
        # Output
        self.output_input.setText('output')
        self.speaker_input.clear()
        self.format_combo.setCurrentText('wav')
        self.trim_silence_check.setChecked(True)
        self.trim_threshold_spin.setValue(-45.0)
        self.trim_chunk_spin.setValue(10)
        self.trim_pad_spin.setValue(50)
        
        self.append_log("Settings reset to defaults", "info")
    
    def load_settings(self):
        settings_file = Path.home() / '.speech_splitter_v4_settings.json'
        if settings_file.exists():
            try:
                with open(settings_file) as f:
                    settings = json.load(f)
                self.apply_settings(settings)
                self.append_log("Settings loaded", "info")
            except Exception as e:
                self.append_log(f"Failed to load settings: {e}", "warning")
    
    def load_settings_dialog(self):
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
        self.model_combo.setCurrentText(settings.get('whisper_model', 'large'))
        self.language_combo.setCurrentText(settings.get('language', 'cs'))
        self.batch_size_spin.setValue(settings.get('batch_size', 6))
        device = settings.get('device', None)
        self.device_combo.setCurrentText('auto' if device is None else device)
        self.output_input.setText(settings.get('output_dir', 'output'))
        self.speaker_input.setText(settings.get('speaker_id', ''))
        self.min_length_spin.setValue(settings.get('min_segment_length', 1.0))
        self.max_length_spin.setValue(settings.get('max_segment_length', 15.0))
        self.format_combo.setCurrentText(settings.get('export_format', 'wav'))
        self.min_confidence_spin.setValue(settings.get('min_confidence', 0.5))
        self.filter_noise_check.setChecked(settings.get('filter_noise', False))
        self.noise_silence_threshold_spin.setValue(settings.get('noise_silence_threshold', 0.3))
        self.filter_music_check.setChecked(settings.get('filter_music', True))
        self.music_threshold_spin.setValue(settings.get('music_threshold', 0.3))
        self.mixed_threshold_spin.setValue(settings.get('mixed_threshold', 0.2))
        self.word_start_buffer_spin.setValue(settings.get('word_start_buffer_ms', 250))
        self.word_end_buffer_spin.setValue(settings.get('word_end_buffer_ms', 250))
        self.check_cutoff_check.setChecked(settings.get('check_cutoff', False))
        self.cutoff_check_mode_combo.setCurrentText(settings.get('cutoff_check_mode', 'both'))
        self.cutoff_sensitivity_spin.setValue(settings.get('cutoff_sensitivity', 0.35))
        self.cutoff_action_combo.setCurrentText(settings.get('cutoff_action', 'skip'))
        self.overlength_action_combo.setCurrentText(settings.get('overlength_action', 'rename'))
        self.trim_silence_check.setChecked(settings.get('trim_silence', True))
        self.trim_threshold_spin.setValue(settings.get('trim_threshold_db', -45.0))
        self.trim_chunk_spin.setValue(settings.get('trim_chunk_ms', 10))
        self.trim_pad_spin.setValue(settings.get('trim_pad_ms', 50))
    
    def save_settings(self):
        settings = self.get_settings()
        settings_file = Path.home() / '.speech_splitter_v4_settings.json'
        
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
