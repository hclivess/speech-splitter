#!/usr/bin/env python3
"""
Speech Splitter GUI - PySide6 Application
Now with YAMNet music filtering
"""

import sys
import os
from pathlib import Path
from typing import List
import json

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QTextEdit, QProgressBar, QGroupBox, QListWidget, QMessageBox,
    QCheckBox, QTabWidget
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont


class ProcessingThread(QThread):
    progress = Signal(int)
    status = Signal(str)
    log = Signal(str)
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
            from speech_splitter_v2 import SpeechSplitter
            
            self.status.emit("Initializing models...")
            self.log.emit(f"Loading Whisper model: {self.settings['model']}")
            if self.settings.get('filter_music', True):
                self.log.emit("Loading YAMNet for music filtering...")
            
            splitter = SpeechSplitter(
                whisper_model=self.settings['model'],
                language=self.settings['language'],
                output_dir=self.settings['output_dir'],
                min_segment_length=self.settings['min_length'],
                max_segment_length=self.settings['max_length'],
                export_format=self.settings['format'],
                pause_threshold=self.settings.get('pause_threshold', 0.5),
                min_confidence=self.settings.get('min_confidence', 0.5),
                filter_noise=self.settings.get('filter_noise', True),
                min_rms=self.settings.get('min_rms', 0.01),
                filter_music=self.settings.get('filter_music', True),
                music_threshold=self.settings.get('music_threshold', 0.3),
                mixed_threshold=self.settings.get('mixed_threshold', 0.2),
                word_start_buffer_ms=self.settings.get('word_start_buffer', 250),
                word_end_buffer_ms=self.settings.get('word_end_buffer', 750)
            )
            
            self.log.emit("✓ Models loaded\n")
            
            total_files = len(self.files)
            results = []
            
            for idx, file_path in enumerate(self.files):
                if not self._is_running:
                    self.log.emit("\n⚠ Processing cancelled")
                    break
                
                self.status.emit(f"Processing {idx + 1}/{total_files}")
                self.log.emit(f"\n{'='*60}")
                self.log.emit(f"File: {Path(file_path).name}")
                self.log.emit(f"{'='*60}\n")
                
                try:
                    speaker_id = self.settings.get('speaker_id') or Path(file_path).stem
                    stats = splitter.process_file(file_path, speaker_id=speaker_id)
                    
                    results.append({
                        'file': file_path,
                        'status': 'success',
                        'segments': stats['total_segments'],
                        'duration': stats['total_duration']
                    })
                    
                    self.log.emit(f"✓ {stats['total_segments']} segments, {stats['total_duration']:.1f}s\n")
                    
                except Exception as e:
                    self.log.emit(f"✗ Error: {str(e)}\n")
                    results.append({'file': file_path, 'status': 'failed', 'error': str(e)})
                
                progress = int((idx + 1) / total_files * 100)
                self.progress.emit(progress)
            
            success = sum(1 for r in results if r['status'] == 'success')
            total_segs = sum(r.get('segments', 0) for r in results)
            
            self.log.emit(f"\n{'='*60}")
            self.log.emit("COMPLETE")
            self.log.emit(f"{'='*60}")
            self.log.emit(f"Success: {success}/{total_files}")
            self.log.emit(f"Segments: {total_segs}")
            self.log.emit(f"Output: {self.settings['output_dir']}")
            self.log.emit(f"{'='*60}\n")
            
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
        self.setWindowTitle("Speech Splitter v2 - TTS Data Generator")
        self.setMinimumSize(1000, 700)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Title
        title = QLabel("Speech Splitter v2")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("Single-Pass Whisper • HuggingFace Compatible • No Audio Resampling")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)
        
        # Main content
        content_layout = QHBoxLayout()
        
        # Left panel - Settings with tabs
        settings_group = QGroupBox("Settings")
        settings_main_layout = QVBoxLayout()
        
        # Create tab widget
        tabs = QTabWidget()
        
        # Tab 1: Basic Settings
        basic_tab = QWidget()
        basic_layout = QVBoxLayout()
        
        basic_layout.addWidget(QLabel("Whisper Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(['tiny', 'base', 'small', 'medium', 'large', 'large-v2'])
        self.model_combo.setCurrentText('large')
        basic_layout.addWidget(self.model_combo)
        
        basic_layout.addWidget(QLabel("Language:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(['cs', 'en', 'de', 'es', 'fr', 'it', 'pl', 'ru'])
        basic_layout.addWidget(self.language_combo)
        
        basic_layout.addWidget(QLabel("Speaker ID (optional):"))
        self.speaker_input = QLineEdit()
        basic_layout.addWidget(self.speaker_input)
        
        basic_layout.addWidget(QLabel("Output Directory:"))
        output_layout = QHBoxLayout()
        self.output_input = QLineEdit("output")
        output_layout.addWidget(self.output_input)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(browse_btn)
        basic_layout.addLayout(output_layout)
        
        basic_layout.addWidget(QLabel("Output Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(['wav', 'mp3'])
        basic_layout.addWidget(self.format_combo)
        
        basic_layout.addStretch()
        basic_tab.setLayout(basic_layout)
        tabs.addTab(basic_tab, "Basic")
        
        # Tab 2: Segment Settings
        segment_tab = QWidget()
        segment_layout = QVBoxLayout()
        
        segment_layout.addWidget(QLabel("Min Segment Length (s):"))
        self.min_length_spin = QDoubleSpinBox()
        self.min_length_spin.setRange(0.5, 10.0)
        self.min_length_spin.setValue(1.0)
        self.min_length_spin.setSingleStep(0.5)
        segment_layout.addWidget(self.min_length_spin)
        
        segment_layout.addWidget(QLabel("Max Segment Length (s):"))
        self.max_length_spin = QDoubleSpinBox()
        self.max_length_spin.setRange(5.0, 30.0)
        self.max_length_spin.setValue(15.0)
        self.max_length_spin.setSingleStep(1.0)
        segment_layout.addWidget(self.max_length_spin)
        
        segment_layout.addWidget(QLabel("Pause Threshold (s):"))
        self.pause_threshold_spin = QDoubleSpinBox()
        self.pause_threshold_spin.setRange(0.1, 2.0)
        self.pause_threshold_spin.setValue(0.5)
        self.pause_threshold_spin.setSingleStep(0.1)
        self.pause_threshold_spin.setToolTip("Duration of pause to split segments (Whisper-based)")
        segment_layout.addWidget(self.pause_threshold_spin)
        
        segment_layout.addWidget(QLabel("Word Start Buffer (ms):"))
        self.word_start_buffer_spin = QSpinBox()
        self.word_start_buffer_spin.setRange(0, 1000)
        self.word_start_buffer_spin.setValue(250)
        self.word_start_buffer_spin.setToolTip("Buffer added at START of segments to capture complete word beginnings")
        segment_layout.addWidget(self.word_start_buffer_spin)
        
        segment_layout.addWidget(QLabel("Word End Buffer (ms):"))
        self.word_end_buffer_spin = QSpinBox()
        self.word_end_buffer_spin.setRange(0, 1000)
        self.word_end_buffer_spin.setValue(750)
        self.word_end_buffer_spin.setToolTip("Buffer added at END of segments to capture complete word endings")
        segment_layout.addWidget(self.word_end_buffer_spin)
        
        segment_layout.addStretch()
        segment_tab.setLayout(segment_layout)
        tabs.addTab(segment_tab, "Segments")
        
        # Tab 3: Quality Filters
        quality_tab = QWidget()
        quality_layout = QVBoxLayout()
        
        quality_layout.addWidget(QLabel("Min Transcription Confidence:"))
        self.min_confidence_spin = QDoubleSpinBox()
        self.min_confidence_spin.setRange(0.0, 1.0)
        self.min_confidence_spin.setValue(0.5)
        self.min_confidence_spin.setSingleStep(0.05)
        quality_layout.addWidget(self.min_confidence_spin)
        
        self.filter_noise_check = QCheckBox("Enable Noise Filtering")
        self.filter_noise_check.setChecked(True)
        quality_layout.addWidget(self.filter_noise_check)
        
        quality_layout.addWidget(QLabel("Min RMS Energy:"))
        self.min_rms_spin = QDoubleSpinBox()
        self.min_rms_spin.setRange(0.001, 0.1)
        self.min_rms_spin.setValue(0.01)
        self.min_rms_spin.setSingleStep(0.005)
        self.min_rms_spin.setDecimals(3)
        quality_layout.addWidget(self.min_rms_spin)
        
        quality_layout.addStretch()
        quality_tab.setLayout(quality_layout)
        tabs.addTab(quality_tab, "Quality")
        
        # Tab 5: Music Filtering (NEW!)
        music_tab = QWidget()
        music_layout = QVBoxLayout()
        
        self.filter_music_check = QCheckBox("Enable YAMNet Music Filtering")
        self.filter_music_check.setChecked(True)
        self.filter_music_check.setToolTip("Use YAMNet to detect and filter out music and mixed music+speech segments")
        music_layout.addWidget(self.filter_music_check)
        
        music_layout.addWidget(QLabel(""))  # Spacer
        
        music_layout.addWidget(QLabel("Music Detection Threshold:"))
        music_threshold_label = QLabel("Higher = stricter filtering (default: 0.3)")
        music_threshold_label.setStyleSheet("color: gray; font-size: 9pt;")
        music_layout.addWidget(music_threshold_label)
        self.music_threshold_spin = QDoubleSpinBox()
        self.music_threshold_spin.setRange(0.1, 0.9)
        self.music_threshold_spin.setValue(0.3)
        self.music_threshold_spin.setSingleStep(0.05)
        self.music_threshold_spin.setDecimals(2)
        music_layout.addWidget(self.music_threshold_spin)
        
        music_layout.addWidget(QLabel(""))  # Spacer
        
        music_layout.addWidget(QLabel("Music+Speech Mix Threshold:"))
        mixed_threshold_label = QLabel("Filters segments with music background (default: 0.2)")
        mixed_threshold_label.setStyleSheet("color: gray; font-size: 9pt;")
        music_layout.addWidget(mixed_threshold_label)
        self.mixed_threshold_spin = QDoubleSpinBox()
        self.mixed_threshold_spin.setRange(0.1, 0.9)
        self.mixed_threshold_spin.setValue(0.2)
        self.mixed_threshold_spin.setSingleStep(0.05)
        self.mixed_threshold_spin.setDecimals(2)
        music_layout.addWidget(self.mixed_threshold_spin)
        
        music_layout.addWidget(QLabel(""))  # Spacer
        
        info_label = QLabel("ℹ YAMNet uses a pre-trained neural network to classify audio.\n"
                          "It will filter out:\n"
                          "• Pure music segments\n"
                          "• Speech with music background\n"
                          "• Mixed music and speech content")
        info_label.setStyleSheet("color: #555; font-size: 9pt; padding: 10px; background: #f0f0f0; border-radius: 5px;")
        info_label.setWordWrap(True)
        music_layout.addWidget(info_label)
        
        music_layout.addStretch()
        music_tab.setLayout(music_layout)
        tabs.addTab(music_tab, "Music Filter")
        
        settings_main_layout.addWidget(tabs)
        settings_group.setLayout(settings_main_layout)
        
        content_layout.addWidget(settings_group, 1)
        
        # Right panel - Files and Log
        right_panel = QVBoxLayout()
        
        # File selection
        files_group = QGroupBox("Audio Files")
        files_layout = QVBoxLayout()
        
        self.file_list = QListWidget()
        files_layout.addWidget(self.file_list)
        
        file_buttons = QHBoxLayout()
        add_files_btn = QPushButton("Add Files")
        add_files_btn.clicked.connect(self.add_files)
        file_buttons.addWidget(add_files_btn)
        
        add_folder_btn = QPushButton("Add Folder")
        add_folder_btn.clicked.connect(self.add_folder)
        file_buttons.addWidget(add_folder_btn)
        
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_files)
        file_buttons.addWidget(clear_btn)
        
        files_layout.addLayout(file_buttons)
        files_group.setLayout(files_layout)
        right_panel.addWidget(files_group, 1)
        
        # Log
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        right_panel.addWidget(log_group, 2)
        
        content_layout.addLayout(right_panel, 2)
        layout.addLayout(content_layout)
        
        # Bottom panel
        bottom_layout = QVBoxLayout()
        
        # Status
        self.status_label = QLabel("Ready")
        bottom_layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        bottom_layout.addWidget(self.progress_bar)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        self.start_button.setMinimumHeight(40)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        save_settings_btn = QPushButton("Save Settings")
        save_settings_btn.clicked.connect(self.save_settings)
        button_layout.addWidget(save_settings_btn)
        
        bottom_layout.addLayout(button_layout)
        layout.addLayout(bottom_layout)
    
    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Audio Files",
            "",
            "Audio Files (*.mp3 *.wav *.flac *.m4a *.ogg);;All Files (*.*)"
        )
        for file in files:
            if file not in self.selected_files:
                self.selected_files.append(file)
                self.file_list.addItem(Path(file).name)
    
    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            audio_exts = ['.mp3', '.wav', '.flac', '.m4a', '.ogg']
            for file in Path(folder).rglob('*'):
                if file.suffix.lower() in audio_exts:
                    file_str = str(file)
                    if file_str not in self.selected_files:
                        self.selected_files.append(file_str)
                        self.file_list.addItem(file.name)
    
    def clear_files(self):
        self.selected_files.clear()
        self.file_list.clear()
    
    def browse_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.output_input.setText(folder)
    
    def start_processing(self):
        if not self.selected_files:
            QMessageBox.warning(self, "No Files", "Please add audio files first")
            return
        
        settings = {
            'model': self.model_combo.currentText(),
            'language': self.language_combo.currentText(),
            'speaker_id': self.speaker_input.text() or None,
            'output_dir': self.output_input.text(),
            'min_length': self.min_length_spin.value(),
            'max_length': self.max_length_spin.value(),
            'pause_threshold': self.pause_threshold_spin.value(),
            'format': self.format_combo.currentText(),
            'min_confidence': self.min_confidence_spin.value(),
            'filter_noise': self.filter_noise_check.isChecked(),
            'min_rms': self.min_rms_spin.value(),
            'filter_music': self.filter_music_check.isChecked(),
            'music_threshold': self.music_threshold_spin.value(),
            'mixed_threshold': self.mixed_threshold_spin.value(),
            'word_start_buffer': self.word_start_buffer_spin.value(),
            'word_end_buffer': self.word_end_buffer_spin.value()
        }
        
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        self.processing_thread = ProcessingThread(settings, self.selected_files)
        self.processing_thread.progress.connect(self.update_progress)
        self.processing_thread.status.connect(self.update_status)
        self.processing_thread.log.connect(self.append_log)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.error.connect(self.processing_error)
        self.processing_thread.start()
    
    def stop_processing(self):
        if self.processing_thread:
            self.processing_thread.stop()
            self.stop_button.setEnabled(False)
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def update_status(self, text):
        self.status_label.setText(text)
    
    def append_log(self, text):
        self.log_text.append(text)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def processing_finished(self, result):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Processing complete")
        self.progress_bar.setValue(100)
        
        QMessageBox.information(
            self,
            "Complete",
            f"Processing finished!\n\n"
            f"Success: {result['success']}\n"
            f"Total segments: {result['total_segments']}"
        )
    
    def processing_error(self, error):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Error occurred")
        QMessageBox.critical(self, "Error", f"Processing failed:\n{error}")
    
    def load_settings(self):
        settings_file = Path.home() / '.speech_splitter_settings.json'
        if settings_file.exists():
            try:
                with open(settings_file) as f:
                    settings = json.load(f)
                    
                self.model_combo.setCurrentText(settings.get('model', 'large'))
                self.language_combo.setCurrentText(settings.get('language', 'cs'))
                self.output_input.setText(settings.get('output_dir', 'output'))
                self.min_length_spin.setValue(settings.get('min_length', 1.0))
                self.max_length_spin.setValue(settings.get('max_length', 15.0))
                self.pause_threshold_spin.setValue(settings.get('pause_threshold', 0.5))
                self.word_start_buffer_spin.setValue(settings.get('word_start_buffer', 500))
                self.word_end_buffer_spin.setValue(settings.get('word_end_buffer', 500))
                self.format_combo.setCurrentText(settings.get('format', 'wav'))
                self.min_confidence_spin.setValue(settings.get('min_confidence', 0.5))
                self.filter_noise_check.setChecked(settings.get('filter_noise', True))
                self.min_rms_spin.setValue(settings.get('min_rms', 0.01))
                self.filter_music_check.setChecked(settings.get('filter_music', True))
                self.music_threshold_spin.setValue(settings.get('music_threshold', 0.3))
                self.mixed_threshold_spin.setValue(settings.get('mixed_threshold', 0.2))
            except Exception as e:
                print(f"Error loading settings: {e}")
    
    def save_settings(self):
        settings = {
            'model': self.model_combo.currentText(),
            'language': self.language_combo.currentText(),
            'output_dir': self.output_input.text(),
            'min_length': self.min_length_spin.value(),
            'max_length': self.max_length_spin.value(),
            'pause_threshold': self.pause_threshold_spin.value(),
            'word_start_buffer': self.word_start_buffer_spin.value(),
            'word_end_buffer': self.word_end_buffer_spin.value(),
            'format': self.format_combo.currentText(),
            'min_confidence': self.min_confidence_spin.value(),
            'filter_noise': self.filter_noise_check.isChecked(),
            'min_rms': self.min_rms_spin.value(),
            'filter_music': self.filter_music_check.isChecked(),
            'music_threshold': self.music_threshold_spin.value(),
            'mixed_threshold': self.mixed_threshold_spin.value()
        }
        
        settings_file = Path.home() / '.speech_splitter_settings.json'
        try:
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            QMessageBox.information(self, "Saved", "Settings saved successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings:\n{e}")


def main():
    app = QApplication(sys.argv)
    window = SpeechSplitterGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
