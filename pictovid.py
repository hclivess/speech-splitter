import sys
import tempfile
import os
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QPushButton, QLabel, QFileDialog, QProgressBar,
                               QSpinBox, QDoubleSpinBox, QHBoxLayout, QLineEdit,
                               QMessageBox, QGroupBox, QFormLayout)
from PySide6.QtCore import QThread, Signal, Qt
from pydub import AudioSegment
from pydub.silence import detect_silence
import subprocess

# Requirements:
# pip install PySide6 pydub
# Also needs ffmpeg installed on system: https://ffmpeg.org/download.html


def check_ffmpeg():
    """Check if ffmpeg is available on the system"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


class AudioProcessorThread(QThread):
    progress = Signal(str, int)  # message, progress_value
    finished = Signal(str)  # output path
    error = Signal(str)
    
    def __init__(self, image_path, audio_path, output_path, silence_thresh, min_silence_len):
        super().__init__()
        self.image_path = image_path
        self.audio_path = audio_path
        self.output_path = output_path
        self.silence_thresh = silence_thresh
        self.min_silence_len = min_silence_len
    
    def run(self):
        try:
            temp_audio_path = None
            
            self.progress.emit("Loading audio...", 10)
            
            # Load audio
            audio = AudioSegment.from_file(self.audio_path)
            
            # Convert to stereo if mono
            if audio.channels == 1:
                self.progress.emit("Converting to stereo...", 25)
                audio = audio.set_channels(2)
            
            # Remove large gaps
            self.progress.emit("Removing silent gaps...", 40)
            processed_audio = self.remove_large_gaps(
                audio, 
                silence_thresh=self.silence_thresh,
                min_silence_len=self.min_silence_len
            )
            
            # Add small fade to prevent clicks
            processed_audio = processed_audio.fade_in(50).fade_out(50)
            
            # Save processed audio to temp file
            self.progress.emit("Saving processed audio...", 60)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                processed_audio.export(temp_audio.name, format="mp3", bitrate="192k")
                temp_audio_path = temp_audio.name
            
            # Create video with ffmpeg
            self.progress.emit("Creating video...", 80)
            
            # Get audio duration in seconds
            duration = len(processed_audio) / 1000.0
            
            # Build ffmpeg command
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file if exists
                "-loop", "1",  # Loop the image
                "-i", self.image_path,  # Input image
                "-i", temp_audio_path,  # Input audio
                "-c:v", "libx264",  # Video codec
                "-tune", "stillimage",  # Optimize for still image
                "-c:a", "aac",  # Audio codec
                "-b:a", "192k",  # Audio bitrate
                "-pix_fmt", "yuv420p",  # Pixel format for compatibility
                "-t", str(duration),  # Duration to match audio
                "-shortest",  # Use shortest stream
                self.output_path
            ]
            
            # Run ffmpeg
            subprocess.run(cmd, check=True, capture_output=True)
            
            self.progress.emit("Complete!", 100)
            self.finished.emit(self.output_path)
            
        except Exception as e:
            self.error.emit(str(e))
        finally:
            # Clean up temp file
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
    
    def remove_large_gaps(self, audio, silence_thresh=-40, min_silence_len=4000, keep_silence=2000):
        """
        Remove large silent gaps from audio
        silence_thresh: dBFS threshold for silence
        min_silence_len: minimum silence length in ms to remove
        keep_silence: how much silence to keep at boundaries in ms
        """
        # Find silent ranges
        silent_ranges = detect_silence(audio, min_silence_len=min_silence_len, 
                                       silence_thresh=silence_thresh)
        
        if not silent_ranges:
            return audio
        
        # Build new audio by keeping non-silent parts
        chunks = []
        last_end = 0
        
        for start, end in silent_ranges:
            # Keep audio before this silent range
            if start > last_end:
                chunks.append(audio[last_end:start])
            
            # Keep a small portion of silence for smooth transitions
            if keep_silence > 0:
                silence_chunk = audio[start:start+keep_silence]
                if len(silence_chunk) > 0:
                    chunks.append(silence_chunk)
            
            last_end = end
        
        # Add the final chunk
        if last_end < len(audio):
            chunks.append(audio[last_end:])
        
        return sum(chunks, AudioSegment.empty())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio + Image to Video")
        self.setMinimumWidth(500)
        
        # File paths
        self.image_path = None
        self.audio_path = None
        
        # Create UI
        self.setup_ui()
    
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # File selection group
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()
        
        # Image selection
        self.image_label = QLabel("No image selected")
        self.image_btn = QPushButton("Select Image")
        self.image_btn.clicked.connect(self.select_image)
        
        file_layout.addWidget(self.image_label)
        file_layout.addWidget(self.image_btn)
        
        # Audio selection
        self.audio_label = QLabel("No audio selected")
        self.audio_btn = QPushButton("Select Audio")
        self.audio_btn.clicked.connect(self.select_audio)
        
        file_layout.addWidget(self.audio_label)
        file_layout.addWidget(self.audio_btn)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Settings group
        settings_group = QGroupBox("Audio Processing Settings")
        settings_layout = QFormLayout()
        
        # Silence threshold
        self.silence_thresh_spin = QDoubleSpinBox()
        self.silence_thresh_spin.setRange(-60, -10)
        self.silence_thresh_spin.setValue(-40)
        self.silence_thresh_spin.setSuffix(" dBFS")
        settings_layout.addRow("Silence Threshold:", self.silence_thresh_spin)
        
        # Min silence length
        self.min_silence_spin = QSpinBox()
        self.min_silence_spin.setRange(100, 10000)
        self.min_silence_spin.setValue(4000)
        self.min_silence_spin.setSuffix(" ms")
        self.min_silence_spin.setSingleStep(100)
        settings_layout.addRow("Min Silence to Remove:", self.min_silence_spin)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Output selection
        output_group = QGroupBox("Output")
        output_layout = QHBoxLayout()
        
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Output video path...")
        self.output_btn = QPushButton("Browse")
        self.output_btn.clicked.connect(self.select_output)
        
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(self.output_btn)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Process button
        self.process_btn = QPushButton("Create Video")
        self.process_btn.clicked.connect(self.process_video)
        self.process_btn.setEnabled(False)
        layout.addWidget(self.process_btn)
        
        # Add stretch
        layout.addStretch()
    
    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.image_path = path
            self.image_label.setText(f"Image: {Path(path).name}")
            self.check_ready()
    
    def select_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio", "", "Audio Files (*.mp3 *.wav *.m4a *.flac *.ogg)"
        )
        if path:
            self.audio_path = path
            self.audio_label.setText(f"Audio: {Path(path).name}")
            self.check_ready()
    
    def select_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Video", "", "MP4 Files (*.mp4)"
        )
        if path:
            if not path.endswith('.mp4'):
                path += '.mp4'
            self.output_edit.setText(path)
            self.check_ready()
    
    def check_ready(self):
        ready = all([
            self.image_path is not None,
            self.audio_path is not None,
            self.output_edit.text()
        ])
        self.process_btn.setEnabled(ready)
    
    def process_video(self):
        # Check for ffmpeg
        if not check_ffmpeg():
            QMessageBox.critical(
                self, "Error", 
                "ffmpeg not found! Please install ffmpeg and add it to your system PATH.\n"
                "Visit: https://ffmpeg.org/download.html"
            )
            return
        
        # Disable UI
        self.set_ui_enabled(False)
        self.progress_bar.setValue(0)
        
        # Get settings
        silence_thresh = self.silence_thresh_spin.value()
        min_silence_len = self.min_silence_spin.value()
        output_path = self.output_edit.text()
        
        # Create and start worker thread
        self.thread = AudioProcessorThread(
            self.image_path, self.audio_path, output_path,
            silence_thresh, min_silence_len
        )
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.processing_finished)
        self.thread.error.connect(self.processing_error)
        self.thread.start()
    
    def set_ui_enabled(self, enabled):
        self.image_btn.setEnabled(enabled)
        self.audio_btn.setEnabled(enabled)
        self.output_btn.setEnabled(enabled)
        self.process_btn.setEnabled(enabled)
        self.silence_thresh_spin.setEnabled(enabled)
        self.min_silence_spin.setEnabled(enabled)
    
    def update_progress(self, message, value):
        self.status_label.setText(message)
        self.progress_bar.setValue(value)
    
    def processing_finished(self, output_path):
        self.set_ui_enabled(True)
        self.status_label.setText("Video created successfully!")
        QMessageBox.information(self, "Success", f"Video saved to:\n{output_path}")
    
    def processing_error(self, error_msg):
        self.set_ui_enabled(True)
        self.status_label.setText("Error occurred")
        QMessageBox.critical(self, "Error", f"An error occurred:\n{error_msg}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
