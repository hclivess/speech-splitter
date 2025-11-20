#!/usr/bin/env python3
"""
PySide6 GUI application for splitting text files based on blank line separators.
"""

import sys
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QSpinBox, QTextEdit, QFileDialog,
    QMessageBox, QGroupBox, QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont


class SplitWorker(QThread):
    """Worker thread for splitting files to avoid blocking the UI."""
    progress = Signal(str)
    finished = Signal(list)
    error = Signal(str)
    
    def __init__(self, input_file, output_dir, min_blank_lines, prefix, min_length):
        super().__init__()
        self.input_file = input_file
        self.output_dir = output_dir
        self.min_blank_lines = min_blank_lines
        self.prefix = prefix
        self.min_length = min_length
    
    def run(self):
        try:
            output_files = self.split_text_file()
            self.finished.emit(output_files)
        except Exception as e:
            self.error.emit(str(e))
    
    def split_text_file(self):
        """Split the text file into multiple files."""
        input_path = Path(self.input_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.progress.emit(f"Reading file: {input_path.name}")
        
        # Read the file
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        self.progress.emit(f"Analyzing {len(lines)} lines...")
        
        # Split into sections
        sections = []
        current_section = []
        blank_count = 0
        
        for line in lines:
            if line.strip() == '':
                blank_count += 1
            else:
                if blank_count >= self.min_blank_lines and current_section:
                    sections.append(current_section)
                    current_section = []
                blank_count = 0
            
            current_section.append(line)
        
        if current_section:
            sections.append(current_section)
        
        self.progress.emit(f"Found {len(sections)} sections before filtering")
        
        # Merge sections that are too short
        if self.min_length > 0:
            filtered_sections = []
            i = 0
            while i < len(sections):
                current = sections[i]
                non_blank_lines = sum(1 for line in current if line.strip())
                
                # If section is too short, merge with next section
                while non_blank_lines < self.min_length and i + 1 < len(sections):
                    self.progress.emit(f"Section {i+1} too short ({non_blank_lines} lines), merging with next...")
                    i += 1
                    current.extend(sections[i])
                    non_blank_lines = sum(1 for line in current if line.strip())
                
                filtered_sections.append(current)
                i += 1
            
            sections = filtered_sections
            self.progress.emit(f"After filtering: {len(sections)} sections")
        
        # Write sections to separate files
        output_files = []
        extension = input_path.suffix
        
        for i, section in enumerate(sections, start=1):
            section_text = ''.join(section).strip() + '\n'
            
            if section_text.strip():
                output_filename = f"{self.prefix}_{i:03d}{extension}"
                output_path = output_dir / output_filename
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(section_text)
                
                output_files.append(str(output_path))
                non_blank = sum(1 for line in section if line.strip())
                self.progress.emit(f"Created: {output_filename} ({non_blank} lines)")
        
        return output_files


class TextSplitterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.input_file = ""
        self.output_dir = ""
        self.worker = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Text File Splitter")
        self.setMinimumSize(700, 600)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        
        # Title
        title_label = QLabel("Text File Splitter")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel("Split text files based on consecutive blank lines")
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setStyleSheet("color: #666;")
        main_layout.addWidget(desc_label)
        
        # Input file section
        input_group = QGroupBox("Input File")
        input_layout = QHBoxLayout()
        
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Select a text file to split...")
        self.input_line.setReadOnly(True)
        input_layout.addWidget(self.input_line)
        
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_input_file)
        browse_btn.setMinimumWidth(100)
        input_layout.addWidget(browse_btn)
        
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)
        
        # Output directory section
        output_group = QGroupBox("Output Directory")
        output_layout = QHBoxLayout()
        
        self.output_line = QLineEdit()
        self.output_line.setPlaceholderText("Select output directory (optional)...")
        self.output_line.setReadOnly(True)
        output_layout.addWidget(self.output_line)
        
        output_browse_btn = QPushButton("Browse")
        output_browse_btn.clicked.connect(self.browse_output_dir)
        output_browse_btn.setMinimumWidth(100)
        output_layout.addWidget(output_browse_btn)
        
        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group)
        
        # Options section
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        
        # Blank lines threshold
        blank_lines_layout = QHBoxLayout()
        blank_lines_layout.addWidget(QLabel("Minimum blank lines to split:"))
        self.blank_lines_spin = QSpinBox()
        self.blank_lines_spin.setMinimum(1)
        self.blank_lines_spin.setMaximum(10)
        self.blank_lines_spin.setValue(2)
        self.blank_lines_spin.setMinimumWidth(80)
        self.blank_lines_spin.valueChanged.connect(self.update_preview_if_file_loaded)
        blank_lines_layout.addWidget(self.blank_lines_spin)
        blank_lines_layout.addStretch()
        options_layout.addLayout(blank_lines_layout)
        
        # Prefix
        prefix_layout = QHBoxLayout()
        prefix_layout.addWidget(QLabel("Output file prefix:"))
        self.prefix_line = QLineEdit()
        self.prefix_line.setText("section")
        self.prefix_line.setMaximumWidth(200)
        prefix_layout.addWidget(self.prefix_line)
        prefix_layout.addStretch()
        options_layout.addLayout(prefix_layout)
        
        # Minimum section length
        min_length_layout = QHBoxLayout()
        min_length_layout.addWidget(QLabel("Minimum section length (lines):"))
        self.min_length_spin = QSpinBox()
        self.min_length_spin.setMinimum(0)
        self.min_length_spin.setMaximum(1000)
        self.min_length_spin.setValue(5)
        self.min_length_spin.setMinimumWidth(80)
        self.min_length_spin.setToolTip("Sections shorter than this will be merged with the next section (0 = no minimum)")
        self.min_length_spin.valueChanged.connect(self.update_preview_if_file_loaded)
        min_length_layout.addWidget(self.min_length_spin)
        min_length_layout.addStretch()
        options_layout.addLayout(min_length_layout)
        
        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)
        
        # Preview section
        preview_group = QGroupBox("Preview / Status")
        preview_layout = QVBoxLayout()
        
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setPlaceholderText("Select a file to see preview and status messages...")
        preview_layout.addWidget(self.preview_text)
        
        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.split_btn = QPushButton("Split File")
        self.split_btn.setMinimumWidth(120)
        self.split_btn.setMinimumHeight(35)
        self.split_btn.clicked.connect(self.split_file)
        self.split_btn.setEnabled(False)
        self.split_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        button_layout.addWidget(self.split_btn)
        
        button_layout.addStretch()
        main_layout.addLayout(button_layout)
    
    def browse_input_file(self):
        """Open file dialog to select input file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Text File",
            "",
            "Text Files (*.txt);;All Files (*.*)"
        )
        
        if file_path:
            self.input_file = file_path
            self.input_line.setText(file_path)
            self.split_btn.setEnabled(True)
            
            # Set default output directory to same as input file
            if not self.output_dir:
                self.output_dir = str(Path(file_path).parent)
                self.output_line.setText(self.output_dir)
            
            # Show preview
            self.show_preview()
    
    def update_preview_if_file_loaded(self):
        """Update preview when settings change, if a file is loaded."""
        if self.input_file:
            self.show_preview()
    
    def browse_output_dir(self):
        """Open dialog to select output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.output_dir or ""
        )
        
        if dir_path:
            self.output_dir = dir_path
            self.output_line.setText(dir_path)
    
    def show_preview(self):
        """Show a preview of the file content with split points marked."""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Analyze where splits will occur
            split_info = self.analyze_splits(lines)
            
            preview = f"File: {Path(self.input_file).name}\n"
            preview += f"Size: {Path(self.input_file).stat().st_size} bytes\n"
            preview += f"Total lines: {len(lines)}\n"
            preview += f"Estimated sections: {split_info['section_count']}\n"
            preview += "-" * 70 + "\n"
            preview += "Preview with split markers:\n"
            preview += "-" * 70 + "\n\n"
            
            # Show preview with split markers
            char_count = 0
            max_preview_chars = 3000
            
            for i, line in enumerate(lines):
                if char_count >= max_preview_chars:
                    preview += f"\n\n... ({len(lines) - i} more lines not shown)\n"
                    break
                
                # Check if this line is a split point
                if i in split_info['split_lines']:
                    section_num = split_info['split_lines'][i]
                    line_count = split_info['section_lengths'][section_num - 1]
                    preview += f"\n{'='*70}\n"
                    preview += f">>> SPLIT HERE - Start of Section {section_num} ({line_count} lines) <<<\n"
                    preview += f"{'='*70}\n\n"
                
                preview += line
                char_count += len(line)
            
            # Add section summary at the end
            preview += "\n" + "-" * 70 + "\n"
            preview += "Section Summary:\n"
            preview += "-" * 70 + "\n"
            for i, length in enumerate(split_info['section_lengths'], start=1):
                preview += f"Section {i}: {length} lines\n"
            
            self.preview_text.setPlainText(preview)
            
        except Exception as e:
            self.preview_text.setPlainText(f"Error reading file: {e}")
    
    def analyze_splits(self, lines):
        """Analyze where splits will occur without actually splitting the file."""
        min_blank_lines = self.blank_lines_spin.value()
        min_length = self.min_length_spin.value()
        
        # First pass: find initial split points
        sections = []
        current_section = []
        blank_count = 0
        
        for line in lines:
            if line.strip() == '':
                blank_count += 1
            else:
                if blank_count >= min_blank_lines and current_section:
                    sections.append(current_section)
                    current_section = []
                blank_count = 0
            
            current_section.append(line)
        
        if current_section:
            sections.append(current_section)
        
        # Second pass: merge sections that are too short
        if min_length > 0:
            filtered_sections = []
            i = 0
            while i < len(sections):
                current = sections[i]
                non_blank_lines = sum(1 for line in current if line.strip())
                
                while non_blank_lines < min_length and i + 1 < len(sections):
                    i += 1
                    current.extend(sections[i])
                    non_blank_lines = sum(1 for line in current if line.strip())
                
                filtered_sections.append(current)
                i += 1
            
            sections = filtered_sections
        
        # Calculate split line positions and section lengths
        split_lines = {}
        section_lengths = []
        current_line = 0
        
        for section_num, section in enumerate(sections, start=1):
            split_lines[current_line] = section_num
            non_blank = sum(1 for line in section if line.strip())
            section_lengths.append(non_blank)
            current_line += len(section)
        
        return {
            'section_count': len(sections),
            'split_lines': split_lines,
            'section_lengths': section_lengths
        }
    
    def split_file(self):
        """Start the file splitting process."""
        if not self.input_file:
            QMessageBox.warning(self, "No File", "Please select an input file first.")
            return
        
        if not self.output_dir:
            self.output_dir = str(Path(self.input_file).parent)
        
        # Disable button and show progress
        self.split_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        self.preview_text.clear()
        self.preview_text.append("Starting split process...\n")
        
        # Create and start worker thread
        self.worker = SplitWorker(
            input_file=self.input_file,
            output_dir=self.output_dir,
            min_blank_lines=self.blank_lines_spin.value(),
            prefix=self.prefix_line.text(),
            min_length=self.min_length_spin.value()
        )
        
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.split_finished)
        self.worker.error.connect(self.split_error)
        self.worker.start()
    
    def update_progress(self, message):
        """Update progress text."""
        self.preview_text.append(message)
    
    def split_finished(self, output_files):
        """Handle successful completion."""
        self.progress_bar.setVisible(False)
        self.split_btn.setEnabled(True)
        
        self.preview_text.append(f"\n{'='*50}")
        self.preview_text.append(f"✓ SUCCESS!")
        self.preview_text.append(f"Created {len(output_files)} files in:")
        self.preview_text.append(f"{self.output_dir}")
        self.preview_text.append(f"{'='*50}")
        
        QMessageBox.information(
            self,
            "Success",
            f"Successfully split file into {len(output_files)} sections!\n\n"
            f"Output directory: {self.output_dir}"
        )
    
    def split_error(self, error_msg):
        """Handle errors."""
        self.progress_bar.setVisible(False)
        self.split_btn.setEnabled(True)
        
        self.preview_text.append(f"\n{'='*50}")
        self.preview_text.append(f"✗ ERROR: {error_msg}")
        self.preview_text.append(f"{'='*50}")
        
        QMessageBox.critical(
            self,
            "Error",
            f"An error occurred:\n\n{error_msg}"
        )


def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = TextSplitterGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
