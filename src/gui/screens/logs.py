"""Logs Screen - View system logs"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QTextEdit, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class LogsScreen(QWidget):
    """Logs screen for viewing system logs"""

    def __init__(self, model_manager=None):
        super().__init__()
        self.model_manager = model_manager
        self.init_ui()
        
    def init_ui(self):
        """Initialize logs UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Title with symbol
        title_label = QLabel("◫ SYSTEM LOGS")
        title_label.setObjectName("titleLabel")
        layout.addWidget(title_label)
        
        # Log Source Selection
        source_container = QWidget()
        source_layout = QHBoxLayout(source_container)
        source_layout.setSpacing(8)
        source_layout.setContentsMargins(0, 0, 0, 0)
        
        source_layout.addWidget(QLabel("◈ LOG SOURCE"))
        
        self.log_source = QComboBox()
        self.log_source.addItems(["Training Logs", "System Logs", "API Logs"])
        self.log_source.currentTextChanged.connect(self.load_logs)
        source_layout.addWidget(self.log_source)
        
        source_layout.addStretch()
        layout.addWidget(source_container)
        
        # Controls
        control_layout = QHBoxLayout()
        
        control_layout.addWidget(QLabel("Level:"))
        self.log_level = QComboBox()
        self.log_level.addItems(["All", "INFO", "WARNING", "ERROR"])
        self.log_level.currentTextChanged.connect(self.filter_logs)
        control_layout.addWidget(self.log_level)
        
        # Action Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        
        refresh_btn = QPushButton("◈ REFRESH")
        refresh_btn.clicked.connect(self.load_logs)
        button_layout.addWidget(refresh_btn)
        
        clear_btn = QPushButton("◉ CLEAR")
        clear_btn.clicked.connect(self.clear_logs)
        button_layout.addWidget(clear_btn)
        
        export_btn = QPushButton("▤ EXPORT")
        export_btn.clicked.connect(self.export_logs)
        button_layout.addWidget(export_btn)
        
        control_layout.addLayout(button_layout)
        layout.addLayout(control_layout)
        
        # Status Label
        self.status_label = QLabel("Ready")
        self.status_label.setFont(QFont("Consolas", 11))
        layout.addWidget(self.status_label)
        
        # Log Display
        log_container = QWidget()
        log_layout = QVBoxLayout(log_container)
        log_layout.setSpacing(8)
        log_layout.setContentsMargins(0, 0, 0, 0)
        
        log_label = QLabel("◫ LOG OUTPUT")
        log_label.setStyleSheet("color: #707070; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;")
        log_layout.addWidget(log_label)
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        log_layout.addWidget(self.log_output)
        
        layout.addWidget(log_container)
        
    def load_logs(self):
        """Load logs from selected source"""
        source = self.log_source.currentText()
        self.log_output.clear()
        
        # Simulate loading logs
        import os
        logs_dir = "logs"
        
        if source == "Training Logs":
            self.log_output.appendPlainText("[INFO] Training log viewer")
            self.log_output.appendPlainText("[INFO] Select a training run to view details")
            if os.path.exists(logs_dir):
                log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
                if log_files:
                    self.log_output.appendPlainText(f"[INFO] Found {len(log_files)} log file(s)")
                    for f in log_files:
                        self.log_output.appendPlainText(f"  - {f}")
                else:
                    self.log_output.appendPlainText("[WARNING] No log files found")
            else:
                self.log_output.appendPlainText("[WARNING] Logs directory not found")
                
        elif source == "System Logs":
            self.log_output.appendPlainText("[INFO] System log viewer")
            self.log_output.appendPlainText(f"[{self.get_timestamp()}] System initialized")
            self.log_output.appendPlainText(f"[{self.get_timestamp()}] GUI started")
            self.log_output.appendPlainText(f"[{self.get_timestamp()}] All modules loaded")
            
        elif source == "API Logs":
            self.log_output.appendPlainText("[INFO] API log viewer")
            self.log_output.appendPlainText("[INFO] API server logs (deprecated)")
            self.log_output.appendPlainText("[INFO] Use TUI instead for full functionality")
        
        self.status_label.setText(f"Loaded {source}")
        self.filter_logs()
        
    def filter_logs(self):
        """Filter logs by level"""
        level = self.log_level.currentText()
        # In a real implementation, this would filter the actual log content
        self.status_label.setText(f"Filtered by {level}")
        
    def export_logs(self):
        """Export logs to file"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Export Logs", "", "Log Files (*.log);;Text Files (*.txt)", options=options
        )
        
        if file_name:
            try:
                with open(file_name, 'w') as f:
                    f.write(self.log_output.toPlainText())
                QMessageBox.information(self, "Export Complete", f"Logs exported to {file_name}")
                self.status_label.setText(f"Exported to {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Could not export logs: {str(e)}")
                
    def clear_logs(self):
        """Clear log output"""
        self.log_output.clear()
        self.status_label.setText("Logs cleared")
        
    def get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
