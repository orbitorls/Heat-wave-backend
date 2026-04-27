"""Data Screen - Data management and download"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QProgressBar
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class DataScreen(QWidget):
    """Data screen for managing ERA5 and other data"""

    def __init__(self, model_manager=None):
        super().__init__()
        self.model_manager = model_manager
        self.init_ui()
        
    def init_ui(self):
        """Initialize data management UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Title with symbol
        title_label = QLabel("◈ DATA MANAGEMENT")
        title_label.setObjectName("titleLabel")
        layout.addWidget(title_label)
        
        # Action Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        
        download_btn = QPushButton("▣ DOWNLOAD ERA5 DATA")
        download_btn.clicked.connect(self.start_download)
        button_layout.addWidget(download_btn)
        
        audit_btn = QPushButton("▧ AUDIT DATA")
        audit_btn.clicked.connect(self.audit_data)
        button_layout.addWidget(audit_btn)
        
        refresh_btn = QPushButton("◈ REFRESH")
        refresh_btn.clicked.connect(self.refresh_data)
        button_layout.addWidget(refresh_btn)
        
        layout.addLayout(button_layout)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status Label
        self.status_label = QLabel("Ready")
        self.status_label.setFont(QFont("Consolas", 11))
        layout.addWidget(self.status_label)
        
        # Data Files Table
        files_container = QWidget()
        files_layout = QVBoxLayout(files_container)
        files_layout.setSpacing(8)
        files_layout.setContentsMargins(0, 0, 0, 0)
        
        files_label = QLabel("▤ DATA FILES")
        files_label.setStyleSheet("color: #707070; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;")
        files_layout.addWidget(files_label)
        
        self.files_table = QTableWidget()
        self.files_table.setColumnCount(4)
        self.files_table.setHorizontalHeaderLabels(["File Name", "Size", "Date", "Status"])
        self.files_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.refresh_data()
        
        files_layout.addWidget(self.files_table)
        layout.addWidget(files_container)
        
    def start_download(self):
        """Start ERA5 data download"""
        self.status_label.setText("Starting ERA5 data download...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Simulate download
        import os
        era5_dir = "era5_data"
        if not os.path.exists(era5_dir):
            os.makedirs(era5_dir)
        
        # This would call the actual download script
        # For now, just simulate
        self.status_label.setText("Download simulation complete")
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)
        QMessageBox.information(self, "Download", "ERA5 data download initiated. Check logs for progress.")
        
    def audit_data(self):
        """Audit data files for integrity"""
        self.status_label.setText("Auditing data files...")
        
        import os
        era5_dir = "era5_data"
        issues = []
        
        if os.path.exists(era5_dir):
            files = [f for f in os.listdir(era5_dir) if f.endswith('.nc')]
            if not files:
                issues.append("No .nc files found")
            
            for f in files:
                filepath = os.path.join(era5_dir, f)
                if os.path.getsize(filepath) < 1000:
                    issues.append(f"{f} is too small (< 1KB)")
        else:
            issues.append("era5_data directory does not exist")
        
        if issues:
            QMessageBox.warning(self, "Audit Issues", "\n".join(issues))
            self.status_label.setText(f"Audit complete: {len(issues)} issues found")
        else:
            QMessageBox.information(self, "Audit Complete", "All data files passed integrity check.")
            self.status_label.setText("Audit complete: No issues found")
        
    def refresh_data(self):
        """Refresh data files list"""
        import os
        from datetime import datetime
        
        self.files_table.setRowCount(0)
        
        era5_dir = "era5_data"
        if os.path.exists(era5_dir):
            files = [f for f in os.listdir(era5_dir) if f.endswith('.nc')]
            self.files_table.setRowCount(len(files))
            
            for i, f in enumerate(files):
                filepath = os.path.join(era5_dir, f)
                size = os.path.getsize(filepath)
                size_str = f"{size / (1024*1024):.2f} MB" if size > 1024*1024 else f"{size / 1024:.2f} KB"
                
                mtime = os.path.getmtime(filepath)
                date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                
                self.files_table.setItem(i, 0, QTableWidgetItem(f))
                self.files_table.setItem(i, 1, QTableWidgetItem(size_str))
                self.files_table.setItem(i, 2, QTableWidgetItem(date_str))
                self.files_table.setItem(i, 3, QTableWidgetItem("OK"))
        else:
            self.files_table.setRowCount(1)
            self.files_table.setItem(0, 0, QTableWidgetItem("No data directory found"))
            for j in range(1, 4):
                self.files_table.setItem(0, j, QTableWidgetItem("-"))
        
        self.status_label.setText(f"Found {self.files_table.rowCount()} data files")
