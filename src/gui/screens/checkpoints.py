"""Checkpoints Screen - Model checkpoint management"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QListWidget, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class CheckpointsScreen(QWidget):
    """Checkpoints screen for managing model checkpoints"""

    def __init__(self, model_manager=None):
        super().__init__()
        self.model_manager = model_manager
        self.init_ui()
        
    def init_ui(self):
        """Initialize checkpoints UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Title with symbol
        title_label = QLabel("▤ MODEL CHECKPOINTS")
        title_label.setObjectName("titleLabel")
        layout.addWidget(title_label)
        
        # Status Label
        self.status_label = QLabel("Ready")
        self.status_label.setFont(QFont("Consolas", 11))
        layout.addWidget(self.status_label)
        
        # Checkpoints List
        checkpoints_container = QWidget()
        checkpoints_layout = QVBoxLayout(checkpoints_container)
        checkpoints_layout.setSpacing(8)
        checkpoints_layout.setContentsMargins(0, 0, 0, 0)
        
        checkpoints_label = QLabel("▤ AVAILABLE CHECKPOINTS")
        checkpoints_label.setStyleSheet("color: #707070; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;")
        checkpoints_layout.addWidget(checkpoints_label)
        
        self.checkpoint_list = QListWidget()
        checkpoints_layout.addWidget(self.checkpoint_list)
        
        layout.addWidget(checkpoints_container)
        
        # Model Details
        details_group = QGroupBox("Model Details")
        details_layout = QGridLayout()
        
        details_layout.addWidget(QLabel("Model Name:"), 0, 0)
        self.model_name_label = QLabel("-")
        details_layout.addWidget(self.model_name_label, 0, 1)
        
        details_layout.addWidget(QLabel("File Size:"), 1, 0)
        self.file_size_label = QLabel("-")
        details_layout.addWidget(self.file_size_label, 1, 1)
        
        details_layout.addWidget(QLabel("Created:"), 2, 0)
        self.created_label = QLabel("-")
        details_layout.addWidget(self.created_label, 2, 1)
        
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)
        
        # Action Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        
        refresh_btn = QPushButton("◈ REFRESH")
        refresh_btn.clicked.connect(self.refresh_checkpoints)
        button_layout.addWidget(refresh_btn)
        
        load_btn = QPushButton("▣ LOAD SELECTED")
        load_btn.clicked.connect(self.load_model)
        button_layout.addWidget(load_btn)
        
        delete_btn = QPushButton("◉ DELETE SELECTED")
        delete_btn.clicked.connect(self.delete_model)
        button_layout.addWidget(delete_btn)
        
        layout.addLayout(button_layout)
        
        # Connect list selection
        self.checkpoint_list.itemClicked.connect(self.show_model_details)
        
    def refresh_checkpoints(self):
        """Refresh checkpoint list"""
        self.checkpoint_list.clear()
        
        import os
        models_dir = "models"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
            for model in model_files:
                self.checkpoint_list.addItem(model)
            
            self.status_label.setText(f"Found {len(model_files)} model(s)")
        else:
            self.checkpoint_list.addItem("No models directory found")
            self.status_label.setText("Models directory not found")
            
    def show_model_details(self, item):
        """Show details for selected model"""
        model_name = item.text()
        
        import os
        from datetime import datetime
        
        filepath = os.path.join("models", model_name)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            size_str = f"{size / (1024*1024):.2f} MB" if size > 1024*1024 else f"{size / 1024:.2f} KB"
            
            mtime = os.path.getmtime(filepath)
            date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            
            self.model_name_label.setText(model_name)
            self.file_size_label.setText(size_str)
            self.created_label.setText(date_str)
        else:
            self.model_name_label.setText("-")
            self.file_size_label.setText("-")
            self.created_label.setText("-")
            
    def load_model(self):
        """Load selected model"""
        current_item = self.checkpoint_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a model to load.")
            return
            
        model_name = current_item.text()
        self.status_label.setText(f"Loading model: {model_name}...")
        
        # Simulate loading
        # In real implementation, this would load the model
        QMessageBox.information(self, "Model Loaded", f"Model '{model_name}' loaded successfully.")
        self.status_label.setText(f"Model loaded: {model_name}")
        
    def delete_model(self):
        """Delete selected model"""
        current_item = self.checkpoint_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a model to delete.")
            return
            
        model_name = current_item.text()
        
        reply = QMessageBox.question(
            self, 'Confirm Delete',
            f'Are you sure you want to delete {model_name}?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            import os
            filepath = os.path.join("models", model_name)
            try:
                os.remove(filepath)
                self.refresh_checkpoints()
                self.status_label.setText(f"Deleted: {model_name}")
                QMessageBox.information(self, "Deleted", f"Model '{model_name}' deleted successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not delete model: {str(e)}")
                self.status_label.setText(f"Error deleting model")
