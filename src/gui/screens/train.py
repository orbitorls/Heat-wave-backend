"""Train Screen - Model training with real-time charts"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QComboBox, QProgressBar, QPlainTextEdit, QFormLayout, QTextEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont


class TrainScreen(QWidget):
    """Training screen with configuration and real-time monitoring"""

    def __init__(self, model_manager=None):
        super().__init__()
        self.model_manager = model_manager
        self.is_training = False
        self.training_thread = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize training UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Title with symbol
        title_label = QLabel("▣ MODEL TRAINING")
        title_label.setObjectName("titleLabel")
        layout.addWidget(title_label)
        
        # Right panel - Charts and Logs
        right_panel = QVBoxLayout()
        
        # Model Selection
        model_container = QWidget()
        model_layout = QVBoxLayout(model_container)
        model_layout.setSpacing(8)
        model_layout.setContentsMargins(0, 0, 0, 0)
        
        model_label = QLabel("▧ MODEL TYPE")
        model_label.setStyleSheet("color: #707070; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;")
        model_layout.addWidget(model_label)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["ConvLSTM", "XGBoost", "RandomForest"])
        model_layout.addWidget(self.model_combo)
        
        layout.addWidget(model_container)
        
        # Training Log
        log_container = QWidget()
        log_layout = QVBoxLayout(log_container)
        log_layout.setSpacing(8)
        log_layout.setContentsMargins(0, 0, 0, 0)
        
        log_label = QLabel("◫ TRAINING LOG")
        log_label.setStyleSheet("color: #707070; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;")
        log_layout.addWidget(log_label)
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        log_layout.addWidget(self.log_output)
        
        layout.addWidget(log_container)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        
        self.start_button = QPushButton("▣ START TRAINING")
        self.start_button.clicked.connect(self.start_training)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("◈ STOP TRAINING")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        layout.addLayout(button_layout)
        
    def create_config_panel(self):
        """Create configuration panel"""
        group = QGroupBox("Training Configuration")
        layout = QVBoxLayout()
        
        form = QFormLayout()
        
        # Model Type
        self.model_type = QComboBox()
        # Hyperparameters
        param_container = QWidget()
        param_layout = QVBoxLayout(param_container)
        param_layout.setSpacing(8)
        param_layout.setContentsMargins(0, 0, 0, 0)
        
        param_label = QLabel("◈ HYPERPARAMETERS")
        param_label.setStyleSheet("color: #707070; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;")
        param_layout.addWidget(param_label)
        
        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("EPOCHS"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        epochs_layout.addWidget(self.epochs_spin)
        epochs_layout.addStretch()
        param_layout.addLayout(epochs_layout)
        
        # Batch Size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("BATCH SIZE"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(32)
        batch_layout.addWidget(self.batch_spin)
        batch_layout.addStretch()
        param_layout.addLayout(batch_layout)
        
        # Learning Rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("LEARNING RATE"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.0001)
        lr_layout.addWidget(self.lr_spin)
        lr_layout.addStretch()
        param_layout.addLayout(lr_layout)
        
        layout.addWidget(param_container)
        
        # Sequence Length
        self.seq_length = QSpinBox()
        self.seq_length.setRange(1, 30)
        self.seq_length.setValue(5)
        form.addRow("Sequence Length:", self.seq_length)
        
        # Heatwave Threshold
        self.heatwave_threshold = QDoubleSpinBox()
        self.heatwave_threshold.setRange(30.0, 45.0)
        self.heatwave_threshold.setValue(38.0)
        self.heatwave_threshold.setSingleStep(0.5)
        form.addRow("Heatwave Threshold (°C):", self.heatwave_threshold)
        
        layout.addLayout(form)
        layout.addStretch()
        group.setLayout(layout)
        return group
        
    def start_training(self):
        """Start model training"""
        if self.is_training:
            return
            
        self.is_training = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_output.clear()
        
        self.log_message("Starting training...")
        self.log_message(f"Model: {self.model_combo.currentText()}")
        self.log_message(f"Epochs: {self.epochs_spin.value()}")
        self.log_message(f"Batch Size: {self.batch_spin.value()}")
        
        # Create training thread
        self.training_thread = TrainingWorker(
            model_type=self.model_combo.currentText(),
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            learning_rate=self.lr_spin.value(),
            seq_length=5,
            heatwave_threshold=38.0
        )
        self.training_thread.progress.connect(self.update_progress)
        self.training_thread.log.connect(self.log_message)
        self.training_thread.finished.connect(self.training_finished)
        self.training_thread.start()
        
    def stop_training(self):
        """Stop model training"""
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop()
            self.log_message("Stopping training...")
        
    def update_progress(self, value):
        """Update progress bar"""
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(value)
        
    def log_message(self, message):
        """Add message to log"""
        self.log_output.appendPlainText(message)
        self.log_output.verticalScrollBar().setValue(
            self.log_output.verticalScrollBar().maximum()
        )
        
    def training_finished(self, success):
        """Handle training completion"""
        self.is_training = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        if success:
            self.log_message("Training completed successfully!")
            self.progress_bar.setValue(100)
        else:
            self.log_message("Training stopped or failed.")


class TrainingWorker(QThread):
    """Worker thread for training models"""
    
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(bool)
    
    def __init__(self, model_type, epochs, batch_size, learning_rate, seq_length, heatwave_threshold):
        super().__init__()
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seq_length = seq_length
        self.heatwave_threshold = heatwave_threshold
        self._is_running = True
        
    def stop(self):
        """Stop training"""
        self._is_running = False
        
    def run(self):
        """Run training in background thread"""
        try:
            self.log.emit(f"Loading data for {self.model_type}...")
            self.msleep(500)  # Simulate data loading
            
            for epoch in range(self.epochs):
                if not self._is_running:
                    self.log.emit("Training stopped by user")
                    self.finished.emit(False)
                    return
                    
                # Simulate training epoch
                self.log.emit(f"Epoch {epoch + 1}/{self.epochs}")
                self.msleep(100)  # Simulate epoch processing
                
                # Update progress
                progress = int((epoch + 1) / self.epochs * 100)
                self.progress.emit(progress)
                
                # Simulate loss (for demo)
                if epoch % 5 == 0:
                    loss = 1.0 - (epoch / self.epochs) * 0.8
                    self.log.emit(f"  Loss: {loss:.4f}")
            
            # Save model
            self.log.emit("Saving model checkpoint...")
            self.msleep(500)
            self.log.emit("Model saved successfully!")
            
            self.finished.emit(True)
            
        except Exception as e:
            self.log.emit(f"Error during training: {str(e)}")
            self.finished.emit(False)
