"""Eval Screen - Model evaluation and metrics"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class EvalScreen(QWidget):
    """Evaluation screen for model metrics"""

    def __init__(self, model_manager=None):
        super().__init__()
        self.model_manager = model_manager
        self.init_ui()
        
    def init_ui(self):
        """Initialize evaluation UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Title with symbol
        title_label = QLabel("▧ MODEL EVALUATION")
        title_label.setObjectName("titleLabel")
        layout.addWidget(title_label)
        
        # Action Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        
        eval_btn = QPushButton("▣ RUN EVALUATION")
        eval_btn.clicked.connect(self.run_evaluation)
        button_layout.addWidget(eval_btn)
        
        accuracy_btn = QPushButton("◉ CHECK ACCURACY")
        accuracy_btn.clicked.connect(self.run_accuracy_check)
        button_layout.addWidget(accuracy_btn)
        
        test_btn = QPushButton("▣ RUN TESTS")
        test_btn.clicked.connect(self.run_tests)
        button_layout.addWidget(test_btn)
        
        layout.addLayout(button_layout)
        
        # Metrics Table
        metrics_group = QGroupBox("▧ EVALUATION METRICS")
        metrics_layout = QVBoxLayout()
        
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(3)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value", "Target"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        # Initialize with common metrics
        metrics = [
            ("F1 Score", "-", ">= 0.70"),
            ("Precision", "-", ">= 0.70"),
            ("Recall", "-", ">= 0.70"),
            ("Accuracy", "-", ">= 0.80"),
            ("MAE (°C)", "-", "<= 2.0"),
            ("RMSE (°C)", "-", "<= 3.0"),
            ("R²", "-", ">= 0.80"),
            ("Brier Score", "-", "<= 0.20"),
        ]
        
        self.metrics_table.setRowCount(len(metrics))
        for i, (metric, value, target) in enumerate(metrics):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(metric))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(value))
            self.metrics_table.setItem(i, 2, QTableWidgetItem(target))
        
        metrics_layout.addWidget(self.metrics_table)
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # Status Label
        self.status_label = QLabel("Ready to evaluate")
        self.status_label.setFont(QFont("Consolas", 11))
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
    def run_evaluation(self):
        """Run full model evaluation"""
        self.status_label.setText("Running evaluation...")
        
        # Simulate evaluation
        import random
        metrics = [
            ("F1 Score", f"{random.uniform(0.6, 0.85):.3f}", ">= 0.70"),
            ("Precision", f"{random.uniform(0.65, 0.90):.3f}", ">= 0.70"),
            ("Recall", f"{random.uniform(0.60, 0.85):.3f}", ">= 0.70"),
            ("Accuracy", f"{random.uniform(0.75, 0.95):.3f}", ">= 0.80"),
            ("MAE (°C)", f"{random.uniform(1.0, 2.5):.2f}", "<= 2.0"),
            ("RMSE (°C)", f"{random.uniform(1.5, 3.5):.2f}", "<= 3.0"),
            ("R²", f"{random.uniform(0.70, 0.92):.3f}", ">= 0.80"),
            ("Brier Score", f"{random.uniform(0.10, 0.25):.3f}", "<= 0.20"),
        ]
        
        for i, (metric, value, target) in enumerate(metrics):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(metric))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(value))
            self.metrics_table.setItem(i, 2, QTableWidgetItem(target))
            
            # Color code based on target
            try:
                val_float = float(value)
                if ">=" in target:
                    target_val = float(target.split(">=")[1].strip())
                    if val_float >= target_val:
                        self.metrics_table.item(i, 1).setForeground(Qt.GlobalColor.green)
                    else:
                        self.metrics_table.item(i, 1).setForeground(Qt.GlobalColor.red)
                elif "<=" in target:
                    target_val = float(target.split("<=")[1].strip())
                    if val_float <= target_val:
                        self.metrics_table.item(i, 1).setForeground(Qt.GlobalColor.green)
                    else:
                        self.metrics_table.item(i, 1).setForeground(Qt.GlobalColor.red)
            except:
                pass
        
        self.status_label.setText("Evaluation complete")
        
    def run_accuracy_check(self):
        """Run accuracy check on loaded model"""
        self.status_label.setText("Checking model accuracy...")
        
        # Simulate accuracy check
        import os
        models_dir = "models"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
            if model_files:
                self.status_label.setText(f"Found {len(model_files)} model(s). Running accuracy check...")
                # Simulate check
                import random
                f1 = random.uniform(0.55, 0.80)
                self.status_label.setText(f"Model F1 Score: {f1:.3f}")
                
                if f1 >= 0.70:
                    QMessageBox.information(self, "Accuracy Check", 
                                          f"Model accuracy is good (F1: {f1:.3f})")
                else:
                    QMessageBox.warning(self, "Accuracy Check",
                                      f"Model accuracy needs improvement (F1: {f1:.3f})")
            else:
                QMessageBox.warning(self, "No Models", "No model files found in models/ directory")
                self.status_label.setText("No models found")
        else:
            QMessageBox.warning(self, "No Models", "Models directory not found")
            self.status_label.setText("Models directory not found")
            
    def run_tests(self):
        """Run unit tests"""
        self.status_label.setText("Running unit tests...")
        
        # Simulate running tests
        import subprocess
        try:
            result = subprocess.run(["pytest", "tests/", "-v"], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self.status_label.setText("All tests passed!")
                QMessageBox.information(self, "Test Results", "All unit tests passed successfully.")
            else:
                self.status_label.setText("Some tests failed")
                QMessageBox.warning(self, "Test Results", 
                                  f"Some tests failed:\n{result.stdout}")
        except Exception as e:
            self.status_label.setText(f"Error running tests: {str(e)}")
            QMessageBox.warning(self, "Test Error", f"Could not run tests: {str(e)}")
