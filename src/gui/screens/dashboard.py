"""Dashboard Screen - System overview and quick actions"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QGroupBox, QLabel, QPushButton, QListWidget, QFrame
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont


class DashboardScreen(QWidget):
    """Dashboard screen showing system status and quick actions"""

    def __init__(self, model_manager=None):
        super().__init__()
        self.model_manager = model_manager
        self.init_ui()
        
    def init_ui(self):
        """Initialize dashboard UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(24)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # Title with subtitle
        title_container = QWidget()
        title_layout = QVBoxLayout(title_container)
        title_layout.setSpacing(4)
        title_layout.setContentsMargins(0, 0, 0, 0)
        
        title_label = QLabel("Dashboard")
        title_label.setObjectName("titleLabel")
        title_layout.addWidget(title_label)
        
        subtitle_label = QLabel("System Overview & Quick Actions")
        subtitle_label.setObjectName("subtitleLabel")
        title_layout.addWidget(subtitle_label)
        
        layout.addWidget(title_container)
        
        # System Info Cards
        info_layout = self.create_info_cards()
        layout.addLayout(info_layout)
        
        # Quick Actions
        actions_group = self.create_quick_actions()
        layout.addWidget(actions_group)
        
        # Recent Activity
        activity_group = self.create_activity_log()
        layout.addWidget(activity_group)
        
        layout.addStretch()
        
    def create_info_cards(self):
        """Create system information cards"""
        layout = QHBoxLayout()
        layout.setSpacing(16)
        
        # GPU Status Card
        gpu_card = self.create_status_card("GPU", "--", "#0088ff", "●")
        self.gpu_label = gpu_card.findChild(QLabel, "valueLabel")
        self.gpu_icon = gpu_card.findChild(QLabel, "iconLabel")
        layout.addWidget(gpu_card)
        
        # Model Status Card
        model_card = self.create_status_card("Models", "--", "#00aa66", "■")
        self.model_label = model_card.findChild(QLabel, "valueLabel")
        self.model_icon = model_card.findChild(QLabel, "iconLabel")
        layout.addWidget(model_card)
        
        # Data Status Card
        data_card = self.create_status_card("Data", "--", "#ffaa00", "◆")
        self.data_label = data_card.findChild(QLabel, "valueLabel")
        self.data_icon = data_card.findChild(QLabel, "iconLabel")
        layout.addWidget(data_card)
        
        # Check status after a short delay
        QTimer.singleShot(500, self.check_system_status)
        
        return layout
        
    def create_status_card(self, title, initial_value, color, indicator):
        """Create a styled status card with geometric indicator"""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border: 1px solid #353535;
                border-radius: 3px;
            }
        """)
        
        layout = QVBoxLayout(card)
        layout.setSpacing(0)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Top indicator bar
        indicator_bar = QFrame()
        indicator_bar.setFixedHeight(2)
        indicator_bar.setStyleSheet(f"background-color: {color};")
        layout.addWidget(indicator_bar)
        
        layout.addSpacing(12)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            color: #707070;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        """)
        layout.addWidget(title_label)
        
        layout.addSpacing(4)
        
        # Value
        value_label = QLabel(initial_value)
        value_label.setObjectName("valueLabel")
        value_label.setStyleSheet(f"""
            color: #e0e0e0;
            font-size: 24px;
            font-weight: 400;
            font-family: 'Consolas', 'Monaco', monospace;
        """)
        layout.addWidget(value_label)
        
        return card
        
    def create_quick_actions(self):
        """Create quick action buttons"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title_label = QLabel("OPERATIONS")
        title_label.setStyleSheet("""
            color: #707070;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        """)
        layout.addWidget(title_label)
        
        layout.addSpacing(4)
        
        # Create button grid
        button_grid = QWidget()
        grid_layout = QGridLayout(button_grid)
        grid_layout.setSpacing(8)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        
        # Action buttons
        actions = [
            ("TRAIN MODEL", self.go_to_train),
            ("RUN PREDICTION", self.go_to_predict),
            ("VIEW MAP", self.go_to_map),
            ("EVALUATE MODEL", self.go_to_eval),
            ("MANAGE DATA", self.go_to_data),
            ("VIEW CHECKPOINTS", self.go_to_checkpoints),
        ]
        
        for i, (title, callback) in enumerate(actions):
            btn = self.create_action_button(title, callback)
            row = i // 2
            col = i % 2
            grid_layout.addWidget(btn, row, col)
        
        layout.addWidget(button_grid)
        return container
        
    def create_action_button(self, title, callback):
        """Create a styled action button"""
        btn = QPushButton(title)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                border: 1px solid #383838;
                border-radius: 2px;
                padding: 14px 20px;
                text-align: left;
                font-size: 11px;
                font-weight: 500;
                letter-spacing: 0.5px;
            }
            QPushButton:hover {
                background-color: #333333;
                border-color: #0088ff;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #1a1a1a;
                border-color: #0066cc;
            }
        """)
        btn.clicked.connect(callback)
        return btn
        
    def create_activity_log(self):
        """Create recent activity log"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title_label = QLabel("SYSTEM LOG")
        title_label.setStyleSheet("""
            color: #707070;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        """)
        layout.addWidget(title_label)
        
        layout.addSpacing(4)
        
        self.activity_list = QListWidget()
        self.activity_list.setStyleSheet("""
            QListWidget {
                background-color: #1a1a1a;
                border: 1px solid #353535;
                border-radius: 2px;
                padding: 0;
            }
            QListWidget::item {
                padding: 8px 12px;
                border-bottom: 1px solid #252525;
            }
            QListWidget::item:selected {
                background-color: #2a2a2a;
            }
        """)
        self.activity_list.addItem("[INIT] System initialized")
        self.activity_list.addItem("[INIT] Dashboard loaded")
        
        layout.addWidget(self.activity_list)
        return container
        
    def check_system_status(self):
        """Check system status and update labels"""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                self.gpu_label.setText("CUDA")
                self.gpu_label.setStyleSheet("color: #0088ff; font-size: 24px; font-weight: 400; font-family: 'Consolas', 'Monaco', monospace;")
            else:
                self.gpu_label.setText("CPU")
                self.gpu_label.setStyleSheet("color: #e0e0e0; font-size: 24px; font-weight: 400; font-family: 'Consolas', 'Monaco', monospace;")
        except ImportError:
            self.gpu_label.setText("N/A")
            self.gpu_label.setStyleSheet("color: #505050; font-size: 24px; font-weight: 400; font-family: 'Consolas', 'Monaco', monospace;")
            
        # Check model status
        import os
        models_dir = "models"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
            if model_files:
                self.model_label.setText(f"{len(model_files)}")
                self.model_label.setStyleSheet("color: #00aa66; font-size: 24px; font-weight: 400; font-family: 'Consolas', 'Monaco', monospace;")
            else:
                self.model_label.setText("0")
                self.model_label.setStyleSheet("color: #505050; font-size: 24px; font-weight: 400; font-family: 'Consolas', 'Monaco', monospace;")
        else:
            self.model_label.setText("0")
            self.model_label.setStyleSheet("color: #505050; font-size: 24px; font-weight: 400; font-family: 'Consolas', 'Monaco', monospace;")
            
        # Check data status
        era5_dir = "era5_data"
        if os.path.exists(era5_dir):
            data_files = [f for f in os.listdir(era5_dir) if f.endswith('.nc')]
            if data_files:
                self.data_label.setText(f"{len(data_files)}")
                self.data_label.setStyleSheet("color: #ffaa00; font-size: 24px; font-weight: 400; font-family: 'Consolas', 'Monaco', monospace;")
            else:
                self.data_label.setText("0")
                self.data_label.setStyleSheet("color: #505050; font-size: 24px; font-weight: 400; font-family: 'Consolas', 'Monaco', monospace;")
        else:
            self.data_label.setText("0")
            self.data_label.setStyleSheet("color: #505050; font-size: 24px; font-weight: 400; font-family: 'Consolas', 'Monaco', monospace;")
            
    def add_activity(self, message):
        """Add message to activity log"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.activity_list.addItem(f"[{timestamp}] {message}")
        self.activity_list.scrollToBottom()
        
    # Navigation methods
    def go_to_train(self):
        """Navigate to train screen"""
        self.window().show_train()
        
    def go_to_predict(self):
        """Navigate to predict screen"""
        self.window().show_predict()
        
    def go_to_map(self):
        """Navigate to map screen"""
        self.window().show_map()
        
    def go_to_eval(self):
        """Navigate to evaluation screen"""
        self.window().show_eval()
        
    def go_to_data(self):
        """Navigate to data screen"""
        self.window().show_data()
        
    def go_to_checkpoints(self):
        """Navigate to checkpoints screen"""
        self.window().show_checkpoints()
