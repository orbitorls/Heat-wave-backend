"""Main Window for Heatwave GUI Application"""

from PyQt6.QtWidgets import (
    QMainWindow, QStackedWidget, QStatusBar, 
    QVBoxLayout, QHBoxLayout, QWidget, QMessageBox, 
    QPushButton, QFrame, QSizePolicy, QLabel
)
from PyQt6.QtGui import QAction

from .styles import STYLESHEET
from .screens.dashboard import DashboardScreen
from .screens.train import TrainScreen
from .screens.predict import PredictScreen
from .screens.map import MapScreen
from .screens.eval import EvalScreen
from .screens.data import DataScreen
from .screens.checkpoints import CheckpointsScreen
from .screens.logs import LogsScreen

# Shared model manager instance
try:
    from src.models.manager import ModelManager
    _model_manager = ModelManager()
except Exception:
    _model_manager = None


class HeatwaveMainWindow(QMainWindow):
    """Main application window for Heatwave GUI"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Heatwave Prediction System - Thailand")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # Shared model manager
        self.model_manager = _model_manager

        # Apply stylesheet
        self.setStyleSheet(STYLESHEET)

        # Initialize UI
        self.init_ui()
        
    def init_ui(self):
        """Initialize all UI components"""
        # Create central widget with horizontal layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Create sidebar
        self.create_sidebar()
        
        # Create content area
        self.content_area = QWidget()
        self.content_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)
        
        # Create stacked widget for screens
        self.stacked_widget = QStackedWidget()
        self.content_layout.addWidget(self.stacked_widget)
        
        self.main_layout.addWidget(self.content_area)
        
        # Create screens
        self.create_screens()
        
        # Create menu bar (still useful for shortcuts)
        self.create_menu_bar()
        
        # Create status bar
        self.create_status_bar()
        
        # Show dashboard by default
        self.show_dashboard()
        
    def create_sidebar(self):
        """Create modern sidebar navigation"""
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setStyleSheet("""
            QFrame#sidebar {
                background-color: #2a2a2a;
                border-right: 1px solid #3a3a3a;
                min-width: 180px;
                max-width: 180px;
            }
        """)
        
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 16, 0, 16)
        sidebar_layout.setSpacing(4)
        
        # App title
        title_container = QWidget()
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(16, 0, 16, 16)
        title_layout.setSpacing(4)
        
        # App icon/title
        app_title = QLabel("HEATWAVE")
        app_title.setStyleSheet("""
            color: #ffffff;
            font-size: 16px;
            font-weight: 700;
            letter-spacing: 1px;
        """)
        title_layout.addWidget(app_title)
        
        # Subtitle
        app_subtitle = QLabel("Thailand")
        app_subtitle.setStyleSheet("""
            color: #a0a0a0;
            font-size: 10px;
            font-weight: 500;
        """)
        title_layout.addWidget(app_subtitle)
        
        sidebar_layout.addWidget(title_container)
        
        # Separator
        separator = QFrame()
        separator.setStyleSheet("background-color: #3a3a3a; max-height: 1px; min-height: 1px;")
        sidebar_layout.addWidget(separator)
        
        # Navigation buttons with geometric symbols
        self.nav_buttons = {}
        nav_items = [
            ("◈ Dashboard", self.show_dashboard, "Ctrl+D"),
            ("▣ Train", self.show_train, "Ctrl+T"),
            ("◉ Predict", self.show_predict, "Ctrl+P"),
            ("▣ Map", self.show_map, "Ctrl+M"),
            ("▧ Evaluation", self.show_eval, "Ctrl+R"),
            ("◈ Data", self.show_data, "Ctrl+A"),
            ("▤ Checkpoints", self.show_checkpoints, "Ctrl+C"),
            ("◫ Logs", self.show_logs, "Ctrl+L"),
        ]
        
        for name, callback, shortcut in nav_items:
            btn = QPushButton(f"  {name}")
            btn.setFlat(True)
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #a0a0a0;
                    border: none;
                    border-radius: 4px;
                    padding: 10px 12px;
                    text-align: left;
                    font-size: 12px;
                    font-weight: 500;
                    margin: 0 8px;
                }
                QPushButton:hover {
                    background-color: #3a3a3a;
                    color: #e0e0e0;
                }
                QPushButton:checked {
                    background-color: #0066cc;
                    color: #ffffff;
                }
            """)
            btn.clicked.connect(lambda checked, cb=callback, b=btn: self.on_nav_clicked(cb, b))
            self.nav_buttons[name] = btn
            sidebar_layout.addWidget(btn)
        
        sidebar_layout.addStretch()
        
        # Bottom section
        bottom_frame = QFrame()
        bottom_layout = QVBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(16, 16, 16, 0)
        
        version_label = QLabel("v1.0.0")
        version_label.setStyleSheet("""
            color: #606060;
            font-size: 10px;
            font-weight: 500;
        """)
        bottom_layout.addWidget(version_label)
        sidebar_layout.addWidget(bottom_frame)
        
        self.main_layout.addWidget(sidebar)
        
    def on_nav_clicked(self, callback, button):
        """Handle navigation button click"""
        # Uncheck all buttons
        for btn in self.nav_buttons.values():
            btn.setChecked(False)
        # Check clicked button
        button.setChecked(True)
        # Call the navigation function
        callback()
        
    def create_screens(self):
        """Create all application screens"""
        # Create screens with shared model manager
        self.dashboard_screen = DashboardScreen(self.model_manager)
        self.stacked_widget.addWidget(self.dashboard_screen)

        # Train Screen
        self.train_screen = TrainScreen(self.model_manager)
        self.stacked_widget.addWidget(self.train_screen)

        # Predict Screen
        self.predict_screen = PredictScreen(self.model_manager)
        self.stacked_widget.addWidget(self.predict_screen)

        # Map Screen
        self.map_screen = MapScreen(self.model_manager)
        self.stacked_widget.addWidget(self.map_screen)

        # Eval Screen
        self.eval_screen = EvalScreen(self.model_manager)
        self.stacked_widget.addWidget(self.eval_screen)

        # Data Screen
        self.data_screen = DataScreen(self.model_manager)
        self.stacked_widget.addWidget(self.data_screen)

        # Checkpoints Screen
        self.checkpoints_screen = CheckpointsScreen(self.model_manager)
        self.stacked_widget.addWidget(self.checkpoints_screen)

        # Logs Screen
        self.logs_screen = LogsScreen(self.model_manager)
        self.stacked_widget.addWidget(self.logs_screen)
        
    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu("&File")
        
        export_action = QAction("Export Report...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_report)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View Menu
        view_menu = menubar.addMenu("&View")
        
        dashboard_action = QAction("Dashboard", self)
        dashboard_action.setShortcut("Ctrl+D")
        dashboard_action.triggered.connect(self.show_dashboard)
        view_menu.addAction(dashboard_action)
        
        train_action = QAction("Train", self)
        train_action.setShortcut("Ctrl+T")
        train_action.triggered.connect(self.show_train)
        view_menu.addAction(train_action)
        
        predict_action = QAction("Predict", self)
        predict_action.setShortcut("Ctrl+P")
        predict_action.triggered.connect(self.show_predict)
        view_menu.addAction(predict_action)
        
        map_action = QAction("Map", self)
        map_action.setShortcut("Ctrl+M")
        map_action.triggered.connect(self.show_map)
        view_menu.addAction(map_action)
        
        eval_action = QAction("Evaluation", self)
        eval_action.setShortcut("Ctrl+R")
        eval_action.triggered.connect(self.show_eval)
        view_menu.addAction(eval_action)
        
        data_action = QAction("Data", self)
        data_action.setShortcut("Ctrl+A")
        data_action.triggered.connect(self.show_data)
        view_menu.addAction(data_action)
        
        checkpoints_action = QAction("Checkpoints", self)
        checkpoints_action.setShortcut("Ctrl+C")
        checkpoints_action.triggered.connect(self.show_checkpoints)
        view_menu.addAction(checkpoints_action)
        
        logs_action = QAction("Logs", self)
        logs_action.setShortcut("Ctrl+L")
        logs_action.triggered.connect(self.show_logs)
        view_menu.addAction(logs_action)
        
        # Tools Menu
        tools_menu = menubar.addMenu("&Tools")
        
        download_data_action = QAction("Download ERA5 Data", self)
        download_data_action.triggered.connect(self.download_data)
        tools_menu.addAction(download_data_action)
        
        check_accuracy_action = QAction("Check Model Accuracy", self)
        check_accuracy_action.triggered.connect(self.check_accuracy)
        tools_menu.addAction(check_accuracy_action)
        
        # Help Menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def create_status_bar(self):
        """Create application status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
    # Screen Navigation Methods
    def show_dashboard(self):
        """Show dashboard screen"""
        self.stacked_widget.setCurrentWidget(self.dashboard_screen)
        self.status_bar.showMessage("Dashboard")
        self.update_nav_button("Dashboard")
        
    def show_train(self):
        """Show training screen"""
        self.stacked_widget.setCurrentWidget(self.train_screen)
        self.status_bar.showMessage("Training")
        self.update_nav_button("Train")
        
    def show_predict(self):
        """Show prediction screen"""
        self.stacked_widget.setCurrentWidget(self.predict_screen)
        self.status_bar.showMessage("Prediction")
        self.update_nav_button("Predict")
        
    def show_map(self):
        """Show map screen"""
        self.stacked_widget.setCurrentWidget(self.map_screen)
        self.status_bar.showMessage("Map Visualization")
        self.update_nav_button("Map")
        
    def show_eval(self):
        """Show evaluation screen"""
        self.stacked_widget.setCurrentWidget(self.eval_screen)
        self.status_bar.showMessage("Model Evaluation")
        self.update_nav_button("Evaluation")
        
    def show_data(self):
        """Show data screen"""
        self.stacked_widget.setCurrentWidget(self.data_screen)
        self.status_bar.showMessage("Data Management")
        self.update_nav_button("Data")
        
    def show_checkpoints(self):
        """Show checkpoints screen"""
        self.stacked_widget.setCurrentWidget(self.checkpoints_screen)
        self.status_bar.showMessage("Model Checkpoints")
        self.update_nav_button("Checkpoints")
        
    def show_logs(self):
        """Show logs screen"""
        self.stacked_widget.setCurrentWidget(self.logs_screen)
        self.status_bar.showMessage("System Logs")
        self.update_nav_button("Logs")
        
    def update_nav_button(self, name):
        """Update navigation button state"""
        for btn_name, btn in self.nav_buttons.items():
            btn.setChecked(btn_name == name)
        
    # Action Methods
    def export_report(self):
        """Export report dialog"""
        self.status_bar.showMessage("Export Report - Feature coming soon")
        QMessageBox.information(self, "Export Report", "Report export feature will be implemented in a future update.")
        
    def download_data(self):
        """Download ERA5 data"""
        self.show_data()
        self.data_screen.start_download()
        
    def check_accuracy(self):
        """Check model accuracy"""
        self.show_eval()
        self.eval_screen.run_accuracy_check()
        
    def show_about(self):
        """Show about dialog"""
        about_text = """
        <h3>Heatwave Prediction System</h3>
        <p>Thailand Heatwave Forecasting Backend</p>
        <p>Version: 1.0.0</p>
        <p><b>Models:</b></p>
        <ul>
            <li>ConvLSTM for sequence-based spatial forecasting</li>
            <li>XGBoost/RandomForest for daily classification</li>
        </ul>
        <p><b>Data:</b> ERA5 + NASA POWER meteorological data</p>
        <hr>
        <p>Developed for Thai heatwave prediction research</p>
        """
        QMessageBox.about(self, "About Heatwave Prediction System", about_text)
        
    def closeEvent(self, event):
        """Handle window close event"""
        # Check if training is in progress
        if hasattr(self.train_screen, 'is_training') and self.train_screen.is_training:
            reply = QMessageBox.question(
                self, 'Confirm Exit',
                'Training is in progress. Are you sure you want to exit?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
        
        event.accept()
