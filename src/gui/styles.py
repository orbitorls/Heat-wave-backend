"""Qt Stylesheet for Heatwave GUI - Scientific Dashboard Theme"""

STYLESHEET = """
/* QMainWindow */
QMainWindow {
    background-color: #1a1a1a;
}

/* Menu Bar */
QMenuBar {
    background-color: #2a2a2a;
    color: #e0e0e0;
    border-bottom: 1px solid #3a3a3a;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 500;
}

QMenuBar::item {
    background-color: transparent;
    padding: 6px 12px;
    margin: 2px;
}

QMenuBar::item:selected {
    background-color: #4a4a4a;
    color: #ffffff;
}

QMenu {
    background-color: #2a2a2a;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    padding: 4px;
}

QMenu::item {
    padding: 8px 16px;
}

QMenu::item:selected {
    background-color: #4a4a4a;
    color: #ffffff;
}

QMenu::separator {
    height: 1px;
    background-color: #3a3a3a;
    margin: 4px 8px;
}

/* Status Bar */
QStatusBar {
    background-color: #2a2a2a;
    color: #a0a0a0;
    border-top: 1px solid #3a3a3a;
    font-size: 11px;
}

/* Push Buttons */
QPushButton {
    background-color: #3a3a3a;
    color: #e0e0e0;
    border: 1px solid #4a4a4a;
    border-radius: 4px;
    padding: 8px 16px;
    font-size: 12px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #4a4a4a;
    border-color: #5a5a5a;
    color: #ffffff;
}

QPushButton:pressed {
    background-color: #2a2a2a;
    border-color: #3a3a3a;
}

QPushButton:disabled {
    background-color: #2a2a2a;
    color: #606060;
    border-color: #3a3a3a;
}

/* Primary Action Buttons */
QPushButton#primaryButton {
    background-color: #0066cc;
    color: #ffffff;
    border-color: #0055aa;
    font-weight: 600;
}

QPushButton#primaryButton:hover {
    background-color: #0077ee;
    border-color: #0066cc;
}

QPushButton#primaryButton:pressed {
    background-color: #0055aa;
    border-color: #004499;
}

/* Group Box */
QGroupBox {
    background-color: #2a2a2a;
    color: #e0e0e0;
    font-weight: 600;
    font-size: 12px;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 12px;
    padding: 16px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 16px;
    padding: 0 8px;
    color: #0088ff;
}

/* Labels */
QLabel {
    color: #e0e0e0;
    font-size: 12px;
}

QLabel#titleLabel {
    color: #ffffff;
    font-size: 16px;
    font-weight: 600;
}

QLabel#subtitleLabel {
    color: #a0a0a0;
    font-size: 12px;
    font-weight: 400;
}

QLabel#metricValue {
    color: #0088ff;
    font-size: 20px;
    font-weight: 600;
}

QLabel#statusLabel {
    color: #00aa66;
    font-size: 11px;
    font-weight: 500;
}

QLabel#warningLabel {
    color: #ffaa00;
}

QLabel#errorLabel {
    color: #ff4444;
}

/* Line Edit */
QLineEdit {
    background-color: #1a1a1a;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 12px;
}

QLineEdit:focus {
    border-color: #0088ff;
    background-color: #2a2a2a;
}

QLineEdit:hover {
    border-color: #4a4a4a;
}

/* Text Edit */
QTextEdit, QPlainTextEdit {
    background-color: #1a1a1a;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 11px;
    padding: 8px;
}

QTextEdit:focus, QPlainTextEdit:focus {
    border-color: #0088ff;
}

/* Table Widget */
QTableWidget {
    background-color: #1a1a1a;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    gridline-color: #2a2a2a;
    selection-background-color: #0066cc;
    selection-color: #ffffff;
    alternate-background-color: #222222;
}

QTableWidget::item {
    padding: 6px 10px;
    border: none;
}

QTableWidget::item:selected {
    background-color: #0066cc;
}

QTableWidget::item:hover {
    background-color: #2a2a2a;
}

QHeaderView::section {
    background-color: #2a2a2a;
    color: #a0a0a0;
    border: none;
    border-bottom: 1px solid #3a3a3a;
    padding: 8px 10px;
    font-weight: 600;
    font-size: 11px;
}

/* List Widget */
QListWidget {
    background-color: #1a1a1a;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    outline: none;
    padding: 4px;
}

QListWidget::item {
    padding: 8px 10px;
    margin: 2px 4px;
}

QListWidget::item:selected {
    background-color: #0066cc;
    color: #ffffff;
}

QListWidget::item:hover {
    background-color: #2a2a2a;
}

/* Combo Box */
QComboBox {
    background-color: #1a1a1a;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 12px;
    min-width: 80px;
}

QComboBox:hover {
    border-color: #4a4a4a;
}

QComboBox:focus {
    border-color: #0088ff;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid #a0a0a0;
}

QComboBox QAbstractItemView {
    background-color: #2a2a2a;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    selection-background-color: #0066cc;
}

/* Spin Box */
QSpinBox, QDoubleSpinBox {
    background-color: #1a1a1a;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 12px;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #0088ff;
}

QSpinBox::up-button, QDoubleSpinBox::up-button {
    background-color: #2a2a2a;
    border: none;
    border-left: 1px solid #3a3a3a;
}

QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #2a2a2a;
    border: none;
    border-left: 1px solid #3a3a3a;
}

/* Progress Bar */
QProgressBar {
    background-color: #1a1a1a;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    text-align: center;
    font-size: 11px;
    font-weight: 500;
}

QProgressBar::chunk {
    background-color: #0066cc;
    border-radius: 3px;
}

/* Scroll Bar */
QScrollBar:vertical {
    background-color: #2a2a2a;
    width: 8px;
    border-radius: 4px;
    margin: 2px;
}

QScrollBar::handle:vertical {
    background-color: #4a4a4a;
    border-radius: 4px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #5a5a5a;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background-color: #2a2a2a;
    height: 8px;
    border-radius: 4px;
    margin: 2px;
}

QScrollBar::handle:horizontal {
    background-color: #4a4a4a;
    border-radius: 4px;
    min-width: 20px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #5a5a5a;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

/* Tab Widget */
QTabWidget::pane {
    background-color: #1a1a1a;
    border: 1px solid #3a3a3a;
    border-radius: 0 4px 4px 4px;
    top: -1px;
}

QTabBar::tab {
    background-color: #2a2a2a;
    color: #a0a0a0;
    border: 1px solid #3a3a3a;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 8px 16px;
    margin-right: 2px;
    font-weight: 500;
}

QTabBar::tab:selected {
    background-color: #1a1a1a;
    color: #ffffff;
    border-bottom: 1px solid #1a1a1a;
}

QTabBar::tab:hover:!selected {
    background-color: #3a3a3a;
    color: #e0e0e0;
}

/* Stacked Widget */
QStackedWidget {
    background-color: #1a1a1a;
}

/* Tool Bar */
QToolBar {
    background-color: #2a2a2a;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    spacing: 4px;
    padding: 4px;
}

QToolBar::separator {
    background-color: #3a3a3a;
    width: 1px;
    margin: 4px;
}

QToolBar QToolButton {
    background-color: transparent;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    color: #e0e0e0;
}

QToolBar QToolButton:hover {
    background-color: #3a3a3a;
    color: #ffffff;
}

/* Slider */
QSlider::groove:horizontal {
    background-color: #2a2a2a;
    height: 4px;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background-color: #0066cc;
    border-radius: 2px;
}

QSlider::add-page:horizontal {
    background-color: #3a3a3a;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background-color: #e0e0e0;
    border: 2px solid #0066cc;
    width: 14px;
    height: 14px;
    border-radius: 7px;
    margin: -5px 0;
}

QSlider::handle:horizontal:hover {
    background-color: #ffffff;
    border-color: #0077ee;
}

/* Check Box */
QCheckBox {
    color: #e0e0e0;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 2px solid #3a3a3a;
    border-radius: 3px;
    background-color: #1a1a1a;
}

QCheckBox::indicator:hover {
    border-color: #4a4a4a;
}

QCheckBox::indicator:checked {
    background-color: #0066cc;
    border-color: #0066cc;
}

/* Radio Button */
QRadioButton {
    color: #e0e0e0;
    spacing: 8px;
}

QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border: 2px solid #3a3a3a;
    border-radius: 8px;
    background-color: #1a1a1a;
}

QRadioButton::indicator:hover {
    border-color: #4a4a4a;
}

QRadioButton::indicator:checked {
    background-color: #0066cc;
    border-color: #0066cc;
}

/* Splitter */
QSplitter::handle {
    background-color: #3a3a3a;
}

QSplitter::handle:horizontal {
    width: 2px;
}

QSplitter::handle:vertical {
    height: 2px;
}

/* Frame */
QFrame {
    border: 1px solid #3a3a3a;
    border-radius: 4px;
}

/* Tooltip */
QToolTip {
    background-color: #2a2a2a;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 11px;
}

/* Dialog */
QDialog {
    background-color: #1a1a1a;
}

/* Message Box */
QMessageBox {
    background-color: #1a1a1a;
}

QMessageBox QPushButton {
    min-width: 80px;
}
"""
