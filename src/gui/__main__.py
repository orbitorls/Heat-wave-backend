"""Entry point for Heatwave GUI Application"""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QLocale, QStandardPaths
from .main_window import HeatwaveMainWindow


def main():
    """Main entry point for the GUI application"""
    # Force Arabic numerals (Western numerals) instead of Thai numerals
    QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = HeatwaveMainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
