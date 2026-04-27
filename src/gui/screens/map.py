"""Map Screen - Interactive heatmap visualization"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QSlider, QToolBar, QSizePolicy, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QAction
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class MapScreen(QWidget):
    """Map screen with interactive Thailand heatmap"""

    def __init__(self, model_manager=None):
        super().__init__()
        self.model_manager = model_manager
        self.init_ui()
        self.generate_sample_heatmap()
        
    def init_ui(self):
        """Initialize map UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Title with geometric symbol
        title_label = QLabel("▣ THAILAND HEATMAP")
        title_label.setObjectName("titleLabel")
        layout.addWidget(title_label)
        
        # Map Canvas with Navigation Toolbar
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(8)
        
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Add matplotlib navigation toolbar for proper zoom/pan
        self.nav_toolbar = NavigationToolbar(self.canvas, canvas_container)
        canvas_layout.addWidget(self.nav_toolbar)
        
        self.ax = self.figure.add_subplot(111)
        canvas_layout.addWidget(self.canvas)
        
        layout.addWidget(canvas_container, 1)
        
        # Controls
        controls_container = QWidget()
        controls_layout = QHBoxLayout(controls_container)
        controls_layout.setSpacing(12)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        
        controls_layout.addWidget(QLabel("THRESHOLD:"))
        
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(30, 45)
        self.temp_slider.setValue(38)
        self.temp_slider.valueChanged.connect(self.update_threshold)
        controls_layout.addWidget(self.temp_slider)
        
        self.temp_label = QLabel("38°C")
        controls_layout.addWidget(self.temp_label)
        
        refresh_btn = QPushButton("REFRESH")
        refresh_btn.clicked.connect(self.generate_sample_heatmap)
        controls_layout.addWidget(refresh_btn)
        
        controls_layout.addStretch()
        layout.addWidget(controls_container)
        
        # Legend
        legend_container = QWidget()
        legend_layout = QHBoxLayout(legend_container)
        legend_layout.setSpacing(16)
        legend_layout.setContentsMargins(0, 0, 0, 0)
        
        legend_items = [
            ('#0066cc', '< 35°C', 'Normal'),
            ('#0099ff', '35-37°C', 'Warm'),
            ('#00cc66', '37-39°C', 'Hot'),
            ('#ffcc00', '39-41°C', 'Very Hot'),
            ('#ff6600', '41-43°C', 'Extreme'),
            ('#cc0000', '> 43°C', 'Danger'),
        ]
        
        for color, range_text, label_text in legend_items:
            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)
            item_layout.setSpacing(6)
            item_layout.setContentsMargins(0, 0, 0, 0)
            
            color_box = QFrame()
            color_box.setFixedSize(16, 16)
            color_box.setStyleSheet(f"background-color: {color}; border: 1px solid #505050;")
            item_layout.addWidget(color_box)
            
            text_label = QLabel(f"{range_text}")
            text_label.setStyleSheet("color: #a0a0a0; font-size: 10px;")
            item_layout.addWidget(text_label)
            
            legend_layout.addWidget(item_widget)
        
        legend_layout.addStretch()
        layout.addWidget(legend_container)
        
    def generate_sample_heatmap(self):
        """Generate realistic Thailand temperature heatmap"""
        try:
            n = 200
            lat = np.linspace(5, 21, n)
            lon = np.linspace(97, 106, n)
            lon_grid, lat_grid = np.meshgrid(lon, lat)

            # Base temperature field: hotter in center-east (Khorat plateau), cooler in north and south
            # Bangkok area ~100.5, 13.7 should be hot
            base_temp = 34.0 + 2.0 * np.sin(np.radians(lat_grid * 3)) + 1.5 * np.cos(np.radians(lon_grid * 2))

            # Add localized heat regions
            # Bangkok heat island
            bkk_dist = np.sqrt((lon_grid - 100.5)**2 + (lat_grid - 13.7)**2)
            base_temp += 2.0 * np.exp(-bkk_dist**2 / 0.5)

            # Northeast heat (Khorat)
            korat_dist = np.sqrt((lon_grid - 102.5)**2 + (lat_grid - 15.5)**2)
            base_temp += 1.5 * np.exp(-korat_dist**2 / 1.0)

            # North cool (Chiang Mai area)
            north_dist = np.sqrt((lon_grid - 99.0)**2 + (lat_grid - 18.8)**2)
            base_temp -= 1.5 * np.exp(-north_dist**2 / 0.4)

            # South peninsula moderate
            south_dist = np.sqrt((lon_grid - 99.5)**2 + (lat_grid - 7.5)**2)
            base_temp -= 1.0 * np.exp(-south_dist**2 / 0.3)

            # No random noise - completely smooth analytical field for clean visualization
            temp = base_temp

            # Clip to realistic range
            temp = np.clip(temp, 28, 46)

            self.temp_data = temp
            self.lat_grid = lat_grid
            self.lon_grid = lon_grid
            self.plot_heatmap(temp)
        except Exception as e:
            print(f"Error generating heatmap: {e}")
            import traceback
            traceback.print_exc()
        
    def plot_heatmap(self, temp):
        """Plot heatmap with given temperature data"""
        try:
            self.ax.clear()

            # Custom colormap
            cmap = self.get_custom_colormap()

            # Plot heatmap with smooth interpolation
            im = self.ax.imshow(temp, cmap=cmap, vmin=30, vmax=45,
                              extent=[97, 106, 5, 21], origin='lower', aspect='auto',
                              interpolation='bicubic', alpha=0.95)

            # Add Thailand borders
            self.add_thailand_borders()

            self.ax.set_xlabel('LONGITUDE (°E)', fontsize=9, fontweight='600', color='#707070')
            self.ax.set_ylabel('LATITUDE (°N)', fontsize=9, fontweight='600', color='#707070')
            self.ax.set_title('THAILAND HEATWAVE FORECAST', fontsize=11, fontweight='600', color='#e0e0e0', pad=15)

            # Set tick colors
            self.ax.tick_params(axis='both', colors='#707070', labelsize=8)

            # Set spine colors
            for spine in self.ax.spines.values():
                spine.set_color('#353535')

            # Set grid
            self.ax.grid(True, alpha=0.15, linewidth=0.5, color='#505050')

            # Add colorbar
            if hasattr(self, 'cbar'):
                self.cbar.remove()
            self.cbar = self.figure.colorbar(im, ax=self.ax, shrink=0.85, pad=0.02)
            self.cbar.set_label('TEMPERATURE (°C)', fontsize=9, fontweight='600', color='#707070')
            self.cbar.ax.tick_params(labelsize=8, colors='#707070')
            for spine in self.cbar.ax.spines.values():
                spine.set_color('#353535')

            # Set background
            self.ax.set_facecolor('#1a1a1a')
            self.figure.patch.set_facecolor('#1a1a1a')

            self.figure.tight_layout()
            self.canvas.draw()
        except Exception as e:
            print(f"Error plotting heatmap: {e}")
            import traceback
            traceback.print_exc()
        
    def add_thailand_borders(self):
        """Add accurate Thailand borders"""
        try:
            # Thailand more accurate coordinates
            thailand_coords = [
            # Southern peninsula
            (99.0, 6.5),
            (99.5, 7.0),
            (100.0, 7.5),
            (100.3, 8.0),
            (100.5, 8.5),
            # East coast
            (100.8, 9.0),
            (101.0, 9.5),
            (101.2, 10.0),
            (101.3, 10.5),
            (101.2, 11.0),
            (101.0, 11.5),
            # Gulf of Thailand coast
            (100.8, 12.0),
            (100.5, 12.5),
            (100.3, 13.0),
            (100.0, 13.5),
            # Central region
            (99.8, 14.0),
            (99.5, 14.5),
            (99.3, 15.0),
            (99.0, 15.5),
            # Northern region
            (98.8, 16.0),
            (98.5, 16.5),
            (98.3, 17.0),
            (98.2, 17.5),
            (98.3, 18.0),
            # Northeastern border
            (98.5, 18.5),
            (99.0, 19.0),
            (99.5, 19.5),
            (100.0, 20.0),
            (100.5, 20.3),
            # Northern border
            (101.0, 20.5),
            (101.5, 20.3),
            (102.0, 20.0),
            # Eastern border
            (102.5, 19.5),
            (103.0, 19.0),
            (103.5, 18.5),
            (104.0, 18.0),
            (104.5, 17.5),
            (105.0, 17.0),
            (105.3, 16.5),
            # Southeastern coast
            (105.0, 16.0),
            (104.5, 15.5),
            (104.0, 15.0),
            (103.5, 14.5),
            (103.0, 14.0),
            (102.5, 13.5),
            (102.0, 13.0),
            (101.5, 12.5),
            (101.2, 12.0),
            (101.0, 11.5),
            # Back to southern
            (100.8, 11.0),
            (100.5, 10.5),
            (100.3, 10.0),
            (100.0, 9.5),
            (99.8, 9.0),
            (99.5, 8.5),
            (99.3, 8.0),
            (99.0, 7.5),
            (98.8, 7.0),
            (98.5, 6.5),
            (99.0, 6.5)   # Close the loop
            ]

            lons, lats = zip(*thailand_coords)

            # Fill Thailand interior with very subtle overlay for clear country boundary
            self.ax.fill(lons, lats, color='#ffffff', alpha=0.03)
            # Thick border with slight glow effect
            self.ax.plot(lons, lats, color='#ffffff', linewidth=2.5, alpha=0.9)
            self.ax.plot(lons, lats, color='#ffffff', linewidth=5, alpha=0.15)

            # Add major cities as clearly visible markers
            cities = [
                (100.5, 13.7, 'Bangkok'),
                (100.9, 12.9, 'Pattaya'),
                (98.9, 18.8, 'Chiang Mai'),
                (98.3, 8.1, 'Phuket'),
            ]
            for lon, lat, name in cities:
                # White circle with dark outline for high visibility on any background
                self.ax.plot(lon, lat, 'o', color='#1a1a1a', markersize=10, alpha=1.0)
                self.ax.plot(lon, lat, 'o', color='#ffffff', markersize=7, alpha=1.0)
                # Label with dark shadow for readability (simplified without bbox to avoid crashes)
                self.ax.text(lon + 0.25, lat + 0.15, name, color='#ffffff', fontsize=8,
                            fontweight='bold', alpha=0.9)
        except Exception as e:
            print(f"Error adding Thailand borders: {e}")
            import traceback
            traceback.print_exc()

    def get_custom_colormap(self):
        """Get custom colormap for temperature - scientific style"""
        import matplotlib.colors as mcolors
        # Scientific color palette with clear progression
        colors = [
            '#0066cc',  # < 35°C (Normal - Blue)
            '#0099ff',  # 35-37°C (Warm - Light Blue)
            '#00cc66',  # 37-39°C (Hot - Green)
            '#ffcc00',  # 39-41°C (Very Hot - Yellow)
            '#ff6600',  # 41-43°C (Extreme - Orange)
            '#cc0000'   # > 43°C (Danger - Red)
        ]
        n_bins = 100
        cmap = mcolors.LinearSegmentedColormap.from_list('temp', colors, N=n_bins)
        return cmap
        
    def zoom_in(self):
        """Zoom in on the map"""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        x_range = (xlim[1] - xlim[0]) / 2 / 1.5
        y_range = (ylim[1] - ylim[0]) / 2 / 1.5
        
        self.ax.set_xlim(x_center - x_range, x_center + x_range)
        self.ax.set_ylim(y_center - y_range, y_center + y_range)
        self.canvas.draw()
        
    def zoom_out(self):
        """Zoom out on the map"""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        x_range = (xlim[1] - xlim[0]) * 1.5
        y_range = (ylim[1] - ylim[0]) * 1.5
        
        self.ax.set_xlim(x_center - x_range, x_center + x_range)
        self.ax.set_ylim(y_center - y_range, y_center + y_range)
        self.canvas.draw()
        
    def reset_view(self):
        """Reset map view to default"""
        self.ax.set_xlim(97, 106)
        self.ax.set_ylim(5, 21)
        self.canvas.draw()
        
    def update_threshold(self, value):
        """Update temperature threshold display"""
        self.temp_label.setText(f"{value}°C")
