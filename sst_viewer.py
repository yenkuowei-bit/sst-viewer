import sys
import os
import re
import datetime
import requests
from bs4 import BeautifulSoup
import numpy as np

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QComboBox, QCheckBox, QLabel, QProgressBar)
from PyQt5.QtCore import QThread, pyqtSignal, Qt

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import json

# Configure Matplotlib fonts to support Chinese display in Windows/macOS/Linux
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang HK', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

BASE_URL = 'https://www.data.jma.go.jp/goos/data/pub/JMA-product/him_sst_pac_D/'


def get_latest_files():
    """Scrape the directory index to find the 10 latest dataset files."""
    resp = requests.get(BASE_URL, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, 'html.parser')
    
    # 1. Find all year directories (e.g., "2026/")
    years = []
    for a in soup.find_all('a'):
        href = a.get('href')
        if href and re.match(r'^\d{4}/?$', href):
            years.append(href.strip('/'))
    # Sort descending to get latest years first
    years.sort(reverse=True)
    
    file_links = []
    # 2. Iterate descending years to gather the latest 10 files
    for y in years:
        y_url = f"{BASE_URL}{y}/"
        r = requests.get(y_url, timeout=10)
        r.raise_for_status()
        s = BeautifulSoup(r.content, 'html.parser')
        
        # Extrapolate .txt links
        links = [a.get('href') for a in s.find_all('a') if a.get('href') and a.get('href').endswith('.txt')]
        links.sort(reverse=True) # Ensure latest dates are first
        
        for link in links:
            file_links.append(y_url + link)
            if len(file_links) >= 10:
                break
        
        if len(file_links) >= 10:
            break
            
    return file_links


class DataLoaderThread(QThread):
    """Background thread to handle network fetching and heavy data parsing."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict) # Signals dict of filename -> (header_str, data_array)
    error = pyqtSignal(str)

    def run(self):
        try:
            self.progress.emit(10, "Fetching file list from JMA...")
            file_links = get_latest_files()
            
            # Ensure local cache directory exists
            os.makedirs('./data', exist_ok=True)
            saved_files = []
            
            # Download missing files
            for i, url in enumerate(file_links):
                fname = url.split('/')[-1]
                fpath = os.path.join('./data', fname)
                saved_files.append(fname)
                
                if not os.path.exists(fpath):
                    self.progress.emit(20 + int(60 * i / 10), f"Downloading {fname}...")
                    r = requests.get(url, timeout=15)
                    r.raise_for_status()
                    with open(fpath, 'wb') as f:
                        f.write(r.content)
            
            # Auto-Cleanup: Delete standard .txt files older than 10 days
            self.progress.emit(85, "Optimizing cache...")
            today = datetime.datetime.now()
            for f in os.listdir('./data'):
                if f.endswith('.txt'):
                    match = re.search(r'him_sst_pac_D(\d{8})\.txt', f)
                    if match:
                        f_date = datetime.datetime.strptime(match.group(1), '%Y%m%d')
                        # If file is older than 10 days and not part of the current latest batch
                        if (today - f_date).days > 10 and f not in saved_files:
                            try:
                                os.remove(os.path.join('./data', f))
                            except Exception:
                                pass # skip errors during deletion
            
            # Parse data locally utilizing high performance numpy techniques
            self.progress.emit(90, "Parsing SST records...")
            results = {}
            for fname in saved_files:
                fpath = os.path.join('./data', fname)
                with open(fpath, 'rb') as f:
                    header = f.readline().decode('ascii').strip()
                    # Strip newlines rapidly in binary form. Data body size MUST be 600 x 800 x 3 = 1,440,000 bytes.
                    raw = f.read().replace(b'\r', b'').replace(b'\n', b'')
                
                if len(raw) == 1440000:
                    # Treat sequence of 3 characters as integers directly from memory layout (no standard loops)
                    # S3 -> 3-byte string; fast conversion to native types. 
                    arr = np.frombuffer(raw, dtype='S3').astype(int).astype(float).reshape(600, 800)
                    # Use masking to clear uncalculated geographic regions seamlessly
                    arr[arr == 999] = np.nan # Map Land to NaN
                    arr[arr == 888] = np.nan # Map Sea Ice to NaN
                    arr *= 0.1               # Apply degree scaling factor
                    results[fname] = (header, arr)
                else:
                    print(f"Skipping {fname} - Malformed data body size.")
            
            self.progress.emit(100, "Processing complete.")
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JMA HIMSST Viewer")
        self.resize(1100, 750)
        
        self.data_dict = {}     
        self.current_filename = None
        self.colorbar = None   # Track existing colorbar to remove before redraw
        
        # ---- UI Setup ----
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 1. Left Control Panel
        left_panel = QVBoxLayout()
        
        self.combo_date = QComboBox()
        self.combo_date.currentIndexChanged.connect(self.on_date_changed)
        self.combo_date.setEnabled(False) # lock during download
        
        self.check_isotherms = QCheckBox("Show Isotherms")
        self.check_isotherms.setChecked(False)
        self.check_isotherms.stateChanged.connect(self.update_plot)
        self.check_isotherms.setEnabled(False)
        
        self.check_kuroshio = QCheckBox("顯示黑潮主流")
        self.check_kuroshio.setChecked(True)   # on by default
        self.check_kuroshio.stateChanged.connect(self.update_plot)
        self.check_kuroshio.setEnabled(False)
        self.check_kuroshio.setStyleSheet("color: #003399; font-weight: bold;")
        
        # Add a stylish header or logo placeholder
        title_lbl = QLabel("JMA SST Data Controller")
        title_lbl.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        left_panel.addWidget(title_lbl)
        left_panel.addWidget(QLabel("Select Data Map Date:"))
        left_panel.addWidget(self.combo_date)
        left_panel.addSpacing(15)
        left_panel.addWidget(self.check_isotherms)
        left_panel.addWidget(self.check_kuroshio)
        left_panel.addStretch()
        
        # 2. Right Plotting Area
        right_panel = QVBoxLayout()
        self.fig = plt.Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        right_panel.addWidget(self.toolbar)
        right_panel.addWidget(self.canvas)
        
        # Combine Layouts
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 4)
        
        # 3. Status Bar
        self.lbl_status = QLabel("Initializing application...")
        self.lbl_hover = QLabel("Hover for info")
        self.lbl_hover.setStyleSheet("font-weight: bold; color: blue;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        
        self.statusBar().addWidget(self.lbl_status)
        self.statusBar().addWidget(self.progress_bar)
        self.statusBar().addPermanentWidget(self.lbl_hover)
        
        # Core Plot Infrastructure
        self.ax = self.fig.add_subplot(1, 1, 1)
        
        # Preload high-resolution (10m) Natural Earth coastline for fine regional detail
        geojson_path = os.path.join('./data', 'ne_10m_coastline.geojson')
        if not os.path.exists(geojson_path):
            os.makedirs('./data', exist_ok=True)
            r = requests.get(
                'https://raw.githubusercontent.com/martynafford/natural-earth-geojson/master/10m/physical/ne_10m_coastline.json',
                verify=False, timeout=30)
            with open(geojson_path, 'wb') as f:
                f.write(r.content)
                
        with open(geojson_path, 'r', encoding='utf-8') as _f:
            self.world_map = json.load(_f)
        
        # Bind Mouse interactions
        self.canvas.mpl_connect('motion_notify_event', self.on_hover)
        
        # Start Automated Process
        self.start_download()
        
    def start_download(self):
        """Invoke data loading thread safely so UI thread stays responsive."""
        self.thread = DataLoaderThread()
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.on_data_loaded)
        self.thread.error.connect(self.on_error)
        self.thread.start()
        
    def update_progress(self, val, msg):
        self.progress_bar.setValue(val)
        self.lbl_status.setText(msg)
        
    def on_error(self, msg):
        self.lbl_status.setText(f"System Error: {msg}")
        self.progress_bar.hide()
        
    def on_data_loaded(self, results):
        self.data_dict = results
        self.progress_bar.hide()
        self.lbl_status.setText(f"Ready. Successfully loaded {len(results)} files.")
        
        self.combo_date.blockSignals(True)
        self.combo_date.clear()
        
        # Populate the combo box
        for fname in sorted(self.data_dict.keys(), reverse=True):
            header_str = self.data_dict[fname][0]
            # Try to format the raw header 'YYYY  MM  DD' cleanly 
            parts = header_str.split()
            if len(parts) >= 3:
                display_date = f"{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d}"
            else:
                display_date = fname
            self.combo_date.addItem(display_date, userData=fname)
            
        self.combo_date.blockSignals(False)
        self.combo_date.setEnabled(True)
        self.check_isotherms.setEnabled(True)
        self.check_kuroshio.setEnabled(True)
        
        if self.combo_date.count() > 0:
            self.combo_date.setCurrentIndex(0)
            self.on_date_changed()
            
    def on_date_changed(self):
        fname = self.combo_date.currentData()
        if fname:
            self.current_filename = fname
            self.update_plot()
            
    def update_plot(self):
        if not self.current_filename or self.current_filename not in self.data_dict:
            return
            
        header, data = self.data_dict[self.current_filename]
        
        # Remove existing colorbar before clearing axes to avoid stacking
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None
        
        # Clear original rendering targets
        self.ax.clear()
        
        # --- Display region: 120–130°E, 20–30°N ---
        LON_MIN, LON_MAX = 120.5, 125.0
        LAT_MIN, LAT_MAX = 21.5, 25.5
        
        # Full-domain extent of the raw data (needed for imshow alignment)
        extent = [100.0, 180.0, 0.0, 60.0]
        
        # Base rendering mapping
        img = self.ax.imshow(data, 
                             extent=extent, 
                             origin='upper', 
                             cmap='jet',
                             vmin=25.0,
                             vmax=28.0)
        
        # Crop the viewed area to the requested region
        self.ax.set_xlim(LON_MIN, LON_MAX)
        self.ax.set_ylim(LAT_MIN, LAT_MAX)
        self.ax.set_xlabel('Longitude (°E)')
        self.ax.set_ylabel('Latitude (°N)')
        parts = header.split()
        if len(parts) >= 3:
            title_date = f"{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d}"
        else:
            title_date = header
        self.ax.set_title(f"JMA HIMSST  —  {title_date}  |  {LON_MIN}°–{LON_MAX}°E / {LAT_MIN}°–{LAT_MAX}°N")
                             
        # Coastline overlay: draw GeoJSON LineString/MultiLineString features
        # Uses built-in json only — no geopandas/fiona/GDAL needed (PyInstaller-safe)
        for feature in self.world_map.get('features', []):
            geom = feature.get('geometry', {})
            gtype = geom.get('type', '')
            coords_list = geom.get('coordinates', [])
            if gtype == 'LineString':
                coords_list = [coords_list]
            elif gtype == 'MultiLineString':
                pass  # already a list of rings
            else:
                continue
            for ring in coords_list:
                xs = [c[0] for c in ring]
                ys = [c[1] for c in ring]
                self.ax.plot(xs, ys, color='black', linewidth=0.8,
                             transform=self.ax.transData, zorder=3)
        self.ax.set_xlim(LON_MIN, LON_MAX)
        self.ax.set_ylim(LAT_MIN, LAT_MAX)
        
        # Optional Features: Isotherms Contour Mask
        if self.check_isotherms.isChecked():
            lons = np.linspace(100.05, 179.95, 800)
            lats = np.linspace(59.95, 0.05, 600)
            X, Y = np.meshgrid(lons, lats)
            cs = self.ax.contour(X, Y, data, colors='black', linewidths=0.5)
            self.ax.clabel(cs, inline=True, fontsize=7, fmt='%.0f°C')
            
        # Kuroshio main path + edge — computed dynamically from SST data
        if self.check_kuroshio.isChecked():
            # Grid coordinates for this dataset
            lons_full = np.linspace(100.05, 179.95, 800)
            lats_full = np.linspace(59.95, 0.05, 600)  # descending (origin='upper')

            # Clip to current view region
            col0 = max(0, int((LON_MIN - 100.05) / 0.1))
            col1 = min(799, int((LON_MAX - 100.05) / 0.1))
            row0 = max(0, int((59.95 - LAT_MAX) / 0.1))
            row1 = min(599, int((59.95 - LAT_MIN) / 0.1))

            r_lons = lons_full[col0:col1 + 1]
            r_lats = lats_full[row0:row1 + 1]   # descending: LAT_MAX → LAT_MIN
            region  = data[row0:row1 + 1, col0:col1 + 1]

            # Warm-core temperature threshold (top 40% of valid SST in region)
            core_thresh = np.nanpercentile(region, 60)

            # Main axis: longitude of max SST per latitude row (warm-core ridge)
            axis_lons, axis_lats = [], []
            for i, lat in enumerate(r_lats):
                row_data = region[i, :]
                if np.all(np.isnan(row_data)):
                    continue
                peak_idx = np.nanargmax(row_data)
                if row_data[peak_idx] >= core_thresh:
                    axis_lons.append(r_lons[peak_idx])
                    axis_lats.append(lat)

            # Reverse so order is S to N (matches Kuroshio flow direction)
            axis_lons = axis_lons[::-1]
            axis_lats = axis_lats[::-1]

            if len(axis_lons) >= 5:
                # Smooth with moving average to remove jaggedness
                k = 7
                if len(axis_lons) > k:
                    sm_lon = np.convolve(axis_lons, np.ones(k) / k, mode='valid')
                    sm_lat = np.convolve(axis_lats, np.ones(k) / k, mode='valid')
                else:
                    sm_lon = np.array(axis_lons)
                    sm_lat = np.array(axis_lats)

                # Draw main axis line
                self.ax.plot(sm_lon, sm_lat,
                             color='#001689', linewidth=2.5, linestyle='-',
                             zorder=5, solid_capstyle='round', label='黑潮主流軸')

                # Directional arrows (4 evenly spaced)
                n = len(sm_lon)
                step = max(1, n // 4)
                for i in range(step // 2, n - 1, step):
                    dx = sm_lon[min(i + 1, n - 1)] - sm_lon[max(i - 1, 0)]
                    dy = sm_lat[min(i + 1, n - 1)] - sm_lat[max(i - 1, 0)]
                    norm = max(np.hypot(dx, dy), 1e-9)
                    scale = 0.08
                    self.ax.annotate('',
                        xy=(sm_lon[i] + dx / norm * scale,
                            sm_lat[i] + dy / norm * scale),
                        xytext=(sm_lon[i] - dx / norm * scale,
                                sm_lat[i] - dy / norm * scale),
                        arrowprops=dict(arrowstyle='->', color='#001689', lw=2.0),
                        zorder=6)

                # Label near midpoint
                mid = n // 2
                self.ax.text(sm_lon[mid] + 0.1, sm_lat[mid],
                             '黑潮主流\nKuroshio',
                             fontsize=9, fontweight='bold',
                             color='#001689', va='center', ha='left', zorder=7,
                             bbox=dict(facecolor='white', alpha=0.75,
                                       edgecolor='#001689', boxstyle='round,pad=0.3'))

            # Edge / thermal front: SST contour at warm-core threshold
            # This isoline marks where Kuroshio warm water meets cooler ambient water
            X_r, Y_r = np.meshgrid(r_lons, r_lats)
            edge_cs = self.ax.contour(X_r, Y_r, region,
                                      levels=[core_thresh],
                                      colors=['#00aaff'],
                                      linewidths=[1.2],
                                      linestyles=['--'],
                                      zorder=4, alpha=0.85)
            self.ax.clabel(edge_cs, fmt=f'{core_thresh:.1f}°C (邊緣)',
                           fontsize=7, inline=True, colors=['#0055cc'])


        # Colorbar (色階表)
        self.colorbar = self.fig.colorbar(img, ax=self.ax, orientation='vertical',
                                          pad=0.02, fraction=0.03,
                                          label='Sea Surface Temperature (°C)')
        
        # UI Custom Watermark
        self.ax.text(0.97, 0.03, "農業部水產試驗所 漁海況研究小組", 
                     transform=self.ax.transAxes, 
                     fontsize=10, 
                     fontweight='bold',
                     color='black', 
                     ha='right', 
                     va='bottom',
                     bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.4'))
        
        # Grid lines
        self.ax.grid(True, linestyle='--', color='gray', alpha=0.5, linewidth=0.5)
                     
        # Flush the buffer updates gracefully onto screen 
        self.canvas.draw()
        
    def on_hover(self, event):
        """Matplotlib callback routine mapping UI interaction mapping space coordinates against source array structures"""
        if event.inaxes != self.ax:
            return
            
        lon, lat = event.xdata, event.ydata
        if lon is None or lat is None:
            return
            
        # Logical data boundaries 
        if not (100.0 <= lon <= 180.0 and 0.0 <= lat <= 60.0):
            self.lbl_hover.setText("Out of map boundaries")
            return
            
        # O(1) mathematical index offset discovery
        x_idx = int((lon - 100.0) / 0.1)
        y_idx = int((60.0 - lat) / 0.1)
        
        x_idx = max(0, min(799, x_idx))
        y_idx = max(0, min(599, y_idx))
        
        if self.current_filename and self.current_filename in self.data_dict:
            _, data = self.data_dict[self.current_filename]
            temp_val = data[y_idx, x_idx]
            
            if np.isnan(temp_val):
                temp_str = "Land / No Data"
            else:
                temp_str = f"{temp_val:.1f} °C"
                
            self.lbl_hover.setText(f"Lon: {lon:.2f} E / Lat: {lat:.2f} N  |  SST: {temp_str} ")


if __name__ == '__main__':
    # Guarantee consistent window layouts during High DPI scaling profiles
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
