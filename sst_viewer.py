import os
import re
import datetime
import requests
from bs4 import BeautifulSoup
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — safe for Streamlit Cloud
import matplotlib.pyplot as plt
import streamlit as st

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JMA HIMSST Viewer",
    page_icon="🌊",
    layout="wide",
)

# ── Fonts / Chinese support ──────────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

BASE_URL = 'https://www.data.jma.go.jp/goos/data/pub/JMA-product/him_sst_pac_D/'

# ── Data helpers (cached so re-runs are fast) ────────────────────────────────

@st.cache_data(show_spinner=False)
def get_latest_file_urls() -> list[str]:
    """Scrape the JMA directory index and return the 10 latest .txt URLs."""
    resp = requests.get(BASE_URL, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, 'html.parser')

    years = []
    for a in soup.find_all('a'):
        href = a.get('href', '')
        if re.match(r'^\d{4}/?$', href):
            years.append(href.strip('/'))
    years.sort(reverse=True)

    file_links: list[str] = []
    for y in years:
        y_url = f"{BASE_URL}{y}/"
        r = requests.get(y_url, timeout=15)
        r.raise_for_status()
        s = BeautifulSoup(r.content, 'html.parser')
        links = sorted(
            [a.get('href') for a in s.find_all('a') if a.get('href', '').endswith('.txt')],
            reverse=True,
        )
        for link in links:
            file_links.append(y_url + link)
            if len(file_links) >= 10:
                break
        if len(file_links) >= 10:
            break

    return file_links


@st.cache_data(show_spinner=False)
def fetch_and_parse_all() -> dict:
    """Download + parse up to 10 latest SST files; returns {filename: (header, array)}."""
    file_links = get_latest_file_urls()
    os.makedirs('./data', exist_ok=True)
    results: dict = {}

    for url in file_links:
        fname = url.split('/')[-1]
        fpath = os.path.join('./data', fname)

        if not os.path.exists(fpath):
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(fpath, 'wb') as f:
                f.write(r.content)

        with open(fpath, 'rb') as f:
            header = f.readline().decode('ascii').strip()
            raw = f.read().replace(b'\r', b'').replace(b'\n', b'')

        if len(raw) == 1_440_000:
            arr = np.frombuffer(raw, dtype='S3').astype(int).astype(float).reshape(600, 800)
            arr[arr == 999] = np.nan
            arr[arr == 888] = np.nan
            arr *= 0.1
            results[fname] = (header, arr)

    # Auto-clean files older than 10 days that are no longer in our batch
    today = datetime.datetime.now()
    for fn in os.listdir('./data'):
        if fn.endswith('.txt') and fn not in results:
            match = re.search(r'him_sst_pac_D(\d{8})\.txt', fn)
            if match:
                f_date = datetime.datetime.strptime(match.group(1), '%Y%m%d')
                if (today - f_date).days > 10:
                    try:
                        os.remove(os.path.join('./data', fn))
                    except Exception:
                        pass

    return results


@st.cache_data(show_spinner=False)
def load_coastline() -> dict:
    """Download (once) and return the Natural Earth 10 m coastline GeoJSON."""
    geojson_path = os.path.join('./data', 'ne_10m_coastline.geojson')
    os.makedirs('./data', exist_ok=True)
    if not os.path.exists(geojson_path):
        r = requests.get(
            'https://raw.githubusercontent.com/martynafford/natural-earth-geojson/'
            'master/10m/physical/ne_10m_coastline.json',
            verify=False, timeout=60,
        )
        with open(geojson_path, 'wb') as f:
            f.write(r.content)
    with open(geojson_path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


# ── Plotting ─────────────────────────────────────────────────────────────────

def make_figure(
    data: np.ndarray,
    header: str,
    world_map: dict,
    show_isotherms: bool,
    show_kuroshio: bool,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    vmin: float,
    vmax: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 7), tight_layout=True)

    extent = [100.0, 180.0, 0.0, 60.0]
    img = ax.imshow(data, extent=extent, origin='upper',
                    cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')

    parts = header.split()
    title_date = (
        f"{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d}"
        if len(parts) >= 3 else header
    )
    ax.set_title(
        f"JMA HIMSST  —  {title_date}  |  "
        f"{lon_min}°–{lon_max}°E / {lat_min}°–{lat_max}°N"
    )

    # ── Coastline ──────────────────────────────────────────────────────────
    for feature in world_map.get('features', []):
        geom = feature.get('geometry', {})
        gtype = geom.get('type', '')
        coords_list = geom.get('coordinates', [])
        if gtype == 'LineString':
            coords_list = [coords_list]
        elif gtype == 'MultiLineString':
            pass
        else:
            continue
        for ring in coords_list:
            xs = [c[0] for c in ring]
            ys = [c[1] for c in ring]
            ax.plot(xs, ys, color='black', linewidth=0.8, zorder=3)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    # ── Isotherms ──────────────────────────────────────────────────────────
    if show_isotherms:
        lons = np.linspace(100.05, 179.95, 800)
        lats = np.linspace(59.95, 0.05, 600)
        X, Y = np.meshgrid(lons, lats)
        cs = ax.contour(X, Y, data, colors='black', linewidths=0.5)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%.0f°C')

    # ── Kuroshio ───────────────────────────────────────────────────────────
    if show_kuroshio:
        lons_full = np.linspace(100.05, 179.95, 800)
        lats_full = np.linspace(59.95, 0.05, 600)

        col0 = max(0, int((lon_min - 100.05) / 0.1))
        col1 = min(799, int((lon_max - 100.05) / 0.1))
        row0 = max(0, int((59.95 - lat_max) / 0.1))
        row1 = min(599, int((59.95 - lat_min) / 0.1))

        r_lons = lons_full[col0:col1 + 1]
        r_lats = lats_full[row0:row1 + 1]
        region = data[row0:row1 + 1, col0:col1 + 1]

        core_thresh = np.nanpercentile(region, 60)

        axis_lons, axis_lats = [], []
        for i, lat in enumerate(r_lats):
            row_data = region[i, :]
            if np.all(np.isnan(row_data)):
                continue
            peak_idx = np.nanargmax(row_data)
            if row_data[peak_idx] >= core_thresh:
                axis_lons.append(r_lons[peak_idx])
                axis_lats.append(lat)

        axis_lons = axis_lons[::-1]
        axis_lats = axis_lats[::-1]

        if len(axis_lons) >= 5:
            k = 7
            if len(axis_lons) > k:
                sm_lon = np.convolve(axis_lons, np.ones(k) / k, mode='valid')
                sm_lat = np.convolve(axis_lats, np.ones(k) / k, mode='valid')
            else:
                sm_lon = np.array(axis_lons)
                sm_lat = np.array(axis_lats)

            ax.plot(sm_lon, sm_lat, color='#001689', linewidth=2.5,
                    linestyle='-', zorder=5, solid_capstyle='round',
                    label='黑潮主流軸')

            n = len(sm_lon)
            step = max(1, n // 4)
            for i in range(step // 2, n - 1, step):
                dx = sm_lon[min(i + 1, n - 1)] - sm_lon[max(i - 1, 0)]
                dy = sm_lat[min(i + 1, n - 1)] - sm_lat[max(i - 1, 0)]
                norm = max(np.hypot(dx, dy), 1e-9)
                scale = 0.08
                ax.annotate('',
                    xy=(sm_lon[i] + dx / norm * scale,
                        sm_lat[i] + dy / norm * scale),
                    xytext=(sm_lon[i] - dx / norm * scale,
                            sm_lat[i] - dy / norm * scale),
                    arrowprops=dict(arrowstyle='->', color='#001689', lw=2.0),
                    zorder=6)

            mid = n // 2
            ax.text(sm_lon[mid] + 0.1, sm_lat[mid],
                    'Kuroshio',
                    fontsize=9, fontweight='bold',
                    color='#001689', va='center', ha='left', zorder=7,
                    bbox=dict(facecolor='white', alpha=0.75,
                              edgecolor='#001689', boxstyle='round,pad=0.3'))

        X_r, Y_r = np.meshgrid(r_lons, r_lats)
        edge_cs = ax.contour(X_r, Y_r, region,
                             levels=[core_thresh],
                             colors=['#00aaff'],
                             linewidths=[1.2],
                             linestyles=['--'],
                             zorder=4, alpha=0.85)
        ax.clabel(edge_cs, fmt=f'{core_thresh:.1f}°C',
                  fontsize=7, inline=True, colors=['#0055cc'])

    # ── Colorbar ───────────────────────────────────────────────────────────
    fig.colorbar(img, ax=ax, orientation='vertical', pad=0.02, fraction=0.03,
                 label='Sea Surface Temperature (°C)')

    # ── Watermark ──────────────────────────────────────────────────────────
    ax.text(0.97, 0.03, "TFRI Fisheries Oceanography Lab",
            transform=ax.transAxes, fontsize=9, fontweight='bold',
            color='black', ha='right', va='bottom',
            bbox=dict(facecolor='white', alpha=0.85,
                      edgecolor='gray', boxstyle='round,pad=0.4'))

    ax.grid(True, linestyle='--', color='gray', alpha=0.5, linewidth=0.5)
    return fig


# ── Streamlit UI ─────────────────────────────────────────────────────────────

st.title("🌊 JMA HIMSST Viewer")
st.caption("Japan Meteorological Agency — High-resolution Sea Surface Temperature (Pacific)")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")

    st.subheader("📅 Data")
    with st.spinner("Loading JMA data…"):
        try:
            data_dict = fetch_and_parse_all()
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.stop()

    if not data_dict:
        st.error("No data files found.")
        st.stop()

    # Build display labels for the selectbox
    def label_for(fname: str) -> str:
        header, _ = data_dict[fname]
        parts = header.split()
        if len(parts) >= 3:
            return f"{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d}"
        return fname

    sorted_fnames = sorted(data_dict.keys(), reverse=True)
    labels = [label_for(f) for f in sorted_fnames]
    selected_label = st.selectbox("Select date", labels)
    selected_fname = sorted_fnames[labels.index(selected_label)]

    st.divider()
    st.subheader("🗺️ Region")
    col1, col2 = st.columns(2)
    with col1:
        lon_min = st.number_input("Lon min (°E)", value=120.5, step=0.5, format="%.1f")
        lat_min = st.number_input("Lat min (°N)", value=21.5, step=0.5, format="%.1f")
    with col2:
        lon_max = st.number_input("Lon max (°E)", value=125.0, step=0.5, format="%.1f")
        lat_max = st.number_input("Lat max (°N)", value=25.5, step=0.5, format="%.1f")

    st.divider()
    st.subheader("🌡️ Colour scale")
    col3, col4 = st.columns(2)
    with col3:
        vmin = st.number_input("Min (°C)", value=25.0, step=0.5, format="%.1f")
    with col4:
        vmax = st.number_input("Max (°C)", value=28.0, step=0.5, format="%.1f")

    st.divider()
    st.subheader("🔧 Overlays")
    show_isotherms = st.checkbox("Show Isotherms", value=False)
    show_kuroshio  = st.checkbox("Show Kuroshio Current", value=True)

# ── Main plot ─────────────────────────────────────────────────────────────────
with st.spinner("Rendering map…"):
    try:
        world_map = load_coastline()
    except Exception as e:
        st.warning(f"Coastline could not be loaded: {e}")
        world_map = {"features": []}

    header, data = data_dict[selected_fname]

    if lon_min >= lon_max or lat_min >= lat_max:
        st.error("Invalid region: min values must be less than max values.")
    else:
        fig = make_figure(
            data=data,
            header=header,
            world_map=world_map,
            show_isotherms=show_isotherms,
            show_kuroshio=show_kuroshio,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            vmin=vmin,
            vmax=vmax,
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Data source: [JMA/GOOS](https://www.data.jma.go.jp/goos/data/pub/JMA-product/him_sst_pac_D/)  "
    "· Coastline: Natural Earth 10 m  "
    "· 農業部水產試驗所 漁海況研究小組"
)
