FROM mambaorg/micromamba:1.5.6-jammy

# Switch to root to install missing X11/GL system dependencies for PyQt5 on Linux
USER root
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1-mesa-glx \
    libegl1-mesa \
    libxrandr2 \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    libxkbcommon-x11-0 \
    libx11-xcb1 \
    libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

# Switch back to the micromamba user
USER micromamba

# Use conda-forge to install Python, PyQt5, Cartopy, and other dependencies reliably
RUN micromamba install -y -n base -c conda-forge \
    python=3.10 \
    pyqt=5 \
    numpy \
    scipy \
    matplotlib \
    cartopy \
    requests \
    beautifulsoup4 \
    && micromamba clean --all --yes

# Set working directory
WORKDIR /app

# Ensure standard entrypoint invokes the micromamba environment
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]

# Default command to run
CMD ["python", "sst_viewer.py"]
