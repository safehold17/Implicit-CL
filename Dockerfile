# =============================================================================
# Nocturne Simulator — PyTorch Release 24.01 (Ubuntu 22.04 / Python 3.10)
# =============================================================================
FROM nvcr.io/nvidia/pytorch:24.01-py3

# ---------------------------------------------------------------------------
# 1. System dependencies
#    - libsfml-dev   : SFML for Nocturne's drawing / visualization layer
#    - pybind11-dev  : C++ → Python bindings
#    - cmake / ninja : build system
#    - ffmpeg / libgl: rendering / OpenGL support for headless envs
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        swig \
        git \
        wget \
        curl \
        pkg-config \
        # SFML runtime + dev headers
        libsfml-dev \
        # pybind11
        pybind11-dev \
        # OpenGL / EGL (headless rendering)
        libgl1-mesa-glx \
        libgl1-mesa-dev \
        libegl1-mesa-dev \
        libgles2-mesa-dev \
        # X11 extension headers required by GLFW during Nocturne build
        libxrandr-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxi-dev \
        # X virtual framebuffer (needed by SFML in headless mode)
        xvfb \
        # misc utilities
        ffmpeg \
        libjpeg-dev \
        libpng-dev \
        zip \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. Make sure we are using Python 3.10 from the base image
#    PyTorch 24.01 ships Python 3.10 — confirm and pin pip
# ---------------------------------------------------------------------------
RUN python3 --version && \
    python3 -m pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir numpy==1.23.0 scipy==1.12.0

RUN pip install --no-cache-dir scikit-build
# ---------------------------------------------------------------------------
# 3. Remove the OpenCV packages that come with the base image
#    Python dependencies via requirements file
#    Install everything into the system (local) Python — no virtualenv/conda.
# ---------------------------------------------------------------------------
RUN python -m pip uninstall -y \
        opencv \
        opencv-python \
        opencv-python-headless \
        opencv-contrib-python \
        opencv-contrib-python-headless || true
RUN rm -rf /usr/local/lib/python3.10/dist-packages/cv2 \
    /usr/local/lib/python3.10/dist-packages/cv2-* \
    /usr/local/lib/python3.10/dist-packages/opencv*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-build-isolation -r /tmp/requirements.txt

# ---------------------------------------------------------------------------
# 4. Clone, build, and install Nocturne — then clean up the source tree
#    Clone into /tmp/nocturne, do a standard (non-editable) install into the
#    system Python, then remove the source so it doesn't bloat the image.
#    Change the branch/tag/commit SHA as needed.
# ---------------------------------------------------------------------------
RUN pip uninstall -y cmake || true

RUN git clone https://github.com/montrealrobotics/ctrl-sim.git /opt/ctrl-sim && \
    cd /opt/ctrl-sim && \
    python setup.py develop
# ---------------------------------------------------------------------------
# 5. Environment variables
#    DISPLAY is required by SFML; point it at the Xvfb server started below.
# ---------------------------------------------------------------------------
ENV DISPLAY=:0

# ---------------------------------------------------------------------------
# 6. Entrypoint helper
#    Start a virtual framebuffer and then run the user's command.
#    Usage examples:
#      docker run --gpus all nocturne python scripts/run_example.py
#      docker run --gpus all -it nocturne bash
# ---------------------------------------------------------------------------
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
