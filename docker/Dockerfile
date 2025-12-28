FROM python:3.11-slim AS builder

LABEL maintainer="cmahnke@gmail.com"
LABEL org.opencontainers.image.source="https://github.com/cmahnke/vintagereality-tools"

COPY requirements.txt /tmp/requirements.txt
COPY ./scripts/spatial.py /usr/local/bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    libimage-exiftool-perl libde265-0 libjpeg62-turbo libgl1 libstdc++6 libatomic1 libbrotli1 libpng16-16 libjxl0.11 libglib2.0-0 \
    libx264-164 libx265-215 libaom3 libdav1d7 libopenh264-8 libhwy1t64 libsharpyuv0 && \
    echo "deb http://deb.debian.org/debian unstable main" >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends libheif1 && \
    pip install --root-user-action ignore --upgrade pip setuptools wheel && \
    pip install --root-user-action ignore jxlpy && \
    pip install --root-user-action ignore -r /tmp/requirements.txt && \
    chmod +x /usr/local/bin/*.py && \
    python3 -c "import pillow_heif; print('pillow_heif python module loaded successfully')" && \
    python3 -c "import jxlpy; print('jxlpy python module loaded successfully')" && \
    python3 -c "import torch; print('PyTorch python module loaded successfully')" && \
    python3 -c "import cv2; print('OpenCV python module loaded successfully')" && \
    rm -rf /root/.cache /tmp/requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


CMD ["/usr/local/bin/spatial.py"]
