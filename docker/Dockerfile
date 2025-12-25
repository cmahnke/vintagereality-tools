FROM python:3.12-alpine AS builder

ENV BUILD_DIR=/tmp/build

RUN apk add --no-cache \
    git cmake make g++ gcc pkgconfig python3-dev libde265-dev libjpeg-turbo-dev libpng-dev x264-dev x265-dev openh264-dev aom-dev aom-static aom libsharpyuv dav1d-dev musl-dev gdk-pixbuf-dev && \
    mkdir $BUILD_DIR && \
    cd $BUILD_DIR && \
    git clone --recursive --depth 1 https://github.com/strukturag/libheif.git && \
    cd $BUILD_DIR/libheif && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Release -DENABLE_EXPERIMENTAL_FEATURES=ON && \
    make -j$(nproc) && \
    make install && \
    pip install --no-binary :all: --root-user-action ignore git+https://github.com/bigcat88/pillow_heif.git

FROM python:3.12-alpine

LABEL maintainer="cmahnke@gmail.com"

COPY --from=builder /usr/lib/libheif* /usr/lib/
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/

RUN apk --update upgrade && \
    apk add --no-cache gdk-pixbuf exiftool libde265 openh264 libjpeg-turbo \
    libpng x264-libs x265-libs aom aom-libs dav1d libsharpyuv libstdc++ && \
    pip install --upgrade pip && \
    pip install --root-user-action ignore pyexiftool numpy opencv-python-headless

RUN python3 -c "import libheif; print('libheif python module loaded successfully')" && \
    exiftool -ver

CMD ["python3"]