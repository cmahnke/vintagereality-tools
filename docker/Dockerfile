FROM python:3.12-alpine AS builder

ENV BUILD_DIR=/tmp/build \
    GIT_URL="https://github.com/libjxl/libjxl.git" \
    DEFAULT_GIT_TAG="v0.11.1"

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
    cd .. && \
    if [ -z "$GIT_TAG" ] ; then \
        GIT_TAG=$DEFAULT_GIT_TAG ; \
    fi && \
    git clone --depth 1 --recursive $GIT_URL --branch "$GIT_TAG" --shallow-submodules && \
    cd libjxl && \
    mkdir build && cd build && \
    export CC=clang CXX=clang++ && \
    cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DJPEGXL_ENABLE_DEVTOOLS:BOOL=ON -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_BUILD_TYPE=Release -DJPEGXL_ENABLE_DOXYGEN:BOOL=OFF -DJPEGXL_ENABLE_BENCHMARK:BOOL=OFF -DJPEGXL_ENABLE_EXAMPLES:BOOL=OFF -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=/usr .. && \
    cmake --build . -- -j$(nproc) && \
    cmake --install . && \
    pip install --root-user-action ignore  jxlpy && \
    pip install --no-binary :all: --root-user-action ignore git+https://github.com/bigcat88/pillow_heif.git

FROM python:3.12-alpine

LABEL maintainer="cmahnke@gmail.com"

COPY --from=builder /usr/lib/libheif* /usr/lib/
COPY --from=builder /usr/lib/cmake/libheif /usr/lib/cmake/libheif/
COPY --from=builder /usr/include/libheif /usr/include/libheif/
COPY --from=builder /usr/lib/pkgconfig/libheif.pc /usr/lib/pkgconfig/libheif.pc
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=builder /usr/include/brotli/  /usr/include/brotli/
COPY --from=builder /usr/lib/pkgconfig/libbrotli* /usr/lib/pkgconfig/
COPY --from=builder /usr/lib/libjxl* /usr/lib/
COPY --from=builder /usr/lib/libbrotli* /usr/lib/
COPY --from=builder /usr/include/jxl/ /usr/include/jxl/
COPY --from=builder /usr/lib/cmake/hwy/ /usr/lib/cmake/hwy/
COPY --from=builder /usr/include/hwy/ /usr/include/hwy/
COPY --from=builder /usr/lib/libhwy.a /usr/lib/libhwy.a
COPY --from=builder /usr/lib/pkgconfig/libhwy.pc /usr/lib/pkgconfig/libhwy.pc

COPY ./scripts/* /usr/local/bin/

RUN apk --update upgrade && \
    apk add --no-cache gdk-pixbuf exiftool libde265 openh264 libjpeg-turbo \
    libpng x264-libs x265-libs aom aom-libs dav1d libsharpyuv libstdc++ py3-numpy py3-opencv && \
    pip install --upgrade pip && \
    pip install --root-user-action ignore pyexiftool opencv-contrib-python-headless && \
    chmod +x /usr/local/bin/*.py

RUN python3 -c "import pillow_heif; print('pillow_heif python module loaded successfully')" && \
    exiftool -ver

RUN rm -rf /var/cache/apk/* /root/.cache

CMD ["python3"]