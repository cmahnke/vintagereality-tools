#!/usr/bin/env bash

if [ -z "$VERSION" ] ; then
  VERSION=$(git describe --exact-match --tags 2>/dev/null || git rev-parse --short HEAD)
fi

set -e

if [[ "$OSTYPE" == "darwin"* ]]; then
    WORKERS=`ioreg -l | grep gpu-core-count|cut -d '=' -f2`
else
    WORKERS=8
fi

rm -rf runs/segment/iiif_urls runs/segment/vintagereality

echo "Using $WORKERS workers for processing."
echo "Setting version to $VERSION"

python scripts/train_segmentation.py --model yolo11n-seg.pt --data data/iiif-urls/data.yaml --output runs/segment/iiif_urls --workers $WORKERS

python scripts/train_segmentation.py --model runs/segment/iiif_urls/weights/best.pt --data data/vintagereality/data.yaml --output runs/segment/vintagereality --workers $WORKERS --export

rm -rf weights
mkdir -p weights

cp runs/segment/vintagereality/weights/best.pt weights/vintagereality-${VERSION}.pt
ln -s weights/vintagereality-${VERSION}.pt weights/vintagereality.pt
cp runs/segment/vintagereality/weights/best.onnx weights/vintagereality-${VERSION}.onnx
ln -s weights/vintagereality-${VERSION}.onnx weights/vintagereality.onnx