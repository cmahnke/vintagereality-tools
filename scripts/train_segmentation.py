#!/usr/bin/env python

import argparse
import os
import sys
from ultralytics import YOLO
import torch
import logging

def main():
    parser = argparse.ArgumentParser(description="Train a YOLO Segmentation Model for Stereogram Cards")
    parser.add_argument("--data", required=True, help="Path to data.yaml file (YOLO format)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--model", default="yolo11n-seg.pt", help="Pretrained model (e.g. yolo11n-seg.pt, yolov8n-seg.pt)")
    parser.add_argument("--output", default="runs/segment/train", help="Output directory")
    parser.add_argument("--export", action="store_true", help="Export to ONNX after training")
    parser.add_argument("--quantize", action="store_true", default=False)
    parser.add_argument('--debug', '-d', help='Debug output', action='store_true')


    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    args = parser.parse_args()
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    logger.info(f"Loading model: {args.model}")
    model = YOLO(args.model) 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    logger.info("Starting training...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=os.path.dirname(args.output) if args.output else None,
        name=os.path.basename(args.output) if args.output else None,
        device=device,
        workers=args.workers,
        cache="disk"
    )

    logger.info(f"Training finished. Results saved to {results.save_dir}")

    if args.export:
        logger.info("Exporting model for browser/deployment...")
            
        try:
            model.export(format="onnx")
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")

        if args.quantize:
            logger.info("Quantizing ONNX model...")
            onnx_path = args.output + "/weights/best.onnx"
            quantized_path = onnx_path.replace(".onnx", ".quant.onnx")
            try:
                from onnxruntime.quantization import quantize_dynamic, QuantType
            except ImportError:
                logger.error("onnxruntime not found. Please install it: pip install onnxruntime")
                sys.exit(1)

            quantize_dynamic(str(onnx_path), str(quantized_path), weight_type=QuantType.QUInt8)

if __name__ == "__main__":
    main()