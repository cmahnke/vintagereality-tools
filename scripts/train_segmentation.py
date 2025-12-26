#!/usr/bin/env python

import argparse
import os
from ultralytics import YOLO
import torch

def main():
    parser = argparse.ArgumentParser(description="Train a YOLOv8 Segmentation Model for Stereogram Cards")
    parser.add_argument("--data", required=True, help="Path to data.yaml file (YOLO format)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--model", default="yolo11n-seg.pt", help="Pretrained model (e.g. yolo11n-seg.pt, yolov8n-seg.pt)")
    parser.add_argument("--output", default="runs/segment/train", help="Output directory")
    parser.add_argument("--export", action="store_true", help="Export to TF.js and ONNX after training")
    
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = YOLO(args.model) 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    print("Starting training...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=os.path.dirname(args.output) if args.output else None,
        name=os.path.basename(args.output) if args.output else None,
        device=device
    )

    print(f"Training finished. Results saved to {results.save_dir}")

    if args.export:
        print("Exporting model for browser/deployment...")
        try:
            model.export(format="tfjs")
        except Exception as e:
            print(f"TF.js export failed (install tensorflowjs to fix): {e}")
            
        try:
            model.export(format="onnx")
        except Exception as e:
            print(f"ONNX export failed: {e}")

if __name__ == "__main__":
    main()