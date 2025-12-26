Expected Data Structure:
------------------------

YOLOv8 uses a YAML file to define the dataset.
    #
1. Directory Structure:
   dataset/
     ├── images/
     │   ├── train/
     │   └── val/
     └── labels/
         ├── train/
         └── val/
    #
   Labels should be in YOLO Segmentation format (.txt files):
   <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
    #
   Label Studio Export Instructions:
   1. Configuration: Ensure your project uses 'PolygonLabels' (not RectangleLabels).
   2. Annotation: Use the Polygon tool to trace the exact shape of the photos (handling arches/corners).
   3. Export: Choose "YOLO" format. Label Studio will export polygons in the correct format for YOLOv8-seg.
    #
    #
   Hint: For fine-tuning on a specific domain like stereogram cards,
   starting with 50-100 annotated images is often sufficient to get
   reasonable results.
    #
2. data.yaml Configuration (--data):
   This file is MANDATORY for training. It defines the dataset root,
   image directories, and class names.
   (The scripts/prepare_dataset.py helper script generates this file automatically)
    #
   Structure:
     path:  ../datasets/stereo  # (Optional) Root directory for the dataset
     train: images/train    Path to training images (relative to 'path')
     val:   images/val      Path to validation images (relative to 'path')
     names:
       0: Left image
       1: Right image

Load a model
yolo11n-seg.pt is the "nano" model - fastest, smallest, good for browsers.
yolo11x-seg.pt is the "extra large" model - most accurate, slowest.