#!/usr/bin/env python

import sys, os
import tempfile
import shutil
import argparse
import math
import logging

import numpy as np
import pillow_heif
from PIL import Image, ImageColor
import cv2
import pathlib
import subprocess
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from stereoscopy import auto_align
from ultralytics import YOLO

import torch
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

HF_REPO_ID = "cmahnke/vintagereality"
HF_FILENAME = "vintagereality.pt"

EXIFTOOL_CONFIG = """
%Image::ExifTool::UserDefined = (
    'Image::ExifTool::XMP::Main' => {
        apple => {
            SubDirectory => {
                TagTable => 'Image::ExifTool::UserDefined::apple',
            },
        },
        spatial => {
            SubDirectory => {
                TagTable => 'Image::ExifTool::UserDefined::spatial',
            },
        },
    },
);

%Image::ExifTool::UserDefined::apple = (
    GROUPS => { 0 => 'XMP', 1 => 'XMP-apple', 2 => 'Image' },
    NAMESPACE => { 'apple' => 'http://ns.apple.com/image/1.0/' },
    WRITABLE => 'string',
    HorizontalFOV => { Writable => 'real' },
    Baseline => { Writable => 'real' },
    HorizontalDisparityAdjustment => { Writable => 'real' },
    CameraModelType => { },
    CameraIntrinsics => { },
    CameraExtrinsicsRotation => { },
    CameraExtrinsicsPosition => { },
    StereoGroupIndex => { Writable => 'integer' },
);

%Image::ExifTool::UserDefined::spatial = (
    GROUPS => { 0 => 'XMP', 1 => 'XMP-spatial', 2 => 'Image' },
    NAMESPACE => { 'spatial' => 'http://ns.google.com/photos/1.0/spatial/' },
    WRITABLE => 'string',
    HasSpatialMetadata => { },
);
"""

def debug_view(left_pil, right_pil, title="Debug viewer"):
    logger.debug(f"Showing debug viewer")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    axes[0].imshow(left_pil)
    axes[0].set_title("Left Eye")
    axes[0].axis('off')

    axes[1].imshow(right_pil)
    axes[1].set_title("Right Eye")
    axes[1].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def load_card_yolo(pil_image, model_path, bg_color=None, use_masks=False, debug=False):
    logger.debug(f"Trying to segment card with model: {model_path}, bg_color: {bg_color}, use_masks: {use_masks}")
    model = YOLO(model_path)
    results = model(pil_image, verbose=False, retina_masks=use_masks)
    
    left_img = None
    right_img = None
    
    r = results[0]
    logger.debug(f"YOLO Detection: {len(r.boxes)} boxes, {len(r.masks.data) if r.masks else 0} masks")
    
    if not r.boxes or len(r.boxes) != 2:
        raise Exception(f"YOLO detected {len(r.boxes) if r.boxes else 0} regions, expected exactly 2.")

    boxes = r.boxes.xyxy.cpu().numpy()
    
    indexed_boxes = sorted(enumerate(boxes), key=lambda x: x[1][0])
    
    left_idx, left_box = indexed_boxes[0]
    right_idx, right_box = indexed_boxes[1]

    masks = None
    if use_masks and r.masks:
         masks = r.masks.data.cpu().numpy()
         if masks.shape[1:] != (pil_image.height, pil_image.width):
             masks_resized = []
             for m in masks:
                 m_resized = cv2.resize(m, (pil_image.width, pil_image.height), interpolation=cv2.INTER_LINEAR)
                 masks_resized.append(m_resized)
             masks = np.array(masks_resized)

    if debug:
        def get_raw_crop(box):
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(pil_image.width, x2), min(pil_image.height, y2)
            return pil_image.crop((x1, y1, x2, y2))
        
        d1 = get_raw_crop(left_box)
        d2 = get_raw_crop(right_box)
        debug_view(d1, d2, "YOLO Raw Results")

    def process_crop(idx, box):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(pil_image.width, x2), min(pil_image.height, y2)
        
        crop = pil_image.crop((x1, y1, x2, y2))

        if use_masks and masks is not None and bg_color is not None:
            mask = masks[idx]
            mask_crop = mask[y1:y2, x1:x2]
            
            if mask_crop.shape[0] != crop.height or mask_crop.shape[1] != crop.width:
                logger.warning(f"Mask shape mismatch: {mask_crop.shape} vs {crop.size[::-1]}. Resizing mask.")
                mask_crop = cv2.resize(mask_crop, (crop.width, crop.height))

            mask_pil = Image.fromarray((mask_crop * 255).astype(np.uint8), mode='L')
            
            if bg_color == 'transparent':
                crop = crop.convert("RGBA")
                crop.putalpha(mask_pil)
            else:
                try:
                    c = ImageColor.getrgb(bg_color)
                    bg = Image.new("RGB", crop.size, c)
                    crop = crop.convert("RGB")
                    bg.paste(crop, (0, 0), mask_pil)
                    crop = bg
                except ValueError:
                    logger.warning(f"Invalid color {bg_color}, ignoring.")
        return crop

    left_img = process_crop(left_idx, left_box)
    right_img = process_crop(right_idx, right_box)
                    
    return left_img, right_img

def remove_borders(left_pil, right_pil):
    logger.debug("Removing borders")
    left_arr = np.array(left_pil)
    right_arr = np.array(right_pil)

    def get_bbox(img_arr):
        if img_arr.ndim == 3:
            if img_arr.shape[2] == 4:
                mask = img_arr[:, :, 3] > 0
            else:
                mask = np.any(img_arr > 0, axis=2)
        else:
            mask = img_arr > 0
        
        if not np.any(mask):
            return 0, 0, img_arr.shape[1], img_arr.shape[0]

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        return x_min, y_min, x_max + 1, y_max + 1

    lx1, ly1, lx2, ly2 = get_bbox(left_arr)
    rx1, ry1, rx2, ry2 = get_bbox(right_arr)

    x1 = max(lx1, rx1)
    y1 = max(ly1, ry1)
    x2 = min(lx2, rx2)
    y2 = min(ly2, ry2)

    if x1 >= x2 or y1 >= y2:
        logger.warning("Border removal would result in empty image. Skipping.")
        return left_pil, right_pil

    left_cropped = left_pil.crop((x1, y1, x2, y2))
    right_cropped = right_pil.crop((x1, y1, x2, y2))
    
    return left_cropped, right_cropped

def match_exposure(left_pil, right_pil):
    logger.debug("Matching exposure")

    left_std = np.std(np.array(left_pil.convert('L')))
    right_std = np.std(np.array(right_pil.convert('L')))

    if left_std > right_std:
        source_pil = right_pil
        reference_pil = left_pil
        target_is_right = True
    else:
        source_pil = left_pil
        reference_pil = right_pil
        target_is_right = False

    src_arr = np.array(source_pil)
    ref_arr = np.array(reference_pil)
    
    src_alpha = None
    if src_arr.ndim == 3 and src_arr.shape[2] == 4:
        src_alpha = src_arr[:, :, 3]
        src = cv2.cvtColor(src_arr, cv2.COLOR_RGBA2BGR)
    else:
        src = cv2.cvtColor(src_arr, cv2.COLOR_RGB2BGR)
        
    if ref_arr.ndim == 3 and ref_arr.shape[2] == 4:
        ref = cv2.cvtColor(ref_arr, cv2.COLOR_RGBA2BGR)
    else:
        ref = cv2.cvtColor(ref_arr, cv2.COLOR_RGB2BGR)

    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)

    s_l, s_a, s_b = cv2.split(src_lab)
    r_l, r_a, r_b = cv2.split(ref_lab)

    def get_cdf(channel):
        hist, _ = np.histogram(channel.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf.max()
        return cdf_normalized

    src_cdf = get_cdf(s_l)
    ref_cdf = get_cdf(r_l)

    lookup_table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        diff = np.abs(ref_cdf - src_cdf[i])
        lookup_table[i] = diff.argmin()

    matched_l = cv2.LUT(s_l, lookup_table)

    matched_lab = cv2.merge([matched_l, s_a, s_b])
    matched_bgr = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)

    matched_rgb = cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2RGB)
    
    matched_pil = Image.fromarray(matched_rgb)
    if src_alpha is not None:
        matched_pil = Image.fromarray(np.dstack((matched_rgb, src_alpha)))
    
    if target_is_right:
        return left_pil, matched_pil
    return matched_pil, right_pil

def create_depthmap(left_pil, right_pil):
    logger.debug("Creating depthmap")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    w, h = left_pil.size
    new_w = ((w + 7) // 8) * 8
    new_h = ((h + 7) // 8) * 8

    img1 = F.to_tensor(left_pil.convert('RGB').resize((new_w, new_h), Image.BILINEAR)).unsqueeze(0)
    img2 = F.to_tensor(right_pil.convert('RGB').resize((new_w, new_h), Image.BILINEAR)).unsqueeze(0)

    img1, img2 = transforms(img1, img2)

    model = raft_large(weights=weights, progress=True).to(device)
    model = model.eval()

    img1 = img1.to(device)
    img2 = img2.to(device)

    with torch.no_grad():
        list_of_flows = model(img1, img2)
        predicted_flow = list_of_flows[-1]

    flow_magnitude = torch.norm(predicted_flow[0], p=2, dim=0)

    flow_magnitude = flow_magnitude.unsqueeze(0)
    flow_magnitude = F.resize(flow_magnitude, [left_pil.height, left_pil.width])
    flow_magnitude = flow_magnitude.squeeze(0)

    denom = flow_magnitude.max() - flow_magnitude.min()
    if denom == 0: denom = 1
    normalized_flow = (flow_magnitude - flow_magnitude.min()) / denom * 255.0

    return Image.fromarray(normalized_flow.byte().cpu().numpy(), mode='L')

def apple_spatial(left_pil, right_pil, output, overwrite=False, hfov=45.0, baseline_mm=65.0, disparity_adj=0.02):
    logger.debug(f"Trying to convert to Apple spatia format: {output}")
    width, height = left_pil.size
    hfov_rad = math.radians(hfov)
    f_pix = (width * 0.5) / math.tan(hfov_rad * 0.5)
    ppx, ppy = width / 2.0, height / 2.0
    intrinsics = f"{f_pix} 0 {ppx} 0 {f_pix} {ppy} 0 0 1"
    
    baseline_m = baseline_mm / 1000.0
    right_pos = f"{baseline_m} 0 0"
    
    heif_file = pillow_heif.from_pillow(left_pil)
    heif_file.add_from_pillow(right_pil)
    
    with tempfile.NamedTemporaryFile(suffix='.heic') as tmp_heic, \
         tempfile.NamedTemporaryFile(mode='w', suffix='.config') as tmp_config:
        
        tmp_config.write(EXIFTOOL_CONFIG)
        tmp_config.flush()

        heif_file.save(tmp_heic.name, quality=95)
        cmd = [
            "exiftool",
            "-config", tmp_config.name,
            "-n",
            "-overwrite_original",
            "-wm", "cg",
            "-XMP-spatial:HasSpatialMetadata=True",
            f"-XMP-apple:HorizontalFOV={hfov}",
            f"-XMP-apple:Baseline={baseline_mm}",
            f"-XMP-apple:HorizontalDisparityAdjustment={disparity_adj}",
            "-XMP-apple:CameraModelType=SimplifiedPinhole",
            f"-XMP-apple:CameraIntrinsics={intrinsics}",
            "-XMP-apple:CameraExtrinsicsRotation=1 0 0 0 1 0 0 0 1",
            f"-XMP-apple:CameraExtrinsicsPosition=0 0 0",
            "-XMP-apple:StereoGroupIndex=1",
            tmp_heic.name
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            if os.path.exists(output):
                if overwrite:
                    os.remove(output)
                else:
                    raise Exception(f"Error: Output file {output} already exists. Use overwrite option to replace.")
            os.rename(tmp_heic.name, output)
            logger.info(f"Successfully generated: {output}")
        except subprocess.CalledProcessError as e:
            logger.error(f"ExifTool Error: {e.stderr.decode()}")
        finally:
            if os.path.exists(tmp_heic.name):
                os.remove(tmp_heic.name)


# TODO: Port image processing from https://github.com/cmahnke/vintagereality/blob/main/scripts/image-splitter.py

def main():
    parser = argparse.ArgumentParser(description='Extract steroscopic images')
    parser.add_argument('--image', '-i', type=pathlib.Path, help='Image to process')
    parser.add_argument('--left', '-l', type=pathlib.Path, help='Left image to process')
    parser.add_argument('--right', '-r', type=pathlib.Path, help='Right image to process')
    parser.add_argument('--output', '-o', help='Image to write', required=True)
    parser.add_argument('--depthmap', '-m', help='Generate depthmap', action='store_true')
    parser.add_argument('--debug', '-d', help='Show as images', action='store_true')
    parser.add_argument('--align', '-a', help='Align images', action='store_true')
    parser.add_argument('--model', default="weights/vintagereality.pt", type=pathlib.Path, help='Path to YOLO model for segmentation')
    parser.add_argument('--background-color', '-b', help='Background color for masked areas (e.g. "transparent", "#000000", "white")', default=None)
    parser.add_argument('--match-exposure', '-e', help='Match exposure between images', action='store_true')
    parser.add_argument('--crop', '-c', help='Remove border from images', action='store_true')

    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    logger.debug("Starting processing")

    if args.image:
      if str(args.image).endswith('.jxl'):
          from jxlpy import JXLImagePlugin

      im = Image.open(args.image)

      if args.model:
          if not args.model.exists():
              logger.info(f"Model not found at {args.model}. Downloading from Hugging Face Hub")
              args.model.parent.mkdir(parents=True, exist_ok=True)
              cached_file = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
              shutil.copy(cached_file, args.model)

          try:
              left_pil, right_pil = load_card_yolo(im, args.model, args.background_color, use_masks=(args.background_color is not None), debug=args.debug)
          except Exception as e:
              logger.warning(f"YOLO detection failed: {e}.")
              sys.exit(1)
      else:
          logger.error("Error: --model is required when using --image.")
          sys.exit(1)

    else:
      left_pil = Image.open(args.left)
      right_pil = Image.open(args.right)

    if args.align:
      logger.debug("Aligning images")
      left_pil, right_pil = auto_align([left_pil, right_pil])

    if args.crop:
        left_pil, right_pil = remove_borders(left_pil, right_pil)

    if args.match_exposure:
        left_pil, right_pil = match_exposure(left_pil, right_pil)

    if args.debug:
        debug_view(left_pil, right_pil, "After processing")

    #if args.debug:
    #  debug_view(left_pil, right_pil)

    if not args.depthmap:
        if str(args.output).lower().endswith('.heic'):
            apple_spatial(left_pil, right_pil, args.output, overwrite=True)
        else:
            mode = 'RGBA' if left_pil.mode == 'RGBA' or right_pil.mode == 'RGBA' else 'RGB'
            w = left_pil.width + right_pil.width
            h = max(left_pil.height, right_pil.height)
            sbs = Image.new(mode, (w, h))
            sbs.paste(left_pil, (0, 0))
            sbs.paste(right_pil, (left_pil.width, 0))
            sbs.save(args.output)
    else:
      depthmap_pil = create_depthmap(left_pil, right_pil)
      #TODO: Scale depth map to input dimensions
      depthmap_pil.save(args.output)

if __name__=="__main__":
    main()
