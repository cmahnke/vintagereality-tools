#!/usr/bin/env python

import cv2
import numpy as np
import pillow_heif
from PIL import Image
import sys, os
import tempfile
import exiftool
import pathlib

def load_card(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    processed_eyes = []

    for cnt in sorted_cnts:
        rect = cv2.minAreaRect(cnt)
        center, size, angle = rect
        
        if angle < -45:
            angle = 90 + angle
            
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
        
        w, h = int(size[0]), int(size[1])
        if angle > 45: w, h = h, w
        
        cropped = cv2.getRectSubPix(rotated, (w, h), center)
        processed_eyes.append(cropped)

    processed_eyes.sort(key=lambda x: cv2.boundingRect(sorted_cnts[processed_eyes.index(x)])[0])
    
    min_h = min(processed_eyes[0].shape[0], processed_eyes[1].shape[0])
    min_w = min(processed_eyes[0].shape[1], processed_eyes[1].shape[1])
    
    left_final = cv2.resize(processed_eyes[0], (min_w, min_h))
    right_final = cv2.resize(processed_eyes[1], (min_w, min_h))

    return Image.fromarray(cv2.cvtColor(left_final, cv2.COLOR_BGR2RGB)), Image.fromarray(cv2.cvtColor(right_final, cv2.COLOR_BGR2RGB))

def match_exposure(source_pil, reference_pil):
    src = cv2.cvtColor(np.array(source_pil), cv2.COLOR_RGB2BGR)
    ref = cv2.cvtColor(np.array(reference_pil), cv2.COLOR_RGB2BGR)

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
    
    return Image.fromarray(cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2RGB))

# TODO: Port image processing from https://github.com/cmahnke/vintagereality/blob/main/scripts/image-splitter.py

def main():
    parser = argparse.ArgumentParser(description='Extract steroscopic images')
    parser.add_argument('--image', '-i', type=pathlib.Path, help='Image to process', required=True)
    parser.add_argument('--output', '-o', help='Image to write', required=True)

    args = parser.parse_args()

    if str(args.image).endswith('.jxl'):
        from jxlpy import JXLImagePlugin

    im = Image.open(args.image)

    left_pil, right_pil = load_card(im)
    right_matched = match_exposure(right_pil, left_pil)
    
    heif_file = pillow_heif.from_pillow(left)
    heif_file.add_from_pillow(right_matched)
    with tempfile.NamedTemporaryFile(suffix='.heic') as tmp_heic:
        heif_file.save(tmp_heic, quality=95)
        tags = {
            "XMP-spatial:HasSpatialMetadata": True,
            "XMP-apple:HorizontalFOV": 45.0,
            "XMP-apple:Baseline": 65.0,
            "XMP-apple:HorizontalDisparityAdjustment": 0.02
        }
    
        with exiftool.ExifToolHelper() as et:
            et.set_tags(
                [tmp_heic],
                tags=tags,
                params=["-overwrite_original"]
            )
        os.rename(tmp_heic, args.output)

if __name__=="__main__":
    main()