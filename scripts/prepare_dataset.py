#!/usr/bin/env python


import argparse
import os
import pathlib
import yaml
import requests
from PIL import Image
import re
import unicodedata

# Register JXL support
try:
    from jxlpy import JXLImagePlugin
except ImportError:
    print("Warning: jxlpy not found. JXL files might not load properly.")

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for YOLO training from JXL images")
    parser.add_argument("--input", "-i", help="Root directory to search for front.jxl files")
    parser.add_argument("--manifest-list", "-m", help="File containing list of IIIF Manifest URLs")
    parser.add_argument("--output", "-o", default="data/vintagereality", help="Output directory for the dataset")
    
    args = parser.parse_args()
    
    if not args.input and not args.manifest_list:
        parser.error("Either --input or --manifest-list must be provided.")

    output_path = pathlib.Path(args.output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    
    files = []
    input_path = None
    if args.input:
        input_path = pathlib.Path(args.input).resolve()
        if not input_path.exists():
            print(f"Error: Input path {input_path} does not exist.")
            return

        # Find all front.jxl and front.jpg files
        print(f"Searching for 'front.jxl' and 'front.jpg' in {input_path}...")
        files = list(input_path.rglob("front.jxl")) + list(input_path.rglob("front.jpg"))
        print(f"Found {len(files)} files.")

    # Create directories
    dirs = {
        "train_img": output_path / "images" / "train",
        "train_lbl": output_path / "labels" / "train",
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        
    def process_files(file_list, img_dir, lbl_dir):
        for src_file in file_list:
            # Generate unique filename based on relative path to avoid collisions
            # e.g. /input/collection/card1/front.jxl -> collection_card1_front.jpg
            try:
                rel_path = src_file.relative_to(input_path).parent
                safe_name = str(rel_path).replace(os.sep, "_") + "_front"
            except ValueError:
                safe_name = src_file.parent.name + "_front"
            
            dest_img_name = f"{safe_name}.jpg"
            dest_lbl_name = f"{safe_name}.txt"
            
            dest_img_path = img_dir / dest_img_name
            dest_lbl_path = lbl_dir / dest_lbl_name
            
            if dest_img_path.exists():
                print(f"Skipping existing image: {dest_img_path}")
                continue
            
            try:
                # Convert and save image
                with Image.open(src_file) as im:
                    rgb_im = im.convert('RGB')
                    rgb_im.save(dest_img_path, quality=95)
                
                # Create empty label stub
                # YOLO format expects a text file. Empty file = no objects (background).
                # This acts as the placeholder for labeling tools.
                dest_lbl_path.touch()
                
            except Exception as e:
                print(f"Error processing {src_file}: {e}")

    def process_manifests(manifest_list_path, img_dir, lbl_dir):
        with open(manifest_list_path, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        for i, url in enumerate(urls):
            try:
                print(f"Fetching manifest: {url}")
                resp = requests.get(url)
                resp.raise_for_status()
                data = resp.json()
                
                img_url = None
                
                def get_full_res_url(node):
                    if 'service' not in node:
                        return None
                    
                    services = node['service']
                    if not isinstance(services, list):
                        services = [services]
                    
                    for service in services:
                        if not isinstance(service, dict):
                            continue
                        
                        url = service.get('@id') or service.get('id')
                        if not url:
                            continue
                            
                        if url.endswith('/info.json'):
                            url = url[:-10]
                        url = url.rstrip('/')
                        
                        # Check for IIIF Image API 3
                        if service.get('type') == 'ImageService3':
                            return f"{url}/full/max/0/default.jpg"
                        
                        profile = service.get('profile')
                        if profile and isinstance(profile, str) and '/3/' in profile:
                            return f"{url}/full/max/0/default.jpg"
                            
                        return f"{url}/full/full/0/default.jpg"
                    return None

                # Try V3
                try:
                    body = data['items'][0]['items'][0]['items'][0]['body']
                    img_url = get_full_res_url(body)
                    if not img_url:
                        img_url = body.get('id')
                except (KeyError, IndexError, TypeError):
                    pass
                
                # Try V2
                if not img_url:
                    try:
                        resource = data['sequences'][0]['canvases'][0]['images'][0]['resource']
                        img_url = get_full_res_url(resource)
                        if not img_url:
                            img_url = resource.get('@id')
                    except (KeyError, IndexError, TypeError):
                        pass
                
                if not img_url:
                    print(f"Could not extract image URL from {url}")
                    continue

                # Heuristic fallback for IIIF Image API to get full resolution
                if 'iiif' in img_url and not img_url.endswith('/default.jpg'):
                     if not any(img_url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                         img_url = img_url.rstrip('/') + '/full/full/0/default.jpg'
                
                label = None
                if 'label' in data:
                    raw_label = data['label']
                    if isinstance(raw_label, dict):
                        for lang in raw_label:
                            if raw_label[lang]:
                                label = raw_label[lang][0]
                                break
                    elif isinstance(raw_label, list):
                        if raw_label:
                            label = str(raw_label[0])
                    else:
                        label = str(raw_label)
                
                if label:
                    label = unicodedata.normalize('NFKD', label).encode('ascii', 'ignore').decode('ascii')
                    label = label.lower()
                    label = re.sub(r'\s+', '-', label)
                    label = re.sub(r'[^a-z0-9-]', '', label)
                    label = re.sub(r'-+', '-', label).strip('-')

                dest_name = label if label else f"manifest_{i}"
                dest_img_path = img_dir / f"{dest_name}.jpg"
                dest_lbl_path = lbl_dir / f"{dest_name}.txt"
                
                if dest_img_path.exists():
                    print(f"Skipping existing image: {dest_img_path}")
                    continue

                print(f"Downloading image: {img_url}")
                img_resp = requests.get(img_url, stream=True)
                img_resp.raise_for_status()
                
                with open(dest_img_path, 'wb') as f_img:
                    for chunk in img_resp.iter_content(chunk_size=8192):
                        f_img.write(chunk)
                
                dest_lbl_path.touch()
                
            except Exception as e:
                print(f"Error processing manifest {url}: {e}")

    if files:
        print("Processing local training files...")
        process_files(files, dirs["train_img"], dirs["train_lbl"])
        
    if args.manifest_list:
        print("Processing IIIF manifests...")
        process_manifests(args.manifest_list, dirs["train_img"], dirs["train_lbl"])
    
    # Generate data.yaml
    data_yaml = {
        "path": str(output_path),
        "train": "images/train",
        "val": "images/train",
        "names": { 0: "Left image", 1: "Right image" }
    }
    
    yaml_path = output_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
        
    print(f"Dataset preparation complete. Config saved to {yaml_path}")

if __name__ == "__main__":
    main()