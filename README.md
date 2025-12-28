VintageReality Tools
====================

This is a updated Toolset for processing files for the [vintageReality](https://vintagereality.projektemacher.org/) page.

It currently contains a Docker image for running the scripts and training data and scripts to train the recognition YOLO Model.

# Setup

## From start

The model is pretrained, it will be pulled from Hugging Face. If you want to make changes, see the "Development" section.

## Docker (Recommended)

To build the Docker image, run:

```bash
docker build -t ghcr.io/cmahnke/vintagereality-tools:latest -f docker/Dockerfile .
```

Otherwise you can just pull a pre build:

```bash
docker pull ghcr.io/cmahnke/vintagereality-tools:latest
```

## Local Installation

To run the scripts locally, you need Python 3.11+ and several system dependencies (ExifTool, libheif, etc.).

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure `exiftool` is installed and in your PATH.

# Usage

## Spatial Tool

The `spatial.py` script is the main tool for processing stereoscopic images. It can extract stereo pairs, align them, and convert them to Apple Spatial HEIC format.

### Basic Usage

#### With Docker

```bash
docker run -v $(pwd):/data vintagereality-tools spatial.py -i /data/input.jpg -o /data/output.heic -a -e -b transparent -c
```

#### With docker and CUDA

```bash
docker run --gpus all -v $(pwd):/data vintagereality-tools spatial.py -i /data/input.jpg -o /data/output.heic -a -e -b transparent -c
```

#### Local

```bash
python scripts/spatial.py -i input.jpg -o output.heic -a -e -b transparent -c
```

### Options

| Option | Short | Description | Default |
| :--- | :--- | :--- | :--- |
| `--image` | `-i` | Input image containing both views | None |
| `--left` | `-l` | Input left image | None |
| `--right` | `-r` | Input right image | None |
| `--output` | `-o` | Output file path | Required |
| `--depthmap` | `-m` | Generate depthmap instead of spatial image | `False` |
| `--debug` | `-d` | Show debug windows | `False` |
| `--align` | `-a` | Auto-align stereo pair | `False` |
| `--model` | | Path to YOLO segmentation model | `weights/vintagereality.pt` |
| `--background-color` | `-b` | Background color for masked areas (e.g., "transparent", "#000000") | None |
| `--match-exposure` | `-e` | Match exposure between left and right images | `False` |
| `--crop` | `-c` | Remove black borders after alignment | `False` |


## Web Viewer

The project includes a web-based viewer for testing models and viewing results.

1. Navigate to the viewer directory:
   ```bash
   cd viewer
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

# Development

## Training the Model

If you want to change, update or improve the model, you need to run `./scripts/train.sh`. The following changes are possible:
* In `train_segmentation.py` one can change the trainig parameters
* In `data` you can either add your own dataset or improve the existing ones.

# TODO

* Move the trainig data to Hugging Face, maybe by providing a better format for the labeks then the YOLO one.