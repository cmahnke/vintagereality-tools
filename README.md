VintageReality Tools
====================

This is a updated Toolset for processing files for the [vintageReality](https://vintagereality.projektemacher.org/) page.

It currently contains a Docker image for running the scripts and training data and scripts to train the recognition YOLO Model.

# Setup

## From start

If you want to change, update or improve the model, you need to run `./scripts/train.sh`. The following changes are possible:
* In `train_segmentation.py` one can change the trainig parameters
* In `data` you can either add your own dataset or improve the existing ones.


# 




# TODO

* Move the trainig data to Hugging Face, maybe by providing a better format for the labeks then the YOLO one.