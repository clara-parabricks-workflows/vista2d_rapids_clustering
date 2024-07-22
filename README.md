# Feature Extraction and Morphology Clustering using Vista2D and RAPIDS

This notebook shows how to use Vista2D to extract features from cell segmentations, and how to use RAPIDS to cluster those features. 

Quick start instructions are provided below, however the full run guide can be found in the accompanying [blog post](). 

## Quick Start  Instructions 

### Start Docker

```
docker run --rm -it \
    -v /full/path/to/repo:/workspace \
    -p 8888:8888 \
    --gpus all \
    nvcr.io/nvidia/pytorch:24.03-py3 \
    /bin/bash
```
### Install additional Python packages  

```
pip install -r requirements.txt 
```

### Start Jupyter

```
jupyter notebook
```
