# Feature Extraction and Morphology Clustering using Vista2D and RAPIDS

This notebook shows how to use Vista2D to extract features from cell segmentations, and how to use RAPIDS to cluster those features. 

Quick start instructions are provided below, however the full run guide can be found in the accompanying [blog post](https://developer.nvidia.com/blog/cell-imaging-feature-extraction-and-morphology-clustering-for-spatial-omics/). 

## Quick Start  Instructions 

### Start Docker

For this example, we will use the [Project MONAI](https://monai.io/) Docker image. 

```
docker run --rm -it \
    -v /absolute/path/to/vista2d_rapids_clustering:/workspace \
    -p 8888:8888 \
    --gpus all \
    projectmonai/monai:1.4.0rc10 \
    /bin/bash
```
### Install additional Python packages  

```
pip install fastremap==1.15.0 roifile==2024.5.24 natsort==8.4.0 plotly 
pip install --no-deps cellpose
```

### Start Jupyter

```
jupyter notebook
```
