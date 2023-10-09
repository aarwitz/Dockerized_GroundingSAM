#!/bin/bash

# Build Docker image
docker build -t pseudolabel_app .

# Run Docker container
nvidia-docker run -it \
    --gpus all \
    -v ~/Dockerized_GroundingSAM/tool_output:/workspace/tool_output \
    -v ~/Dockerized_GroundingSAM/gradio_demo/uploaded_images:/workspace/gradio_demo/uploaded_images \
    pseudolabel_app \
    python gradio_demo/gradio_demo.py
# nvidia-docker run -it --gpus all -v ~/Dockerized_GroundingSAM/tool_output:/workspace/tool_output -v ~/Dockerized_GroundingSAM/gradio_demo/uploaded_images:/workspace/gradio_demo/uploaded_images pseudolabel_app python gradio_demo/gradio_demo.py