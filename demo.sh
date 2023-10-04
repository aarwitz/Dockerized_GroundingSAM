#!/bin/bash

# Build Docker image
docker build -t pseudolabel_app .

# Run Docker container
nvidia-docker run -it --gpus all -v ~/Dockerized_GroundingSAM/tool_output:/workspace/tool_output pseudolabel_app python gradio_demo/gradio_demo.py
# docker exec -it pseudolabel_app /bin/bashbash