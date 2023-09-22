#!/bin/bash

# Build Docker image
docker build -t pseudolabel_app .

# Run Docker container
nvidia-docker run -it --gpus all pseudolabel_app
