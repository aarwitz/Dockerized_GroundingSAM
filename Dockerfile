
# Use an official NVIDIA CUDA base image with Ubuntu as the base OS
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04


# Set environment variables
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA="1"
ENV CUDA_VISIBLE_DEVICES="0"
ENV NVCC_PATH=/usr/local/cuda-11.8/bin/nvcc
ENV PATH="/usr/local/cuda-11.8/bin:${PATH}"
# Install essential packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0\
    ffmpeg=7:* \
    libsm6=2:* \
    libxext6=2:* \
    nano \
    python3 \
    python3-pip \
    python3-dev \
    ninja-build \
    build-essential \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/* \
    nvidia-cuda-toolkit

# Set default python version
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install torch
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy source to workspace
WORKDIR /workspace
ADD requirements.txt /workspace/
RUN pip3 install -r /workspace/requirements.txt

# Verify CUDA installation
RUN nvcc --version

# Clone GroundingDINO repo and weights
WORKDIR /workspace
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git
WORKDIR /workspace/GroundingDINO
RUN git checkout -q 57535c5a79791cb76e36fdb64975271354f10251
RUN pip3 install -e .
RUN mkdir weights
WORKDIR /workspace/GroundingDINO/weights
RUN wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
# RUN cd /workspace/GroundingDINO && python setup.py install  # Should not be necessary


# Clone Segment Anything (SAM) repo and weights
WORKDIR /workspace
RUN pip install 'git+https://github.com/facebookresearch/segment-anything.git@6fdee8f'
RUN mkdir weights
WORKDIR /workspace/weights
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Add example images
WORKDIR /workspace/example_images
ADD example_images /workspace/example_images/
# WORKDIR /workspace/Ferrag/
# ADD Ferrag /workspace/Ferrag/
WORKDIR /workspace/Packages2Overlay
ADD Packages2Overlay /workspace/Packages2Overlay


WORKDIR /workspace/
ADD otcempty1.bmp /workspace/


# Cleanup
RUN apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Set up a working directory (optional)
WORKDIR /workspace
RUN mkdir synthetic_overlays

ADD requirements.txt Overlay2.py Pseudolabel_Demo.py cartel2roLabelImg.py padimg4labeling.py min_in_image_area_rect.py /workspace/
# CMD ["python", "/workspace/GroundingDINO/setup.py", "install"]
