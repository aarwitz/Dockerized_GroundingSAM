# Use an official NVIDIA CUDA base image with Ubuntu as the base OS
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04


# Set environment variables
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV DEBIAN_FRONTEND=noninteractive
# ENV FORCE_CUDA="1"
# ENV CUDA_VISIBLE_DEVICES="0"
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

# Requirements
WORKDIR /workspace
ADD requirements.txt /workspace/
RUN pip3 install -r /workspace/requirements.txt
RUN pip install -U openmim
RUN mim install mmengine
RUN mim install "mmcv>=2.0.0"
RUN mim install mmdet
RUN mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .

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
WORKDIR /workspace/
ADD empty_conveyor.bmp /workspace/

# Cleanup
RUN apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Copy source to workspace
# COPY /home/aaron/Dockerized_GroundingSAM/utilities/cartel2roLabelImg.py /workspace/
ADD /utilities /workspace/utilities/
ADD /gradio_demo /workspace/gradio_demo/
ADD label_app.py /workspace/

# Load models
# ENTRYPOINT ["/bin/bash","-c","python label_app.py"]
# CMD ["tail", "-f", "/dev/null"]
# ENTRYPOINT [["/bin/bash"]]
# CMD ["sh", "-c", "python label_app.py"]

# RUN python label_app.py
