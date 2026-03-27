# Dockerized_GroundingSAM

Dockerized pseudolabeling workflow for prompt-based object detection and segmentation using **GroundingDINO** + **Segment Anything**.

This project was built around a warehouse/logistics labeling use case: generate labels from natural-language prompts, filter them geometrically, and export visualization artifacts plus rotated-box annotations.

## What it does
- Runs the labeling stack inside Docker
- Uses text prompts to detect target objects in images
- Applies segmentation masks to refine detections
- Filters detections by ROI, area, and IoU
- Writes:
  - labeled visualizations
  - masks
  - synthetic overlays
  - JSON label output

## Key files
- `pseudolabel.sh` — builds and launches the Docker environment
- `Dockerfile` — container definition
- `label_app.py` — main labeling application
- `gradio_demo/` — demo UI experiments
- `utilities/` — filtering, file management, visualization, and bounding-box utilities
- `example_images/` — sample input imagery

## Quick start
Build the image and launch the container:

```bash
./pseudolabel.sh
```

The launch script currently:
1. builds the Docker image as `pseudolabel_app`
2. starts a GPU-enabled container with an output volume mounted at `/workspace/tool_output`

## Example invocation inside the container
```bash
python label_app.py \
    --image_path '/workspace/Packages2Overlay' \
    --output_path '/workspace/Packages2Overlay_labeled' \
    --confidence_score 0.3 \
    --prompt 'parcel,package,clothing bag,jeans,bag,box,envelope,plastic,white square' \
    --background_path '/workspace/otcempty1.bmp' \
    --maxmin_area 2231850 70000 \
    --max_iou 0.01
```

## Expected environment
This repo assumes a development environment with:
- Docker
- NVIDIA GPU runtime support
- model weights mounted or available inside the container
- GroundingDINO / SAM dependencies installed through the image

## Outputs
The pipeline writes artifacts under the configured output directory, including visualized labels, masks, and generated overlay images for inspection.

## Related repo
This repo is the more containerized counterpart to:
- [`aarwitz/GroundingDINO_Autolabeling`](https://github.com/aarwitz/GroundingDINO_Autolabeling)
