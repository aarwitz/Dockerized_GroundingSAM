# Dockerized_GroundingSAM

## Install
./pseudolabel.sh

# Example
python label_app.py \
    --image_path '/workspace/Packages2Overlay' \
    --output_path '/workspace/Packages2Overlay_labeled' \
    --confidence_score 0.3 \
    --prompt 'parcel,package,clothing bag,jeans,bag,box,envelope,plastic,white square' \
    --background_path '/workspace/otcempty1.bmp' \
    --maxmin_area 2231850 70000 \
    --max_iou 0.01
