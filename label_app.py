"""
Input a folder of images and a natural language prompt
Output a folder with rotated bounding boxes around prompted items and a json storing position and class of bbox
 
Uses an adapted cv2.minAreaRect() to create rotated bounding boxes of objects beyond the image's edge
    #           . P
    #          /|\
    #        /  | \
    # -----/----.--\--------------------
    # |  /      B   \                  |
    # |  \           \ Q               |
    # |   \         /                  |
    # |    \      /                    |
    # |     \   /                      |
    # |      \/                        |
    # |                                |
    # |                                |
    # ----------------------------------
 
"""
import argparse
from PIL import Image
from pathlib import Path
import math
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from typing import List, Tuple
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import json
import shutil
from utilities.min_in_image_area_rect import min_in_image_area_rect
import utilities.cartel2roLabelImg as cartel2roLabelImg
from utilities.padimg4labeling import add_padding_to_images
from utilities.file_mgmt import suppress_stdout, empty_directory_and_subdirectories
from utilities.filters import *

# Create empty json to store labels in
cartel_json = {
"categories": {
    "0": {}
},
"samples": {}
}

def dino_detect(
        image: np.ndarray, 
        confidence_score: float, 
        prompt: list,
        roi: tuple,
        minmax_area: tuple,
        max_iou: float
        ) -> np.ndarray:
    # Define model config and weights
    config_path = r"/workspace/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    weights_path = r"/workspace/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    # Define classes and thresholds
    CLASSES = prompt
    CLASSES_EXCLUDED_IDX = []    # can insert a negative prompt here
    BOX_TRESHOLD = confidence_score
    TEXT_TRESHOLD = confidence_score
    # Create instance of model, pass in images with prompt and thresholds
    with suppress_stdout():   # suppress prints from GroundingDINO module
        grounding_dino_model = Model(model_config_path=config_path, model_checkpoint_path=weights_path)
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
    # xyxy bbox min/max corner points from all detections
    xyxy = detections.xyxy
    # class_ids of these detections, in same order as bboxes
    class_ids = detections.class_id
    # Create a boolean mask based on the target class IDs
    detections_mask = np.logical_not(np.isin(class_ids, CLASSES_EXCLUDED_IDX))
    print('\n-----------------------------------------------------------------\n')
    count = 0
    for class_id in class_ids:
        if class_id is None:
             class_ids[count] = -1
             class_id = -1
             detections_mask[count] = False
        count+=1
    print('Detected objects:')
    for i in range(len(detections_mask)):
         if detections_mask[i] == True:
              print(CLASSES[class_ids[i]],end = ", confidence = ")
              print(detections.confidence[i])
    #### Filter: Classes extracted from language prompt
    print('\nDetected object bounding boxes in [xmin ymin xmax ymax] format:')
    classfiltered_xyxy = xyxy[detections_mask]
    print(classfiltered_xyxy)
    #### Filter: Define ROI
    roifiltered_xyxy = filter_bboxes_by_roi(roi, classfiltered_xyxy)
    print('\nApply ROI filter:')
    print(roifiltered_xyxy)
    #### Filter: Define max area
    areafiltered_xyxy = filter_bboxes_by_area(roifiltered_xyxy, minmax_area[0], minmax_area[1])
    print('\nApply area filter:')
    print(areafiltered_xyxy)   
    #### Filter: Define IOU thresholds to filter out duplicate detections
    ioufiltered_xyxy = filter_bboxes_by_IOU(areafiltered_xyxy, max_iou)
    print('\nApply IOU filter:')
    print(ioufiltered_xyxy)
    return ioufiltered_xyxy

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)
 
def run_SAM(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    SAM_ENCODER_VERSION = "vit_h"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam_weights_path = r"/workspace/weights/sam_vit_h_4b8939.pth"
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=sam_weights_path).to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    mask = segment(
    sam_predictor=sam_predictor,
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    xyxy=bbox
    )
    return mask
   
def add_dino_pseudolabel_to_json(box_info: list, count: int, image_fname: str) -> None:
    bbox_list = [] # empty list for bboxes
    for bbox in box_info:
        center_x, center_y, angle, width, height = bbox
        bbox_data = {
            "angle": str(np.deg2rad(angle)),
            "category_id": "0",
            "center_x": str(center_x),
            "center_y": str(center_y),
            "height": str(height),
            "width": str(width)
        }
        bbox_list.append(bbox_data)
    cartel_json["samples"][str(count)] = {
        "bboxes": bbox_list,
        "image_id": image_fname
    }

def rotate_bbox(image: np.ndarray, masks: np.ndarray, image_fname: str, output_path: Path, count: int) -> np.ndarray:
    box_info_4json = []
    cv2.imwrite(str(output_path / "labels_imagenet" / image_fname), image)
    for mask in masks:
        # get white pixels in mask
        coords = np.column_stack(np.where(mask.transpose() > 0))
        coords = coords.astype(np.int32)
        # get rotated rectangle that bounds the mask
        rotrect = cv2.minAreaRect(coords)   # substitute with propietary version later (uncomment below)
        # height, width, _ = image.shape
        # rotrect = min_in_image_area_rect(coords, (width, height))  # or use cv2.minAreaRect()
        box = np.intp(cv2.boxPoints(rotrect))
        # Draw the rotated rectangle on the original image
        cv2.drawContours(image, [box], 0, (0,0,255), 4)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    # Uncomment this line if you input BGR images
        box_info_4json += [get_rotated_bounding_box_info(box)]
    cv2.imwrite(str(output_path / "labels_visualized" / image_fname), image)
    print('Visualizing rotated bounding box label in: /Dockerized_GroundingSAM/tool_output/labels_visualized/'+ image_fname)
    return box_info_4json

def get_rotated_bounding_box_info(box):
    # Calculate the center point of the bounding box
    center_x = np.mean(box[:, 0])
    center_y = np.mean(box[:, 1])
    # Calculate the angle of rotation
    dx = box[1, 0] - box[0, 0]
    dy = box[1, 1] - box[0, 1]
    angle = np.arctan2(dy, dx) * (180 / math.pi)
    # Calculate the width and height of the bounding box
    width = np.linalg.norm(box[1] - box[0])
    height = np.linalg.norm(box[2] - box[1])
    return center_x, center_y, angle, width, height
 
def stack_masks(mask: np.ndarray, image_shape: Tuple, image_fname: str) -> np.ndarray:
    if mask.ndim < 3:
        return np.ones(image_shape, dtype=np.uint8)
    elif len(mask) > 1:
        # Stack the masks along a new dimension (axis 0)
        stacked_masks = np.stack(mask, axis=0)
        # Combine masks using logical OR along the new dimension
        combined_mask = np.logical_or.reduce(stacked_masks).astype(np.uint8)
        mask = combined_mask  # Use the combined mask as the final mask
    mask = np.squeeze(mask).astype(np.uint8)
    cv2.imwrite('/workspace/tool_output/masks/'+image_fname,mask*255)
    return mask

def overlay(old_path: Path,new_path: Path, mask: np.ndarray, count: int) -> None:
    background = Image.open(new_path).convert('RGB')
    if len(mask.shape) < 2 or np.all(mask==1):
        return
    foreground = Image.open(old_path).convert('RGB')
    height, width = mask.shape
    foreground = foreground.resize((width, height))
    background = background.resize((width, height))
    mask = cv2.resize(mask, (width, height))
    # Convert the images to numpy arrays
    foreground = np.array(foreground)
    background = np.array(background)
    # Create an alpha channel for the mask
    alpha = np.expand_dims(mask, axis=-1)
    # Create a new image by combining the foreground and background using the mask as the alpha channel
    new_image = np.where(alpha == 1, foreground, background)
    # Convert the new image to a PIL Image and display it
    new_image = Image.fromarray(new_image.astype(np.uint8))
    #new_image.show()
    synthetic_output_path = r"/workspace/tool_output/synthetic_overlays/"
    synthetic_image_path = synthetic_output_path + old_path.name
    new_image.save(synthetic_image_path)
    print('Saving synthetic image to: /Dockerized_GroundingSAM/tool_output/labels_visualized/' + old_path.name)

def create_output_directory(output_path: Path) -> None:
    if os.path.exists(str(output_path)):
        empty_directory_and_subdirectories(output_path)
    # Create the base directory
    output_path.mkdir(parents=False, exist_ok=True)
    # Create cartel (.json) labels folder
    (subdirectory1_path := output_path / "labels_imagenet").mkdir(parents=True, exist_ok=True)
    # Create imagenet (.xml) labels folder
    (subdirectory2_path := output_path / "labels_cartel").mkdir(parents=True, exist_ok=True)
    # Create folder for saving masks of detections
    (subdirectory3_path := output_path / "masks").mkdir(parents=True, exist_ok=True)
    # Create folder for visualizing detections (rotated bounding boxes)
    (subdirectory4_path := output_path / "labels_visualized").mkdir(parents=True, exist_ok=True)
    # Create synthetic overlays folder
    (subdirectory5_path := output_path / "synthetic_overlays").mkdir(parents=True, exist_ok=True)

def copy_xml(source_dir, destination_dir) -> None:
    # List all files in the source directory
    files = os.listdir(source_dir)
    # Iterate through the files and copy .xml files to the destination directory
    for file in files:
        if file.endswith(".xml"):
            source_file = os.path.join(source_dir, file)
            destination_file = os.path.join(destination_dir, file)
            shutil.copy2(source_file, destination_file)
            print(f"Copied: {file} to {destination_dir}")
    print("All .xml files have been copied.")

def main(image_path: Path = Path('/workspace/example_images'), 
         output_path: Path = Path('/workspace/tool_output'), 
         confidence_score: float = 0.3,
         prompt: str = "item on conveyor",
         background_path: Path = "none",
         roi: tuple = (0, 0, 10e10, 10e10),
         minmax_area: tuple = (0, 10e10),
         max_iou: float = 0.5) -> None:
    # create output directory structure
    create_output_directory(output_path=output_path)
    # Define ROI using first image in dataset
    count = 0
    for image_fname in os.listdir(image_path):
        # Load image
        image = cv2.imread(str(image_path / image_fname))
        # Detect with GroundingDINO
        bboxes = dino_detect(image, confidence_score, prompt, roi, minmax_area, max_iou)
        # Pass detection to SAM
        masks = run_SAM(image,bboxes)
        # Rotate bboxes using masks and write annotation to image and save
        box_info = rotate_bbox(image, masks, image_fname, output_path, count)
        # Store label in Cartel json
        add_dino_pseudolabel_to_json(box_info, count, image_fname)
        # Combine masked items into one mask
        mask = stack_masks(masks,image.shape,image_fname)
        # Ovelay masked objects onto background
        if str(background_path) != "none":
            print(str(background_path))
            overlay(old_path = image_path / image_fname, new_path = background_path, mask = mask, count = count)
        count+=1
    # Write json
    cartel_json_output_path = output_path / "labels_cartel" / "data.json"
    with open(cartel_json_output_path, 'w') as file:
        json.dump(cartel_json, file, indent=4)
    print(f"Cartel data saved to {cartel_json_output_path} as JSON.")
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Input image pa")
   
    # Define command-line arguments
    parser.add_argument('--image_path', type=str, required=True,
                        help="Path in docker container to the folder containing images.")
    parser.add_argument('--confidence_score', type=float, required=True,
                        help="Confidence threshold for GroundingDINO detections.")
    parser.add_argument('--prompt', type=str, required=True,
                        help="Prompt for GroundingDINO detections.")    
    parser.add_argument('--background_path', type=str, required=False, default = "none",
                        help="Path to a single background image to overlay masks onto.")
    parser.add_argument('--roi', type=int, nargs='+', required=False, default = (0, 0, 10e10, 10e10),
                        help="Region of interest (xmin,ymin,xmax,ymax).")  
    parser.add_argument('--minmax_area', type=int, nargs='+', required=False, default = (0, 10e10),
                        help="Minimum and maximum are of detections")    
    parser.add_argument('--max_iou', type=float, required=False, default = 0.5,
                        help="IOU threshold for detections")
    args = parser.parse_args()
    tool_output = Path('/workspace/tool_output')
    main (
        image_path = Path(args.image_path),
        output_path = tool_output,
        confidence_score = args.confidence_score,
        prompt = args.prompt.split(','),
        background_path = Path(args.background_path),
        roi = tuple(args.roi),
        minmax_area = tuple(args.minmax_area),
        max_iou = args.max_iou
    )

    # Store the labels in both imagenet .xml and cartel .json
    cartel2roLabelImg.main(
        pseudolabel_output_path = tool_output
    )
    # shutil.rmtree(Path(args.output_path) / "labels_cartel") # comment out to keep cartel labels
    copy_xml(Path(args.output_path)/"labels_imagenet", '/workspace/synthetic_overlays/')
    #     Pad images with 200px of black on each side, to allow for out-of-bounds labeling
    # ### Leave as optional
    # add_padding_to_images(
    #      input_folder = output_path / "labels_imagenet_original",
    #      output_folder = output_path / "labels_imagenet"
    # )

"""
python label_app.py \
--image_path '/workspace/example_images' \
--confidence_score 0.2 \
--prompt 'box,package,parcel,item on conveyor' \
--background_path '/workspace/empty_conveyor.bmp' \
--roi 100 0 2300 2048 \
--minmax_area 70000 2231850 \
--max_iou 0.3
"""