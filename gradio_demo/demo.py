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
from pathlib import Path
import math
import torch
import cv2
import os
from typing import List, Tuple
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from utilities.file_mgmt import suppress_stdout, empty_directory_and_subdirectories
from utilities.filters import *
from utilities.visualize_mask import visualize_mask
from utilities.file_mgmt import create_timestamped_dir
from utilities.filters import filter_bboxes_by_IOU
from mmdet.apis import DetInferencer

def load_models():
    # declare global variables
    global grounding_dino_model
    global sam_predictor
    global mmdet_inferencer
    # GroundingDINO model and config
    config_path = r"/workspace/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    weights_path = r"/workspace/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    grounding_dino_model = Model(model_config_path=config_path, model_checkpoint_path=weights_path)
    # Sam model and config
    SAM_ENCODER_VERSION = "vit_h"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam_weights_path = r"/workspace/weights/sam_vit_h_4b8939.pth"
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=sam_weights_path).to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    # GroundingDINO finetuned blue marker model
    config_path = r"workspace/mmdet/groundingdino/bluemarker_config.py"
    weights_path = r"workspace/mmdet/groundingdno/bluemarker.pth"
    mmdet_inferencer = DetInferencer(model=config_path, weights=weights_path)

def set_search_params(prompt: str, confidence_score: float):
    global CLASSES
    global BOX_TRESHOLD
    global TEXT_TRESHOLD
    prompt = prompt.split(',')
    CLASSES = [item.strip() for item in prompt]
    BOX_TRESHOLD = confidence_score
    TEXT_TRESHOLD = confidence_score
    print('in set search paramers, CLASSES:', CLASSES)

def dino_detect(image: np.ndarray, max_iou: float) -> np.ndarray:
    # pass in images with prompt and thresholds
    print('tuple(Classes)',tuple(CLASSES))
    with suppress_stdout():   # suppress prints from GroundingDINO module
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=tuple(CLASSES),
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
    # xyxy bbox min/max corner points from all detections
    xyxy = detections.xyxy
    # class_ids of these detections, in same order as bboxes
    class_ids = detections.class_id
    # Create a boolean mask based on the target class IDs
    CLASSES_EXCLUDED_IDX = []
    detections_mask = np.logical_not(np.isin(class_ids, CLASSES_EXCLUDED_IDX))  # to implement negative prompts
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
    print('classfiltered_xyxy: ', classfiltered_xyxy)
    #### Filter: Define IOU thresholds to filter out duplicate detections
    ioufiltered_xyxy = filter_bboxes_by_IOU(classfiltered_xyxy, max_iou)
    print('\nApply IOU filter:')
    print(ioufiltered_xyxy)
    return ioufiltered_xyxy


def dino_blue_marker_detect(image: np.ndarray, max_iou: float) -> np.ndarray:
    inferencer = DetInferencer(model='/home/aaron/mmdetection/configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_cat.py', weights='/home/aaron/mmdetection/marker_work_dir/epoch_62.pth')
    results_dict = inferencer(inputs='/home/aaron/mmdetection/demo/demo.jpg',texts='bench .',pred_score_thr=.3,out_dir='outputs')
    # Assuming results_dict is your dictionary
    predictions = results_dict['predictions'][0]
    # Filter by score
    filtered_bboxes_xyxy = []
    while predictions['scores'][i] > max_iou:
        filtered_bboxes_xyxy += [predictions['bboxes'][i]]
        i+=1
    #### Filter: Define IOU thresholds to filter out duplicate detections
    ioufiltered_xyxy = filter_bboxes_by_IOU(filtered_bboxes_xyxy, max_iou)
    return np.array(ioufiltered_xyxy)

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
    mask = segment(
    sam_predictor=sam_predictor,
    image=image,
    xyxy=bbox
    )
    return mask

def rotate_bbox(image: np.ndarray, masks: np.ndarray, image_fname: str, output_path: Path) -> np.ndarray:
    box_info_4json = []
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
        box_info_4json += [get_rotated_bounding_box_info(box)]
    cv2.imwrite(str(output_path / "labels_visualized" / image_fname), image)
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
 
def stack_masks(mask: np.ndarray, image_shape: Tuple, image_fname: str, output_path: Path) -> np.ndarray:
    if mask.ndim < 3:
        return np.ones(image_shape, dtype=np.uint8)
    elif len(mask) > 1:
        # Stack the masks along a new dimension (axis 0)
        stacked_masks = np.stack(mask, axis=0)
        # Combine masks using logical OR along the new dimension
        combined_mask = np.logical_or.reduce(stacked_masks).astype(np.uint8)
        mask = combined_mask  # Use the combined mask as the final mask
    mask = np.squeeze(mask).astype(np.uint8)
    # cv2.imwrite(str(output_path/image_fname),mask*255)
    return mask

def create_output_directory(output_path: Path) -> None:
    if os.path.exists(str(output_path)):
        empty_directory_and_subdirectories(output_path)
    # Create the base directory
    output_path.mkdir(parents=False, exist_ok=True)
    # Create folder for saving masks of detections
    # (subdirectory3_path := output_path / "masks").mkdir(parents=True, exist_ok=True)
    # Create folder for visualizing detections (rotated bounding boxes)
    (subdirectory4_path := output_path / "labels_visualized").mkdir(parents=True, exist_ok=True)

def main(
    image_path: Path = Path('/workspace/example_images'),
    output_path: Path = Path('/workspace/tool_output'),
    confidence_score: float = 0.2,
    prompt: str = "package on conveyor",
    max_iou: float = 0.0,
    model_selection: str = 'SwinT'
    ) -> Path:
    stamped_output_path = create_timestamped_dir(base_path=output_path)
    set_search_params(prompt,confidence_score)
    create_output_directory(output_path=stamped_output_path)
    print('tool_output to:', stamped_output_path)
    for image_fname in os.listdir(image_path):
        # Load image
        image = cv2.imread(str(image_path / image_fname))
        # Detect with GroundingDINO
        if model_selection == 'SwinT':
            bboxes = dino_detect(image, max_iou=max_iou)
        else:
            bboxe = dino_blue_marker_detect(image, max_iou=max_iou)
        if len(bboxes) < 1:
            return None
        # Pass detection to SAM
        masks = run_SAM(image,bboxes)
        # Rotate bboxes using masks and write annotation to image and save
        box_info = rotate_bbox(image, masks, image_fname, stamped_output_path)
        # Combine masked items into one mask
        mask = stack_masks(masks,image.shape,image_fname,stamped_output_path)
        # Visualize rotated bbox and labels
        if len(masks) > 0:
            visualize_mask(stamped_output_path / "labels_visualized", image_fname, mask)
    return stamped_output_path

def parse_args():
    parser = argparse.ArgumentParser()
    # Define command-line arguments
    parser.add_argument('--image_path', type=str, required=False,
                        help="Path in docker container to the folder containing images.")
    parser.add_argument('--confidence_score', type=float, required=False,
                        help="Confidence threshold for GroundingDINO detections.")
    parser.add_argument('--prompt', type=str, required=False,
                        help="Prompt for GroundingDINO detections.")
    parser.add_argument('--max_iou', type=float, required=False, default = 0,
                        help="IOU threshold for detections")
    args = parser.parse_args()
    return args
 
if __name__ == '__main__':
    args = parse_args()
    tool_output = Path('/workspace/tool_output')
    if args.image_path is None:
        load_models()  # If no arguments are provided, just load models
    else:
        load_models()
        main (
            image_path = Path(args.image_path),
            output_path = tool_output,
            confidence_score = args.confidence_score,
            prompt = args.prompt,
            max_iou=args.max_iou
        )