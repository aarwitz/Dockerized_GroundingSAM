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
from min_in_image_area_rect import min_in_image_area_rect
import cartel2roLabelImg
from padimg4labeling import add_padding_to_images
 
# Create empty cartel json to store labels in
cartel_json = {
"categories": {
    "0": {}
},
"samples": {}
}
 
def calculate_iou(box1, box2):
    # Calculate the intersection area
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])
   
    intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)
   
    # Calculate the union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
   
    # Calculate IOU
    iou = intersection_area / union_area
    return iou
 
 
 
def filter_bboxes_by_roi(roi: Tuple[int, int, int, int], bboxes: np.ndarray) -> np.ndarray:
    if len(bboxes.shape) == 1:
         return np.array([])
    filtered_bboxes = []
    roi_x_min, roi_y_min, roi_x_max, roi_y_max = roi
   
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        # Check if the bounding box exceeds the roi boundaries
        if x_min >= roi_x_min and y_min >= roi_y_min and x_max <= roi_x_max and y_max <= roi_y_max:
            filtered_bboxes.append(bbox)
   
    return np.array(filtered_bboxes)
 
def filter_bboxes_by_area(bboxes: np.ndarray, max_area: int, min_area: int) -> np.ndarray:
    # Calculate the areas of each bounding box
    if len(bboxes.shape) == 1:
        return np.array([])
 
    # Calculate the areas of each bounding box
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
 
    # Find indices of bounding boxes that satisfy the area condition
    indices_to_keep = np.where((areas <= max_area) & (areas >= min_area))[0]
 
    # Filter the bounding boxes based on the indices
    filtered_bboxes = bboxes[indices_to_keep]
   
    return filtered_bboxes
 
def filter_bboxes_by_IOU(bboxes: np.ndarray, max_iou: int) -> np.ndarray:
    if len(bboxes.shape) == 1:
         return np.array([])
    num_boxes = len(bboxes)
    duplicates = set()
   
    for i in range(num_boxes):
        if i not in duplicates:
            for j in range(i + 1, num_boxes):
                if j not in duplicates:
                    iou = calculate_iou(bboxes[i], bboxes[j])
                    if iou > max_iou:
                        duplicates.add(j)
    filtered_bboxes = np.delete(bboxes, list(duplicates), axis=0)
    return filtered_bboxes
 

def dino_detect(
        image: np.ndarray, 
        confidence_score: float, 
        prompt: list,
        maxmin_area: tuple,
        max_iou: float
        ) -> np.ndarray:
    # Define model
    config_path = r"/workspace/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    weights_path = r"/workspace/GroundingDINO/weights/groundingdino_swint_ogc.pth"

    grounding_dino_model = Model(model_config_path=config_path, model_checkpoint_path=weights_path)

    # Define classes and thresholds
    CLASSES = prompt
    CLASSES_EXCLUDED_IDX = []
    BOX_TRESHOLD = confidence_score
    TEXT_TRESHOLD = confidence_score

    # detect object
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
    ########### Filters ###################

    # Class filter
    print('\n')
    print('-------------------------------------------------------')
    print('\n')
    print('Detected items: ')
    count = 0
    for class_id in class_ids:
        if class_id is None:
             class_ids[count] = -1
             class_id = -1
             print(CLASSES[class_id])
             detections_mask[count] = False
        if isinstance(class_id, int) or isinstance(class_id, np.int64):
            print(CLASSES[class_id])
        count+=1
    print('\n')

 

    # Print item classes to be labeled
    print('Labeled items: ')
    for i in range(len(detections_mask)):
         if detections_mask[i] == True:
              print(CLASSES[class_ids[i]],end = ", confidence = ")
              print(detections.confidence[i])

    print('\n')

    # Filter the rows based on the mask
    classfiltered_xyxy = xyxy[detections_mask]

    #### Filter: Define ROI
    roi = (0, 0, 50000, 600000)
    roifiltered_xyxy = filter_bboxes_by_roi(roi, classfiltered_xyxy)
    # Print changes made from ROI filter
    print('[ xmin, ymin, xmax, ymax ]')
    print('pre-filter for roi')
    print(classfiltered_xyxy)
    print('post-filter for roi')
    print(roifiltered_xyxy)

    ## Filter: Define max area
    # max_area = 1531850
    # min_area = 15000
    areafiltered_xyxy = filter_bboxes_by_area(roifiltered_xyxy, maxmin_area[0], maxmin_area[1])

    # Print changes made from ROI filter
    print('\nArea')
    print('pre-filter for area')
    print(roifiltered_xyxy)
    print('post-filter for area')
    print(areafiltered_xyxy)   

    ### Filter: Define IOU thresholds to filter out duplicate detections
    # max_iou = 0.5
    ioufiltered_xyxy = filter_bboxes_by_IOU(areafiltered_xyxy, max_iou)

    # Print changes made from ROI filter
    print('\nIOU')
    print('pre-filter for IOU')
    print(areafiltered_xyxy)
    print('post-filter for IOU')
    print(ioufiltered_xyxy)

    print('----------------------------------------------------------')

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
   
def add_dino_autolabel_to_json(box_info: list, count: int, image_fname: str) -> None:
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
    #mport matplotlib.pyplot as plt
    box_info_4json = []
    cv2.imwrite(str(output_path / "labels_imagenet_original" / image_fname), image)
    for mask in masks:
        # get white pixels in mask
        coords = np.column_stack(np.where(mask.transpose() > 0))
        coords = coords.astype(np.int32)
        # get rotated rectangle that bounds the mask
        # rotrect = cv2.minAreaRect(coords)
        height, width, _ = image.shape
        rotrect = min_in_image_area_rect(coords, (width, height))
        # rotated rectangle box points
        box = np.int0(cv2.boxPoints(rotrect))
        # Draw the rotated rectangle on the original image
        cv2.drawContours(image, [box], 0, (0,0,255), 4)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        box_info_4json += [get_rotated_bounding_box_info(box) ]
    # cv2.imwrite(str(output_path / "labels_visualized" / (str(count) + ".BMP"), image) 
    cv2.imwrite(str(output_path / "labels_visualized" / image_fname), image)
    print(str(output_path / "labels_visualized" / image_fname))

    print('Visualizing label in: ',str(output_path / "labels_visualized" / (str(count) + ".BMP")))
    mounted_volume_dir = Path("/workspace/output_labels/labels_visualized/" + str(count) + ".BMP")
    print(mounted_volume_dir)
    cv2.imwrite(str(mounted_volume_dir),image)
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
 
def create_output_directory(output_path: Path) -> None:
    if os.path.exists(str(output_path)):
        shutil.rmtree(str(output_path))
    # Create the base directory
    output_path.mkdir(parents=False, exist_ok=False)
    # Create cartel (.json) labels folder
    (subdirectory1_path := output_path / "labels_imagenet").mkdir(parents=True, exist_ok=False)
    #  Create cartel (.json) labels folder
    (subdirectory1_path := output_path / "labels_imagenet_original").mkdir(parents=True, exist_ok=False)
    # Create imagenet (.xml) labels folder
    (subdirectory1_path := output_path / "labels_cartel").mkdir(parents=True, exist_ok=False)
    # Create labeled_images folder
    (subdirectory2_path := output_path / "labels_visualized").mkdir(parents=True, exist_ok=False)
 



def overlay(old_path: Path,new_path: Path, mask: np.ndarray, count: int):
    print('mask.ndim: ', mask.ndim)
    print(type(mask))
    if mask.ndim < 3:
        print('No detections to overlay.')
        return
    elif len(mask) > 1:
        # Stack the masks along a new dimension (axis 0)
        stacked_masks = np.stack(mask, axis=0)

        # Combine masks using logical OR along the new dimension
        combined_mask = np.logical_or.reduce(stacked_masks).astype(np.uint8)
        mask = combined_mask  # Use the combined mask as the final mask
    
    mask = np.squeeze(mask).astype(np.uint8)
    print(mask.shape)

    # if mask.ndim > 2:
    #     mask1 = mask[0]  # First mask
    #     mask2 = mask[1]  # Second mask

    #     # Combine masks using logical OR (| operator)
    #     mask = np.logical_or(mask1, mask2)
    # visualize the predicted masks
    #plt.imshow(mask, cmap='gray')
    #plt.show()
    foreground = Image.open(old_path).convert('RGB')
    background = Image.open(new_path).convert('RGB')
    height, width = mask.shape
    foreground = foreground.resize((width, height))
    background = background.resize((width, height))
    mask = cv2.resize(mask, (width, height))

    # Convert the images to numpy arrays
    #mask = np.array(mask)
    foreground = np.array(foreground)
    background = np.array(background)
    # Create an alpha channel for the mask
    
    alpha = np.expand_dims(mask, axis=-1)

    # Create a new image by combining the foreground and background using the mask as the alpha channel
    new_image = np.where(alpha == 1, foreground, background)

    # Convert the new image to a PIL Image and display it
    new_image = Image.fromarray(new_image.astype(np.uint8))
    #new_image.show()
    output_path = r"/workspace/synthetic_overlays/"
    # new_image.save(output_path + "synthetic_blender" + str(count) + ".png")
    # print('Saving image to',output_path+ "synthetic_blender" + str(count) + ".png")
    synthetic_image_path = output_path + old_path.name
    new_image.save(synthetic_image_path)
    print('Saving synthetic image to: ', synthetic_image_path)

def copy_xml(source_dir, destination_dir):

    # List all files in the source directory
    files = os.listdir(source_dir)

    # Iterate through the files and copy .xml files to the destination directory
    for file in files:
        if file.endswith(".xml"):
            source_file = os.path.join(source_dir, file)
            destination_file = os.path.join(destination_dir, file)
            
            # Copy the .xml file to the destination directory
            shutil.copy2(source_file, destination_file)
            print(f"Copied: {file} to {destination_dir}")

    print("All .xml files have been copied.")

def main(image_path: Path, output_path: Path, confidence_score: float, prompt: str, background_path: Path, maxmin_area: tuple, max_iou: float) -> None:
    # create output directory structure
    create_output_directory(output_path=output_path)
    # Define ROI using first image in dataset
    count = 0
    for image_fname in os.listdir(image_path):
        # Load image
        image = cv2.imread(str(image_path / image_fname))
        # Detect with GroundingDINO
        bboxes = dino_detect(image, confidence_score, prompt, maxmin_area, max_iou)
        # Pass detection to SAM
        masks = run_SAM(image,bboxes)
        # Rotate bboxes using masks and write annotation to image and save
        box_info = rotate_bbox(image, masks, image_fname, output_path, count)
        # Store label in Cartel json
        add_dino_autolabel_to_json(box_info, count, image_fname)
        

        # Ovelay
        overlay(old_path = image_path / image_fname, new_path = background_path, mask = masks, count = count)
        
        count+=1
    # Write json
    cartel_json_output_path = output_path / "labels_cartel" / "data.json"
    with open(cartel_json_output_path, 'w') as file:
        json.dump(cartel_json, file, indent=4)
    print(f"Cartel data saved to {cartel_json_output_path} as JSON.")

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Input image path and labels output path.")
   
    # Define command-line arguments
    parser.add_argument('--image_path', type=str, required=True,
                        help="Path in docker container to the folder containing images.")
    parser.add_argument('--output_path', type = str, required=True,
                        help = "Path in docker container to output pseudolabels to")
    parser.add_argument('--confidence_score', type=float, required=True,
                        help="Confidence threshold for GroundingDINO detections.")
    parser.add_argument('--prompt', type=str, required=True,
                        help="Prompt for GroundingDINO detections.")    
    parser.add_argument('--background_path', type=str, required=True,
                        help="Path to a single background image to overlay masks onto.")  
    parser.add_argument('--maxmin_area', type=int, nargs='+', required=True,
                        help="Path to a single background image to overlay masks onto.")    
    parser.add_argument('--max_iou', type=float, required=True,
                        help="IOU threshold for detections")
    args = parser.parse_args()

    # Pass arguments to main
    main (
        image_path = Path(args.image_path),
        output_path = Path(args.output_path),
        confidence_score = args.confidence_score,
        prompt = args.prompt.split(','),
        background_path = Path(args.background_path),
        maxmin_area = tuple(args.maxmin_area),
        max_iou = args.max_iou
    )
 
    # Store the labels in both imagenet .xml and cartel .json
    cartel2roLabelImg.main(
        pseudolabel_output_path = Path(args.output_path)
    )
    shutil.rmtree(Path(args.output_path) / "labels_cartel") # comment out to keep cartel labels
    # Copy labels to mounted volume on host
    shutil.copytree(Path(args.output_path)/"labels_imagenet","/workspace/output_labels/labels/")
    copy_xml(Path(args.output_path)/"labels_imagenet_original", '/workspace/synthetic_overlays/')
    #     Pad images with 200px of black on each side, to allow for out-of-bounds labeling
    # ### Leave as optional
    # add_padding_to_images(
    #      input_folder = output_path / "labels_imagenet_original",
    #      output_folder = output_path / "labels_imagenet"
    # )
