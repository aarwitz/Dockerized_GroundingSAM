from typing import List, Tuple
import numpy as np
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
 
def filter_bboxes_by_area(bboxes: np.ndarray, min_area: int, max_area: int) -> np.ndarray:
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
 