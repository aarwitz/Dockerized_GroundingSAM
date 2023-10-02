import cv2
import numpy as np

def create_roi(image_path, target_width, target_height):

    # Read image
    img_raw = cv2.imread(image_path)

    # Resize the image
    img_resized = cv2.resize(img_raw, (target_width, target_height))

    # Select ROI function on the resized image
    roi_resized = cv2.selectROI(img_resized)

    # Calculate the scale factor for resizing back to the original size
    scale_factor_x = img_raw.shape[1] / img_resized.shape[1]
    scale_factor_y = img_raw.shape[0] / img_resized.shape[0]

    # Scale the ROI back to the original size
    roi = (
        int(roi_resized[0] * scale_factor_x),
        int(roi_resized[1] * scale_factor_y),
        int(roi_resized[2] * scale_factor_x),
        int(roi_resized[3] * scale_factor_y)
    )

    # Crop selected ROI from raw image
    roi_cropped = img_raw[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]

    # Show cropped image
    cv2.imshow("ROI", roi_cropped)
    cv2.waitKey(0)

    # Calculate roi area
    roi_area = (roi[2] - roi[0]) * (roi[3] - roi[1])

    return roi, roi_area

# Path to the image
image_path = r"/home/aaron/Dockerized_GroundingSAM/example_images/Image_0001_20230719180951.bmp"

# Target dimensions for resizing
target_width = 1280
target_height = 960

# Call the function to define ROI
roi, roi_area = create_roi(image_path, target_width, target_height)
print('roi:', roi)
print('roi_area (minmax_area):', roi_area)
