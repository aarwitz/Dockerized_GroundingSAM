import cv2
import numpy as np

def create_roi(image_path):

    #read image

    img_raw = cv2.imread(image_path)

    #select ROI function
    roi = cv2.selectROI(img_raw)

    #print rectangle points of selected roi
    print(roi)

    #Crop selected roi from raw image
    roi_cropped = img_raw[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

    #show cropped image
    cv2.imshow("ROI", roi_cropped)
    cv2.waitKey(0)

    # Calculate roi area
    roi_area = (roi[2]-roi[0])*(roi[3]-roi[1])

    return roi, roi_area


# Path to the image
image_path = r"/home/aaron/Downloads/labels_visualized/9.BMP"

# Call the function to define ROI
roi, roi_area = create_roi(image_path)
print('roi:',roi)
print('roi_area:',roi_area)