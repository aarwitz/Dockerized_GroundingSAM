import cv2
import numpy as np
import os

def visualize_mask(original_image_path, image_fname, mask ):

    # Load your mask (assuming it's a single-channel image, e.g., grayscale)
    # mask = cv2.imread(str(output_path/'masks'/image_fname), cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(str(original_image_path/image_fname))
    # Ensure the mask has an alpha channel (transparency)
    mask_with_alpha = cv2.merge([mask, mask, mask, mask])

    # Set the color tint for the masked region (adjust as needed, here it's green)
    color_tint = [0, 255, 0]  # BGR color for green

    # Set the transparency level (adjust as needed)
    transparency = 0.5

    # Apply the color tint to the masked region
    masked_region = original_image.copy()
    masked_region[np.where((mask_with_alpha[:, :, 0] > 0))] = color_tint

    # Create a transparent overlay by blending the original image and the tinted mask
    overlay = cv2.addWeighted(original_image, 1 - transparency, masked_region, transparency, 0)

    # Display the result or save it to a file
    # cv2.imshow('Overlay', overlay)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # img_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(original_image_path/image_fname),overlay)