from PIL import Image
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

def add_padding_to_images(input_folder, output_folder, padding=300):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if not f.endswith('.xml')]

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        xml_file = os.path.join(input_folder, os.path.splitext(image_file)[0] + ".xml")
        
        # Open the image using PIL
        image = Image.open(image_path)

        # Get the original image dimensions
        original_width, original_height = image.size

        # Calculate the new dimensions with padding
        new_width = original_width + 2 * padding
        new_height = original_height + 2 * padding

        # Create a new image with the specified dimensions and fill it with a transparent background
        new_image = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))

        # Calculate the position to paste the original image in the center of the new image
        offset_x = padding
        offset_y = padding

        # Paste the original image onto the new image
        new_image.paste(image, (offset_x, offset_y))

        # Save the new image with padding to the output folder
        output_image_path = os.path.join(output_folder, image_file)
        new_image.save(output_image_path)

        # Process the XML label
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for obj in root.findall('object'):
            robndbox = obj.find('robndbox')
            cx = float(robndbox.find('cx').text) + padding
            cy = float(robndbox.find('cy').text) + padding
            robndbox.find('cx').text = str(cx)
            robndbox.find('cy').text = str(cy)

        # Save the modified XML label to the output folder
        output_xml_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + ".xml")
        tree.write(output_xml_path)
    # Remove directory containing original unpadded images and labels
    shutil.rmtree(input_folder)

if __name__ == "__main__":
    # Set the input and output folders here
    input_folder = Path("/home/g5_team3/Pictures/SBS_Labeled/ChewyRoller")
    output_folder = Path("/home/g5_team3/Pictures/ChewyRoller_Labeled")

    # Add padding to the images and corresponding XML labels
    add_padding_to_images(input_folder, output_folder)

