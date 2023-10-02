

from PIL import Image
import json
import cv2
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple
import xml.etree.ElementTree as ET
import os
import shutil
from xml.dom import minidom




def get_image_depth(image_path: Path):
    image = Image.open(image_path)
    image_depth = image.mode
    return image_depth

def get_image_size(image_path: Path):
    with Image.open(image_path) as image:
        width, height = image.size
        return str(width), str(height)

def create_object_element(annotation: ET.Element, bbox: dict) -> None:
    object_elem = ET.SubElement(annotation, 'object')
    type_elem = ET.SubElement(object_elem, 'type')
    type_elem.text = 'robndbox'
    name_elem = ET.SubElement(object_elem, 'name')
    name_elem.text = ''
    pose_elem = ET.SubElement(object_elem, 'pose')
    pose_elem.text = 'Unspecified'
    truncated_elem = ET.SubElement(object_elem, 'truncated')
    truncated_elem.text = '0'
    difficult_elem = ET.SubElement(object_elem, 'difficult')
    difficult_elem.text = '0'

    robndbox_elem = ET.SubElement(object_elem, 'robndbox')
    cx_elem = ET.SubElement(robndbox_elem, 'cx')
    cx_elem.text = bbox["center_x"]
    cy_elem = ET.SubElement(robndbox_elem, 'cy')
    cy_elem.text = bbox["center_y"]
    w_elem = ET.SubElement(robndbox_elem, 'w')
    w_elem.text = str(bbox['width'])
    h_elem = ET.SubElement(robndbox_elem, 'h')
    h_elem.text = str(bbox['height'])
    angle_elem = ET.SubElement(robndbox_elem, 'angle')
    angle_elem.text = str(bbox['angle'])

def create_imagenet_xml(pseudolabel_output_path: Path) -> ET.Element:
    with open(str(pseudolabel_output_path / 'labels_cartel' / 'data.json'),'r') as cartel_json:
        data = json.load(cartel_json)

    for sample in tqdm(data["samples"]):
        annotation = ET.Element('annotation')
        annotation.set('verified', 'yes')

        image_path = str(pseudolabel_output_path / 'labels_imagenet_original' / data["samples"][sample]["image_id"])
        folder = ET.SubElement(annotation, 'folder')
        folder.text = image_path.split('/')[-2]
        filename = ET.SubElement(annotation, 'filename')
        filename.text = data["samples"][sample]["image_id"]
        path = ET.SubElement(annotation, 'path')
        path.text = str(pseudolabel_output_path / "labels_imagenet_original" / data["samples"][sample]["image_id"])
        source = ET.SubElement(annotation, 'source')
        database = ET.SubElement(source, 'database')
        database.text = 'Unknown'
        
        size = ET.SubElement(annotation, 'size')
        width = ET.SubElement(size, 'width')
        height = ET.SubElement(size, 'height')
        width.text, height.text =  get_image_size(image_path)
        depth = ET.SubElement(size, 'depth')
        depth.text = get_image_depth(image_path)
        
        segmented = ET.SubElement(annotation, 'segmented')
        segmented.text = '0'

        for bbox in data["samples"][sample]["bboxes"]:
            create_object_element(annotation, bbox)

        # Create the XML tree and write it to a file
        #tree = ET.ElementTree(annotation)
        tree_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent='  ')

        xml_name = data["samples"][sample]["image_id"].split('.')[0] + '.xml'
        with open(pseudolabel_output_path / 'labels_imagenet_original' / xml_name, 'w') as xml_file:
            xml_file.write(tree_str)




def main(pseudolabel_output_path: Path):
    # Convert from cartel to imagenet format and save xml
    create_imagenet_xml(pseudolabel_output_path)

