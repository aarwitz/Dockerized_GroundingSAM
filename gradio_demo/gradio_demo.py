

import numpy as np
import gradio as gr
from pathlib import Path
import cv2
import sys
import os

outer_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(outer_dir)

from label_app import main

# def sepia(input_img):
#     img_path = Path('/workspace/gradio_demo/uploaded_images/test.png')
#     print(type(input_img))
#     print(input_img.shape)
#     cv2.imwrite(str(img_path),input_img)
#     sepia_filter = np.array([
#         [0.393, 0.769, 0.189], 
#         [0.349, 0.686, 0.168], 
#         [0.272, 0.534, 0.131]
#     ])
#     sepia_img = input_img.dot(sepia_filter.T)
#     sepia_img /= sepia_img.max()
#     return sepia_img

def sepia(input_img):
    img_path = Path('/workspace/gradio_demo/uploaded_images')
    cv2.imwrite(str(img_path / 'test.png'),input_img)
    main (
        image_path = img_path
        p
    )
    sepia_filter = np.array([
        [0.393, 0.769, 0.189], 
        [0.349, 0.686, 0.168], 
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()

    labeled_img_path = Path('/workspace/tool_output/labels_visualized')
    labeled_img = cv2.imread(str(labeled_img_path / 'test.png'))
    print('type(labeled_img)',type(labeled_img))
    print(labeled_img.shape)
    # print(type(sepia_img))
    # print(sepia_img.shape)
    return sepia_img

demo = gr.Interface(sepia, gr.Image(shape=(200, 200)), "image")
demo.launch(share=True)
# demo.launch()