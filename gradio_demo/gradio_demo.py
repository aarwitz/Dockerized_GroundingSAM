

import numpy as np
import gradio as gr
from pathlib import Path
import cv2
import sys
import os

outer_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(outer_dir)

from label_app import main, load_models

load_models() # load models before starting demo

def run_app(input_img,language_prompt, confidence_score = 0.5):
    prompt = language_prompt.lower()
    img_path = Path('/workspace/gradio_demo/uploaded_images')
    cv2.imwrite(str(img_path / 'test.png'),input_img)
    main (
        image_path = img_path,
        prompt=prompt,
        confidence_score=confidence_score
    )
    labeled_img_path = Path('/workspace/tool_output/labels_visualized')
    labeled_img = cv2.imread(str(labeled_img_path / 'test.png'))
    img_rgb = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB)
    return labeled_img

demo = gr.Interface(
    fn = run_app, 
    inputs = [
        gr.Image(label="Input Image"), 
        gr.Text(label="Language Prompt",description = "Enter a description of objects to detect", placeholder = "box, shipping label, tote"),
        gr.Slider(minimum=0,maximum=1.0,label="Confidence Score",description="Enter a confidence score in range: 0-1)",placeholder=0.6)
        ],
    outputs = "image")
demo.launch(share=True)
# demo.launch()