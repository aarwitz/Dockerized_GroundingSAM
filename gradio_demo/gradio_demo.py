import numpy as np
import gradio as gr
from pathlib import Path
import cv2
import sys
import os

outer_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(outer_dir)
from demo import main, load_models
from utilities.file_mgmt import create_timestamped_dir, img_timestamped_fname

load_models() # load models before starting demo

def run_app(input_img,language_prompt, model_selection, confidence_score = 0.5,max_iou = 0.5):
    prompt = language_prompt.lower()
    img_path = create_timestamped_dir(base_path=Path("/workspace/gradio_demo/uploaded_images"))
    img_name = img_timestamped_fname()
    cv2.imwrite(str(img_path / img_name),input_img)
    output_path = main (
        image_path = img_path,
        prompt=prompt,
        confidence_score=confidence_score,
        max_iou=max_iou,
        model_selection = model_selection
    )
    if output_path is None:
        return input_img
    labeled_img_path = output_path / 'labels_visualized'
    labeled_img = cv2.imread(str(labeled_img_path / img_name))
    img_rgb = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB)
    return labeled_img

model_options = ["Swin-T","Swin-T finetuned blue expo marker"]

demo = gr.Interface(
    fn = run_app, 
    inputs = [
        gr.Image(label="Input Image"), 
        gr.Text(label="Language Prompt",description = "Enter a description of objects to detect", placeholder = "box, shipping label, tote"),
        gr.Dropdown(label="Model Selection", choices=model_options, default=model_options[0]),  # Add this line for model selection
        gr.Slider(minimum=0,maximum=1.0,label="Confidence Score",description="Enter a confidence score in range: 0-1)",placeholder=0.6),
        gr.Slider(minimum=0,maximum=1.0,label="IOU Threshold",description="Enter a maximum IOU that is accepted for overlapping detections")
        ],
    outputs = "image")
demo.queue()
demo.launch(share=True)