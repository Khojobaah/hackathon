import os
import shutil
import torch
from natsort import natsorted
from glob import glob
import cv2
import numpy as np
import gradio as gr
from PIL import Image

os.system("pip install einops")
os.system("git clone https://github.com/swz30/Restormer.git")
os.chdir('Restormer')

# Download pretrained models
os.system("wget https://github.com/swz30/Restormer/releases/download/v1.0/real_denoising.pth -P Denoising/pretrained_models")
os.system("wget https://github.com/swz30/Restormer/releases/download/v1.0/single_image_defocus_deblurring.pth -P Defocus_Deblurring/pretrained_models")
os.system("wget https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth -P Motion_Deblurring/pretrained_models")
os.system("wget https://github.com/swz30/Restormer/releases/download/v1.0/deraining.pth -P Deraining/pretrained_models")

# Download sample images
os.system("wget https://github.com/swz30/Restormer/releases/download/v1.0/sample_images.zip -P demo")
shutil.unpack_archive('demo/sample_images.zip', 'demo/')
os.remove('demo/sample_images.zip')


examples = [['demo/sample_images/Real_Denoising/degraded/117355.png', 'Denoising'],
            ['demo/sample_images/Single_Image_Defocus_Deblurring/degraded/engagement.jpg', 'Defocus Deblurring'],
            ['demo/sample_images/Motion_Deblurring/degraded/GoPro-GOPR0854_11_00-000090-input.jpg','Motion Deblurring'],
            ['demo/sample_images/Deraining/degraded/Rain100H-77-input.jpg','Deraining']]


title = "Restormer"
description = """
Gradio demo for Restormer: Efficient Transformer for High-Resolution Image Restoration, CVPR 2022--ORAL. <a href='https://arxiv.org/abs/2111.09881'>[Paper]</a><a href='https://github.com/swz30/Restormer'>[Github Code]</a>\n 
With Restormer, you can perform: (1) Image Denoising, (2) Defocus Deblurring, (3)  Motion Deblurring, and (4) Image Deraining. 
To use it, simply upload your own image, or click one of the examples provided below.
"""
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2111.09881'>Restormer: Efficient Transformer for High-Resolution Image Restoration </a> | <a href='https://github.com/swz30/Restormer'>Github Repo</a></p>"


def inference(img,task):
    os.system('mkdir temp')
    max_res = 904
    width, height = img.size
    if max(width,height) > max_res:
      scale = min(width,height)/max(width,height)
      if width > max_res:
        width = max_res
        height = int(scale*max_res)
      if height > max_res:
        height = max_res
        width = int(scale*max_res)
      img = img.resize((width,height), Image.ANTIALIAS)
  
    img.save("temp/image.jpg", "JPEG")

    if task == 'Motion Deblurring':
      task = 'Motion_Deblurring'
      os.system("python demo.py --task 'Motion_Deblurring' --input_dir './temp/image.jpg' --result_dir './temp/'")
  
    if task == 'Defocus Deblurring':
      task = 'Single_Image_Defocus_Deblurring'
      os.system("python demo.py --task 'Single_Image_Defocus_Deblurring' --input_dir './temp/image.jpg' --result_dir './temp/'")
  
    if task == 'Denoising':
      task = 'Real_Denoising'
      os.system("python demo.py --task 'Real_Denoising' --input_dir './temp/image.jpg' --result_dir './temp/'")
  
    if task == 'Deraining':
      os.system("python demo.py --task 'Deraining' --input_dir './temp/image.jpg' --result_dir './temp/'")
  
    return f'temp/{task}/image.png'
    
gr.Interface(
    inference,
    [
        gr.inputs.Image(type="pil", label="Input"),
        gr.inputs.Radio(["Denoising", "Defocus Deblurring", "Motion Deblurring", "Deraining"], default="Denoising", label='task type')
    ],
    gr.outputs.Image(type="file", label="Output"),
    title=title,
    description=description,
    article=article,
    examples=examples,
    allow_flagging=False,
    ).launch(debug=True,enable_queue=True)