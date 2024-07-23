import os
# import shutil
# import torch
# from natsort import natsorted
# from glob import glob
import cv2
# from Restormer.demo import restore_image
from Restormer.demo2 import restore_image
import numpy as np
import gradio as gr
from PIL import Image




def inference(img, task):
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
      img = img.resize((width, height), Image.LANCZOS)

    restore_image('./TT.jpg', task)

    return f'temp/{task}/image.jpg'


# image_01 = cv2.imread('TT.jpg')
# cv2_img = np.array(image_01)
image_01 = Image.open('TT.jpg')

os.chdir('Restormer')

print(inference(image_01, "Real_Denoising"))
# print(inference(image_01, 'Gaussian_Gray_Denoising'))
# print(inference('TT.jpg', 'Real_Denoising'))
# print(inference('TT.jpg', 'Real_Denoising'))
# gr.Interface(
#     inference,
#     [
#         gr.inputs.Image(type="pil", label="Input"),
#         gr.inputs.Radio(["Denoising", "Defocus Deblurring", "Motion Deblurring", "Deraining"], default="Denoising", label='task type')
#     ],
#     gr.outputs.Image(type="file", label="Output"),
#     title=title,
#     description=description,
#     article=article,
#     examples=examples,
#     allow_flagging=False,
#     ).launch(debug=True,enable_queue=True)