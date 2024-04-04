import torch
from diffusers.models import AutoencoderKL
from safetensors.torch import load_file
import numpy as np
import random
from PIL import Image
import os
from diffusers import PixArtAlphaPipeline, LCMScheduler, Transformer2DModel

model_path = "weights/PixArt-alpha/PixArt-XL-2-512x512"
transformer_path = "weights/Luo-Yihong/yoso_pixart1024"

transformer = Transformer2DModel.from_pretrained(
    transformer_path, torch_dtype=torch.float16).to('cuda')

pipe = PixArtAlphaPipeline.from_pretrained(model_path, 
                            transformer=transformer,
                            torch_dtype=torch.float16, use_safetensors=True)


pipeline = pipe.to('cuda')
pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler.config.prediction_type = "v_prediction"

steps = 1

folder_path = 'infer-pics/pic-yoso_pixart1024/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

output_width = 2 # 列数
output_height = 2 # 行数
nums=output_width*output_height
width = 1024
image_size = (width, width) # 设置每张图片的大小，可以根据实际需要调整

prompt_list = [
'A photo of a man, XT3',
"A photo of a corgi in forest, highly detailed, 8k, XT3.",
"portrait photo of a girl, photograph, highly detailed face,\
 depth of field, moody light, golden hour",
"a motorcycle",
"a black background with a large yellow square",
"Pirate ship trapped in a cosmic maelstrom nebula, rendered in cosmic \
    beach whirlpool engine, volumetric lighting, spectacular, ambient lights, \
        light pollution, cinematic atmosphere, art nouveau style, illustration \
            art artwork by SenseiJaye, intricate detail.",
]

for prompt in prompt_list:
    image_files = []

    for i in range(nums):

        seed = i

        #image = pipe(prompt=prompt, num_inference_steps=25, cross_attention_kwargs={"scale": 1.0}).images[0]
        image = pipeline(prompt=prompt,
                    num_inference_steps=steps, 
                    num_images_per_prompt = 1,
                    generator = torch.Generator(device="cuda").manual_seed(seed),
                    guidance_scale=1.,
                ).images[0]
        image_files.append(image)

    output_image = Image.new("RGB", (output_width * image_size[0], output_height * image_size[1]))

    # 遍历每张图片并粘贴到合成图上
    for i in range(output_width):
        for j in range(output_height):
            index = i * output_width + j
            if index < len(image_files):
                img = image_files[index]
                output_image.paste(img, (j * image_size[0], i * image_size[1]))

    filename = prompt[:18] if len(prompt) > 17 else prompt
    output_image.save(folder_path + filename + ".png")
















