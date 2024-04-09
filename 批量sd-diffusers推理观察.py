from diffusers import AutoPipelineForText2Image
import torch


pipe = AutoPipelineForText2Image.from_pretrained("/private/dj/models/sdxl-turbo",
                                                 torch_dtype=torch.float16, 
                                                 variant="fp16")
pipe.to("cuda")

import torch
from diffusers.models import AutoencoderKL
import os
from diffusers import DiffusionPipeline, LCMScheduler



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
"A portrait of a young woman with a mysterious allure, her deep and story-filled eyes sparkling with wisdom and curiosity. Her skin has a natural glow, with delicate and elegant facial features. Her hair is wavy dark brown, gently resting on her shoulders. She wears a simple yet elegant black top, adorned with a delicate necklace. The focus of the image is on her portrait, from the shoulders up, emphasizing her expressive face and captivating gaze.",
"A regal lioness donning a flowing, golden Renaissance gown, complete with intricate embroidery that mimics her majestic mane. She holds court from atop a grand throne, a scepter in one paw, her demeanor both noble and wise.",
"beautiful scenery nature glass bottle landscape, , purple galaxy bottle",
"1girl, solo, long hair, looking at viewer, simple background, (black hair:1.2), closed mouth, upper body, artist name, from side, looking to the side, makeup, straight hair, brown background, red lips, (red Hanfu:1.2), (realistic:1.2), red robe, Chinese style",
"A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
]

negetive_prompt = ""

steps = 1
cfg = 0 #1.


folder_path = f'/private/lujunda/infer/infer-pics-secondweek/pic-sdxlturbo-{steps}step-{cfg}cfg/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

seed = 0

for prompt in prompt_list:
    

    image = pipe(prompt=prompt,
                    num_inference_steps=steps, 
                    num_images_per_prompt = 1,
                    generator = torch.Generator(device="cuda").manual_seed(seed),
                    guidance_scale=cfg,
                    negetive_prompt = negetive_prompt,
                ).images[0]
    
    filename = prompt[:20] if len(prompt) > 19 else prompt
    image.save(folder_path + filename + ".png")
    
