from diffusers import AutoPipelineForText2Image
import torch


pipe = AutoPipelineForText2Image.from_pretrained("/private/dj/models/sdxl-turbo",
                                                 torch_dtype=torch.float16, 
                                                 variant="fp16")
pipe.to("cuda")

# pipe.enable_xformers_memory_efficient_attention()

# Enable memory optimizations.
# pipe.enable_model_cpu_offload()

'''
When using torch >= 2.0, you can improve the inference speed by 20-30% with torch.compile. Simple wrap the unet with torch compile before running the pipeline:

pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

If you are limited by GPU VRAM, you can enable cpu offloading by calling pipe.enable_model_cpu_offload instead of .to("cuda"):

- pipe.to("cuda")
+ pipe.enable_model_cpu_offload()











加载单一组件
safetensors或bin

transformer = Transformer2DModel.from_pretrained(
    "weights/PixArt-Alpha-DMD-XL-2-512x512", #可以换成repo_id,直接下载。内部机制是先查询为本地路径，然后查huggingface
    subfolder='transformer', 
    torch_dtype=weight_dtype,
    use_safetensors=False,
)


pipe = PixArtSigmaPipeline.from_pretrained(
    "newlytest/PixArt-sigma/output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers",
    transformer=transformer,
    torch_dtype=weight_dtype,
    use_safetensors=True,
)

#pipe.scheduler = LCMScheduler.from_pretrained("weights/PixArt-Alpha-DMD-XL-2-512x512", subfolder="scheduler")

pipe.scheduler = DDPMScheduler.from_pretrained("weights/PixArt-Alpha-DMD-XL-2-512x512", 
                                         subfolder="scheduler")                         
#pipe.scheduler.config.prediction_type = "v_prediction"



以在多个pipeline中可以重复使用相同的组件，以避免将权重加载到RAM中2次

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
 
model_id = "runwayml/stable-diffusion-v1-5"
stable_diffusion_txt2img = StableDiffusionPipeline.from_pretrained(model_id, use_safetensors=True)
 
components = stable_diffusion_txt2img.components
可以将components传递到另一个pipeline中，无需将权重重新加载到RAM中：

stable_diffusion_img2img = StableDiffusionImg2ImgPipeline(**components)







from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
 
repo_id = "runwayml/stable-diffusion-v1-5"
 
ddpm = DDPMScheduler.from_pretrained(repo_id, subfolder="scheduler")
ddim = DDIMScheduler.from_pretrained(repo_id, subfolder="scheduler")
pndm = PNDMScheduler.from_pretrained(repo_id, subfolder="scheduler")
lms = LMSDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
euler_anc = EulerAncestralDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
euler = EulerDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
dpm = DPMSolverMultistepScheduler.from_pretrained(repo_id, subfolder="scheduler")
 
# replace `dpm` with any of `ddpm`, `ddim`, `pndm`, `lms`, `euler_anc`, `euler`
pipeline = StableDiffusionPipeline.from_pretrained(repo_id, scheduler=dpm, use_safetensors=True)


'''

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
    



'''
端侧部署    

Optimum
Optimum provides a Stable Diffusion pipeline compatible with both OpenVINO and ONNX Runtime.

OpenVINO
To install Optimum with the dependencies required for OpenVINO :

pip install optimum[openvino]

'''