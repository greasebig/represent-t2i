import safetensors.torch
import torch
import numpy as np
from PIL import Image
import os

def load_torch_file(ckpt, safe_load=False, device=None):
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    else:
        raise ValueError("仅支持 safetensors 后缀")
    return sd

device = torch.device("cuda")
ldm_ic_model_state_dict = load_torch_file(unet_path, device=device)

content = str(ldm_ic_model_state_dict.keys())
# 将内容按逗号分隔并换行
lines = content.split(",")
# 定义文件名
filename = "ldm_iclight_output.txt"
# 将内容写入txt文件，每个元素占一行
with open(filename, "w", encoding="utf-8") as file:
    for line in lines:
        file.write(line + "\n")
print(f"内容已成功保存到 {filename}")

diffusers_ic_model_state_dict = load_torch_file(unet_path, device=device)

content = str(diffusers_ic_model_state_dict.keys())
# 将内容按逗号分隔并换行
lines = content.split(",")
# 定义文件名
filename = "diffusers_iclight_output.txt"
# 将内容写入txt文件，每个元素占一行
with open(filename, "w", encoding="utf-8") as file:
    for line in lines:
        file.write(line + "\n")
print(f"内容已成功保存到 {filename}")

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")

sd_origin = unet.state_dict()

content = str(sd_origin.keys())
# 将内容按逗号分隔并换行
lines = content.split(",")
# 定义文件名
filename = "diffusers_unet_output.txt"
# 将内容写入txt文件，每个元素占一行
with open(filename, "w", encoding="utf-8") as file:
    for line in lines:
        file.write(line + "\n")
print(f"内容已成功保存到 {filename}")

filename = "ldm-iclight-output.txt"

