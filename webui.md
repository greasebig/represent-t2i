## 3090
### 黑图和xyz
真实感大模型sdxl加载失败    
File "/teams/ai_model_1667305326/WujieAITeam/private/lujunda/stable-diffusion-webui-master/modules/sd_disable_initialization.py", line 219, in load_state_dict
    state_dict = {k: v.to(device="meta", dtype=v.dtype) for k, v in state_dict.items()}
  File "/teams/ai_model_1667305326/WujieAITeam/private/lujunda/stable-diffusion-webui-master/modules/sd_disable_initialization.py", line 219, in <dictcomp>
    state_dict = {k: v.to(device="meta", dtype=v.dtype) for k, v in state_dict.items()}
RuntimeError: dictionary changed size during iteration

Applying attention optimization: Doggettx... done.
Model loaded in 182.3s (calculate hash: 76.8s, load weights from disk: 5.3s, create model: 0.7s, apply weights to model: 98.9s, calculate empty prompt: 0.3s).



1 girl    
Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 1957970079, Size: 512x512, Model hash: a2e2de4c7a, Model: 0307_Rocky_sdxl_PrivateImaging_model, Version: 1.8.0-RC

sdxl turbo和原模型一样大.推理时8g-10g跳动，才像是cfg起作用，以及可能是采样器问题，只是快    

xyz plot    
AssertionError: Error: Resulting grid would be too large (367 MPixels) (max configured size is 200 MPixels)    
sdxl-turbo   
![alt text](assets/webui/image.png)   
这些图片要一小时才能生成完    
大概3,4it/s    
全是黑图    
![alt text](assets/webui/xyz_grid-0000.png)

关了hires   
使用v1.5   
![alt text](assets/webui/image-1.png)    
![alt text](assets/webui/image-2.png)    

    File "/teams/ai_model_1667305326/WujieAITeam/private/lujunda/stable-diffusion-webui-master/repositories/k-diffusion/k_diffusion/sampling.py", line 701, in sample_dpmpp_3m_sde
        h_1, h_2 = h, h_1
    UnboundLocalError: local variable 'h' referenced before assignment
虽然也有这些错误，但是还是能正常生图    
dpm 2m 3m都会有这个问题

使用turbo sdxl    
![alt text](assets/webui/image-3.png)   
估计被打码了？？？   

sdxl私人摄影也是   
![alt text](assets/webui/image-4.png)   

网上说可能是显卡半精度问题    
需要在启动webui时候 no half, full   
--precision full --no-half   

sdxl私人摄影    
![alt text](assets/webui/image-5.png)     
![alt text](assets/webui/image-6.png)   
![alt text](assets/webui/image-8.png)   
推理步数，采样器都一致   
也需要1分钟，15g   


sdxl turbo fp16     
![alt text](assets/webui/image-7.png)   
1分钟    
15g   

真实感大模型加载成功   
15g   
![alt text](assets/webui/image-9.png)   
1 girl    
Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 0, Size: 512x512, Model hash: dfe73aa42d, Model: LEOSAM_HelloWorld_新世界_SDXL真实感大模型_v3.2_AutoDPO, Version: 1.8.0-RC

Time taken: 5.1 sec.

A: 13.96 GB, R: 15.27 GB, Sys: 14.6/23.6914 GB (61.8%)

开hires refiner    
![alt text](assets/webui/image-10.png)    
1 girl   
Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 0, Size: 512x512, Model hash: dfe73aa42d, Model: LEOSAM_HelloWorld_新世界_SDXL真实感大模型_v3.2_AutoDPO, Denoising strength: 0.7, Hires upscale: 2, Hires upscaler: Latent, Version: 1.8.0-RC

Time taken: 18.1 sec.

A: 16.59 GB, R: 18.12 GB, Sys: 19.0/23.6914 GB (80.3%)


seed 0
![alt text](assets/webui/image-11.png)    
Sampler: DPM++ 2M SDE Karras,   

![alt text](assets/webui/image-12.png)
Sampler: DPM++ 3M SDE Karras,


![alt text](assets/webui/image-13.png)   
![alt text](assets/webui/image-14.png)     
18分钟    
推理过程一般18g    
![alt text](assets/webui/image-15.png)   
50mb   







### webui启动方式
python webui.py   
python launch.py    
--precision full --no-half   
--xformers   

而comfyui会自动调用xformers



## comfyui denoise
当您以 1.0 运行 Ksampler 时，它会完全模糊任何传入的图像或噪声，然后按照给定的步骤数对其进行处理。

如果您以 0.6 运行 ksampler，它会模糊 60% 的强度，并根据给定的步数对其进行降噪。

两者之间的区别在于，100% 时它使用的是原始噪声或图像的极小部分。

而在 60% 时，它使用了大部分原始图像的颜色、明暗信息。









# 结尾







