# few step gen



## SDXS
Real-Time One-Step Latent Diffusion Models with Image Conditions    
为了提高迭代采样的速度，作者引入了模型小型化和减少采样步骤的方法。该方法利用知识蒸馏来简化 U-Net 和 图像解码器 的架构，并引入了一种新的one step DM 训练技术，其使用了特征匹配和分数蒸馏。共推出了两种模型：SDXS-512 和 SDXS-1024，在单个 GPU 上分别实现了约 100 FPS（比 SD v1.5 快 30 倍）和 30 FP（比 SDXL 快 60 倍）的推理速度。此外，该训练方法尤其有利于图像条件控制和图像到图像的任务。    
文章 https://arxiv.org/abs/2403.16627   


2024.3.25 推出 SDXS-512-0.9 is a old version of SDXS-512. For some reasons, we are only releasing this version for the time being, and will gradually release other versions.

Model Information:

Teacher DM: SD Turbo   
Offline DM: SD v2.1 base   
VAE: TAESD  (10MB)



该模型与1.0版本的主要区别在于三个方面：

该版本采用TAESD，当weight_type为float16时，可能会产生低质量的图像。我们的图像解码器与当前版本的扩散器不兼容，因此现在不会提供。   
该版本没有执行实现细节部分中提到的LoRA-GAN微调，这可能会导致图像细节稍差。   
该版本在最高分辨率阶段用交叉注意力取代了自注意力，与直接删除它们相比，这引入了最小的开销。    

```
weight_type = torch.float32     # or float16

# use original VAE   335MB 
# pipe.vae = AutoencoderKL.from_pretrained("IDKiro/sdxs-512-0.9/vae_large")

```


### 原理
#### Model Acceleration
![alt text](assets/README/image.png)    
We train an extremely light-weight image decoder to mimic the original VAE decoder’s output through a combination of output distillation loss and GAN loss. (335 -> 10 MB)   
We also leverage the block removal distillation strategy to efficiently transfer the knowledge from the original U-Net to a more compact version.    








### 效果




## YOSO

只有一个lora模型。可以调用社区的sd1.5。   
"You Only Sample Once: Taming One-Step Text-To-Image Synthesis by Self-Cooperative Diffusion GANs"   




### 实践
#### 1步推理  
目前仅允许基于 SD v1.5 进行 1 步推理。   
 And you should prepare the informative initialization according to the paper for better results.
即需要额外准备图像latent   
```
import torch
from diffusers import DiffusionPipeline, LCMScheduler
pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype = torch.float16)
pipeline = pipeline.to('cuda')
pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
pipeline.load_lora_weights('Luo-Yihong/yoso_sd1.5_lora')
generator = torch.manual_seed(318)
steps = 1
bs = 1
latents = ... # maybe some latent codes of real images or SD generation
latent_mean = latent.mean(dim=0)
noise = torch.randn([1,bs,64,64])
input_latent = pipeline.scheduler.add_noise(latent_mean.repeat(bs,1,1,1),noise,T)
imgs= pipeline(prompt="A photo of a dog",
                    num_inference_steps=steps, 
                    num_images_per_prompt = 1,
                        generator = generator,
                        guidance_scale=1.5,
                    latents = input_latent,
                   )[0]
imgs

```
如果没有质量会较差   
```
pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype = torch.float16)
pipeline = pipeline.to('cuda')
pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
pipeline.load_lora_weights('Luo-Yihong/yoso_sd1.5_lora')
generator = torch.manual_seed(318)
steps = 1
imgs = pipeline(prompt="A photo of a corgi in forest, highly detailed, 8k, XT3.",
                    num_inference_steps=1, 
                    num_images_per_prompt = 1,
                        generator = generator,
                        guidance_scale=1.,
                   )[0]
imgs[0]


```





#### 2步推理   
我们注意到，小的 CFG 可用于提高图像质量。   





# prompt 数据集
## parti prompts
https://huggingface.co/datasets/nateraw/parti-prompts   
tsv 或 parque 读取   
几百KB   

![alt text](assets/README/image-2.png)   
![alt text](assets/README/image-3.png)    
![alt text](assets/README/image-1.png)  

PartiPrompts (P2) is a rich set of over 1600 prompts in English that we release as part of this work. P2 can be used to measure model capabilities across various categories and challenge aspects.   
![alt text](assets/README/image-4.png)   

数据集来源：   
谷歌2022年推出 Pathways Autoregressive Text-to-Image model (Parti), an autoregressive text-to-image generation model    

Parti 和 Imagen 在探索两个不同的生成模型系列（分别是自回归模型和扩散模型）方面是互补的   
Parti 将文本到图像的生成视为序列到序列的建模问题，类似于机器翻译——这使其能够受益于大型语言模型的进步，尤其是通过扩展数据和模型大小来解锁的功能。     
In this case, the target outputs are sequences of image tokens instead of text tokens in another language.    

