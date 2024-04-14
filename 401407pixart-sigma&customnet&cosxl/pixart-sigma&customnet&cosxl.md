# CustomNet
: Object Customization with Variable-Viewpoints in Text-to-Image Diffusion Models.   
CustomNet: Zero-shot Object Customization with Variable-Viewpoints in Text-to-Image Diffusion Models    





需要额外安装  
basicsr   
但依赖tb-nightly    
清华源不存在tb-nightly    
pip install tb-nightly -i https://mirrors.aliyun.com/pypi/simple    


CustomNet is novel unified customization method that can generate harmonious customized images without test-time optimization. CustomNet supports explicit viewpoint, location, text controls while ensuring object identity preservation.    
![alt text](assets/pixart-sigma&customnet/image.png)   
不知道background-image在哪里设置，示例代码好像没有   


![alt text](assets/pixart-sigma&customnet/image-1.png)
可以通过文字描述或用户定义的背景来实现位置控制和灵活的背景控制。   
`Some` methods `finetune T2I models for each object individually at test-time`, which tend to be overfitted and time-consuming     
`Others train an extra encoder` to extract object visual information for customization efficiently but struggle to preserve the object’s identity.       
we incorporates `3D novel view synthesis` capabilities into the customization process    
we propose a `dataset construction pipeline` to better handle real-world objects and complex backgrounds.    
Additionally, we introduce delicate designs that enable `location control and flexible background control` through textual descriptions or user-defined backgrounds. Our method allows for object customization without the need of test-time optimization     


将定制对象合并到图像生成中是文本到图像 (T2I) 生成的一个有吸引力的功能。一些方法在测试时单独微调每个对象的 T2I 模型，这往往会过度拟合且耗时。其他人训练额外的编码器来提取对象视觉信息以进行有效的定制，但很难保留对象的身份。为了解决这些限制，我们提出了 CustomNet，这是一个基于编码器的统一对象定制框架，它明确地将 3D 新颖视图合成功能合并到定制过程中。这种集成有助于空间位置和视点的调整，产生不同的输出，同时有效地保留对象的身份。为了有效地训练我们的模型，我们提出了一个数据集构建管道，以更好地处理现实世界的对象和复杂的背景。此外，我们还引入了精致的设计，可以通过文字描述或用户定义的背景来实现位置控制和灵活的背景控制。我们的方法允许对象定制，无需测试时优化，提供对视点、位置和文本的同步控制。实验结果表明，我们的方法在身份保存、多样性和和谐性方面优于其他定制方法。

使用方法：输入一张白背景的物品图片作为参考图图，输入prompt编辑背景，可以通过参数改变物品在图片中的坐标位置和3D视角。   
测试模型：CustomNet   
测试参数：   
DDIM，采样50步，GUI无法修改    
测试结论：人物恢复效果差，文本控制不太准确，上下视角不太准确，小图时， 绘制效果不好。图片会被预处理成256*256。    










# PixArt-Σ 
Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation   

(🔥 New) Apr. 6, 2024. 💥 PixArt-Σ checkpoint 256px & 512px are released!   
(🔥 New) Mar. 29, 2024. 💥 PixArt-Σ training & inference code & toy data are released!!!   

华为诺亚方舟实验室、大连理工大学、香港大学、香港科技大学    
https://pixart-alpha.github.io/PixArt-sigma-project/    
https://arxiv.org/abs/2403.04692    
[Submitted on 7 Mar 2024 (v1), last revised 17 Mar 2024 (this version, v2)]





## 该组织前期研究
https://arxiv.org/abs/2310.00426   
[Submitted on 30 Sep 2023 (v1), last revised 29 Dec 2023 (this version, v3)]    
PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis    
被yoso用来微调模型    
few_step_gen folder有简略介绍   



https://arxiv.org/abs/2401.05252    
[Submitted on 10 Jan 2024]   
PIXART-δ: Fast and Controllable Image Generation with Latent Consistency Models    

## 推理
使用gradio推理   
尚不支持diffusers   
可以训练和推理   
有256 512 1024模型    
后续还会出dmd模型     



nvcc11.8,torch 2.0.0没说明cu版本   
好像默认11.7   

    File "/root/miniconda3/envs/pixart/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 459, in _conv_forward
        return F.conv2d(input, weight, bias, self.stride,
    RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR

pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118      
卸载重装     
还是cudnn错误   

但是这个3090是可以运行webui推理    

有人说 其实就是gpu显存不够，减小点工作量就可以了    

 2、手动使用cudnn

    import torch
    torch.backends.cudnn.enabled = False



## 原理
主要模型结构与PixArt-α相同   




# cosxl
Cos Stable Diffusion XL 1.0 and Cos Stable Diffusion XL 1.0 Edit   

可以一致的生成2k 4k 8k   
显存恒为25g左右    
清晰度没有上升，就是图片大小变大了    
所以是为什么能够一致性的生成？？？？？？    
而且速度还挺快   

尝试使用realistic_vision_v51进行4096*4096生图   
Ran out of memory when regular VAE encoding, retrying with tiled VAE encoding.     
显存2g上升到5g   
42s/it     
42steps 
![alt text](assets/pixart-sigma&customnet&cosxl/WeChat93b104784546b5c1c61f7d31dd987388.jpg)
半个小时    


![alt text](assets/pixart-sigma&customnet&cosxl/image.png)   
正常写法生图效果不好    
![alt text](assets/pixart-sigma&customnet&cosxl/image-2.png)   
确切地说需要使用eular42步    
![alt text](assets/pixart-sigma&customnet&cosxl/image-3.png)   
使用dpm ++ 2m不太正常   


要加上ip2p写法    
![alt text](assets/pixart-sigma&customnet&cosxl/image-1.png)   



![alt text](assets/pixart-sigma&customnet&cosxl/image-4.png)


类似ip2p工作  
[Submitted on 28 Jan 2023 (v1), last revised 2 Nov 2023 (this version, v2)]     
SEGA: Instructing Text-to-Image Models using Semantic Guidance         

