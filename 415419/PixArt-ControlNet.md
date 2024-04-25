PixArt-δ-1024-ControlNet     
有512 1024两个模型     


https://huggingface.co/PixArt-alpha/PixArt-ControlNet/tree/main     
三个月前发布    

提出了PixArt-alpha的lcm和controlnet。但是这两个东西不是连着用的。     


# 论文信息
PIXART-δ: Fast and Controllable Image Generation with Latent Consistency Models    
https://arxiv.org/abs/2401.05252     
[Submitted on 10 Jan 2024]    
提出了PixArt-alpha的lcm和controlnet。但是这两个东西不是连着用的。  

Alpha的大小写形式分别是 Α 和 α 。它是希腊字母表中的第1个字母。
Delta(大写 Δ,小写 δ),是第四个希腊字母      
英语名称： sigma ，汉语名称：西格玛（大写Σ，小写σ）Sigma是希腊字母的第十八个字母     

本技术报告介绍了 PIXART-{\delta}，这是一种文本到图像合成框架，它将潜在一致性模型 (LCM) 和 ControlNet 集成到先进的 PIXART-{\alpha} 模型中。 PIXART-{\alpha} 因其通过非常高效的训练过程生成 1024px 分辨率的高质量图像的能力而受到认可。 PIXART-{\delta}中 LCM 的集成显着加快了推理速度，只需 2-4 个步骤即可生成高质量图像。值得注意的是，PIXART-{\delta} 在生成 1024x1024 像素图像方面突破了 0.5 秒，比 PIXART-{\alpha} 提高了 7 倍。此外，PIXART-{\delta} 设计为可在一天内在 32GB V100 GPU 上进行高效训练。凭借其 8 位推理能力（von Platen 等人，2023），PIXART-{\delta} 可以在 8GB GPU 内存限制内合成 1024px 图像，大大增强了其可用性和可访问性。此外，`结合类似 ControlNet 的模块可以对文本到图像扩散模型进行细粒度控制。我们引入了一种新颖的 ControlNet-Transformer 架构，专为 Transformer 量身定制，可在生成高质量图像的同时实现明确的可控性。`作为最先进的开源图像生成模型，PIXART-{\delta} 为稳定扩散模型系列提供了一种有前景的替代方案，为文本到图像的合成做出了重大贡献。



## 同篇论文的其他部分 lcm
![alt text](<assets/PixArt-ControlNet /截屏2024-04-25 09.57.31.png>)    
delta的lcm模型没有sigma的dmd效果好    

# 原理
pixart ControlNet        
![alt text](<assets/PixArt-ControlNet /image-2.png>)      
我们结合了类似 ControlNet（https://github.com/lllyasviel/ControlNet）模块，可以对文本到图像扩散模型进行细粒度控制。我们引入了一种新颖的 ControlNet-Transformer 架构，专为 Transformer 量身定制，可在生成高质量图像的同时实现明确的可控性。

有关PixArt-ControlNet的更多详细信息，请查看技术报告PixArt-δ。






















# 官方示例
![alt text](<assets/PixArt-ControlNet /image.png>)      
![alt text](<assets/PixArt-ControlNet /image-1.png>)      





# 其他代码
## hed
https://github.com/s9xie/hed   
九年前   

HED shows a clear advantage in consistency over Canny. The history of computational edge detection is extremely rich; we now highlight a few representative works that have proven to be of great practical importance.

Notice how the Canny edge detector is not able to preserve the object boundary of the cat, mountains, or the rock the cat is sitting on. HED, on the other hand, is able to preserve all of those object boundaries



## t5
该模型被分成两个9g的依次加载    







# 结尾