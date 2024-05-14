💫CoMat: Aligning Text-to-Image Diffusion Model with Image-to-Text Concept Matching    
💫CoMat：将文本到图像扩散模型与图像到文本概念匹配对齐



# 论文信息：
[Submitted on 4 Apr 2024]
CoMat: Aligning Text-to-Image Diffusion Model with Image-to-Text Concept Matching

Dongzhi Jiang1,2, Guanglu Song2, Xiaoshi Wu1, Renrui Zhang1,3, Dazhong Shen3, Zhuofan Zong2,
Yu Liu2, Hongsheng Li1    
1CUHK MMLab,    
2SenseTime Research, 3Shanghai AI Laboratory    
港中文    

[2024.04.30] 🔥 We release the training code of CoMat.

[2024.04.05] 🚀 We release our paper on arXiv.

我们提出了💫CoMat，一种具有图像到文本概念匹配机制的端到端扩散模型微调策略。我们利用图像字幕模型来测量图像到文本的对齐情况，并指导扩散模型重新访问被忽略的标记。

训练   
我们目前支持SD1.5和SDXL。


https://github.com/CaraJ7/CoMat





# 原理
扩散模型在文本到图像生成领域取得了巨大成功。然而，减轻文本提示和图像之间的错位仍然具有挑战性。未对准背后的根本原因尚未得到广泛调查。我们观察到这种`错位是由于令牌注意力激活不足`引起的。我们进一步将这种现象归因于扩散模型的条件利用不足，这是由其训练范式造成的。为了解决这个问题，我们提出了 CoMat，一种具有图像到文本概念匹配机制的端到端扩散模型微调策略。我们`利用图像字幕模型来测量图像到文本的对齐情况，并指导扩散模型重新访问被忽略的标记`。还提出了一种`新颖的属性集中模块来解决属性绑定问题`。在没有任何图像或人类偏好数据的情况下，我们`仅使用 20K 文本提示`来微调 SDXL 以获得 CoMat-SDXL。大量实验表明，CoMat-SDXL 在两个文本到图像对齐基准测试中显着优于基线模型 SDXL，并实现了最先进的性能。


The text-to-image diffusion model (T2I-Model) first generates an image according to the text prompt. Then the image is sent to the Concept Matching module, Attribute Concentration module, and Fidelity Preservation module to compute the loss for fine-tuning the online T2I-Model.

![alt text](assets/CoMat/image-2.png)

具体来说，我们利用图像字幕模型来监督扩散模型，以充分关注概念匹配模块中文本提示中的每个概念。在属性集中模块中，我们促进每个实体的名词和属性的注意力图的一致性。最后，在保真度保持模块中，我们引入了一种新颖的对抗性损失来保持在线微调模型的生成质量。

Specifically, we leverage an image captioning model to supervise the diffusion model to sufficiently attend to each concept in the text prompt in the Concept Matching module. In the Attribute Concentration module, we promote the consistency of the attention map of each entity's noun and attributes. Finally, in the Fidelity Preservation module, we introduce a novel adversarial loss to conserve the generation quality of the online fine-tuning model.




# 效果 
![alt text](assets/CoMat/image.png)

![alt text](assets/CoMat/image-1.png)








#  结尾