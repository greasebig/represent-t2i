MoMA: Multimodal LLM Adapter for Fast Personalized Image Generation     
MoMA：用于快速生成个性化图像的多模态 LLM 适配器    

即插即用适配器    
我们的模型是一个通用适配器，因为我们在训练阶段冻结了原始扩散模型。它可以推广到从同一基本模型微调的自定义模型检查点。在下图中，我们在 HuggingFace 和 CivitAi 的社区模型上验证了这一点，包括 Realistic Vision V4.0、ReV-Animated、Anything v4 和 Esthetic Retro Anime。这些模型都是从 `SD v1.5` 开始进行微调的。 MoMA可以直接应用于这些社区模型，无需任何修改。      

和ipadapter对比     
和直接修改背景对比？？    




# 论文信息
字节   
罗格斯大学(美国公立大学系统)    






[Submitted on 8 Apr 2024]   
MoMA: Multimodal LLM Adapter for Fast Personalized Image Generation     
https://arxiv.org/abs/2404.05674    

项目地址    
https://github.com/bytedance/MoMA     
https://moma-adapter.github.io/    

https://huggingface.co/KunpengSong/MoMA_llava_7b

发布    
[2024/04/20] 🔥 我们在 GitHub 上发布了模型代码。   
[2024/04/22] 🔥 我们添加 HuggingFace 存储库并发布检查点。   






模型类型： MoMA 是一个开源图像个性化模型。它具有新的注意力层和从 LLaVA-7B 微调的多模态大语言模型。

![alt text](assets/MOMA/image-7.png)

没有issue     
结果应该不好    




# 原理

`SD v1.5`模型   

我们推出 MoMA：一种开放词汇、免训练的个性化图像模型，拥有灵活的零样本功能。随着基础文本到图像模型的快速发展，对强大的图像到图像转换的需求不断增长。为了满足这一需求，`MoMA 专门研究主题驱动的个性化图像生成`。利用开源的多模态大语言模型 (MLLM)，我们`训练 MoMA 充当特征提取器和生成器`的双重角色。该方法有效地协同参考图像和文本提示信息以产生有价值的图像特征，促进图像扩散模型。为了更好地利用生成的特征，我们进一步`引入了一种新颖的自注意力捷径方法，该方法可以有效地将图像特征转移到图像扩散模型，从而提高生成图像中目标对象的相似度`。值得注意的是，作为一个免调整的`即插即用`模块，我们的模型`仅需要单个参考图像`，并且在生成具有高细节保真度、增强的身份保留和即时忠实度的图像方面优于现有方法。我们致力于将我们的工作开源，从而让所有人都能获得这些进步。

![alt text](assets/MOMA/image.png)    
我们推出了 MoMA，这是一种通过细粒度特征传输增强的多模式 LLM 适配器。整体架构如下图所示。我们的方法由三个部分组成：（1）利用生成多模态解码器`从参考图像中提取图像特征，并根据目标提示对其进行编辑，产生上下文化图像特征`； （2）我们将原始图像的背景替换为白色，`只留下目标像素，并利用原始UNet的自注意力层来提取目标图像特征`； (3)最后，在新图像生成过程中，我们分别`使用专门训练的上下文交叉注意层和对象交叉注意层将上下文图像特征和对象图像特征注入`到UNet扩散模型中。    

为了实现最佳模型性能，我们提出了两阶段预训练策略。首先，我们提出了一个多模态生成学习阶段，我们对多模态图像特征解码器进行预训练，使其学习根据目标提示组合主题的图像特征，并输出目标图像的 CLIP 嵌入。其次，训练主题和上下文交叉注意层来注入这种嵌入。为了进一步增强细节的真实度，我们涉及图像自注意力特征转移并应用掩蔽机制

![alt text](assets/MOMA/image-4.png)


与其他方法比较    
零样本定性比较。我们在上面板中共享重新上下文化，在下面板中共享纹理编辑。我们的结果为上下文编辑提供了明显更准确的细节，并在纹理编辑中的提示和图像保真度之间实现了更好的平衡。    

![alt text](assets/MOMA/image-5.png)

即插即用适配器    
我们的模型是一个通用适配器，因为我们在训练阶段冻结了原始扩散模型。它可以推广到从同一基本模型微调的自定义模型检查点。在下图中，我们在 HuggingFace 和 CivitAi 的社区模型上验证了这一点，包括 Realistic Vision V4.0、ReV-Animated、Anything v4 和 Esthetic Retro Anime。这些模型都是从 `SD v1.5` 开始进行微调的。 MoMA可以直接应用于这些社区模型，无需任何修改。      


![alt text](assets/MOMA/image-6.png)    



# 使用
![alt text](assets/MOMA/image-1.png)


新上下文：    
![alt text](assets/MOMA/image-2.png)

新纹理：   
![alt text](assets/MOMA/image-3.png)    



超参数：

在“更改上下文”中，您可以增加strength以获得更准确的详细信息。大多数情况下，strength=1.0是最好的。建议strength不大于1.2。

在“更改纹理”中，您可以更改strength细节精度和提示保真度之间的平衡。为了获得更好的提示保真度，只需减少strength。大多数情况下，strength=0.4是最好的。建议strength不大于0.6。

diffusers加载扩散模型      

    VAE: stabilityai--sd-vae-ft-mse
    StableDiffusion: Realistic_Vision_V4.0_noVAE
    MoMA: 
        Multi-modal LLM: MoMA_llava_7b (13 GB)
        Attentions and mappings: attn_adapters_projectors.th (151 Mb)


# 其他
## DeepFloyd IF
新的生图模型DeepFloyd IF来了，可以拳打Stable Diffusion，脚踢Dall-E？

2023.05    
https://github.com/deep-floyd/IF    

Stability AI与它的多模式AI研究实验室DeepFloyd共同宣布研究版本DeepFloyd IF的发布,这是一款强大的文text-to-image级联像素扩散模型（cascaded pixel diffusion model），复现了Google的Imagen（Text-to-Image Diffusion Models）。

对比Stable Diffusion（可以看我以前的文章：北方的郎：深入浅出讲解Stable Diffusion原理，新手也能看明白），Imagen也依赖于一个冻结的文本编码器：先将文本提示转换为嵌入，然后由扩散模型解码成图像。但不同的是，Imagen并没有使用多模态训练的CLIP，而是使用了大型T5-XXL语言模型。而StabilityAI推出的DeepFloyd IF复刻的正是这一架构。同时DeepFloyd IF在像素空间工作，与Stable Diffusion不同，扩散是在像素级实现的。

这些特点使它可以更精确的生成图像，例如生成带有特定文本的图片。在测试中，DeepFloyd IF直接超越了谷歌的Imagen，以及一众竞品（包括兄弟产品Stable Diffusion）。

DeepFloyd IF，具有高度的照片级真实感和语言理解。DeepFloyd IF 是一个由冻结的文本编码器和三个级联像素扩散模块组成的模块：

一个基于文本提示（Text Prompt）生成 64x64 像素图像的基本模型和两个超分辨率模型，每个模型都旨在生成分辨率不断提高的图像：256x256 像素和 1024x1024 像素。该模型的所有阶段都利用基于 T5 转换器的冻结文本编码器来提取文本嵌入，然后将其馈送到通过交叉注意力和注意力池增强的 UNet 架构中。

结果是一个高效的模型，优于当前最先进的模型，在COCO数据集上实现了6.66的 zero-shot FID分数。研究者的工作体现了更大的UNet架构在级联扩散模型第一阶段的潜力，并描绘了文本到图像合成的光明未来。
描述和特征

•深度文本提示(text prompt)理解:

利用大型语言模型T5-XXL-1.1作为文本编码器。大量的文本-图像交叉注意力层(text-image cross-attention layers)也提供了更好的提示和图像联盟。

•将文本描述应用于图像:

结合T5模型的智能性,DeepFloyd IF生成连贯清晰的文本以及出现在各种空间关系中的不同属性的对象。到目前为止,这些用例对大多数文本到图像模型来说都是具有挑战性的。

•高度写真性:

这一特点反映在令人印象深刻的 zero-shot FID得分6.66上,该得分是在COCO dataset上获得的(FID是评估文本到图像模型性能的主要指标;分数越低,性能越好)。

•宽高比转换:

生成非标准宽高比的图像的能力,垂直或水平的,以及标准的方形宽高比。

•级联:

DeepFloyd IF以级联方式对高分辨率数据进行建模,使用不同分辨率下单独训练的一系列模型。该过程从生成唯一低分辨率样本的基本模型(“player”)开始,然后由连续的超分辨率模型(“amplifiers”)上采样以产生高分辨率图像。

![alt text](assets/MOMA/861ff9f788944c40b9e862c9c8313c08.png)


这幅生成流程图代表三个阶段的工作：文本提示通过冻结的T5-XXL语言模型传递,将其转换为定性文本表示。

第一阶段:基本扩散模型将定性文本转换为64x64图像。DeepFloyd团队已训练三个版本的基本模型,每个模型的参数都不同:IF-I 400M、IF-I 900M和IF-I 4.3B。

第二阶段:为了“放大”图像,应用两个文本条件超分辨率模型(Efficient U-Net)对基本模型的输出。第一个模型将64x64图像放大到256x256图像。同样,该模型也有几个版本可用:IF-II 400M和IF-II 1.2B。

第三阶段:应用第二个超分辨率扩散模型产生生动的1024x1024图像。最终的第三阶段模型IF-III有700M个参数。注意:研究者还没有发布这个第三阶段模型;然而,IF模型的模块化特性允许他们在第三阶段使用其他放大模型 - 如Stable Diffusion x4 Upscaler。


数据集训练

DeepFloyd IF在定制的LAION-A数据集上训练,该数据集由10亿对高质量图像和文本组成。LAION-A是LAION-5B数据集英语部分的优化后的子集，包括基于相似性哈希进行去重、额外清理以及对原始数据集的其他修改。DeepFloyd的定制过滤器用于删除带水印的、不适合工作环境的和其他不恰当的内容。


![alt text](assets/MOMA/image-8.png)


这个实验只有DeepFloyd IF正确显示了文字。   
这个实验只有DeepFloyd IF比较正确显示了文字（4张图就1张图多了一个t）。   
Prompt: a neon sign says "It's Saturday"   
不正确     



运行

目前DeepFloyd IF模型也已经集成到了diffusers库

I. Dream

Dream is the text-to-image mode of the IF model

II. Zero-shot Image-to-Image Translation

III. Super Resolution

For super-resolution, users can run IF-II and IF-III or 'Stable x4' on an image that was not necessarely generated by IF (two cascades):=



IV. Zero-shot Inpainting


## Kohya Trainer
https://github.com/Linaqruf/kohya-trainer

page太乱     







# 结尾