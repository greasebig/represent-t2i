Hyper-SD     
Hyper-SD: Trajectory Segmented Consistency Model for Efficient Image Synthesis

又是一个单步推理模型


# 论文信息
字节    

[Submitted on 21 Apr 2024]
Hyper-SD: Trajectory Segmented Consistency Model for Efficient Image Synthesis


最近，出现了一系列扩散感知蒸馏算法，以减轻与扩散模型（DM）的多步骤推理过程相关的计算开销。当前的蒸馏技术通常分为两个不同的方面：i) ODE 轨迹保持； ii) ODE 轨迹重构。然而，这些方法会遭受严重的性能下降或域转移。为了解决这些限制，我们提出了 Hyper-SD，这是一种新颖的框架，它协同地融合了 ODE 轨迹保存和重构的优点，同时在步骤压缩期间保持近乎无损的性能。首先，我们引入轨迹分段一致性蒸馏，在预定义的时间步段内逐步进行一致性蒸馏，这有利于从高阶角度保存原始 ODE 轨迹。其次，我们结合人类反馈学习来提高模型在低步状态下的性能，并减轻蒸馏过程带来的性能损失。第三，我们集成了分数蒸馏，以进一步提高模型的低步生成能力，并首次尝试利用统一的LoRA来支持所有步骤的推理过程。大量实验和用户研究表明，Hyper-SD 对于 SDXL 和 SD1.5 来说都可以通过 1 到 8 个推理步骤实现 SOTA 性能。例如，在 1 步推理中，Hyper-SDXL 在 CLIP 分数上超过 SDXL-Lightning +0.68，在 Aes 分数上超过 +0.51。


Introduction    
Hyper-SD is one of the new State-of-the-Art diffusion model acceleration techniques. In this repository, we release `the models distilled from SDXL Base 1.0 and Stable-Diffusion v1-5。`





# 模型信息
https://huggingface.co/ByteDance/Hyper-SD

Project Page: https://hyper-sd.github.io/


News🔥🔥🔥

    Apr.30, 2024. 💥💥💥 Our 8-Steps CFG-Preserved Hyper-SDXL-8steps-CFG-LoRA and Hyper-SD15-8steps-CFG-LoRA is available now(support 5~8 guidance scales), we strongly recommend making the 8-step CFGLora a standard configuration for all SDXL and SD15 models!!! (the 4-steps version will be coming soon)💥💥💥

    Apr.28, 2024. ComfyUI workflows on 1-Step Unified LoRA 🥰 with TCDScheduler to inference on different steps are released! Remember to install ⭕️ ComfyUI-TCD in your ComfyUI/custom_nodes folder!!! You're encouraged to adjust the eta parameter to get better results 🌟!
    Apr.26, 2024. 💥💥💥 Our CFG-Preserved Hyper-SD15/SDXL that facilitate negative prompts and larger guidance scales (e.g. 5~10) will be coming soon!!! 💥💥💥
    Apr.26, 2024. Thanks to @Pete for contributing to our scribble demo with larger canvas right now 👏.
    Apr.24, 2024. The ComfyUI workflow and checkpoint on 1-Step SDXL UNet ✨ is also available! Don't forget ⭕️ to install the custom scheduler in your ComfyUI/custom_nodes folder!!!
    Apr.23, 2024. ComfyUI workflows on N-Steps LoRAs are released! Worth a try for creators 💥!
    Apr.23, 2024. Our technical report 📚 is uploaded to arXiv! Many implementation details are provided and we welcome more discussions👏.
    Apr.21, 2024. Hyper-SD ⚡️ is highly compatible and work well with different base models and controlnets. To clarify, we also append the usage example of controlnet here.
    Apr.20, 2024. Our checkpoints and two demos 🤗 (i.e. SD15-Scribble and SDXL-T2I) are publicly available on HuggingFace Repo.



Hyper-SD Scribble demo host on 🤗 scribble

Hyper-SDXL One-step Text-to-Image demo host on 🤗 T2I


Checkpoints

    Hyper-SDXL-Nstep-lora.safetensors: Lora checkpoint, for SDXL-related models.

    Hyper-SD15-Nstep-lora.safetensors: Lora checkpoint, for SD1.5-related models.

    Hyper-SDXL-1step-unet.safetensors: Unet checkpoint distilled from SDXL-Base.

Hyper-SD is one of the new State-of-the-Art diffusion model acceleration techniques. In this repository, we release the models distilled from SDXL Base 1.0 and Stable-Diffusion v1-5。    

![alt text](assets/Hyper-SD/image-2.png)    

# 原理
Hyper-SD 采用两阶段渐进稠度蒸馏。第一阶段涉及两个独立时间段的一致性蒸馏：[0，T/2]和[T/2，T]以获得两个段的一致性ODE。然后，采用该ODE轨迹在后续阶段训练全局一致性模型     
![alt text](assets/Hyper-SD/image.png)     


原理：
Hyper-SD协同地融合了 ODE 轨迹保存和重构的优点，同时在步骤压缩期间保持近乎无损的性能。
1. 引入轨迹分段一致性蒸馏，在预定义的时间步段内逐步进行一致性蒸馏，这有利于从高阶角度保存原始 ODE 轨迹。
Trajectory Segmented Consistency Distillation （TSCD）
2. 结合人类反馈学习来提高模型在低步状态下的性能，并减轻蒸馏过程带来的性能损失。
3. 集成了分数蒸馏，以进一步提高模型的低步生成能力





# 效果
ByteDance/Hyper-SDXL-1Step-T2I      

the word 'START'   
![alt text](assets/Hyper-SD/image-1.png)    
单步与论文不一致，文字不遵循    
可能是多步的结果   


The unified LoRAs of Hyper-SD are compatible with ControlNet. The examples are conditioned on either scribble or canny images.    

Hyper-SD15-Scribble     
![alt text](assets/Hyper-SD/image.jpeg)    
a photo of a cat   
lcm   
网页上两三秒出图  
![alt text](assets/Hyper-SD/image-1.jpeg)      
![alt text](assets/Hyper-SD/image-2.jpeg)    

TCD


## lora
1. lora加速   
lora加速生成的人物图片的美观度和清晰度普遍降低。受蒸馏底模影响。     
lora加速生成，在4步以内优势明显，超过8步后不如现在常用的采样方法使用高效采样计划进行采样。
Hyper-SDXL-1step-lora可以和社区底模搭配使用，由TCD改进而来。结合汉服模型1步可以生成，但是画质较差，4步以内生成最好在1024*1024分辨率上使用，否则人物畸形严重。8步生成图片有小的瑕疵。cfg取1附近的值。    
搭配公司底模，换不同prompt会产生人脸很像的情况    
这个lora在使用在1024*1024分辨率效果比较好    
sdxl_lighrning_2step_lora至少需要2步生成。    
LCM&TurboMix LoRA 至少需要6步生成，CFG为2.0时效果最好。4步时图片相当模糊。     
lcm-lora-sdxl 至少需要4步生成，CFG为2.0时效果最好。    
lcm-lora-sdxl人物没那么像。其他不同prompt都挺像    
sdxl_lightning_2step_lora稍微没那么像，但是质感是sdxl base经典的卡通质感    

2. 以上这些加速lora的cfg都比较小     
Hyper-LORA系列有一个 Hyper-SDXL-8steps-CFG-lora 少步推理条件下支持大cfg: 5-8。可以与AYS等加速采样方式对比。   
生成质量差，眼睛一般有问题   


3. 蒸馏主模型    
Hyper-SDXL-1step-Unet效果比lightning好一些，比turbo好。    
文字生成优于另外两种     
lightning偏向卡通和畸形和模糊    


4. 公司已有模型加速采样推荐采用两种方法，这两种都能比较好保留原模型质量   
4.1 自己蒸馏主模型。   
4.2 高效采样计划进行10步或15步采样。如AYS, sgm_uniform。少数情况会出不好的图。   











# 其他
## 混合专家模型 Mixture of Experts，简称MoE

随着不同应用场景的实际需求，大模型的参数会变得越来越大，复杂性和规模不断的增加，尤其是在多模态大模型的开发中，每个数据集可能完全不同，有来自文本的数据、图像的数据、语音的数据等，包含不同的模式，特征和标注之间的关系可能也大有不同，这不但增加了训练的难度，也提高了推理的成本，如何将大模型的训练难度和推理成本降低已经是各大研究机构和大厂都在攻克的任务。为了解决这些问题，混合专家（MoE）方法应运而生。

一、什么是混合专家模型？     
混合专家（Mixture of Experts，简称MoE）是一种集成学习方法，它通过将多个专业化的子模型（即“专家”）组合起来，形成一个整体模型，每一个“专家”都在其擅长的领域内做出贡献。而决定哪个“专家”参与解答特定问题的，是一个称为“门控网络”的机制。每个专家模型可以专注于解决特定的子问题，而整体模型则能够在复杂的任务中获得更好的性能。

MoE提出的前提是如果有一个包括了多个领域知识的复杂问题，我们该使用什么样的方法来解决呢？最简单的办法就是把各个领域的专家集合到一起来攻克这个任务，当然我们事先要把不同的任务先分离出来，这样才便于分发给不同领域的专家，让他们来帮忙处理，最后再汇总结论。

二、结构和原理    
混合专家模型（MoE）是一种稀疏门控制的深度学习模型，由两个关键组成部分构成：门控网络（GateNet）和专家网络（Experts）。

门控网络：负责根据输入数据的特征，动态地决定哪个专家模型应该被激活以生成最佳预测。    
专家网络：是一组独立的模型，每个模型都负责处理某个特定的子任务。    


通过门控网络，输入数据将被分配给最适合的专家模型进行处理，并根据不同模型的输出进行加权融合，得到最终的预测结果。

混合专家模型在训练过程中通过门控模型实现“因材施教”，进而在推理过程中实现专家模型之间的“博采众长”。MoE的专家模型可以是小型的MLP或者复杂的LLM。


## TCD
TCD受一致性模型的启发，是一种新的蒸馏技术，可将预训练扩散模型中的知识蒸馏到少步采样器中。

TCD的优势：

    ● 灵活的NFEs: 对于TCD, NFEs可以任意变化(与Turbo相比)，而不会对结果质量产生不利影响(与LCMs相比)，其中LCM在高NFEs时质量显著下降。
    ● 优于Teacher: TCD在高NFEs下保持了卓越的生成质量，甚至超过了origin SDXL的DPM-Solver++(2S)的性能。值得注意的是，在训练期间没有包括额外的鉴别器或LPIPS监督。
    ● 自由改变细节: 在推理过程中，可以通过调整一个超参数gamma简单地修改图像中的细节水平。该选项不需要引入任何其他参数。
    ● 通用性: 与LoRA技术集成，TCD可以直接应用于共享相同骨干网的各种模型(包括自定义社区模型、styled LoRA、ControlNet、IP-Adapter)。



## LCM&TurboMix LoRA
webui strength      
Same way you'd change any other word weight, <lora:$NAME:0.8>      






## UniFL

[Submitted on 8 Apr 2024]    
UniFL: Improve Stable Diffusion via Unified Feedback Learning

扩散模型彻底改变了图像生成领域，导致高质量模型和多样化下游应用的激增。然而，尽管取得了这些重大进步，当前的竞争解决方案仍然存在一些局限性，包括视觉质量差、缺乏美感、推理效率低，而且还没有全面的解决方案。为了应对这些挑战，我们提出了 UniFL，这是一个利用反馈学习来全面增强扩散模型的统一框架。 UniFL 是一种通用、有效且可推广的解决方案，适用于各种扩散模型，例如 SD1.5 和 SDXL。值得注意的是，UniFL 包含三个关键组成部分：感知反馈学习，提高视觉质量；解耦反馈学习，提高审美吸引力；以及对抗性反馈学习，可优化推理速度。深入的实验和广泛的用户研究验证了我们提出的方法在提高生成模型的质量及其加速方面的卓越性能。例如，UniFL 在生成质量方面超过 ImageReward 17% 用户偏好，并在 4 步推理中比 LCM 和 SDXL Turbo 分别高出 57% 和 20%。此外，我们还验证了我们的方法在下游任务中的有效性，包括 Lora、ControlNet 和 AnimateDiff。




## CreaPrompt Lightning

![alt text](assets/Hyper-SD/image-3.png)

![alt text](assets/Hyper-SD/image-4.png)





## SDXL-Lightning

ByteDance/SDXL-Lightning

[Submitted on 21 Feb 2024 (v1), last revised 2 Mar 2024 (this version, v3)]   
SDXL-Lightning: Progressive Adversarial Diffusion Distillation

SDXL-Lightning：渐进式对抗扩散蒸馏

我们提出了一种扩散蒸馏方法，该方法在基于 SDXL 的一步/少步 1024px 文本到图像生成中实现了新的最先进技术。我们的方法结合了渐进式和对抗式蒸馏，以实现质量和模式覆盖范围之间的平衡。在本文中，我们讨论了理论分析、鉴别器设计、模型制定和训练技术。我们将经过精炼的 SDXL-Lightning 模型作为 LoRA 和完整的 UNet 权重进行开源。

有lora和unet     






## sdxl-turbo
stabilityai/sdxl-turbo

SDXL-Turbo 是SDXL 1.0的精炼版本，经过实时合成训练。 SDXL-Turbo 基于一种称为对抗扩散蒸馏 (ADD) 的新颖训练方法（请参阅技术报告），该方法允许在高图像质量下以 1 到 4 个步骤对大规模基础图像扩散模型进行采样。这种方法使用分数蒸馏来利用大规模现成的图像扩散模型作为教师信号，并将其与对抗性损失相结合，以确保即使在一个或两个采样步骤的低步骤状态下也能确保高图像保真度。

使用完整模型comfyui仿佛没办法识别文字

只能512*512左右      












# 结尾


