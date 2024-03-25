# Elucidating the Design Space of Diffusion-Based Generative Models

## 项目背景
diffuser库在2024三月份上新   
v0.27.0: Stable Cascade, Playground v2.5, EDM-style training, IP-Adapter image embeds, and more   
需要实测模型效果    
大致看来可以达到加速，以及质量不降低的特效   
类似LCM   

更新概览：    

Stable Cascade    
Stable Cascade 系列管道与 Stable Diffusion 的不同之处在于，它们建立在三个不同的模型之上，并允许对患者图像进行分层压缩，从而实现卓越的输出。   
allow for hierarchical compression of image patients, achieving remarkable outputs.


Playground v2.5   
PlaygroundAI 发布了新的 v2.5 模型（playgroundai/playground-v2.5-1024px-aesthetic），该模型在美观方面尤其出色。除了一些调整之外，该模型紧密遵循 Stable Diffusion XL 的架构。   

EDM-style training   
EDM 是指以下论文中介绍的训练和采样技术：Elucidating the Design Space of Diffusion-Based Generative Models。我们在脚本中引入了对使用 EDM 公式进行训练的支持train_dreambooth_lora_sdxl.py。  
要stabilityai/stable-diffusion-xl-base-1.0使用 EDM 公式进行训练，您只需--do_edm_style_training在训练命令中指定标志即可   
采用 EDM 公式的新调度程序   
为了更好地支持 Playground v2.5 模型和 EDM 式培训，我们提供了对EDMDPMSolverMultistepScheduler和 的支持EDMEulerScheduler。DPMSolverMultistepScheduler它们分别支持和 的EDM 公式EulerDiscreteScheduler。    

Trajectory Consistency Distillation   
轨迹一致性蒸馏 (TCD) 使模型能够以更少的步骤生成更高质量和更详细的图像。此外，由于蒸馏过程中有效的误差缓解，即使在推理步骤较大的情况下，TCD 也表现出优越的性能。它是在轨迹一致性蒸馏中提出的。   
此版本提供了支持TCDScheduler这种快速采样的功能。与 LCM-LoRA 非常相似，TCD 需要额外的适配器来加速。   

IP-Adapter 图像嵌入和屏蔽 embeddings and masking     
所有支持 IP 适配器的管道都接受 ip_adapter_image_embeds 参数。    
我们还引入了对提供二进制掩码的支持，以指定应将输出图像的哪一部分分配给 IP 适配器。对于每个输入 IP 适配器图像，必须提供二进制掩码和 IP 适配器。   


合并 LoRA 指南    
合并 LoRA 是一种有趣且富有创意的方式来创建新的、独特的图像。set_adaptersDiffusers 通过连接 LoRA 权重进行合并的方法提供合并支持。   
现在，Diffusers 还支持add_weighted_adapterPEFT 库中的方法，解锁更高效的合并方法，如 TIES、DARE、线性，甚至这些合并方法的组合，如dare_ties.    

LEDITS++   
我们正在添加对名为 LEDITS++ 的真实图像编辑技术的支持：使用文本到图像模型进行无限图像编辑 Limitless Image Editing using Text-to-Image Models ，这是一种无参数方法，不需要微调或任何优化。   
为了编辑真实图像，LEDITS++ 管道首先反转图像 DPM-solver++ 调度程序，该调度程序有助于通过 只需要 20 个扩散步骤进行编辑，以实现反转和推理相结合。LEDITS++ 指导的定义使其既反映了编辑的方向（如果我们想要远离/靠近编辑概念）又反映了效果的强度。该指南还包括一个专注于相关图像区域的屏蔽术语，特别是对于多次编辑，可确保每个概念的相应指南术语大部分保持隔离，从而限制干扰。    




## 原理

摘要   
我们认为，基于扩散的生成模型的理论和实践目前不必要地复杂化，并试图通过提出一个明确区分具体设计选择的设计空间来弥补这种情况。这让我们能够识别采样和训练过程以及评分网络的预处理的一些变化。总之，我们的改进使 CIFAR-10 在类条件设置下的 FID 达到 1.79，在无条件设置下达到 1.97，采样速度比之前的设计快得多（每个图像 35 个网络评估）。为了进一步证明其模块化性质，我们表明我们的设计更改极大地提高了之前工作中预训练评分网络的效率和质量，包括将之前训练的 ImageNet-64 模型的 FID 从 2.07 提高到接近 SOTA 1.55 ，并在使用我们建议的改进进行重新训练后达到新的 SOTA 1.36。   




Tero Karras 等研究者在论文《Elucidating the design space of diffusionbased generative models》中对扩散模型的设计空间进行了分析，并确定了 3 个阶段，分别为   
i) 选择噪声水平的调度，  
ii) 选择网络参数化（每个参数化生成一个不同的损失函数），  
iii) 设计采样算法。     

本来是想直接consistency model的，但是发现consistency model基本被Karras method全文贯穿了，所以索性就直接从Karras method开始了  

Karras method 来源于

'' Karras T, Aittala M, Aila T, et al. Elucidating the design space of diffusion-based generative models[C]. NIPS, 2022. '' https://arxiv.org/pdf/2206.00364.pdf

这篇文章主要干了3件事情：

1. 给出了几种diffusion模型的通用框架，并且将这些模型分成了几个部分，然后加以分析，看一看什么是对模型影响最大的，哪些需要调整之类的事情。

2. 着眼于sampling过程，即图像生成部分，琢磨怎么有效减少生成图片的步数。

3. 反思了神经网络结构部分，并且尝试修改这个部分，以便他达到更好的效果。


总体来说Karras文章告诉我们的事情是，如果把整个diffusion模型的架构统一起来，那么就可以发现不少可以加速模型，提升效果的点。

### Consistency Model

''Consistency Models Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever In the 40th International Conference on Machine Learning, 2023.'' https://arxiv.org/pdf/2303.01469.pdf

Consistency Model的目标是让目标生成的速度尽可能快，比如一步到位，同时也可以支持基于zero-shot的图像生成问题，比如图像填充之类的问题。这是因为Consistency Model不光可以从0开始重新训练，也可以先获得一个预训练模型，然后在这个预训练模型的基础上进行修正，以此达到完成既定的任务。

参考链接 https://zhuanlan.zhihu.com/p/630353542


## 测试







## 训练











## dora
DoRA：权重分解低阶自适应中提出， DoRA与 LoRA 非常相似，不同之处在于它将预训练的权重分解为大小和方向两个部分，并采用 LoRA 进行定向更新，以有效地最小化可训练参数的数量。作者发现，通过使用 DoRA，LoRA 的学习能力和训练稳定性都得到了增强，并且在推理过程中没有任何额外的开销。    

LoRA 似乎比 DoRA 收敛得更快（因此训练 LoRA 时可能导致过度拟合的一组参数可能对 DoRA 效果很好）
DoRA 质量优于 LoRA，尤其是在较低等级中，等级 8 的 DoRA 和等级 8 的 LoRA 的质量差异似乎比训练等级为 32 或 64 时更显着。
这也与论文中显示的一些定量分析相一致。
用法





# 结尾