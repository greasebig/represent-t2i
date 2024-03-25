# Elucidating the Design Space of Diffusion-Based Generative Models

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




