# few step gen



## SDXS
Real-Time One-Step Latent Diffusion Models with Image Conditions    
小米2024 3.25     


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
没看懂    

为了提高迭代采样的速度，作者引入了模型小型化和减少采样步骤的方法。该方法利用`知识蒸馏来简化 U-Net 和 图像解码器 的架构`，并引入了一种新的one step DM 训练技术，其使用了`特征匹配和分数蒸馏`。共推出了两种模型：SDXS-512 和 SDXS-1024，在单个 GPU 上分别实现了约 100 FPS（比 SD v1.5 快 30 倍）和 30 FP（比 SDXL 快 60 倍）的推理速度。此外，该训练方法尤其有利于图像条件控制和图像到图像的任务。    
文章 https://arxiv.org/abs/2403.16627   

此外，我们提出的方法还可以训练 ControlNet。   
![alt text](assets/README/image-18.png)   


methods   
#### Model Acceleration
![alt text](assets/README/image.png)    
unet去掉mid   
ctrlnet去掉mid   
image decoder没看懂    



We train an extremely light-weight image decoder to mimic the original VAE decoder’s output through a combination of output distillation loss and GAN loss. (335 -> 10 MB)   
我们训练一个极其轻量级的图像解码器，通过输出蒸馏损失和 GAN 损失的组合来模仿原始 VAE 解码器的输出。   
We also leverage the block removal distillation strategy to efficiently transfer the knowledge from the original U-Net to a more compact version.    
我们还利用块去除蒸馏策略来有效地将知识从原始 U-Net 转移到更紧凑的版本。   

U网。  LDM 采用 U-Net 架构 [36]，结合残差块和 Transformer 块作为其核心去噪模型。 为了利用预训练的 U-Net 的功能，同时减少计算需求和参数数量，我们采用了受 BK-SDM [16] 的块去除训练策略启发的知识蒸馏策略。 这涉及有选择地从 U-Net 中删除残差和 Transformer 块，旨在训练一个更紧凑的模型，该模型仍然可以有效地再现原始模型的中间特征图和输出。 图 2 (b) 说明了提取微型 U-Net 的训练策略。 知识蒸馏是通过输出知识蒸馏（OKD）和特征知识蒸馏（FKD）损失来实现的：   
![alt text](assets/README/image-27.png)    
![alt text](assets/README/image-28.png)    




甚至在 GPU 上实现了 100 FPS 的 512x512 图像生成和 30 FPS 的 1024x1024 图像生成。   
实测a800 512*512   
1 step   
46fps   
显存消耗3.5g   
  
![alt text](assets/README/image-19.png)

#### 文本转图像
为了减少 NFE(number of function evaluations)，我们建议拉直采样轨迹straightening the sampling trajectory ，并通过用所提出的特征匹配损失替换蒸馏损失函数by replacing the distillation loss function with the proposed feature matching loss，快速将多步模型微调为一步模型。然后，我们扩展了 Diff-Instruct 训练策略，使用所提出的特征匹配损失的梯度来替换时间步后半部分由分数蒸馏提供的梯度。   


![alt text](assets/README/image-20.png)    
feature matching loss   

![alt text](assets/README/image-21.png)   

Diff-Instruct [37] 将分数蒸馏带回到图像生成中，依赖于两个分布 p、q 之间的 Integral Kullback-Leibler (IKL) 散度的定义：     
![alt text](assets/README/image-26.png)   
其中 qt 和 pt 表示时间 t 时扩散过程的边际密度。  (3)中q0和p0之间的IKL梯度为   








#### 图像到图像
我们将我们提出的训练策略扩展到 ControlNet 的训练，依靠将预训练的 ControlNet 添加到评分函数中。 the score function.   

![alt text](assets/README/image-22.png)   
 specifically for transformations involving canny edges and depth maps.   
 ![alt text](assets/README/image-23.png)    









### 效果

```
# Ensure using 1 inference step and CFG set to 0.
image = pipe(
    prompt, 
    num_inference_steps=1, 
    guidance_scale=0,
    generator=torch.Generator(device="cuda").manual_seed(seed)
).images[0]

image.save("output.png")

```










## YOSO
"You Only Sample Once: Taming One-Step Text-To-Image Synthesis by Self-Cooperative Diffusion GANs"   
香港科技大学 2024.3.29



Note that YOSO-PixArt-α-512 is trained on JourneyDB with 512 resolution. YOSO-PixArt-α-1024 is obtained by directly merging YOSO-PixArt-α-512 with PixArt-XL-2-1024-MS, without extra explicit training on 1024 resolution.   

(YOSO-PixArt-α-512) This model is fine-tuning from PixArt-XL-2-512x512, enabling one-step inference to perform text-to-image generation.


论文好像只讲lora   
没调通

transformer训练在哪里？？？？    
公式真多   


### 原理
这是通过将扩散过程与 GAN 集成实现的。   
我们通过去噪生成器本身来平滑分布，执行自合作学习。   
by the denoising generator itself, performing self-cooperative learning   

我们提供了史上第一个`DiT`，可以在 512 分辨率上进行训练并生成图像，具有无需明确训练即可适应 1024 分辨率的能力。 

![alt text](assets/README/image-24.png)   

（2022|ICLR，扩散 GAN，少量步扩散，对抗散度，非饱和 GAN）用去噪扩散 GAN 解决生成学习难题

（2024，EBGAN，扩散，变分近似）通过扩散过程改进基于能量的对抗模型

扩散-GAN 混合模型。扩散模型中的一个问题是，当去噪步长较大时，真实的 q(x_(t−1) | xt) 不再是一个高斯分布。因此，Diffusion GANs [52] 提议不再使用参数化的高斯分布来最小化负 ELBO，而是提出最小化模型 pθ(x′_(t−1) | xt) 和 q(x_(t−1) | xt) 之间的对抗差异：  

![alt text](assets/README/image-25.png)  
其中，pθ(x0|xt) 是由 GAN 生成器强加的。 基于 GAN 的 pθ(x′_(t−1) | xt) 公式的能力，使得更大的去噪步长（即 4 步）成为可能，相比之下高斯分布要小得多。   

#### 方法
自协同扩散 GANs   
公式比较多   








#### Diffusions+GANs
然而扩散模型的核心缺点是缓慢的生成过程，哪怕采用最先进的采样器，也一般需要20步以上获得高质量的生成结果。

而另一方面，GANs这个生成模型领域在扩散模型之前无可争议的最强模型，其生成过程天然就是一步完成的。但是GANs的问题集中在训练过程的不稳定以及模式崩溃上。前者使得把GANs很难扩展到大型的图文数据集以及大型的模型上，需要极其复杂的模型设计，正则化设计，训练目标设计。这并不利于人们发展先进的文生图GANs技术，因此目前关注文生图的pure GANs不仅少，而且效果也不如扩散模型。

回顾现有的Diffusions+GANs

现有的文生图Diffusions+GANs的工作并不多，总结起来是两条路线：

1）结合diffusions的蒸馏技术和GANs技术：

SD-Turbo[1]是这条路线的代表之作，他们并没有改动GANs的目标，直接把diffusions的去噪结果当成假图片，真实数据当成真图片来进行GANs的训练。这里存在一个问题：GANs的对抗训练基本是原汁原味的，容易出现模式崩溃和训练不稳定。因此SD-Turbo是在数据空间上定义的鉴别器，并且使用了预训练的Dino v2作为鉴别器骨干，这可以大大稳定训练过程。但是这种做法的缺点很明显：`Stable Diffusion的动机就是把训练转移到latent space，这个鉴别器设计又把训练搬回去data space了，造成了训练成本的大大增加，说白了也就是stability AI卡大气粗玩得起：）。当然后续是SD3-turbo，他们也注意到了这个问题，把鉴别器又搬回latent space了`。。。

2）把对抗训练和加噪结合：

UFOGen[2]暂时是这条路线里唯一的作品（至少是笔者知道的范围里）。与SD-Turbo不同，UFOGen认为原始的GANs训练目标不稳定，因此需要使用额外的技术光滑训练分布。所以他们采用了和diffusions统一的设计，也就是说他们定义了不同的加噪级别，把diffusions的去噪结果和真实数据都加一样程度的噪声，并在加噪图片上进行GANs的训练。这种操作带来的好处不言而喻：真实分布和虚假分布更难区分，从而稳定GANs的对抗训练。但是这种在加噪图片上进行的对抗训练，会导致次优的一步生成学习。























#### PixArt-XL-2-512x512

PixArt-XL-2-512x512 模型结构    
![alt text](assets/README/image-12.png)   

Pixart-α 由用于潜在扩散的纯 Transformer 块组成：它可以在单个采样过程中根据文本提示直接生成 1024px 图像。  
`DiT`    

原始单步就能生成。yoso经过微调   
It is a Transformer Latent Diffusion Model that uses one fixed, pretrained text encoders (T5)) and one latent feature encoder (VAE).   
PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis（2023.9）


本文介绍PIXART-A，一种基于 Transformer 的 T2I 扩散模型，其图像生成质量可与最先进的图像生成器（例如 Imagen、SDXL，甚至 Midjourney）竞争，达到接近商业应用的标准。此外，它还支持高达 1024px 分辨率的高分辨率图像合成，训练成本较低，如图 1 和图 2 所示。为了实现这一目标，提出了三个核心设计：`（1）训练策略分解：我们设计了三个不同的训练步骤分别优化像素依赖性、文本图像对齐和图像美学质量； （2）高效的T2I Transformer：我们将交叉注意力模块合并到Diffusion Transformer（DiT）中，以注入文本条件并简化计算密集型的类条件分支； （3）高信息数据：我们强调文本-图像对中概念密度的重要性，并利用大型视觉语言模型自动标记密集的伪标题以辅助文本-图像对齐学习。`结果，PIXART-A训练速度明显超越现有的大规模T2I模型，例如PIXART-A仅花费 Stable Diffusion v1.5 训练时间的 10.8%（A100 GPU 675 天 vs. 6,250 天），节省近300,000美元（ 26,000美元vs. 320,000美元），并减少 90% 的二氧化碳排放。而且，与更大的SOTA模型RAPHAEL相比，我们的训练成本仅为1%。大量实验证明 PIXART-A在图像质量、艺术性和语义控制方面表现出色。我们希望PIXART-A将为 AIGC 社区和初创公司提供新的见解，以加速从头开始构建自己的高质量且低成本的生成模型。    

我的理解是采用了高质量 少量数据 训练出了较好的效果   
数据量2500万张，很少    
训练策略，使用dit 密集类条件注入   




它更适合训练和推理，并且将添加最先进的扩散采样器（如`SA-Solver`）    

##### SA-Solver
：用于扩散模型快速采样的随机 Adams 求解器     
SA-Solver: Stochastic Adams Solver for Fast Sampling of Diffusion Models   
扩散概率模型（DPM）在生成任务中取得了相当大的成功。由于从 DPM 采样相当于求解扩散 SDE 或 ODE，非常耗时，因此提出了许多基于改进微分方程求解器的快速采样方法。大多数此类技术都考虑求解扩散常微分方程，因为它具有卓越的效率。然而，随机抽样可以在生成多样化和高质量的数据方面提供额外的优势。在这项工作中，我们从方差控制扩散SDE和线性多步SDE求解器两个方面对随机采样进行了全面分析。根据我们的分析，我们提出了 SA-Solver，它是一种改进的高效随机 Adams 方法，用于求解扩散 SDE 以生成高质量数据。我们的实验表明，SA-Solver 实现了：1）与现有最先进的少步采样采样方法相比，性能得到改进或相当； 2) 在适当数量的功能评估 (NFE) 下，在大量基准数据集上获得 SOTA FID 分数。    

![alt text](assets/README/image-13.png)  

Limitations   
The model does not achieve perfect photorealism  
The model cannot render legible text   
The model struggles with more difficult tasks which involve compositionality, such as rendering an image corresponding to “A red cube on top of a blue sphere”   
fingers, .etc in general may not be generated properly.   
The autoencoding part of the model is lossy.   
模型的自动编码部分是有损的。    
 



### 实践
#### 第一种方式：加载lora 和 sd1.5
只有一个lora模型。可以调用社区的sd1.5。   
加载lora失败

##### 1步推理  
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





##### 2步推理   
我们注意到，小的 CFG 可用于提高图像质量。   





#### 第二种方式 yoso_pixart512
加载yoso_pixart512 transfomer. 主模型加载PixArt-XL-2-512x512。lcm schedu . v_prediction    
需要加装pip install sentencepiece（t5模型）(PixArt-XL-2-512x512)     




##### sentencepiece
Unsupervised text tokenizer for Neural Network-based text generation.    
17年发布，到现在仍在更新   
谷歌的语言模型bert，文生图模型PixArt 一般都要加装sentencepiece   
openai则使用另外的分词方式：bpe。如clip sd   

the vocabulary size is predetermined prior to the neural model training.   
其中词汇量大小在神经模型训练之前预先确定  
SentencePiece implements subword units (e.g., byte-pair-encoding (BPE) [Sennrich et al.]) and unigram language model [Kudo.]) with the extension of direct training from raw sentences.   
SentencePiece allows us to make a purely end-to-end system that does not depend on language-specific pre/postprocessing.   


技术亮点   
纯粹数据驱动：SentencePiece 从句子训练标记化和去标记化模型。预标记化（Moses tokenizer / MeCab / KyTea）并不总是需要的。   
与语言无关：SentencePiece 将句子视为 Unicode 字符序列。不存在依赖于语言的逻辑。  
多子字算法：BPE [ Sennrich 等人。 ] 和一元语言模型[ Kudo。 ] 支持。  
子词正则化：SentencePiece 实现了子词采样以进行子词正则化和BPE-dropout，这有助于提高 NMT 模型的鲁棒性和准确性。   
快速且轻量级：分词速度约为 50k 句/秒，内存占用约为 6MB。   
自包含：只要使用相同的模型文件，就可以获得相同的标记化/去标记化。  
直接词汇id生成：SentencePiece管理词汇到id的映射，并且可以直接从原始句子生成词汇id序列。  
基于 NFKC 的标准化：SentencePiece 执行基于 NFKC 的文本标准化。  











#### 第3种方式 yoso_pixart1024
加载yoso_pixart1024 transfomer. 主模型加载PixArt-XL-2-512x512。lcm schedu . v_prediction



## 其他模型
LCM [30] 基于一致性蒸馏（consistency distillation） [49]，将其调整为稳定扩散（SD），并使图像在 4 步中具有可接受的质量。然而，减少采样步骤导致生成质量较差。

InstaFlow [27] 基于矫正流（Rectified Flows） [26, 25]，将其调整为稳定扩散，促进了仅一步生成文本到图像。尽管如此，生成图像的保真度仍然较低。此外，由于模型在固定数据噪声配对上的训练，失去了支持可变数据噪声组合的灵活性，这对于像图像修改或可以提高生成图像质量的迭代多步采样任务是有害的



### SDXL-Lightning
可以看到 SDXL-Lightning 在 2-8 步之间的生成效果都不错，8 步时质量最稳定，甚至与 SDXL 模型 32 步的生成效果不相上下；即使在 4 步的条件下，在图像质量以及风格多样性也比 Turbo 和 LCM 模型好很多。   
SDXL-Lightning 模型是从 Stability AI 的   stable-diffusion-xl-base-1.0 模型中，使用了一种结合渐进式和对抗式蒸馏的扩散蒸馏方法提炼出来的。渐进式蒸馏使提炼后的模型能保留原模型的图像风格和种类，对抗式蒸馏则用于提升图像生成质量，二者结合使 SDXL-Lightning 在图像的快速生成和高质量、多样化之间找到了一个平衡点， 使其在快速出图的同时，依旧能保持较高的图像质量，并且能够覆盖广泛的图像模式。   

最近，探索了将 DM 和 GAN 结合用于一步文本到图像生成。UFOGen 将 DDGANs [52] 和 SSIDMs [57] 扩展到稳定扩散，通过将重建损失的计算从损坏样本改为清洁（clean）样本来进行修改。然而，它仍然使用损坏样本进行对抗匹配。     
ADD [44] 提出了基于稳定扩散的一步文本到图像生成。它遵循早期研究，使用预训练的图像编码器 DINOv2 [35] 作为鉴别器的骨干来加速训练。然而，鉴别器设计将训练从潜在空间移动到像素空间，这大大增加了计算需求。此外，它们直接在清洁的真实数据上执行对抗匹配，增加了训练的挑战。这需要更好但昂贵的鉴别器设计和昂贵的 R1 正则化来稳定训练。    

相比之下，我们通过用自生成数据替换真实数据来进行对抗训练，以平滑地实现真实数据。此外，与 UFOGen 和 ADD 相比，我们的方法可以从头开始训练以执行一步生成，这是他们没有展示的。此外，我们将我们的方法不仅扩展到 Stable Diffusion，还扩展到基于扩散 Transformer [37] 的 PixArt-α [6]。这证明了我们提出的 YOSO 的广泛应用。    









# prompt 数据集
## parti prompts
https://huggingface.co/datasets/nateraw/parti-prompts   
https://github.com/google-research/parti   
tsv 或 parque 读取   
几百KB   

![alt text](assets/README/image-2.png)   
![alt text](assets/README/image-3.png)    
![alt text](assets/README/image-1.png)  

PartiPrompts (P2) is a rich set of over 1600 prompts in English that we release as part of this work. P2 can be used to measure model capabilities across various categories and challenge aspects.   
![alt text](assets/README/image-4.png)     
P2 提示可以很简单，让我们能够衡量扩展的进度。它们也可以很复杂，例如我们为文森特·梵高的《星夜》（1889）创建的以下 67 字描述：   
Oil-on-canvas painting of a blue night sky with roiling energy. A fuzzy and bright yellow crescent moon shining at the top. Below the exploding yellow stars and radiating swirls of blue, a distant village sits quietly on the right. Connecting earth and sky is a flame-like cypress tree with curling and swaying branches on the left. A church spire rises as a beacon over rolling blue hills.    


### 典型案例
针对sdxl有两个prompt来发现其缺点   
"a motorcycle",  
"a black background with a large yellow square",  

playground进行了针对性优化


### 数据集来源：   
谷歌2022年推出 Pathways Autoregressive Text-to-Image model (Parti), an autoregressive text-to-image generation model    

Parti 和 Imagen 在探索两个不同的生成模型系列（分别是自回归模型和扩散模型）方面是互补的   
Parti 将文本到图像的生成视为序列到序列的建模问题，类似于机器翻译——这使其能够受益于大型语言模型的进步，尤其是通过扩展数据和模型大小来解锁的功能。     
In this case, the target outputs are `sequences of image tokens` instead of text tokens in another language.    

Parti 使用功能强大的图像标记器ViT-VQGAN将图像编码为离散标记序列  
通过将 Parti 的编码器-解码器参数扩展到 200 亿个参数，持续提高质量。
MS-COCO 上最先进的零样本 FID 得分为 7.23，微调后的 FID 得分为 3.22。   
我们对 Localized Narratives 和 PartiPrompts 的分析在各种类别和难度方面的有效性，这是我们作为这项工作的一部分发布的 1600 多个英语提示的新整体基准。   

![alt text](assets/README/image-5.png)   

从 350M 参数扩展到 20B 参数   

The 20B model especially excels at prompts that are abstract, require world knowledge, specific perspectives, or writing and symbol rendering.   
![alt text](assets/README/image-6.png)   
已经能写字   
![alt text](assets/README/image-7.png)   




provide a detailed example of it in the paper, where we build a very complex prompt and strategies to produce an image that fully reflects the description.    
https://arxiv.org/abs/2206.10789   
Scaling Autoregressive Models for Content-Rich Text-to-Image Generation   
![alt text](assets/README/image-8.png)   
  



### 难例分析
https://sites.research.google/parti/    

Two baseballs to the left of three tennis balls.   
A rhino beetle this size of a tank grapples a real life passenger airplane on the tarmac.   
A portrait of a statue of Anubis with a crown and wearing a yellow t-shirt that has a space shuttle drawn on it. A white brick wall is in the background.   
A cream colored labradoodle next to a white cat with black-tipped ears.   
A plate that has no bananas on it. there is a glass without orange juice next to it.   
A robot painted as graffiti on a brick wall. The words "Fly an airplane" are written on the wall. A sidewalk is in front of the wall, and grass is growing out of cracks in the concrete.   
A shiny robot wearing a race car suit and black visor stands proudly in front of an F1 race car. The sun is setting on a cityscape in the background. comic book illustration.    



## DiffusionDB
![alt text](assets/README/image-9.png)   
DiffusionDB is the first large-scale text-to-image prompt dataset. It contains 14 million images generated by Stable Diffusion using prompts and hyperparameters specified by real users.   
数据集中的文本大部分是英文。它还包含其他语言，例如西班牙语、中文和俄语。

DiffusionDB 提供两个子集（DiffusionDB 2M 和 DiffusionDB Large）来支持不同的需求。    
![alt text](assets/README/image-10.png)   
两个子集具有相似数量的独特提示，但 DiffusionDB Large 具有更多的图像。 DiffusionDB Large 是 DiffusionDB 2M 的超集。   
DiffusionDB 2M中的图像按格式存储png； DiffusionDB Large 中的图像使用无损webp格式。  


我们使用模块化的文件结构来分发 DiffusionDB。 DiffusionDB 2M 中的 200 万张图像被分为 2,000 个文件夹，其中每个文件夹包含 1,000 个图像和一个 JSON 文件，该文件将这 1,000 个图像链接到它们的提示和超参数。同样，DiffusionDB Large 中的 1400 万张图像被分为 14000 个文件夹。   

```
# DiffusionDB 2M
./
├── images
│   ├── part-000001
│   │   ├── 3bfcd9cf-26ea-4303-bbe1-b095853f5360.png
│   │   ├── 5f47c66c-51d4-4f2c-a872-a68518f44adb.png
│   │   ├── 66b428b9-55dc-4907-b116-55aaa887de30.png
│   │   ├── [...]
│   │   └── part-000001.json
│   ├── part-000002
│   ├── part-000003
│   ├── [...]
│   └── part-002000
└── metadata.parquet



```

```
# DiffusionDB Large
./
├── diffusiondb-large-part-1
│   ├── part-000001
│   │   ├── 0a8dc864-1616-4961-ac18-3fcdf76d3b08.webp
│   │   ├── 0a25cacb-5d91-4f27-b18a-bd423762f811.webp
│   │   ├── 0a52d584-4211-43a0-99ef-f5640ee2fc8c.webp
│   │   ├── [...]
│   │   └── part-000001.json
│   ├── part-000002
│   ├── part-000003
│   ├── [...]
│   └── part-010000
├── diffusiondb-large-part-2
│   ├── part-010001
│   │   ├── 0a68f671-3776-424c-91b6-c09a0dd6fc2d.webp
│   │   ├── 0a0756e9-1249-4fe2-a21a-12c43656c7a3.webp
│   │   ├── 0aa48f3d-f2d9-40a8-a800-c2c651ebba06.webp
│   │   ├── [...]
│   │   └── part-000001.json
│   ├── part-010002
│   ├── part-010003
│   ├── [...]
│   └── part-014000
└── metadata-large.parquet



```



```
{
  "f3501e05-aef7-4225-a9e9-f516527408ac.png": {
    "p": "geodesic landscape, john chamberlain, christopher balaskas, tadao ando, 4 k, ",
    "se": 38753269,
    "c": 12.0,
    "st": 50,
    "sa": "k_lms"
  },
}

Data Fields
key: Unique image name
p: Prompt
se: Random seed
c: CFG Scale (guidance scale)
st: Steps
sa: Sampler



```


然而，生成具有所需细节的图像很困难，因为它需要用户编写正确的提示来指定确切的预期结果。开发此类提示需要反复试验，并且常常让人感觉随意且无原则。西蒙·威利森将编写提示比作巫师学习“魔法咒语”：用户不明白为什么某些提示有效，但他们会将这些提示添加到他们的“咒语书”中。例如，为了生成高度详细的图像，在提示中添加“artstation 上的趋势”和“虚幻引擎”等特殊关键字已成为常见做法。
 “trending on artstation” and “unreal engine”    



我们通过在官方 Stable Diffusion Discord 服务器上抓取用户生成的图像来构建 DiffusionDB。我们选择稳定扩散，因为它是目前唯一开源的大型文本到图像生成模型，并且所有生成的图像都具有 CC0 1.0 通用公共领域奉献许可证，该许可证放弃所有版权并允许用于任何目的。我们选择官方的Stable Diffusion Discord 服务器是因为它是公开的，并且它对生成和共享非法、仇恨或 NSFW（不适合工作，例如性和暴力内容）图像有严格的规则。服务器还不允许用户编写或共享带有个人信息的提示。









### Parquet
我们以 Parquet 格式存储这些表，因为 Parquet 是基于列的：您可以有效地查询各个列（例如提示），而无需读取整个表。

下面是来自 的三个随机行metadata.parquet。

metadata.parquet为 (2000000, 13)， 的形状metatable-large.parquet为 (14000000, 13)。两个表共享相同的架构，每行代表一个图像   

![alt text](assets/README/image-11.png)   


方法 3. 使用metadata.parquet（仅限文本）   
如果您的任务不需要图像，那么您可以轻松访问表中的所有 200 万个提示和超参数metadata.parquet。   
```
from urllib.request import urlretrieve
import pandas as pd

# Download the parquet table
table_url = f'https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet'
urlretrieve(table_url, 'metadata.parquet')

# Read the table using Pandas
metadata_df = pd.read_parquet('metadata.parquet')

```


## JourneyDB 
Neural Information Processing Systems (NeurIPS) 2023   
1CUHK MMLab,   2The University of Hong Kong,   3Shanghai AI Laboratory,   4Nanjing University,   5CPII   


是一个大规模生成的图像理解数据集，包含4,429,295 张 高分辨率的 Midjourney 图像，并附有相应的文本提示、图像标题和 视觉问答等注释。

如图 1 所示，对于每个图像实例，我们获取相应的文本提示，用于使用 Midjourney 生成图像。此外，我们使用 GPT3.5 来生成标题和 VAQ groundtruth。     
![alt text](assets/README/image-14.png)   
![alt text](assets/README/image-15.png)   

每个实例都包含一个Midjourney图像、一个文本提示、一个标题和一个 VAQ 基本事实。

我们将风格提示分为 344 个类别，包括 3.2 节中介绍的相机参数、灯光、艺术家风格、配色方案等，并在每个类别中进行风格提示检索，这显着缩小了搜索空间。为了建立基准，我们使用 CLIP 来执行零样本检索评估。结果如表5所示。我们注意到整体风格提示空间的检索导致召回率极低。当在每个类别的子空间上检索时，模型的表现要好得多。  
![alt text](assets/README/image-16.png)   


我们设置了任务：提示反演，它采用单个图像并预测相应的提示。  
现有模型很难捕获输入图像的细节和风格相关信息，并且在传统数据集中的表现也不佳。    
![alt text](assets/README/image-17.png)   




# 结尾