# 论文信息：

IC-Light 是一个控制图像照明的项目。

“IC-Light”这个名称代表“Impose Constant Light”（我们将在本页末尾简要描述这一点）。

目前，我们发布了两种类型的模型：文本条件重新光照模型和背景条件模型。两种类型都将前景图像作为输入。



Related Work

Also read ...

Total Relighting: Learning to Relight Portraits for Background Replacement

Relightful Harmonization: Lighting-aware Portrait Background Replacement

SwitchLight: Co-design of Physics-driven Architecture and Pre-training Framework for Human Portrait Relighting
About

型号注释 

iclight_sd15_fc.safetensors - 默认的重新照明模型，以文本和前景为条件。您可以使用初始潜伏来影响重新照明。

iclight_sd15_fcon.safetensors - 与“iclight_sd15_fc.safetensors”相同，但使用偏移噪声进行训练。请注意，在用户研究中，默认的“iclight_sd15_fc.safetensors”稍微优于此模型。这就是为什么默认模型是没有偏移噪声的模型的原因。    
Same as "iclight_sd15_fc.safetensors" but trained with offset noise. Note that the default "iclight_sd15_fc.safetensors" outperform this model slightly in a user study. And this is the reason why the default model is the model without offset noise.      
历史经验可以得到更纯的图片颜色

iclight_sd15_fbc.safetensors - 以文本、前景和背景为条件的重新照明模型。




# 原理

换背景，换灯光方向（四个：上下左右）     
控制光照柔和与强烈程度，光种     


## Text-Conditioned Model   
输入：提供人物图片（会被自动提取未前景再输入模型），删去背景获取前景图

(Note that the "Lighting Preference" are just initial latents - eg., if the Lighting Preference is "Left" then initial latent is left white right black.)      

Prompt: beautiful woman, detailed face, warm atmosphere, at home, bedroom

Lighting Preference: Left

## Background-Conditioned Model     
输入：提供人物图片，纯背景图    

背景图可以flip     


![alt text](assets/IC-Light/image-2.png)



来自“外观混合”和“光源混合”的两个图像是一致的（理想情况下，在 HDR 空间中数学上是等效的）。

在训练重新照明模型时，我们强加了这种一致性（在潜在空间中使用 MLP）。

因此，该模型能够产生高度一致的重新光照 -如此一致，甚至可以将不同的重新光照合并为法线贴图！尽管事实上这些模型是潜在扩散的。
As a result, the model is able to produce highly consistent relight - so consistent that different relightings can even be merged as normal maps! Despite the fact that the models are latent diffusion.

从扩散模型的角度实现打光，几十年前的技术复现


![alt text](assets/IC-Light/image-4.png)
从左到右依次是输入、模型输出、重新照明、分割的阴影图像和合并的法线贴图。请注意，该模型未使用任何法线贴图数据进行训练。这个正常的估计来自于重新点亮的一致性。










## 施加一致的光
在 HDR 空间中，照明具有所有光传输都是独立的属性。     
illumination has a property that all light transports are independent.



因此，不同光源的外观混合相当于混合光源的外观：   
the blending of appearances of different light sources is equivalent to the appearance with mixed light sources:
![alt text](assets/IC-Light/image-3.png)








# 其他
## fooocus
https://github.com/lllyasviel/Fooocus


About    
Focus on prompting and generating

![alt text](assets/IC-Light/image-1.png)

Fooocus is an image generating software (based on Gradio).

Fooocus is a rethinking of Stable Diffusion and Midjourney’s designs:

Learned from Stable Diffusion, the software is offline, open source, and free.

Learned from Midjourney, the manual tweaking is not needed, and users only need to focus on the prompts and images.




## Swarm UI
alternate comfyui

https://github.com/Stability-AI/StableSwarmUI


StableSwarmUI, A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility.

![alt text](assets/IC-Light/image.png)


## 早期研究也能控制打光。而且数据集完善


### Acquiring the Reflectance Field of a Human Face
https://www.pauldebevec.com/Research/LS/

https://www.pauldebevec.com/Research/LS/debevec-siggraph2000-high.pdf

获取人脸反射场    
Paul Debevec、Tim Hawkins、Chris Tchou、Haarm-Pieter Duiker、Westley Sarokin 和Mark Sagar      
SIGGRAPH 2000 会议论文集

2004 年 4 月 10 日



摘要：

我们提出了一种获取人脸反射场的方法，并使用这些测量结果在光照和视点的任意变化下渲染人脸。我们首先使用光台在入射照明方向的密集采样下从一小组视点获取面部图像。然后，我们根据照明方向空间上的每个观察到的图像像素的值构建反射函数图像。根据反射率函数，我们可以以任何形式的采样或计算照明从原始视点直接生成面部图像。为了改变视点，我们使用皮肤反射率模型来估计新视点的反射率函数的外观。我们通过在新颖的照明和视点下合成人脸的渲染来演示该技术。

![alt text](assets/IC-Light/image-6.png)




## GeoWizard
GeoWizard: Unleashing the Diffusion Priors for 3D Geometry Estimation from a Single Image    

[Submitted on 18 Mar 2024]     
GeoWizard: Unleashing the Diffusion Priors for 3D Geometry Estimation from a Single Image     

https://github.com/fuxiao0719/GeoWizard

我们引入了 GeoWizard，一种新的生成基础模型，旨在从单个图像估计几何属性，例如深度和法线。尽管该领域已经进行了大量研究，但由于公开数据集的多样性低和质量差，进展受到很大限制。因此，先前的工作要么受限于有限的场景，要么无法捕捉几何细节。在本文中，我们证明生成模型与传统判别模型（例如 CNN 和 Transformer）相反，可以有效解决固有的不适定问题。我们进一步表明，利用扩散先验可以显着提高泛化能力、细节保留和资源使用效率。具体来说，我们扩展了原始的稳定扩散模型来联合预测深度和法线，从而允许两种表示之间的相互信息交换和高度一致性。更重要的是，我们提出了一种简单而有效的策略，将各种场景的复杂数据分布分离成不同的子分布。这种策略使我们的模型能够识别不同的场景布局，以卓越的保真度捕获 3D 几何图形。 GeoWizard 为零镜头深度和法线预测设定了新的基准，显着增强了许多下游应用，例如 3D 重建、2D 内容创建和新颖的视点合成。     



在微调过程中，GeoWizard 将图像、GT 深度和 GT 法线通过冻结的 VAE 编码到潜在空间中，并形成两个串联的几何组。每个组都被输入 U-Net，在几何切换器的指导下生成深度或正常域的输出。此外，还引入了场景提示，以使用三种可能的场景布局（室内/室外/物体）之一生成结果。在推理过程中，给定图像、场景提示、初始深度噪声和法线噪声，GeoWizard 可以联合生成高质量的深度和法线。

![alt text](assets/IC-Light/image-7.png)





## switchlight
https://arxiv.org/pdf/2402.18848




## Total Relighting:
Learning to Relight Portraits for Background Replacement   
https://augmentedperception.github.io/total_relighting/   
SIGGRAPH 2021 技术视频

![alt text](assets/IC-Light/image-5.png)    
老方法效果已经很好     
包括现在腾讯会议的换背景，就是有时候有点抖。ic-light主打的打光甚至都不能算新颖技术    
可能是对于limit的优化吧      


我们提出了一种用于人像重新照明和背景替换的新颖系统，该系统可保持高频边界细节并准确合成新颖照明下的主体外观，从而为任何所需场景生成逼真的合成图像。我们的技术包括通过 Alpha 抠图、重新照明和合成进行前景估计。我们证明，这些阶段中的每一个都可以在顺序管道中处理，无需使用先验（例如已知背景或已知照明），也无需专门的采集技术，仅使用单个 RGB 肖像图像和新颖的目标 HDR 照明环境作为输入。我们使用在光级计算照明系统中捕获的对象的重照肖像来训练我们的模型，该系统记录了多种照明条件、高质量的几何形状和准确的 alpha 遮罩。为了执行逼真的重新照明以进行合成，我们在深度学习框架中引入了一种新颖的每像素照明表示，它明确地模拟了外观的漫反射和镜面反射分量，生成具有令人信服的渲染非朗伯效果（如镜面高光）的重新照明肖像。多次实验和比较表明了所提出的方法应用于野外图像时的有效性。




## Relightful Harmonization
[Submitted on 11 Dec 2023 (v1), last revised 7 Apr 2024 (this version, v2)]      
Relightful Harmonization: Lighting-aware Portrait Background Replacement

肖像协调旨在将拍摄对象合成到新的背景中，调整其灯光和颜色以确保与背景场景的和谐。现有的协调技术通常只专注于调整前景的全局颜色和亮度，而忽略了背景中的关键照明线索，例如明显的照明方向，从而导致不切实际的构图。我们推出 Relightful Harmonization，这是一种照明感知扩散模型，旨在使用任何背景图像无缝协调前景肖像的复杂照明效果。我们的方法分三个阶段展开。首先，我们引入一个照明表示模块，该模块允许我们的扩散模型对来自目标图像背景的照明信息进行编码。其次，我们引入了一个对齐网络，它将从图像背景中学习到的照明特征与从全景环境地图中学习到的照明特征对齐，这是场景照明的完整表示。最后，为了进一步提高所提出方法的真实感，我们引入了一种新颖的数据模拟管道，该管道可以从各种自然图像中生成合成训练对，用于细化模型。我们的方法在视觉保真度和照明连贯性方面优于现有基准，在现实测试场景中表现出卓越的泛化能力，突出了其多功能性和实用性。




## Photon (sd1.5底模)
Photon aims to generate photorealistic and visually appealing images effortlessly.

Recommendation for generating the first image with Photon:

Prompt: A simple sentence in natural language describing the image.

Negative: "cartoon, painting, illustration, (worst quality, low quality, normal quality:2)"

Sampler: DPM++ 2M Karras | Steps: 20 | CFG Scale: 6

Size: 512x768 or 768x512

Hires.fix: R-ESRGAN 4x+ | Steps: 10 | Denoising: 0.45 | Upscale x 2

(avoid using negative embeddings unless absolutely necessary)


### development
The development process was somewhat chaotic but essentially:

It started from an old mix.

LORAs were trained on various topics using AI-generated photorealistic images.

These LORAs were mixed within the model using different weights.

In the midst of this mixing, hand generation broke.

LORAs were generated and remixed in an attempt to fix hand generation (not entirely successful).

### limit
In future versions, I will try to:

Completely eliminate the need for a negative prompt to generate high-quality images.

Fix the hand generation issue to minimize instances of poorly drawn hands.

Explore more automated training processes. I would love to have 5,000 or 50,000 high-quality AI-generated photorealistic images for training purposes.







# 结尾