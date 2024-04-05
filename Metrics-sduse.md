## 生成模型指标 (Metrics of Generative Models)
生成模型 (genrative model) 的生成结果通常没有直接的 ground truth 来计算样本的生成质量. 比如图像生成既要考虑生成图像的噪声要低, 清晰度要高, 同时生成的多样性要高. 我们可以直接找一些人来做图灵测试, 从而评估图像生成的效果, 但这样评估的成本显然是比较高的. 本文介绍几种论文中常用的近似指标 IS, FID, NLL. 既然是近似指标, 那么就说明, 这些指标好不代表生成的样本一定好, 它们仅仅是在一定程度上可以反映生成样本的质量.  

### Inception Score, IS
Salimans 等人在 2016 年 《Improved Techniques for Training GANs》 一文中提出了 Inception Score 来衡量生成模型的结果. 作者发现 IS
 的得分与人类评估的结果是匹配的不错的. IS
 的定义基于 Inception-V3 分类模型, 该模型是 Szegedy 等人在 2015 年 《Rethinking the Inception Architecture for Computer Vision》 一文中提出的, 用于 ImageNet 1000 个类别的分类.


 ### Fréchet Inception Distance, FID
 区别于 IS 是在 Inception-V3 输出的分布上计算的, FID 是在高层特征上计算真假图片之间的距离. FID 是 Heusel 等人在 2017 年 《GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium》 一文中提出的用于衡量生成样本质量的指标. 其计算方式如下：   

 $\text{FID} = \Vert \mu_r - \mu_g \Vert^2 + Tr\left( \Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{\frac12} \right)$

 tex command 才能在 md 写公式

![alt text](assets/Metrics/image.png)   

## 损失
### LPIPS 损失
LPIPS 是一种衡量图像相似度的方法，它通过深度学习模型来评估两个图像之间的感知差异。LPIPS 认为，即使两个图像在像素级别上非常接近，人类观察者也可能将它们视为不同。因此，LPIPS 使用预训练的深度网络（如 VGG、AlexNet）来提取图像特征，然后计算这些特征之间的距离，以评估图像之间的感知相似度。        

来源于CVPR2018的一篇论文《The Unreasonable Effectiveness of Deep Features as a Perceptual Metric》     
LPIPS 比传统方法（比如L2/PSNR, SSIM, FSIM）更符合人类的感知情况。LPIPS的值越低表示两张图像越相似，反之，则差异越大。       
![alt text](assets_picture/Metrics-sduse/image.png)    

由传统的评价标准如L2、SSIM、PSNR等评价结果和人体认为的大不相同，这是传统方法的弊端。如果图片平滑，那么传统的评价方式则大概率会失效。而目前GAN尤其是VAE等生成模型生成结果都过于平滑。     


### 感知损失（Perceptual loss）
计算方法：
感知损失通常用于图像重建或生成任务中，如风格迁移、超分辨率等，其目的是使生成的图像在感知上更接近目标图像。感知损失通过使用深度网络（如 VGG）的中间层激活来比较原始图像和生成图像。不同于直接比较像素值，感知损失关注于图像的高级特征（如纹理、形状等），这更接近于人类的视觉感知。     

两者之间的关系
相同点：

目标相似性：两者都旨在通过深度学习模型捕捉图像的感知质量，更接近人类的视觉系统。
实现方式：它们都使用了深度卷积网络（如 VGG）来提取图像的特征表示。这些网络经过训练能够捕捉到图像的关键视觉特征，从而用于计算图像间的差异或相似性。
不同点：

应用场景的差异：尽管两者在理论上都关注于感知质量，但LPIPS更多用于评估图像相似度，如图像质量评估任务，而感知损失通常作为一种损失函数来使用，广泛应用于图像生成和重建任务，帮助生成的图像在感知上更加自然和真实。           



## diffusers快速生图
jupyter notebook


## diffusers多图生成方式
```
generator = torch.manual_seed(318)
steps = 2
img_list = []
for age in [2,20,30,50,60,80]:
    imgs = pipeline(prompt=f"A photo of a cute girl, {age} yr old, XT3",
                        num_inference_steps=steps, 
                        num_images_per_prompt = 1,
                            generator = generator,
                            guidance_scale=1.1,
                       )[0]
    img_list.append(imgs[0])
make_image_grid(img_list,rows=1,cols=len(img_list))


```
![alt text](assets/Metrics-sduse/image.png)   


下面这样生成的图片和每个图片绑定seed结果不太一样   
```
seed = 0

#image = pipe(prompt=prompt, num_inference_steps=25, cross_attention_kwargs={"scale": 1.0}).images[0]
image = pipe(
        prompt, 
        num_inference_steps=1, 
        num_images_per_prompt = 8,
        guidance_scale=0,
        generator=torch.Generator(device="cuda").manual_seed(seed)
    )
print(image.images)
print(len(image.images))


folder_path = '/home/WujieAITeam/private/lujunda/infer-pics/pic-sdxs-512-0.9/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
for i in range(len(image.images)):
# 返回的image.images就是单纯四个PIL文件  
    image.images[i].save(folder_path + prompt[:10] + str(i) + ".png")

```








# 结尾

惶惶然不知所往   
宿命感   
往往这时候忘记都是因为压力降临，有紧迫任务而忘记   
明白但是仅此而已    
五斗米折腰   
