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


## sd多图生成方式
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
