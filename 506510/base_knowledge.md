# 矢量图
又称矢量图形（Vector Graphic），是一种面向对象的图像或绘图图像。  
与通常的位图图像（Pixel Graphic）不同，矢量图矢量图是根据几何特性来绘制的，只能靠软件生成。


矢量图具备以下显著特点：   
1、无限缩放性： 与位图图像不同，矢量图可以在任何尺寸下无损缩放，无需担心像素失真或图像质量下降。这是因为矢量图由数学公式生成，可以轻松适应各种输出设备的不同分辨率。

2、小文件体积： 矢量图文件通常较小，因为它们仅包含图像的数学描述，而不存储每个像素的颜色信息。这使得它们在网络传输速度更快，并且占用更少的存储空间。

3、高度可编辑： 矢量图易于编辑和修改，因为它们由一系列路径、线条和形状组成，可以通过图形设计软件进行精确编辑。

4、透明背景： 矢量图可以具有透明背景，这意味着可以轻松叠加到其他图像或背景上，无需复杂的抠图过程。

5、多用途性： 矢量图通常用于创建图标、标志、矢量艺术、图表、地图等各种图形元素。它们在印刷品、网站设计、动画制作和多媒体项目中都有广泛的应用。

矢量图，也称为面向对象的图像或绘图图像，在数学上定义为一系列由点连接的线。矢量文件中的图形元素称为对象。每个对象都是一个自成一体的实体，它具有颜色、形状、轮廓、大小和屏幕位置等属性。

# Clip Skip
描述画面的准确程度与数值的大小成反比，数值越小表示对图像的控制度越高。最佳使用区间是1-2。

![alt text](assets_picture/base_knowledge/image.png)

感觉有点cfg控制的意思。      
但是其实是文本输入不准确     

类似lora的几个参数一起控制强度


# sampler
DPM adaptive是个例外，在高CFG Scale下能够稳定出图。

DPM++ SDE Karras采样器，风格独特，出图与其他采样器在相同参数下，完全不一样。

Euler、Heun、DPM2、DPM++ 2M、LMS Karras、DPM2 Karras、DPM++ 2M Karras，以及Euler a、DPM2 a、DPM adaptive*、DPM++ 2S a、DPM++ 2S a Karras在相同参数下，从整体布局，到人物动作及衣物细节等存在相同或者相似的情况。

Heun、DPM adaptive、LMS几种采样器，出图细节比较稳定，不随着迭代步数增加而发生细节变化。



# 宽高
1:1头像图，1:2手机屏幕，4:3文章配图，3:4社交媒体，16:9电脑壁纸，9:16宣传海报




# cn
## openpose
OpenPose_face 预处理器是 OpenPose 预处理器的基础上增加脸部关键点的检测与标识，效果如下所示：


![alt text](assets_picture/base_knowledge/image-1.png)


OpenPose_faceonly 预处理器仅检测脸部的关键点信息，如果我们想要固定脸部，改变其他部位的特征的话

![alt text](<assets_picture/base_knowledge/image (1).png>)



openpose_hand 预处理器能够识别图像中人物的整体骨架+手部关键点，

![alt text](<assets_picture/base_knowledge/image (2).png>)


openpose_full 预处理器能够识别图像中人物的整体骨架+脸部关键点+手部关键点

![alt text](assets_picture/base_knowledge/image-3.png)

dw_openpose_full 预处理器是目前 OpenPose 算法中最强的预处理器，其不但能够人物的整体骨架+脸部关键点+手部关键点，而且精细程度也比 openpose_full 预处理器更好，其效果如下图所示：

![alt text](assets_picture/base_knowledge/image-2.png)





## 边缘检测-Canny
，硬边缘检测。
Canny 的预处理可以对图片内的所有元素通过类似 PS 中的硬笔工具进行勾勒出轮廓和细节
再通过 Canny 模型作用于绘图中，让我们生成类似于原图轮廓和细节的图片。如下图示例：这里需要注意 canny 所提取的图片看起来很像是线稿，但其实和线稿完全不一样。


![alt text](assets_picture/base_knowledge/image-4.png)



invert (from white bg & black line)则是对白色背景黑色线条的图片进行反转


特殊参数   
canny中有两个特殊的参数最低阈值和最高阈值两个参数。它们决定了哪些边缘可以被视为强边缘，哪些可以被视为弱边缘，以及哪些应该被完全忽略。


下面是这两个参数的具体作用：   
1. 最低阈值: 这个参数用来决定哪些边缘是不重要的，可以直接被舍弃的。边缘强度小于最低阈值的部分会被认为是噪声或颜色变化不明显的区域，因此在边缘检测的最后结果中不予显示。100   
2. 最高阈值: 这个参数用来确定哪些边缘是显著的，应该被保留的。边缘强度高于最高阈值的区域会被当作真正的边缘。这部分通常代表了图像中亮度变化最大的地方，反映了明显的边界。200  

当使用双阈值时，对于边缘强度介于这两个阈值之间的边缘像素，如果它们与已标记的强边缘像素相邻，则被认为是弱边缘，可以保留；否则，为了降低误报，这些部分也会被舍弃。   

![alt text](assets_picture/base_knowledge/image-5.png)




## 轮廓识别-softedge      
Canny（硬边缘检测）预处理器和模型，但相对 Canny 我们发现其线条太过于严格，让 AI 绘画发挥的空间比较小。
如果我们需要让 AI 更多的发挥空间，让线条更宽松一些，这里我们就引入了 SoftEdge（软边缘）预处理及模型。
这是 softedge 的效果


边缘检测可以理解为用铅笔提取边缘，而轮廓识别则是用毛笔

![alt text](assets_picture/base_knowledge/image-6.png)

预处理质量 HED 最高也最常用：HED>PiDiNet>HEDSafe>PiDiNetSafe      
质量越高消耗显存越多，所以效率方面：PiDiNetSafe>HEDSafe>PiDiNet>HED


## 涂鸦上色-Scribble
Canny（硬边缘检测）和 SoftEdge (软边缘检测)线条类 ControlNet 模型，更多的区别在于对线条的宽松度，让 AI 发挥空间的大小，如果我们需要更宽松的空间让 AI 发挥，这里就引入了 Scribble 。
涂鸦上色能够提取图片中曝光对比度比较明显的区域，生成黑白稿，涂鸦成图，其比其他的轮廓控制更加自由，也可以用于对手绘线稿进行着色处理。
涂鸦上色非常适合进行创意绘画和儿童绘画，你只需要用画笔简单画一些图案交给 AI，它就可以生成精美得分图片。

预处理器
涂鸦上色的预处理器有四个：Scribble-hed, Scribble-pidinet, Scribble-xdog，t2ia_sketch_pidi

这里我们可以把这 4 种预处理器分为 2 类：   
第一类：HED、PiDiNet、Sketh_PiDi   
第二类：Xdog   
从下图中可以看出，第一类质量由大到小为：HED > Sketh_PiDi > PiDiNet，同时消耗显存会增加。

相比前两个预处理器控制程度较高。阈值越小精细程度就越高，阈值越高精细程度就越低。

## 线稿提取-lineart
线稿提取是专门提取线稿的功能，可以针对不同类型的图片进行不同的处理。     
lineart 是一个专门提取线稿的模型，可以针对不同类型的图片进行不同的处理，主要可以处理一下几种图片类型：动漫图片提取线稿，真实照片提取线稿，素描图片提取线稿，黑白线稿提取


1. lineart_coarse 是素描线稿预处理器，它比较适用于简单线条的场景，比如肖像与静物画，但对线条非常复杂的场景就很不适用，会出现非常多的错误。
2. lineart_anime 专为处理动漫风格的图像设计。通常动漫风格的线稿有清晰的轮廓线和特定的风格特征，这种预处理器会优化这些特定的线条特点。它的优点是，擅长处理浅色图片能够较为完整的保留图片的线稿；缺点则是对深色图片的线稿可能会发生错误，图片会保留许多干扰信息。
3. lineart_anime_denoise 是上述 lineart_anime 预处理器的一个变体，它不仅进行常规的优化，而且可包含去除图片噪声的功能。这对于提取从非专业的或低质量的源图像中提取线条特别有用。它的优点则是，可以处理深色图片通过去噪保留正确的线稿；缺点则是因为去噪所以可能会消除原图中许多的线段。
4. lineart_realistic 预处理器可能是用来处理更为真实主义风格的线稿，其中的线条可能要表现出不同的质感和深度，以及真实世界的细节和阴影。优点则是适用于现实生活中的图片，或者复杂场景的图片，缺点则是细节捕捉太多，可能造成不必要的麻烦。



## 直线检测-mlsd
MLSD 模型是一个专门检测直线的模型   
基于 MLSD 模型的特性，MLSD 模型在建筑、室内方向的处理上是比较好的选择   
可以很好的检测出原图的直线线条   
举个例子若原图室内环境中有人物出现，但是新生成的图片中不希望有人物，那么使用 MLSD 模型就可以很好的避开人物线条的检测。


看看预处理出来的图，都只有直线，有弧度的线条都会被忽略掉。



![alt text](assets_picture/base_knowledge/image-7.png)

其中，价值阈是对输入图片中直线进行打分，我们可以通过提高价值阈的数值来屏蔽一些混乱的直线。

![alt text](assets_picture/base_knowledge/image-8.png)


## 深度检测-depth

深度图也被称为距离影像，指的是图像采集器采集到图像中各个场景区域的距离，深度图会使用灰阶数值 0~255 组成图像，灰阶数值为 0 的区域表示图像中最远的区域，灰阶数值 255 表示图像中最近的区域，所以我们在深度图中可以看到不同灰度的区域组成的图像。
![alt text](assets_picture/base_knowledge/image-10.png)    
depth_leres   
depth_leres 预处理器的成像焦点在中间景深层，这样的好处是能有更远的景深，且中距离物品边缘成像会更清晰，但近景图像的边缘会比较模糊。
![alt text](assets_picture/base_knowledge/image-9.png)

`depth_leres++`预处理器在 depth_leres 预处理器的基础上做了优化，能够有更多的细节，但处理速度相对更慢。   
![alt text](assets_picture/base_knowledge/image-11.png)

depth_midas 预处理器是经典的深度估计器，也是最常用的深度估计器，处理速度最快。   
![alt text](assets_picture/base_knowledge/image-12.png)

depth_zoe 预处理器的参数量是最大的，所以处理速度比较慢，实际效果上更倾向于强化前后景深对比，更适合处理复杂场景。
![alt text](assets_picture/base_knowledge/image-13.png)

## 语义分割-segment

预处理器
预处理器：seg_ofcoco，seg_ofade20k，seg_ufade20k
预处理器功能：生成图片的物品分割图
模型功能：依据标注的分割图进行精准控图    
注意：推荐使用 of 开头的预处理器，coco 代表 coco 数据集标记方法，ade20k 也是一个标记方法



## 法线贴图-normal
NormalMap 算法根据图片生成一张记录凹凸纹理信息的法线贴图，通过提取输入图片中的 3D 物体的法线向量，以法线为参考绘制出一副新图，同时给图片内容进行更好的光影处理。


![alt text](assets_picture/base_knowledge/image-14.png)


法线贴图有 Bae 和 Midas 两种预处理器，默认 Bae 预处理器，Midas是比较早期的版本，一般不再使用。

## 参考生图-Reference 
Reference 模型是一种预处理器，可以根据导入的素材图片，参考图片的配色、色调、画风、画中的事物创建出新图片，使画中事物仍然和原图的相似性。这个模型在涂鸦或线稿生成等场景中应用较广，能产生与参考图风格类似但细节不同、元素多样的图片。

Reference算法可以直接将SD模型的注意力机制与图像进行映射，这样SD模型就可以将输入的图像作为参考（图像提示词）。   
参考生图一共有三种预处理器，Reference adain、Reference only、Reference adain+attn，一般选择reference only，模型选择为none

Reference 中有一个特殊参数：Style Fidelity (only for "Balanced" mode)
风格保真度，值越大生成的图片风格和参考图的差异就越小，反之亦然。    
Reference 出图更依赖于大模型，对于一些大模型，Style Fidelity 值太大，会导致崩图，所以我们出图需要更接近参考图的时，可以设置 0.5-0.9 之间，如果需要更多风格的改变，Style Fidelity 值调整为 0.5 以下。


reference_adain    
在具体使用过程中发现该预处理器保持原画面的相似性较差，主要是将参考图的风格/纹理迁移到结果图中，以生成模型为主，参考图为辅的形式进行。  
reference_only   
固定参考图特征，在这个基础之上进行发散，基本上保留原图的风格   
reference_adain+attn   
Reference_adain+attn 预处理器是 reference_adain 一种高级的预处理器   
它融合了 reference_adain 和 reference_only 两个算法，在具体使用过程中发现该预处理器保持原画面的相似性最好。  

![alt text](assets_picture/base_knowledge/image-15.png)

![alt text](assets_picture/base_knowledge/image-16.png)

魔法色子

![alt text](assets_picture/base_knowledge/image-17.png)



## 风格迁移-shuffle
Shuffle将图片进行随机打乱，经过训练可以重组图片，同时可以将其他图片的画风快速转移到自己的照片上。
注意这里风格转移和一些大模型无关，这是一个纯 ControlNet    
可以理解为，我将漫画图片的动漫风格转移到真实照片上，即使大模型选择的是真实模型而不是动漫模型，也不影响结果成为动漫风格。

## 局部修图-inpaint

inpaint_global_harmonious     
是整张图进行重绘，重绘之后整体融合比较好，但是重绘之后的整个图片都会有轻微的改变

inpaint_only    
只重绘涂绘区，其他地方不作任何改变

inpaint_only+lama    
这个预处理器是新增的一个预处理器，看起来与 inpaint_only 类似，但有点“干净”：更简单、更一致、随机对象更少。这使得 inpaint_only+lama 适合图像修复延展或对象移除

## 稳定像素-ip2p


给照片加特效，比如让房子变成冬天、让房子着火  
这里需要输入的关键词比较特殊   
需要在关键词里面输入：make it.... （让它变成...）
比如让大厦着火，就输入：make it fire


## 细节增强-tile
Tile 模型具备增补图片细节的能力。简单来说，它能将解析度较低的图片放大到高解析度，并在放大过程中保持图片的细节和清晰度，从而使放大后的图片看起来非常逼真。



Tile 模型的关键优势在于，它可以生成新的细节，并且忽略现有的图像细节。因此，你可以使用该模型删除不适合的细节，比如调整图像大小导致的模糊，并添加更精细的细节。

需要注意的是，Tile 模型并不是创建超高解析度图像的模型，而是专注于忽略图像中的细节并`生成新的细节`。这意味着`即使图像本身已有良好的细节，你仍然可以使用此模型进行内容替换或优化。`


如果你希望将图片放大到高解析度，你可以将 Tile 模型与 Tiled Upscaling 工具（如 Ultimate SD Upscale）配合使用。

在绘图过程中，Tile 模型会忽略全局提示词。换句话说，绘图的主要内容会依据图片内的物体为主，即使收到提示词的指示，只要图片里没有这样的物体，提示词就会失效。

预处理器   
tile有none，tile_resample，tile_colorfix，tile_colorfix+sharp   
不选择预处理器直接 tile

tile_resample   
tile_resample 的作用是缩小原图片，缩小后的图片更有利于模型产生更加多样化的结构。   
tile_resample 中有一个特殊参数：最低采样率

最低采样率是缩小倍率，数值越大图片被缩的越小
比如原图分辨率为 1024*1024

    数值为 1 时，图片大小不变，仍然为 1024*1024
    数值为 2 时，图片缩小 2 倍，变为 512*512
    数值为 4 时，图片缩小 4 倍，变为 256*256
理论上数值越大，生成的图像被补充的细节越多，和原图的差距越大，所以我们在修复图片细节的时候尽量不要把数值调太大，正常保持 1-2 即可。

tile_colorfix   
tile_colorfix 可以很好的保留原图片的色彩。

![alt text](assets_picture/base_knowledge/image-18.png)



变化可以控制区块颜色的变化，数值越大区块颜色变化越大，这里不宜将数值调太大，容易导致颜色不可控


tile_colorfix+sharp   
tile_colorfix+sharp 是 tile_colorfix 的升级版，可以消除 tile_colorfix 的模糊效果。   
tile_colorfix+sharp 预处理器其实就是对 tile_colorfix 预处理器的改良。改良的关键就是 sharp，翻译过来就是锐化，是防止 Tile 模型在生成图片时出现的模糊问题。

Sharpness：就是锐化度  
数值越大锐化程度越高，建议数值控制在 0.7~1 之间


blur_gaussian    
blur_gaussian 预处理器主要作用是向图像增加一个高斯模糊，然后将模糊后的图片交给 Tile 模型去添加更多的细节，这样我们生成的图片就带有景深了。



## 光影控制-brightness
光影控制与其他条件生图不同，在界面上并没有预处理器，只有两个模型选择，分别是：

control_v1p_sd15_brightness，control_v1p_sd15_illumination；


仿佛是iclight

![alt text](assets_picture/base_knowledge/image-19.png)

一般会用control_v1p_sd15_brightness，control_v1p_sd15_brightness，control_v1p_sd15_illumination的区别在于：


![alt text](assets_picture/base_knowledge/image-20.png)


光影控制的玩法多种多样，比如你可以用光影控制来将照片或者文字隐藏在图片中

当然还可以直接将你想要的光线放入到光影控制中进行图片的生成。

![alt text](assets_picture/base_knowledge/image-21.png)      

fbc

在使用光影控制中时，权重建议0.4-0.65，这里数值越大，光线图案就会越明显，但相对的，光线和图片的融合度也会越差。      
同时可以调整开始与结束控制步数，结束步数参数建议0.6-0.75，代表着条件生图什么时候停止介入，数值越大后面留给模型处理融合的时间就越少，融合度就会变差，数值越小模型介入过早就会破坏已有的结构导致光线无法出现。


## 二维码控制-qrcode
创意二维码设计主要解决营销活动二维码问题，可以制作出富有创意的营销二维码。
流程：
1. 二维码控制+光影控制，配合模型风格做创意二维码
2. 发送到条件生图，选择细节增强-底图


修改参数，权重：0.9-1.15
开始控制步数：0.2-0.4
结束控制步数：0.6-0.8


光影控制       
修改参数，权重：0.3-0.55
开始控制步数：0.3-0.4
结束控制步数：0.7-0.85

![alt text](assets_picture/base_knowledge/image-22.png)


fbc也可以

![alt text](assets_picture/base_knowledge/image-23.png)


![alt text](assets_picture/base_knowledge/image-24.png)

![alt text](assets_picture/base_knowledge/image-25.png)


## 元素融合-ip-adapter
IP Adapter 它的作用是将你输入的图像作为图像提示词，本质上就像 MJ 的垫图功能。   
这个功能与 Reference (参考生图)有些类似，但 IP Adapter 比 reference 的效果要好，而且会快很多，适配于各种 stable diffusion 模型，还能和 controlnet 一起用。

![alt text](assets_picture/base_knowledge/image-26.png)

![alt text](assets_picture/base_knowledge/image-27.png)

![alt text](assets_picture/base_knowledge/image-28.png)


![alt text](assets_picture/base_knowledge/image-29.png)


![alt text](assets_picture/base_knowledge/image-30.png)


有时候v2只是增加功能，并不能优化质量     
甚至有些只是加速，减少显存等     


# 蒙版
![alt text](assets_picture/base_knowledge/image-31.png)

## 蒙版模糊度
类似 PS 这种绘图工具中的“羽化”功能。
““羽化”（Feather）是指在选中区域的边缘添加一层透明的、渐变式的效果，使得选中区域的边缘更加柔和自然。 


蒙版模糊度越高，那么重绘区域和未被重绘的区域会形成一种丝滑的过度，如果我们将蒙版模糊拉到最低，那么就相当于没有设置羽化值，重绘区域和未被重绘的区域交界处会显得非常生硬。    
一般保持在10以下，根据区域的大小选择相应的数值，蒙版较大可以选择较大的数值，较小的区域则选择较小的数值。

![alt text](assets_picture/base_knowledge/image-32.png)


## 蒙版模式
inpaint masked是重绘蒙版内容，就是把蒙版中白色部分重新画一遍。   
就如下面这张图，我们选择inpaint masked模式，杯子将会发生重绘

inpaint not masked是重绘非蒙版内容，就是把蒙版中黑色部分重绘

## 蒙版内容
填充、原图、潜变量噪声、潜变量数值零等四个是通过四个不同的算法去进行重绘的。不过根据实际的操作来看，四个生成的图片区别不大，通常认为填充和原图会更稳定。

特别在一个大图中，人物脸部分辨率比较小，人物脸部崩掉的情况，可以通过仅蒙版模式来加大脸部的像素进行高清修复，可以说又是一个高清修复的利器！


# 分块绘图

【分块绘图（Multi Diffusion + Tiled VAE）】图生图、条件生图支持使用分块绘图的功能

备注：需关闭图像精绘模型，采样器建议使用DDIM，可适当提高CFG参数


分块宽度与高度的调整即：将所生成的图片切割的尺寸。数值越高切割越大，生成速度越快。数值越低切割越小，生成速度越慢。    
采样模式：二次元建议使用R-ESRGAN 4x+ Anime6B，真人建议使用R-ESRGAN 4x+ 

如果需要AI生成更多的细节，可不开启画面稳定选项

![alt text](assets_picture/base_knowledge/image-33.png)

小图看上去没什么区别   
只是在放大的时候能看出模糊与否

飞书的500倍放大，看来可以把分块结果再扩大十倍

# 多区域控制
多区域绘制，支持使用融合模型生成图像，扩展用法支持多人多融合模型。    
![alt text](assets_picture/base_knowledge/image-35.png)    
![alt text](assets_picture/base_knowledge/image-36.png)    
![alt text](assets_picture/base_knowledge/image-34.png)

# 插件
分别为脸部修复，无损放大，表情修复，手部修复与超分辨。


# 工作流
第一步：选择分割一切制作模版   
第二步：选择法线贴图制作特效    
![alt text](assets_picture/base_knowledge/image-37.png)

![alt text](assets_picture/base_knowledge/image-38.png)

![alt text](assets_picture/base_knowledge/image-39.png)

![alt text](assets_picture/base_knowledge/image-40.png)




# 图生视频
1.生成时长可以选择2秒与4秒，直接图生视频，并不需要输入描述词。   
2.运动幅度可以进行调整，范围在1-255，数值越大，运动幅度越大   
3 .
噪点强度影响着添加到输入图像的噪声量，较高的噪声会降低视频与输入图像的相似度，提高数值会产生更大
的运动效果。


# 透明图  
生成透明图像   
透明图功能可以轻松生成单个透明图像或者多个透明图层，且支持前后景的图片融合和图层拆分

![alt text](assets_picture/base_knowledge/1716012293552.png)

![alt text](assets_picture/base_knowledge/1716012342620.png)

不仅能生成透明玻璃杯，就连“凌乱的发丝”这样平时很难抠图的复杂图像也能完美生成。这几乎为设计工作者提供了无限的商用素材，同时还省去了一些P图的时间。


在专业版使用“透明图”时，有多种模式可供选择。其中"Only Generate Transparent lmage (AttentionInjection)"和"Only Generate Transparentlmage(Conv Injection)"都是直接生成透明图像的，两者功能-样，只是处理方式有所不同。    
"From Foreground to Blending”可以设置前景，固定前景的图像不变，生成背景后和前景融合成一张新的图片。

![alt text](assets_picture/base_knowledge/1716012502804.png)


# 训练素材获取
## 人物素材    
训练人物lora时，需要同一个人物的不同角度的素材至少20-30张。     
确实字节的星绘强一些，人脸相似度也很高     
而且只是引流到抖音

(1)制作动漫同人lora时，可寻找某个动漫角色的素材，图片素材尽量高清无遮挡。    
(2)制作真人lora时，可寻找某个人的照片素材，图片素材尽量高清无遮挡无曝光。注意：训练真人lora时，避免纠纷，素材图片需寻找不侵权的照片，如：公开自己形象授权的人物、明星等。    
(3)制作AI人lora时，可先使用无界的人脸lora搭配真人模型出图，使用人脸lora可以保持AI人的脸保持一致。图片素材尽量不要有缺陷。      


## 风格素材
训练画风、风格lora时，需要同一种风格的素材至少70-100张。
风格虽无版权与侵权的说法，但是为了避免纠纷，建议使用国外或作者声明开放免费使用的素材训练风格。


## 点击增加标签
首先需要熟悉并观察上传的素材图片，找到素材中同一特征，并将其作为标签。标签的可以起到特征隔离、增加模型泛化性等作用。   
如：一组素材中， 同一个角色身着不同颜色的衣服。可以通过增加标签，将不同颜色衣服隔离分类。增加“黑色西装”数据标签，将符合黑色西装特征的图片增加至标签下，增加“红色卫衣”标签，将符合红色卫衣特征的图片增加至标签下，增加“白色衬衫”标签，将符合白色衬衫特征的图片增加至标签下。以此类推，可增加“黑色肌肤”、“白色长发”、“黑色束发”等等特征词并增加数据。这些标签训练完成后可通过输入提示词触发模型效果。标签支持中文标注。

分类模型


![alt text](assets_picture/base_knowledge/image-41.png)

选择训练尺寸（选填）   
可选择512*512或768*768，尺寸越大训练时间越长，效果越好   
4.选择训练的基础模型（选填）   
当前支持SD1.5的训练，后续更新会开放支持SDXL的训练

参数设置   
1.学习率（可不调整）    
参数影响：学习率越高，学习速度越快，但是有可能导致矫枉过正，对于素材训练学习失败。学习率低可以让AI更细致的学习，但过低的学习率可能导致不拟合，与素材图片不相似，并且消耗更长的时间。（建议新手使用默认参数）   
2*10-6

2.训练轮数（可不调整）    
训练轮数的含义为：数据集内所有素材学习一遍为一轮，20轮即是训练集中的素材图片每一张训练20遍。   
理论上，训练轮数越高，AI就能更好的读懂与训练素材。   
但是在实际训练过程中，训练轮数过高会导致过拟合，让AI对图片的认知固化，失去泛化性能力。   


![alt text](assets_picture/base_knowledge/image-42.png)



1.数据重复次数（可不调整）   
数据重复次数的含义为：重复训练轮数。如：上方设置训练20轮，数据重复次数为5，即是20轮重复5次，最终素材训练为100轮。


2.Unet LR（可不调整）  
默认即可   
5*10-5



3.Clip skip（可不调整）    
Clip skip（跳过层）指的是控制图像生成过程中CLIP模型的使用频率的参数，它影响了图像生成的过程中
使用的CLIP模型的次数。

Clip skip的取值范围是1到12   
值越小，生成的图像就越接近原始图像或输入图像 。   
值越大，生成的图像就越偏离原始图像或输入图像，甚至可能出现黑屏或无关的人物 。  
一般设为2，出图效果最佳。    

这个应该是收到CLIP训练损失设置的影响，相当于对vgg,resnet特征提取能力的调用      

4.Network Dim（可不调整）   
Dim表示神经网络的维度，维度越大，模型的生成表现能力越强，模型的体积越大

Network Dim并不是越高越好，4个参数为四个维度，维度提升有助于学习更多的细节，但是模型训练收敛速度会变慢，需要更长的训练时间，更容易过拟合。

训练集素材的分辨率越高，需要的Dim维度越高


Dim维度与LoRa模型输出文件大小有关联    
Network Dim 128，训练的lora大小约140MB+   
Network Dim 64，训练的lora大小约70MB+   
Network Dim 32，训练的lora大小约40MB+   
Network Dim 16，训练的lora大小约20MB+   
建议：   
训练二次元时，Network Dim维度选择32或64   
训练人物时，Network Dim维度选择32或64或128   
训练实物、风景时，Network Dim维度选择128    


5.Network Alpha（可不调整）   
Alpha的数值小于或等于Dim，一般设置与Dim相同或者是Dim的一半   
如：Dim设置128，Alpha可设置128或64

![alt text](assets_picture/base_knowledge/1716013644570.png)

![alt text](assets_picture/base_knowledge/1716013680600.png)

![alt text](assets_picture/base_knowledge/1716013720263.png)

使用分层控制的效果会更加接近于动画电影模型的效果，未使用

![alt text](assets_picture/base_knowledge/image-43.png)

![alt text](assets_picture/base_knowledge/image-44.png)

# 融合模型 Lora
![alt text](assets_picture/base_knowledge/1716013872662.png)


![alt text](assets_picture/base_knowledge/1716013912597.png)


![alt text](assets_picture/base_knowledge/1716013940016.png)

![alt text](assets_picture/base_knowledge/1716013968962.png)


![alt text](assets_picture/base_knowledge/1716014017184.png)


![alt text](assets_picture/base_knowledge/1716014042346.png)






# 结尾