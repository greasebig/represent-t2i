Flashface

flashface-sd1.5模型本身的质量不好，生成的图片质量低    
相似度也没有比instantID高


本文针对以前的zero-shot工作提出了几个问题
● 过去的工作难以保持面部的形状和细节，这主要是因为在将参考面孔编码成一个或几个标记时，空间表征可能会丢失。
● 其次，在实现精确的语言控制方面面临挑战，例如在使用年轻参考图像生成老年人面部时，往往不能很好地遵循语言提示。
● 这些问题的原因在于面部标记和文本标记被平等对待并集成到U-Net的同一位置，导致控制信号纠缠。
● 此外，数据构建流程中的目标图像裁剪也使得模型倾向于复制参考图像而非遵循语言提示。
总结就是，部分细节丢失、对于提示词的控制不够准确

(a) 基于embedding的方法   （b）FlashFace

![alt text](assets/flashface/image.png)

方法的新颖之处在于：
● 将面部编码为一系列特征图而不是多个token  （以保留更精细的细节）
● 使用单独的参考层和文本控制层进行解耦集成  （增强文本控制能力，缺点是导致模型过大~10GB)


![alt text](assets/flashface/image-1.png)



