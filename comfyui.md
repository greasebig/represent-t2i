# 安装自定义节点
## manager
## 手动
git clone repo 到 ComfyUI/custom_nodes/    
其他不用改    
模型那些遵照readme即可   

特殊的，如果comfyui官方未支持节点的一些底层推理    
需要在comfyui根路径git apply patch      


# comfyui特有denoise
当您以 1.0 运行 Ksampler 时，它会完全模糊任何传入的图像或噪声，然后按照给定的步骤数对其进行处理。

如果您以 0.6 运行 ksampler，它会模糊 60% 的强度，并根据给定的步数对其进行降噪。

两者之间的区别在于，100% 时它使用的是原始噪声或图像的极小部分。

而在 60% 时，它使用了大部分原始图像的颜色、明暗信息。








# 结尾




