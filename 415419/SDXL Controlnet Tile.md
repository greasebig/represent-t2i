# v2 信息

最新更新2024/4/13：

以下是 Tile V2 更新说明的精炼版本：

-引入新的 Tile V2，通过大幅改进的训练数据集和更广泛的训练步骤进行增强。

-Tile V2 现在可以自动识别更广泛的对象，无需明确提示。

- 强大的文本重组能力，可以通过风格转移过程保留最清晰的文本。

-我对色彩偏移问题进行了重大改进。如果您仍然看到明显的偏移，这是正常的，只需添加提示或使用颜色修复节点即可。

-控制强度更加鲁棒，在某些情况下可以替代canny+openpose。



如果您遇到 t2i 或 i2i（尤其是 i2i）的边缘光晕问题，请确保预处理为 controlnet 图像提供足够的模糊效果。如果输出太锐利，可能会产生“光晕”——边缘周围具有高对比度的明显形状。在这种情况下，请在将其发送到控制网之前应用一些模糊。如果输出太模糊，这可能是由于预处理过程中过度模糊，或者原始图片可能太小。

享受 Tile V2 的增强功能！



# 历史版本信息
v1 在3.2放出     
v2 在4.13放出   


这是一个基于 SDXL 的 controlnet Tile 模型，使用 Huggingface 扩散器集进行训练，适合稳定扩散 SDXL controlnet。

它是为我自己的真实模型进行原始训练的，用于终极高档处理以增强图片细节。通过适当的工作流程，它可以为高细节、高分辨率图像修复提供良好的结果。

由于最开源的版本中没有可用的 SDXL Tile，我决定分享这个。


# 使用
风格变更应用说明和高档简单工作流程的更新：

更新 comfyui 的样式更改工作流程：

https://openart.ai/workflows/gJQkI6ttORrWCPAiTaVO

Part1 样式和背景更改应用程序：     
Part1 for style and background change application:   


打开 A1111 WebUI。

选择要用于 controlnet 磁贴的图像

记住设置是这样的，make 100% preprocessor 是 none。而控制方式是我的提示更重要。

在正负文本框中输入提示，生成您想要的图像。如果你想改变布料，像穿着黄色T恤的女人一样打字，并像在购物中心一样改变背景，

支持雇用修复！


## 用法二
Part2 for ultimate sd upscale application     
终极 SD 高端应用的第 2 部分

这是终极升级的简化工作流程，您可以根据实际情况对图像进行修改和添加预处理。就我而言，我通常会对真实的低质量图像（例如 600*400 到 1200*800）以 0.1 降噪率制作图像到图像，然后再将其进入最终的高档过程。

如果您需要相同的人脸，请添加 IPA 处理，对于低质量图像 i2i，请在原始预处理中添加 IPA。请记住，提高低分辨率图像的质量始终是提高低分辨率图像质量的最佳方法。

https://civitai.com/models/333060/simplified-workflow-for-ultimate-sd-upscale

# 基本作者信息
开发者： TT星球

模型类型： Controlnet Tile


Important: Tile model is not a upscale model!!! 

重要提示：瓷砖模型不是高档模型！！！它增强或改变原始尺寸图像的细节，在使用它之前记住这一点！

该模型不会显着改变基础模型的风格。它只是将功能添加到放大的像素块......

--Just use a regular controlnet model in Webui by select as tile model and use tile_resample for Ultimate Upscale script.

--只需在 Webui 中使用常规 controlnet 模型，选择作为图块模型，并使用tile_resample 作为 Ultimate Upscale 脚本。

--只需使用comfyui中的负载controlnet模型并应用于控制网络情况。

--如果您尝试在webui t2i中使用它，需要正确的提示设置，否则它将显着修改原始图像颜色。我不知道原因，因为我并没有真正使用这个功能。


——它对于数据集中的图像确实表现得更好。然而，对于 i2i 型号来说，一切都很好，通常终极高档应用在什么地方！

--另请注意，这是一个现实的训练集，因此不承诺漫画、动画应用。

--对于高档瓷砖，将降噪设置在 0.3-0.4 左右以获得良好的效果。

--对于controlnet强度，设置为0.9会是更好的选择

--对于人体图像修复，IPA 和 controlnet 上的提前停止将提供更好的结果

--挑选一个好的逼真基础模型很重要！


realistic base model

模糊恢复：

换衣服但保持姿势和人物：

除了基本功能外，Tile还可以根据您的模型更改图片风格，请将预处理器选择为“无”（不重新采样！！！）您可以通过良好的控制从一张图片构建不同的风格！


建议    
使用 comfyui 构建自己的 Upscale 流程，效果很好！

特别感谢Controlnet构建者lllyasviel张吕敏（LyuminZhang）给我们带来了这么多的乐趣，也感谢huggingface制作的训练集让训练如此顺利。


# 效果 
比肩 Magnific.ai



# 实践
## upscale


    When loading the graph, the following node types were not found: 

        Fast Groups Bypasser (rgthree) (In group node 'workflow/control')
        workflow/control

    CR Image Input Switch (In group node 'workflow/control')
    Cfg Literal (In group node 'workflow/首次采样缩放控制')
    workflow/首次采样缩放控制

        UltimateSDUpscale
        StableSRColorFix
        GetImageSize
        Image Comparer (rgthree)
        Float
        Gemini_API_S_Zho
        DisplayText_Zho
        Simple String Combine (WLSH)
        Text box

    Nodes that have failed to load will show as red on the graph.



## change style

    When loading the graph, the following node types were not found: 

        Text box
        SDXLPromptStyler
        ACN_AdvancedControlNetApply
        ControlNetLoaderAdvanced
        JWImageResizeByLongerSide
        easy imageSize
        ScaledSoftControlNetWeights

    Nodes that have failed to load will show as red on the graph.














# 结尾