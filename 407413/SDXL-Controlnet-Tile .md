TTPLanet_SDXL_Controlnet_Tile_Realistic

TODO:   
对比两个模型测效果    
七八张图    

https://civitai.com/models/330313?modelVersionId=370104   

以下是 Tile V2 更新说明的精炼版本：

- 引入新的 Tile V2，通过大幅改进的训练数据集和更广泛的训练步骤进行增强。

- Tile V2 现在可以自动识别更广泛的对象，无需明确提示。

- 强大的文本重组能力，可以通过风格转移过程保留最清晰的文本。

- 我对色彩偏移问题进行了重大改进。如果您仍然看到明显的偏移，这是正常的，只需添加提示或使用颜色修复节点即可。

- 控制强度更加鲁棒，在某些情况下可以替代canny+openpose。




# 工作流
comfyui
## style change
3D粗糙人物转绘。提供一张3D粗糙人物图片，根据prompt生成高质量真实图片
使用模型：底模 + tile

## Ultimate SD Upscale
使用模型：底模 + tile + upscale

经过Ultimate SD Upscale ，扩大两倍同时ctrlnet细化     
最后还使用stableSR的小波变换（没用模型）进行颜色匹配   

    Requested to load SDXLClipModel
    Loading 1 new model
    Canva size: 2184x1952
    Image size: 1089x976
    Scale factor: 3
    Upscaling iteration 1 with scale factor 3
    Tile size: 1024x1024
    Tiles amount: 6
    Grid: 2x3
    Redraw enabled: True
    Seams fix mode: NONE
    Requested to load AutoencoderKL
    Loading 1 new model
    Requested to load SDXL
    Requested to load ControlNet
    Loading 2 new models


    Canva size: 2184x1704
    Image size: 546x426
    Scale factor: 4
    Upscaling iteration 1 with scale factor 4
    Tile size: 1024x1024
    Tiles amount: 6
    Grid: 2x3
    Redraw enabled: True
    Seams fix mode: NONE



