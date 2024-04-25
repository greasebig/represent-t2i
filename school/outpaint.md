# outpaint

![alt text](assets/README/291712641102_.pic.jpg)   
![alt text](assets/README/281712641096_.pic.jpg)    
prompt: explosion, white, black    


一般来说全参调，或者lora,或者controlnet   
去网上，社区找开源模型    
2.controlnet 局部重绘 inpaint_only_lama control_v11p_sd15_inpaint    
powerpaint   
等   

涂鸦 inpaint都可以视作扩图方法 都做mask

## glid
https://github.com/Jack000/glid-3-xl-stable    
https://huggingface.co/Jack000/glid-3-xl-stable/tree/main/default   
运行训练有些问题   
可以推理   
![alt text](assets/README/271712641089_.pic.jpg)

拆开ckpt:   

    # split checkpoint
    python split.py sd-v1-4.ckpt

    # you should now have diffusion.pt and kl.pt

    # alternatively
    wget -O diffusion.pt https://huggingface.co/Jack000/glid-3-xl-stable/resolve/main/default/diffusion-1.4.pt
    wget -O kl.pt https://huggingface.co/Jack000/glid-3-xl-stable/resolve/main/default/kl-1.4.pt

kl模型暂不知用意是什么，拆开vae? 只在训练时候输入模型路径，推理不用？？？    





### 代码

    elif args.outpaint == 'left':
        input_image = torch.zeros(1, 4, im.shape[2], im.shape[3]+32, device=device)
        input_image[:,:,:,32:32+im.shape[3]] = im
        input_image_mask = torch.zeros(1, 1, im.shape[2], im.shape[3]+32, device=device, dtype=torch.bool)
        input_image_mask[:,:,:,32:32+im.shape[3]] = True
1.创建一个与输入图像 im 相同大小的张量，但是宽度增加了32个像素的空白区域。这个张量是用零填充的，表示黑色背景。        
2.将原始图像 im 复制到这个新创建的张量中，复制到新张量的右侧，即向左填充32个像素的空白区域。      
3.创建一个与输入图像 im 相同大小的布尔型张量，同样是宽度增加了32个像素的空白区域，用于表示图像的掩码。       
4.将掩码的相应区域设置为 True，表示在这个区域内需要进行处理。     

这段代码的作用是在输入图像的左侧添加一个32像素宽的空白区域，同时为该区域生成一个掩码，用于后续的图像处理。     


```
from PIL import Image
import os

def resize_images(input_folder, output_folder_resize, output_folder_crop):
    # 创建保存文件夹
    os.makedirs(output_folder_resize, exist_ok=True)
    os.makedirs(output_folder_crop, exist_ok=True)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 确保是图片文件
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            # 打开图片
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)

            # Resize图片为512x512
            resized_img = img.resize((512, 512))

            # 保存resize后的图片到文件夹1
            output_path_resize = os.path.join(output_folder_resize, filename)
            resized_img.save(output_path_resize)

            # 切割图片
            width, height = resized_img.size
            left = width // 2
            top = 0
            right = width
            bottom = height
            cropped_img = resized_img.crop((left, top, right, bottom))

            # 保存切割后的图片到文件夹2
            output_path_crop = os.path.join(output_folder_crop, filename)
            cropped_img.save(output_path_crop)

    print("任务完成！")

# 设置输入文件夹和输出文件夹
input_folder = "input_folder_path"
output_folder_resize = "output_folder1_path"
output_folder_crop = "output_folder2_path"

# 调用函数
resize_images(input_folder, output_folder_resize, output_folder_crop)



```

178张

```
from PIL import Image
import os
def resize_images(input_folder, output_folder_crop):
    # 创建保存文件夹
    os.makedirs(output_folder_crop, exist_ok=True)
    i=0
    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 确保是图片文件
        i += 1
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            # 打开图片
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)


            # 切割图片
            width, height = img.size
            left = width // 2
            top = 0
            right = width
            bottom = height
            cropped_img = img.crop((left, top, right, bottom))

            # 保存切割后的图片到文件夹2
            #output_path_crop = os.path.join(output_folder_crop, filename)
            #cropped_img.save(output_path_crop)

    print(f"{i} 任务完成！")

```



    else:
        input_image_pil = Image.open(fetch(args.edit)).convert('RGB')

        im = transforms.ToTensor()(input_image_pil).unsqueeze(0).to(device)
        im = 2*im-1
        im = ldm.encode(im).sample()
        有些奇怪 
取一半outpaint，感觉不正确，因为图原本已经裁一半了    
但是生出的还是挺大的    
32 改成 im.shape[3]//2    

在latant空间加减像素吗？？？    
另外该程序最后还不是resize。是填充灰色    
![alt text](assets/outpaint/image-2.png)     






### 报错
    Traceback (most recent call last):
    File "/data/lujunda/sd/glid-3-xl-stable-master/sample.py", line 32, in <module>
        from transformers import CLIPTokenizer, CLIPTextModel
    File "/home/lujunda/.conda/envs/glid-sd/lib/python3.9/site-packages/transformers/__init__.py", line 43, in <module>
        from . import dependency_versions_check
    File "/home/lujunda/.conda/envs/glid-sd/lib/python3.9/site-packages/transformers/dependency_versions_check.py", line 41, in <module>
        require_version_core(deps[pkg])
    File "/home/lujunda/.conda/envs/glid-sd/lib/python3.9/site-packages/transformers/utils/versions.py", line 94, in require_version_core
        return require_version(requirement, hint)
    File "/home/lujunda/.conda/envs/glid-sd/lib/python3.9/site-packages/transformers/utils/versions.py", line 85, in require_version
        if want_ver is not None and not ops[op](version.parse(got_ver), version.parse(want_ver)):
    File "/home/lujunda/.conda/envs/glid-sd/lib/python3.9/site-packages/packaging/version.py", line 54, in parse
        return Version(version)
    File "/home/lujunda/.conda/envs/glid-sd/lib/python3.9/site-packages/packaging/version.py", line 200, in __init__
        raise InvalidVersion(f"Invalid version: '{version}'")
    packaging.version.InvalidVersion: Invalid version: '0.10.1,<0.11'

需要升级transformers   
numpy也是和torch版本一一对应      

    Traceback (most recent call last):
    File "/data/lujunda/sd/glid-3-xl-stable-master/sample.py", line 252, in <module>
        ldm = instantiate_from_config(kl_config.model)
    File "/data/lujunda/sd/glid-3-xl-stable-master/latent-diffusion/ldm/util.py", line 85, in instantiate_from_config
        return get_obj_from_str(config["target"])(**config.get("params", dict()))
    File "/data/lujunda/sd/glid-3-xl-stable-master/latent-diffusion/ldm/util.py", line 93, in get_obj_from_str
        return getattr(importlib.import_module(module, package=None), cls)
    File "/home/lujunda/.conda/envs/glid-sd/lib/python3.9/importlib/__init__.py", line 127, in import_module
        return _bootstrap._gcd_import(name[level:], package, level)
    File "<frozen importlib._bootstrap>", line 1030, in _gcd_import
    File "<frozen importlib._bootstrap>", line 1007, in _find_and_load
    File "<frozen importlib._bootstrap>", line 986, in _find_and_load_unlocked
    File "<frozen importlib._bootstrap>", line 680, in _load_unlocked
    File "<frozen importlib._bootstrap_external>", line 850, in exec_module
    File "<frozen importlib._bootstrap>", line 228, in _call_with_frames_removed
    File "/data/lujunda/sd/glid-3-xl-stable-master/latent-diffusion/ldm/models/autoencoder.py", line 2, in <module>
        import pytorch_lightning as pl
    File "/home/lujunda/.conda/envs/glid-sd/lib/python3.9/site-packages/pytorch_lightning/__init__.py", line 20, in <module>
        from pytorch_lightning import metrics  # noqa: E402
    File "/home/lujunda/.conda/envs/glid-sd/lib/python3.9/site-packages/pytorch_lightning/metrics/__init__.py", line 15, in <module>
        from pytorch_lightning.metrics.classification import (  # noqa: F401
    File "/home/lujunda/.conda/envs/glid-sd/lib/python3.9/site-packages/pytorch_lightning/metrics/classification/__init__.py", line 14, in <module>
        from pytorch_lightning.metrics.classification.accuracy import Accuracy  # noqa: F401
    File "/home/lujunda/.conda/envs/glid-sd/lib/python3.9/site-packages/pytorch_lightning/metrics/classification/accuracy.py", line 18, in <module>
        from pytorch_lightning.metrics.utils import deprecated_metrics, void
    File "/home/lujunda/.conda/envs/glid-sd/lib/python3.9/site-packages/pytorch_lightning/metrics/utils.py", line 22, in <module>
        from torchmetrics.utilities.data import get_num_classes as _get_num_classes
    ImportError: cannot import name 'get_num_classes' from 'torchmetrics.utilities.data' (/home/lujunda/.conda/envs/glid-sd/lib/python3.9/site-packages/torchmetrics/utilities/data.py)


pip install pytorch-lightning  --upgrade
要使用torch2.2与nvcc不匹配

使用nvcc 11.3 torch1.12    
pip --no-cache-dir install pytorch-lightning==2.1.0    


    This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.


pip3 install opencv-python==4.1.2.30     
找不到

For me, it worked by using a opencv-python version prior to 4.2 version that just got released. 

pip3 install opencv-python==3.4.18.65   
找不到别的

    qt.qpa.xcb: could not connect to display 
    qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
    This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

    Available platform plugins are: eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, wayland-egl, wayland, wayland-xcomposite-egl, wayland-xcomposite-glx, webgl, xcb.

    已放弃 (核心已转储)

无法用 学校a100可能需要   
For Ubuntu users,

sudo apt-get install qt5-default fixes the issue.

(I'm using OpenCV 4.4)

### 2080
还是已放弃 (核心已转储)   
安装 sudo apt-get install qt5-default    
找不到      

sudo apt-get install qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools qtcreator

无法发起与 mirrors.tuna.tsinghua.edu.cn:80 (

![alt text](assets_picture/outpaint/1714059930951.png)    
过于麻烦，后续再看






## stable-diffusion-infinity-xl
装环境可以，运行app.py报错：   

    (sd-inf) root@q1yOYo:/private/lujunda/stable-diffusion-infinity-xl-main# python app.py
    patch_match compiling failed, will fall back to edge_pad
    [Taichi] version 1.7.0, llvm 15.0.4, commit 2fd24490, linux, python 3.10.14
    Found 1 CUDA devices
    Device 0: NVIDIA GeForce RTX 3090
    SMs: 82
    Global mem: 24260 MB
    CUDA Cap: 8.6
    [PIE]Successfully initialize PIE grid solver with cuda backend
    Traceback (most recent call last):
    File "/private/lujunda/stable-diffusion-infinity-xl-main/app.py", line 1148, in
    setup_button.click(
    TypeError: EventListener._setup..event_trigger() got an unexpected keyword argument '_js'
作者已经不维护   


### webui inpaint script
https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#inpainting-model-sd2     
https://github.com/runwayml/stable-diffusion#inpainting-with-stable-diffusion


v1.5    
#### 1. 直接resize大小，往左右两边扩展       
但这不是需求  

#### 2. poor man's outpainting   
可以选择方向    
![alt text](assets/README/431712639349_.pic-1.jpg)    
![alt text](assets/README/631712642102_.pic.jpg)    
![alt text](assets/README/image.png)    
![alt text](assets/README/image-1.png)   

#### 3. outpainting mk2   
![alt text](assets/README/image-2.png)    
![alt text](assets/README/image-3.png)    
![alt text](assets/README/image-4.png)     
![alt text](assets/README/image-5.png)   
参数比较难调   

采用专门对inpaint优化的模型   
sd2.1基准    
https://huggingface.co/webui/stable-diffusion-2-inpainting/tree/main    
![alt text](assets/outpaint/image.png)

controlnet：   
https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint   
sd1.5基准   
![alt text](assets/outpaint/image-1.png)    
ControlNet插件inpaint局部重绘模型对于接缝处的处理 确实比图生图自带的局部重绘功能处理的要好太多了。     
https://zhuanlan.zhihu.com/p/633750880?utm_id=0    


### webui插件支持outpaint


比较难以安装    
相比于comfyui   

直接使用inpaint    
![alt text](assets/README/431712639349_.pic.jpg)   
![alt text](assets/README/581712640260_.pic.jpg)   


装插件    
1. masoic    
16步，比较模糊   
原理扩展加masoic然后又有另一张mask图片，通过这些去做inpaint   
我的理解是输入前处理latent，生图。获取的结果通过mask过滤   
可以选择方向，功能齐全，效果略差    
可以使用controlnet   
后期也许可以考虑叠加lora，   
![alt text](assets/README/241712589773_.pic.jpg)    
![alt text](assets/README/251712589784_.pic.jpg)   
![alt text](assets/README/261712589802_.pic.jpg)   






2. 另一个是infinite zoom   
介绍是生视频的，生出五张图，没有方向控制    
https://youtube.com/shorts/Erju6TzEAEM?feature=share   



3. 另一个是画板形式插件，类似stable-diffusion-infinity-xl   
但是还不了解如何作画，使用    










