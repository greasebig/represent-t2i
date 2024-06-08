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

# glid



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





## 模型信息
GLID-3-xl-stable is stable diffusion back-ported to the OpenAI `guided diffusion` codebase, for easier development and training.

Commits on Aug 21, 2022     



## 代码

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






## 报错
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

## 2080
还是已放弃 (核心已转储)   
安装 sudo apt-get install qt5-default    
找不到      

sudo apt-get install qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools qtcreator

无法发起与 mirrors.tuna.tsinghua.edu.cn:80 (

![alt text](assets_picture/outpaint/1714059930951.png)    
过于麻烦，后续再看




## 顺利安装
只需要矩池云就可以顺利安装所有的环境       

推理运行时：       
ImportError: cannot import name 'masks_to_boxes' from 'torchvision.ops' (/root/miniconda3/envs/ldm/lib/python3.8/site-packages/torchvision/ops/__init__.py)       

torch2.3   torchvision 0.8.1   

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

    File "sample.py", line 32, in <module>
        from transformers import CLIPTokenizer, CLIPTextModel
    File "/root/miniconda3/envs/ldm/lib/python3.8/site-packages/transformers/__init__.py", line 43, in <module>
    from . import dependency_versions_check

    File "/root/miniconda3/envs/ldm/lib/python3.8/site-packages/packaging/version.py", line 200, in __init__
        raise InvalidVersion(f"Invalid version: '{version}'")
    packaging.version.InvalidVersion: Invalid version: '0.10.1,<0.11'



    pip show packaging
    Name: packaging
    Version: 24.0


报错原因

    packaging.version版本过高

    解决方向
    降低packaging版本；
    先尝试降低2个版本：

    pip install packaging==21.3 -i https://pypi.tuna.tsinghua.edu.cn/simple/
    1
    问题解决结果（成功解决）

接下来


    Traceback (most recent call last):
    File "sample.py", line 32, in <module>
        from transformers import CLIPTokenizer, CLIPTextModel
    ImportError: cannot import name 'CLIPTokenizer' from 'transformers' (unknown location)


    pip show transformers
    Name: transformers
    Version: 4.3.1

    pip install transformers==4.8.0


终于版本检查通过，可以运行进去了        


ImportError:      
cannot import name 'get_num_classes' from 'torchmetrics.utilities.data' (/root/miniconda3/envs/ldm/lib/python3.8/site-packages/torchmetrics/utilities/data.py)

pip --no-cache-dir install pytorch-lightning==2.1.0    

    Attempting uninstall: pytorch-lightning
        Found existing installation: pytorch-lightning 1.4.2
        Uninstalling pytorch-lightning-1.4.2:
        Successfully uninstalled pytorch-lightning-1.4.2
    Successfully installed pytorch-lightning-2.1.0

还是这个    

    making attention of type 'vanilla' with 512 in_channels
    draw the area for inpainting, then close the window
    qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
    This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

    Available platform plugins are: eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, wayland-egl, wayland, wayland-xcomposite-egl, wayland-xcomposite-glx, webgl, xcb.


pip install opencv-python-headless

sudo apt-get install qt5-default

解决不了，这个版本已经对了 opencv-python             4.1.2.30           

pip install --upgrade pyqt5_tools

    Attempting uninstall: pyqt5
        Found existing installation: PyQt5 5.15.10
        Uninstalling PyQt5-5.15.10:
        Successfully uninstalled PyQt5-5.15.10
    Successfully installed pyqt5-5.15.9 pyqt5-plugins-5.15.9.2.3 pyqt5-tools-5.15.9.3.3 python-dotenv-1.0.1 qt5-applications-5.15.2.2.3 qt5-tools-5.15.2.1.3

单单是运行就已经报错改了三次还不行        
启动还很慢      


    Version:0.9 StartHTML:0000000170 EndHTML:0000004125 StartFragment:0000000206 EndFragment:0000004089 SourceURL:https://github.com/NVlabs/instant-ngp/discussions/300 I was trying hard, but could not fix it. I was downloading literally hundreds of different packages, over and over again, trying all sorts of "hacks" found online, eventually trying anaconda, a massive download...
    In the end, I don't know if that was the fix, because my system is about to be reinstalled after all of my "trying", BUT, after I wrote this in console:
    export QT_QPA_PLATFORM=offscreen

## 解决关键 export QT_QPA_PLATFORM=offscreen



第四次好像成功了，但是linux没有图形界面

    Working with z of shape (1, 4, 32, 32) = 4096 dimensions.
    making attention of type 'vanilla' with 512 in_channels
    draw the area for inpainting, then close the window
    QStandardPaths: XDG_RUNTIME_DIR not set, defaulting to '/tmp/runtime-root'



用linux服务器没有图形界面，qt一直打不开，后续应该只能直接使用mask_file做输入才能出结果，这个我还得看下怎么改

运行显存 8620MiB         


    警告或报错问题：XDG_RUNTIME_DIR not set, defaulting to ‘/tmp/runtime-root‘
    解决方法：
    方法一：设置环境变量：终端输入export XDG_RUNTIME_DIR=/usr/lib/
    方法二：在/etc/profile末尾增加两句

    export XDG_RUNTIME_DIR=/usr/lib/
    export RUNLEVEL=3
    1
    2
    然后刷新全局变量

    source /etc/profile



在MacOS系统上安装Putty，得先安装MacPorts，它是一个类似brew的包管理工具。    
因为putty工具在Linux和macOS下，都是使用的GTK+图形界面，所以需要安装GTK支持。打开终端，使用port命令安装GTK+，命令如下：   

四、安装XQuartz

XQuartz提供Mac下的绘图层支持，请打开下方链接下载dmg文件：

五、安装Putty

sudo port install putty

稍微麻烦 四步完成，不如直接提供mask_file

需要本地电脑装putty     
服务器再装一个可视化服务器       
gpt        










## gui这一步总算是运行成功
但是需要一个mask_file     
应该就可以了


module 'PIL.Image' has no attribute 'ANTIALIAS'         


pip uninstall -y Pillow       
pip install Pillow==9.5.0


Successfully uninstalled pillow-10.3.0


![alt text](assets_picture/outpaint/image.png)


## 终于

    Using device: cuda:0
    making attention of type 'vanilla' with 512 in_channels
    Working with z of shape (1, 4, 32, 32) = 4096 dimensions.
    making attention of type 'vanilla' with 512 in_channels
    sample.py:362: DeprecationWarning: ANTIALIAS is deprecated and will be removed in Pillow 10 (2023-07-01). Use LANCZOS or Resampling.LANCZOS instead.
    mask_image = mask_image.resize((input_image.shape[3],input_image.shape[2]), Image.ANTIALIAS)
    100%|█████████████████████████████| 50/50 [00:09<00:00,  5.10it/s]
    100%|█████████████████████████████| 50/50 [00:10<00:00,  4.97it/s]
    100%|█████████████████████████████| 50/50 [00:10<00:00,  4.97it/s]
    100%|█████████████████████████████| 50/50 [00:10<00:00,  4.96it/s]


显存峰值10676mb

好像是ddim     
50步         
然后使用类似tile渲染      

python sample.py --model_path inpaint.pt --edit 1.png --text "explosion, grepscale" --outpaint left --kl_path kl.pt --mask mask1.png    
![alt text](assets/outpaint/521715010081_.pic.jpg)

![alt text](assets/outpaint/571715011105_.pic_hd.jpg)

![alt text](assets/outpaint/541715010425_.pic.jpg)

![alt text](assets/outpaint/501715010042_.pic.jpg)

![alt text](assets/outpaint/531715010097_.pic.jpg)

大小正确但是不可控     


## 更改prompt
mask1     
![alt text](assets/outpaint/mask1.png)     
512*512     

左黑右白可用     

按照32扩展    
python sample.py --model_path inpaint.pt --edit 1.png --text "explosion black white" --outpaint left --kl_path kl.pt --mask mask1.png --negative "color object human" --seed 0 --prefix "explosion" --guidance_scale 5.0     
![alt text](assets/outpaint/explosion00000.png)

按照一半扩展    
python sample_my.py --model_path inpaint.pt --edit 1.png --text "explosion black white" --outpaint left --kl_path kl.pt --mask mask1.png --negative "color object human" --seed 0 --prefix "explosion" --guidance_scale 5.0    
11916mb    
![alt text](assets/outpaint/explosion00000-1.png)      
1024*680      

making attention of type 'vanilla' with 512 in_channels    
Working with z of shape (1, 4, 32, 32) = 4096 dimensions.    
making attention of type 'vanilla' with 512 in_channels    
im.shape[3]= 64    
im.shape[2]= 85     

![alt text](assets/outpaint/image-3.png)     

mask全白       

扩展32       
![alt text](assets/outpaint/explosion00000-3.png)   
结果差不多一样，但是比上面那个mask更好     
上面少了一些信息     

扩展一半     
![alt text](assets/outpaint/explosion00000-2.png)

全白mask生效分析     
mask1 = np.ones((height, width, 1), np.uint8) * 255
cv2.imwrite('mask-white.png', mask1)    

0黑，255是白      
灰度图和彩色图区别为：组成不同、通道不同、表示不同。      


## mask代码
推理代码处理 

    input_image = torch.zeros(1, 4, im.shape[2], im.shape[3]+32, device=device)
    input_image[:,:,:,32:32+im.shape[3]] = im
    input_image_mask = torch.zeros(1, 1, im.shape[2], im.shape[3]+32, device=device, dtype=torch.bool)
    input_image_mask[:,:,:,32:32+im.shape[3]] = True
    这行代码创建了一个全零的张量（tensor），使用了 torch.zeros 函数。该张量是一个布尔类型（torch.bool），并且被设定在特定的 device 上（这个设备由代码中的变量 device 决定，可能是 GPU 或 CPU）。
    这段代码的作用是创建一个和输入图像同样大小的掩码，这个掩码的宽度比输入图像的宽度大 32 个像素，掩码中除了与输入图像宽度相同的部分外，其余部分都被标记为无效。

    mask1 = (mask > 0.5)
    input_image_mask *= mask1

    扩展后的图片全部设置成true然后inpaint
    但是是如何保证原油的不变？？
    因为这个相乘操作。原本扩展部分就是默认是false的，所以整个input_image_mask属于正常逻辑的mask,不需要特别提供mask_file，给个全白即可，自动outpaint

    image_embed = torch.cat(args.batch_size*2*[input_image], dim=0).float()


    kwargs = {
        "context": torch.cat([text_emb, text_emb_blank], dim=0).float(),
        "clip_embed": None,
        "image_embed": image_embed
    }


    后续
    overlap = 32
    这个overlap导致图片空白边缘出现。为了满足模型的一些限制   



    if args.edit:
        for i in range(args.num_batches):
            output = input_image.detach().clone()
            output *= input_image_mask.repeat(1, 4, 1, 1).float()
            output就是im所在部分true 

            mask = input_image_mask.detach().clone()

            box = masks_to_boxes(~mask.squeeze(0))[0]

            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2] + 1)
            y1 = int(box[3] + 1)

            x_num = math.ceil(((x1-x0)-overlap)/(64-overlap))
            y_num = math.ceil(((y1-y0)-overlap)/(64-overlap))

            if x_num < 1:
                x_num = 1
            if y_num < 1:
                y_num = 1

            for y in range(y_num):
                for x in range(x_num):
                    offsetx = x0 + x*(64-overlap)
                    offsety = y0 + y*(64-overlap)

                    if offsetx + 64 > x1:
                        offsetx = x1 - 64
                    if offsetx < 0:
                        offsetx = 0

                    if offsety + 64 > y1:
                        offsety = y1 - 64
                    if offsety < 0:
                        offsety = 0

                    patch_input = output[:,:, offsety:offsety+64, offsetx:offsetx+64]
                    patch_mask = mask[:,:, offsety:offsety+64, offsetx:offsetx+64]

                    if not torch.any(~patch_mask):
                        # region does not require any inpainting
                        output[:,:, offsety:offsety+64, offsetx:offsetx+64] = patch_input
                        continue

                    mask[:,:, offsety:offsety+64, offsetx:offsetx+64] = True

                    patch_init = None
    
    
                    if args.skip_timesteps > 0:
                        patch_init = input_image[:,:, offsety:offsety+64, offsetx:offsetx+64]
                        patch_init = torch.cat([patch_init, patch_init], dim=0)

                    skip_timesteps = args.skip_timesteps

                    if not torch.any(patch_mask):
                        # region has no input image, cannot use init
                        patch_init = None
                        skip_timesteps = 0

                    patch_kwargs = {
                        "context": kwargs["context"],
                        "clip_embed": None,
                        "image_embed": torch.cat([patch_input, patch_input], dim=0)
                    }


                    cur_t = diffusion.num_timesteps - 1

                    samples = sample_fn(
                        model_fn,
                        (2, 4, 64, 64),
                        clip_denoised=False,
                        model_kwargs=patch_kwargs,
                        cond_fn=cond_fn,
                        device=device,
                        progress=True,
                        init_image=patch_init,
                        skip_timesteps=skip_timesteps,
                    )

                    for j, sample in enumerate(samples):
                        cur_t -= 1
                        output[0,:, offsety:offsety+64, offsetx:offsetx+64] = sample['pred_xstart'][0]
                        if j % 25 == 0:
                            save_sample(i, output, square=(offsetx, offsety))

                    save_sample(i, output)







## 接下来
换prompt     
进去看隐变量大小    



# 训练glid
Training/Fine tuning 和 Train inpainting 的 arg 参数一致     

Train with same flags as guided diffusion. Data directory should contain image and text files with the same name (image1.png image1.txt)

A custom inpainting/outpainting model trained for an additional 100k steps

对于正常训练模型会merge会sd14      

    model_path = sys.argv[1]
    diffusion_path = sys.argv[2]

    state = torch.load(model_path)
    diffusion = torch.load(diffusion_path)

    diffusion_prefix = 'model.diffusion_model.'

    for key in diffusion.keys():
        state['state_dict'][diffusion_prefix + key] = diffusion[key]

    torch.save(state, 'model-merged.pt')

## 代码差异观察
inpaint多了    

    blur = transforms.GaussianBlur(kernel_size=35, sigma=(0.1, 5)
    定义了一个高斯模糊的变换，它将被用于生成随机的遮罩(mask)。



    emb_cond = emb.detach().clone()

    for i in range(batch.shape[0]):
        if random.randint(0,100) < 20:
        以20%的概率执行以下操作。
            emb_cond[i,:,:,:] = 0 # unconditional
        else:
            if random.randint(0,100) < 50: # random mask
            随机掩码：以一定的概率，生成一个随机的掩码（mask），通过模糊化处理后，将其应用于输入的特征向量（emb_cond）。这个掩码是一个二元的张量，与输入的特征向量形状相同，用于控制哪些元素被保留（值为1）或者被遮蔽（值为0）。
                mask = torch.randn(1, emb.shape[2], emb.shape[3]).to(dist_util.dev())
                mask = blur(mask)
                mask = (mask > 0)
                mask = mask.repeat(4, 1, 1)
                mask = mask.float()
                emb_cond[i] *= mask
                生成一个与输入特征向量（emb）同形状的随机张量作为掩码，然后通过blur函数进行模糊处理，将其二值化为0或1，并将其复制多份以覆盖整个特征向量的空间维度。最后将掩码应用于输入特征向量，将对应位置的元素置为0。
            else:
            随机遮蔽矩形：以一定的概率，对输入的特征向量应用随机数量的矩形遮罩。每个矩形的位置和大小都是随机生成的，并且会将这些矩形区域内的元素置为0，从而达到遮蔽的效果。
                # mask out 4 random rectangles
                for j in range(random.randint(1,4)):
                随机生成1到4之间的数值，确定要生成的矩形数量。然后对每个矩形，随机生成其宽度和高度，并根据特征向量的尺寸确定其位置，将该区域内的元素置为0，以达到遮蔽的效果。
                    max_area = emb.shape[2]*emb.shape[3]//2

                    w = random.randint(1,emb.shape[3])
                    h = random.randint(1,emb.shape[2])
                    if w*h > max_area:
                        if random.randint(0,100) < 50:
                            w = max_area//h
                        else:
                            h = max_area//w
                    if w == emb.shape[3]:
                        offsetx = 0
                    else:
                        offsetx = random.randint(0, emb.shape[3]-w)
                    if h == emb.shape[2]:
                        offsety = 0
                    else:
                        offsety = random.randint(0, emb.shape[2]-h)
                    emb_cond[i,:, offsety:offsety+h, offsetx:offsetx+w] = 0


    model_kwargs["image_embed"] = emb_cond
    将处理后的条件化的嵌入张量 emb_cond 存储在模型参数字典 model_kwargs 中，以便后续传递给模型。


    defaults['image_condition'] = True









## lora解决
这是全量微调的代码     

考虑直接训练一个爆炸图片的lora     
trainer不知道行不    
到时加载底模，训练框架有些问题     
必须用trainer或diffusers训练lora，那里比较成熟。但是底模不知道能不能加载上原本的      

训练和推理有些问题    
倒是可以训练一个任意底模的LORA，用SD-SCRIPT    
但是推理时候怎么合到源代码里面？     


glid源代码模型加载方式     

    model_config = model_and_diffusion_defaults()
    model_config.update(model_params)

    # Load models
    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(model_state_dict, strict=True)
    model.requires_grad_(False).eval().to(device)



加载lora比较麻烦     
直接全参数训练吧     

## 全参数训练解决
继承于 guided_diffusion

其模型结构     

    def model_and_diffusion_defaults():
        """
        Defaults for image training.
        """
        res = dict(
            image_size=64,
            num_channels=128,
            num_res_blocks=2,
            num_heads=4,
            num_heads_upsample=-1,
            num_head_channels=-1,
            attention_resolutions="16,8",
            channel_mult="",
            dropout=0.0,
            class_cond=False,
            use_checkpoint=True,
            use_scale_shift_norm=True,
            resblock_updown=False,
            use_fp16=False,

            use_spatial_transformer=True,
            context_dim=768,

            clip_embed_dim=None,
            image_condition=False,
            super_res_condition=False
        )
        res.update(diffusion_defaults())
        return res



24g bs1训练不了



## 学校a100装环境报错
mpi4py    
ERROR: Could not build wheels for mpi4py, which is required to install pyproject.toml-based projects      

训练不起来    


    collect2: error: ld returned 1 exit status
        failure.
        removing: _configtest.c _configtest.o
        error: Cannot link MPI programs. Check your configuration!!!
        [end of output]
    
    note: This error originates from a subprocess, and is likely not a problem with pip.
    ERROR: Failed building wheel for mpi4py
    Failed to build mpi4py
    ERROR: Could not build wheels for mpi4py, which is required to install pyproject.toml-based projects


解决方法:     
换conda安装     
conda install mpi4py       

StackOverflow也有说sudo装一些东西，但没必要了         

## 训练过程

bs1 训练显存33g

    MODEL_FLAGS="--actual_image_size 512 --lr_warmup_steps 10000 --ema_rate 0.9999 --attention_resolutions 64,32,16 --class_cond False --diffusion_steps 1000 --image_size 64 --learn_sigma False --noise_schedule linear --num_channels 320 --num_heads 8 --num_res_blocks 2 --resblock_updown False --use_fp16 True --use_scale_shift_norm False "
    TRAIN_FLAGS="--lr 5e-5 --batch_size 1 --log_interval 10 --save_interval 10000 --kl_model checkpoint/kl-1.4.pt --resume_checkpoint checkpoint/ema_0.9999_100000.pt"
    export OPENAI_LOGDIR=./logs_inpaint/
    python scripts/image_train_inpaint.py --data_dir datasets/mydata $MODEL_FLAGS $TRAIN_FLAGS

两张图片训练       
![alt text](assets_picture/outpaint/1715179619944.png)       


A custom inpainting/outpainting model trained for an additional 100k steps

readme说额外训练10万步       
不知道他的数据量多少      

model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

有些奇怪         


OPENAI_
LOGDIR=./logs_inpaint1/ python -m debugpy --l
isten 7890 --wait-for-client scripts/image_tr
ain_inpaint.py --data_dir datasets/mydata --lr 5e
-5 --batch_size 1 --log_interval 10 --save_in
terval 10000 --kl_model checkpoint/kl-1.4.pt 
--resume_checkpoint checkpoint/ema_0.9999_100
000.pt --actual_image_size 512 --lr_warmup_st
eps 10000 --ema_rate 0.9999 --attention_resol
utions 64,32,16 --class_cond False --diffusio
n_steps 1000 --image_size 64 --learn_sigma Fa
lse --noise_schedule linear --num_channels 32
0 --num_heads 8 --num_res_blocks 2 --resblock
_updown False --use_fp16 True --use_scale_shi
ft_norm False --lr_anneal_steps 15000 结束步数


OPENAI_
LOGDIR=./logs_inpaint1/ python scripts/image_tr
ain_inpaint.py --data_dir datasets/mydata --lr 5e
-5 --batch_size 1 --log_interval 10 --save_in
terval 10000 --kl_model checkpoint/kl-1.4.pt 
--resume_checkpoint checkpoint/ema_0.9999_100
000.pt --actual_image_size 512 --lr_warmup_st
eps 10000 --ema_rate 0.9999 --attention_resol
utions 64,32,16 --class_cond False --diffusio
n_steps 1000 --image_size 64 --learn_sigma Fa
lse --noise_schedule linear --num_channels 32
0 --num_heads 8 --num_res_blocks 2 --resblock
_updown False --use_fp16 True --use_scale_shi
ft_norm False --lr_anneal_steps 15000 结束步数



## 扩图方向增加
这个guidance diffusion模型应该在 webui inpaint 用不了，现在都是直接支持sd1.5 或者 sdxl类型的，这个早版本的架构没人用的。应该不支持    








# 通用训练 inpaint
使用sd1.5 sdxl框架     
然后用webui     


## train_dreambooth_inpaint_lora
diffusers/examples/research_projects/dreambooth_inpaint/train_dreambooth_inpaint_lora.py

diffusers还算注释详细，有点良心的      
功能较齐全     

### glid代码mask处理

    input_image = torch.zeros(1, 4, im.shape[2], im.shape[3]+32, device=device)
    input_image[:,:,:,32:32+im.shape[3]] = im
    input_image_mask = torch.zeros(1, 1, im.shape[2], im.shape[3]+32, device=device, dtype=torch.bool)
    input_image_mask[:,:,:,32:32+im.shape[3]] = True
    这行代码创建了一个全零的张量（tensor），使用了 torch.zeros 函数。该张量是一个布尔类型（torch.bool），并且被设定在特定的 device 上（这个设备由代码中的变量 device 决定，可能是 GPU 或 CPU）。
    这段代码的作用是创建一个和输入图像同样大小的掩码，这个掩码的宽度比输入图像的宽度大 32 个像素，掩码中除了与输入图像宽度相同的部分外，其余部分都被标记为无效。

    mask1 = (mask > 0.5)
    input_image_mask *= mask1


### mask准备     

    def prepare_mask_and_masked_image(image, mask):
        image = np.array(image.convert("RGB"))
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        mask = np.array(mask.convert("L"))
        mask = mask.astype(np.float32) / 255.0
        mask = mask[None, None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

        masked_image = image * (mask < 0.5)
        逻辑与glid相反，左黑右白可用。也就是glid白色才是正确mask原图位置。glid逻辑比较多，负负得正的效果，比较乱     
        

        return mask, masked_image


    # generate random masks
    def random_mask(im_shape, ratio=1, mask_full_image=False):
        mask = Image.new("L", im_shape, 0)
        draw = ImageDraw.Draw(mask)
        size = (random.randint(0, int(im_shape[0] * ratio)), random.randint(0, int(im_shape[1] * ratio)))
        随机size
        # use this to always mask the whole image
        if mask_full_image:
            size = (int(im_shape[0] * ratio), int(im_shape[1] * ratio))
        limits = (im_shape[0] - size[0] // 2, im_shape[1] - size[1] // 2)
        控制 center 范围不要超出
        center = (random.randint(size[0] // 2, limits[0]), random.randint(size[1] // 2, limits[1]))
        随机 center
        draw_type = random.randint(0, 1)
        if draw_type == 0 or mask_full_image: 长方形
            draw.rectangle(
                (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
                fill=255,
            )
        else: 画椭圆
            draw.ellipse(
                (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
                fill=255,
            )

        return mask

        glid中好像是有1-4个正方形，随机遮


### mask collate_fn

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]
            pior_pil = [example["class_PIL_images"] for example in examples]

        masks = []
        masked_images = []
        for example in examples:
            pil_image = example["PIL_images"]
            # generate a random mask
            mask = random_mask(pil_image.size, 1, False)
            # prepare mask and masked image
            mask, masked_image = prepare_mask_and_masked_image(pil_image, mask)

            masks.append(mask)
            masked_images.append(masked_image)

        if args.with_prior_preservation:
            for pil_image in pior_pil:
                # generate a random mask
                mask = random_mask(pil_image.size, 1, False)
                # prepare mask and masked image
                mask, masked_image = prepare_mask_and_masked_image(pil_image, mask)

                masks.append(mask)
                masked_images.append(masked_image)

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids
        masks = torch.stack(masks)
        masked_images = torch.stack(masked_images)
        batch = {"input_ids": input_ids, "pixel_values": pixel_values, "masks": masks, "masked_images": masked_images}
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn
    )        







### mask使用

    # Convert masked images to latent space
    masked_latents = vae.encode(
        batch["masked_images"].reshape(batch["pixel_values"].shape).to(dtype=weight_dtype)
    ).latent_dist.sample()
    masked_latents = masked_latents * vae.config.scaling_factor

    masks = batch["masks"]
    # resize the mask to latents shape as we concatenate the mask to the latents
    mask = torch.stack(
        [
            torch.nn.functional.interpolate(mask, size=(args.resolution // 8, args.resolution // 8))
            for mask in masks
        ]
    ).to(dtype=weight_dtype)
    mask = mask.reshape(-1, 1, args.resolution // 8, args.resolution // 8)

    # concatenate the noised latents with the mask and the masked latents
    latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

### Training with prior-preservation loss
Prior-preservation is used to avoid overfitting and language-drift.

For prior-preservation we first generate images using the model with a class prompt and then use those during training along with our data.        
According to the paper, it's recommended to generate num_epochs * num_samples images for prior-preservation. 200-300 works well for most cases.


Training with gradient checkpointing and 8-bit optimizer:       
With the help of gradient checkpointing and the 8-bit optimizer from bitsandbytes it's possible to run train dreambooth on a 16GB GPU.


Fine-tune text encoder with the UNet.       
The script also allows to fine-tune the text_encoder along with the unet. It's been observed experimentally that fine-tuning text_encoder gives much better results especially on faces. Pass the --train_text_encoder argument to the script to enable training text_encoder.


### loss

    # Convert images to latent space
    # Convert masked images to latent space
    # resize the mask to latents shape as we concatenate the mask to the latents

    mask = torch.stack(
        [
            torch.nn.functional.interpolate(mask, size=(args.resolution // 8, args.resolution // 8))
            for mask in masks
        ]
    ).to(dtype=weight_dtype)
    mask = mask.reshape(-1, 1, args.resolution // 8, args.resolution // 8)

    noise = torch.randn_like(latents)

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # concatenate the noised latents with the mask and the masked latents
    latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

    # Get the text embedding for conditioning
    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

    # Predict the noise residual
    noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample


    # Get the target for loss depending on the prediction type
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    if args.with_prior_preservation:
        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
        target, target_prior = torch.chunk(target, 2, dim=0)

        # Compute instance loss
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()
        reduction="none" 参数指定了不进行降维操作，因此它会返回一个与输入相同形状的张量，其中每个元素代表对应位置的损失值。接着使用 .mean([1, 2, 3]) 对每个样本的损失进行求均值，最后再使用 .mean() 对所有样本的损失值再进行一次求均值，得到最终的损失值。
        .mean([1, 2, 3]) 表示在指定的维度上求均值。在PyTorch中，对张量调用 .mean() 方法时可以传入一个维度参数，该参数告诉函数在哪些维度上计算均值。在这种情况下，传入 [1, 2, 3] 表示在第1、2、3个维度上分别求均值。

        假设你有一个4维张量，形状为 [batch_size, channels, height, width]，那么 .mean([1, 2, 3]) 就会在通道、高度和宽度三个维度上分别计算均值，最终得到每个样本的均值。



        # Compute prior loss
        prior_loss = F.mse_loss(noise_pred_prior.float(), target_prior.float(), reduction="mean")
        通过 reduction="mean" 参数对所有元素的损失值进行求均值，得到最终的损失值。

        # Add the prior loss to the instance loss.
        loss = loss + args.prior_loss_weight * prior_loss

        好简单，就一个scale控制
    else:
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
    
直接就是 noisy_latent(加噪的latent)和mask和mask_latent 一起丢入unet，预测噪声，然后于真实噪声算mse            
mask_latent是没有加噪的。     


### 保存权重
    # Save the lora layers
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unet.save_attn_procs(args.output_dir)


### lora初始化

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    经验之谈
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    好像comfyui是给vae用xformers

    # now we will add new LoRA weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Set correct lora layers
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        这些个不是很理解
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            这些个不是很理解
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
    这个代码写法很像MoMA的set_ip_adapter


        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)

    accelerator.register_for_checkpointing(lora_layers)









### 学习目标：dreambooth 和 dreambooth_lora 和 lora 训练的区别

dreambooth先验保留的图片如何生成

dreambooth_lora: 我理解是使用 dreambooth 损失训练lora, 这样可能稍微好一些    








## train_dreambooth_inpaint
diffusers/examples/research_projects/dreambooth_inpaint/train_dreambooth_inpaint.py


### 训练命令

    export MODEL_NAME="runwayml/stable-diffusion-inpainting"
    export INSTANCE_DIR="path-to-instance-images"
    export CLASS_DIR="path-to-class-images"
    export OUTPUT_DIR="path-to-save-model"

    accelerate launch train_dreambooth_inpaint.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --train_text_encoder \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="a photo of sks dog" \
    --class_prompt="a photo of dog" \
    --resolution=512 \
    --train_batch_size=1 \
    --use_8bit_adam \
    --gradient_checkpointing \
    --learning_rate=2e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=200 \
    --max_train_steps=800

dreambooth读数据      



相关输入参数

    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")

    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )

就是说如果没有准备够，模型会先自己采样的，其实也不需要额外手动准备        

class DreamBoothDataset(Dataset):

    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

    def __getitem__(self, index):
        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            class_image = self.image_transforms_resize_and_crop(class_image)
            example["class_images"] = self.image_transforms(class_image)
            example["class_PIL_images"] = class_image
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example



### class文件夹不够数量自动生成


    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:


            num_new_images = args.num_class_images - cur_class_images

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(
                sample_dataset, batch_size=args.sample_batch_size, num_workers=1
            )

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):

            这个是inpaint训练代码，生成的竟然还要做mask？？？？
            没有mask还能理解dreambooth，这有了之后就不理解了
                bsz = len(example["prompt"])
                fake_images = torch.rand((3, args.resolution, args.resolution))
                transform_to_pil = transforms.ToPILImage()
                fake_pil_images = transform_to_pil(fake_images)

                fake_mask = random_mask((args.resolution, args.resolution), ratio=1, mask_full_image=True)
            
            全部mask掉，不就是输入空的latent吗，为了统一模型的输入吧
            和后面的统一


                images = pipeline(prompt=example["prompt"], mask_image=fake_mask, image=fake_pil_images).images

                for i, image in enumerate(images):
                    hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


### 保存权重
这个脚本才能训练unet和text_encoder       
上个脚本只有个lora   

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
        )
        pipeline.save_pretrained(args.output_dir)


## 混淆项 train_dreambooth_lora.py

这些不是inpaint模型的      

https://github.com/huggingface/diffusers/tree/main/examples/dreambooth

https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora.py


但是输入命令，数据集都可以是一致的，只是训练脚本内部稍微不一样       




# 训练 dreambooth lora
diffutorch201环境不需要特别安装，我看了一下都满足    
数据集 /data/lujunda/207/explosion   
模型在上一个文件夹   
 
代码位置 /data/lujunda/new-diffuser/diffuser
s-main/examples/research_projects/dreambooth_inpaint     
    

需要inpaint模型     


https://github.com/runwayml/stable-diffusion#inpainting-with-stable-diffusion

https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#inpainting-model-sd2  

Inpainting Model SD2      
Model specifically designed for inpainting trained on SD 2.0 512 base.

512 inpainting (2.0) - (model+yaml) - .safetensors      
inpainting_mask_weight or inpainting conditioning mask strength works on this too.

https://huggingface.co/webui/stable-diffusion-2-inpainting/tree/main


diffusers这种文档缺陷好大       
不能自定义每个图片的prompt       
不能分桶      

唯一优势在于khoya没有专门的inpaint lora训练






# stable-diffusion-infinity-xl
这个应该可以的，因为是sd1.5 sdxl框架      


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


# prompt
--text "explosion black white" --negative "color object human"



# webui inpaint script 局限
https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#inpainting-model-sd2     

https://github.com/runwayml/stable-diffusion#inpainting-with-stable-diffusion

功能里面没有image_condition的设置，全图输入，这个局限太大了   


v1.5    
## 1. 直接resize大小，往左右两边扩展       
但这不是需求  

## 2. poor man's outpainting   
可以选择方向    
![alt text](assets/README/431712639349_.pic-1.jpg)    
![alt text](assets/README/631712642102_.pic.jpg)    
![alt text](assets/README/image.png)    
![alt text](assets/README/image-1.png)   

## 3. outpainting mk2   
![alt text](assets/README/image-2.png)    
![alt text](assets/README/image-3.png)    
![alt text](assets/README/image-4.png)     
![alt text](assets/README/image-5.png)   
参数比较难调   

### 采用专门对inpaint优化的模型   
sd2.1基准    
https://huggingface.co/webui/stable-diffusion-2-inpainting/tree/main    
![alt text](assets/outpaint/image.png)

也有   
https://huggingface.co/webui/stable-diffusion-inpainting/tree/main

这两个都是2023.1.26的

#### 复现

原图   
371 × 681像素     
![alt text](assets/outpaint/下载.jpeg)    

##### 结果一


explosion black white   
Negative prompt: color object human     
Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 3527031196, Size: 512x704, Model hash: b29e2ed9a8, Model: 512-inpainting-ema, Denoising strength: 0.75, Conditional mask weight: 1.0, Version: f0.0.17v1.8.0rc-latest-276-g29be1da7   
Time taken: 3.5 sec.

A: 3.31 GB, R: 3.68 GB, Sys: 8.3/23.6914 GB (34.9%)   


512 × 681像素     
![alt text](assets/outpaint/image-14.png)


left   
pixel expand 128 
mask blur  8   
fall off exponent 1     
color variation 0.05     

Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: 0.8      


接续扩展      
640 × 681像素     
![alt text](assets/outpaint/image-15.png)



##### 结果二
一次性扩展最大256       
640 × 681像素      
![alt text](assets/outpaint/image-16.png)

Time taken: 2.3 sec.

A: 3.31 GB, R: 3.68 GB, Sys: 8.3/23.6914 GB (35.0%)

再二次扩展可能需要根据黑边prompt作为输入，具体看需求

###### 二次扩展
896 × 681像素    
![alt text](assets/outpaint/image-17.png)


black white,  black background   
![alt text](assets/outpaint/image-18.png)

black white,  ((black background))     
![alt text](assets/outpaint/image-19.png)   

black white,  ((pure black background))    
color object human, explosion    
![alt text](assets/outpaint/image-20.png)    

((pure black background))   
color object human, explosion   
![alt text](assets/outpaint/image-21.png)    

((pure black background))   
color object human, explosion, smoke    
![alt text](assets/outpaint/image-22.png)    

((pure black background))   
color object human, (explosion, smoke), white    
![alt text](assets/outpaint/image-23.png)    









# controlnet优化模型：   
局限，没有参照框限制重绘和参考区域

https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint   
sd1.5基准   
![alt text](assets/outpaint/image-1.png)    
ControlNet插件inpaint局部重绘模型对于接缝处的处理 确实比图生图自带的局部重绘功能处理的要好太多了。     
https://zhuanlan.zhihu.com/p/633750880?utm_id=0    


![alt text](assets/outpaint/image-24.png)












# webui插件


比较难以安装    
相比于comfyui   

直接使用inpaint    
![alt text](assets/README/431712639349_.pic.jpg)   
![alt text](assets/README/581712640260_.pic.jpg)   


装插件    
## masoic    
16步，比较模糊   
原理扩展加masoic然后又有另一张mask图片，通过这些去做inpaint   
我的理解是输入前处理latent，生图。获取的结果通过mask过滤   
可以选择方向，功能齐全，效果略差    
可以使用controlnet   
后期也许可以考虑叠加lora，   
![alt text](assets/README/241712589773_.pic.jpg)    
![alt text](assets/README/251712589784_.pic.jpg)   
![alt text](assets/README/261712589802_.pic.jpg)   






## infinite zoom   
介绍是生视频的，生出五张图，没有方向控制    
https://youtube.com/shorts/Erju6TzEAEM?feature=share   



## OpenOutpaint
类似stable-diffusion-infinity-xl   
但是还不了解如何作画，使用    

OpenOutpaint

![alt text](assets_picture/outpaint/930dc31b29162c3b2b7394ad1568277.jpg)


https://blog.csdn.net/ddrfan/article/details/130316244


https://www.bilibili.com/video/BV1TM411P7a8/?spm_id_from=333.337.search-card.all.click&vd_source=15487431cfc74ae938beabdb124d750a


一直在offline       
用不了？？？       




# inpaint模型结构有什么改变？

训练加mask     
添加方式略有了解     

loss怎么计算      

    RGB图像（3个通道）：原始的RGB图像包含三个通道（红色、绿色和蓝色），这些通道表示图像的颜色信息。

    掩码图像（3个通道）：在许多修复任务中，需要修复的区域通过掩码来指示。掩码版本的RGB图像（也是3个通道）可以作为输入，其中掩码区域可能用特定的值（例如黑色或其他占位符值）填充。

    二值掩码（1个通道）：二值掩码指示图像中哪些部分缺失并需要修复。这个掩码有一个通道，每个像素要么是0（表示像素缺失），要么是1（表示像素存在）。

    额外的引导信息（2个通道）：一些模型使用额外的通道提供更多的上下文或引导信息用于修复任务。这些信息可以包括边缘图、梯度、深度信息或其他形式的辅助数据，这些数据可以帮助模型理解缺失区域的结构和上下文。





















# 社区模型
Anything V3-inpainting       
hako-mikan/sd-webui-supermerger    

![alt text](assets_picture/outpaint/image-1.png)

![alt text](assets_picture/outpaint/image-2.png)


这个融合方法感觉上只是保存到本地，不好借鉴到iclight


# fooocus outpaint
原理应该是一样的









# 其他
## guided_diffusion
从DDPM到GLIDE：基于扩散模型的图像生成算法进展    
前几天，OpenAI在Arxiv上挂出来了他们最新最强的文本-图像生成GLIDE [1]，如头图所示，GLIDE能生成非常真实的结果。GLIDE并非基于对抗生成网络或是VQ-VAE类模型所设计，而是采用了一种新的图像生成范式 - 扩散模型（Diffusion Model）。作为一种新的生成模型范式，扩散模型有着和GAN不同且有趣的很多特质。    

发布于 2021-12-27 10:34・IP 属地未知

### 一、扩散模型与DDPM

![alt text](assets/outpaint/image-6.png)




### 二、Guided Diffusion - 基于类别引导的扩散模型

https://github.com/openai/guided-diffusion     

 [Submitted on 11 May 2021 (v1), last revised 1 Jun 2021 (this version, v4)]     
Diffusion Models Beat GANs on Image Synthesis





通常而言，对于通用图像生成任务，加入类别条件能够比无类别条件生成获得更好的效果，这是因为加入类别条件的时候，实际上是大大减小了生成时的多样性。OpenAI的Guided Diffusion [4]就提出了一种简单有效的类别引导的扩散模型生成方式。Guided Diffusion的核心思路是在逆向过程的每一步，用一个分类网络对生成的图片进行分类，再基于分类分数和目标类别之间的交叉熵损失计算梯度，用梯度引导下一步的生成采样。这个方法一个很大的优点是，不需要重新训练扩散模型，只需要在前馈时加入引导既能实现相应的生成效果。


基于条件的逆向过程

在DDPM中，无条件的逆向过程可以用![alt text](assets/outpaint/image-4.png)
来描述，在加入类别条件 后，逆向过程可以表示为     
![alt text](assets/outpaint/image-5.png)

![alt text](assets/outpaint/image-7.png)

扩散模型结构改进

guided diffusion 中，还对DDPM中采用的U-Net 结构的Autoencoder进行了一些结构上的改进。包括加深网络、增加attention head数量、增加添加attention layer的尺度数量、采用BigGAN的残差模块结构。此外，在这篇工作中还采用了一种称为Adaptive Group Normalization （AdaGN）的归一化模块。


### 三、Semantic Guidence Diffusion - 更多的扩散引导形式（图片/文本）

在Guided Diffusion 中，每一步逆向过程里通过引入朝向目标类别的梯度信息，来实现针对性的生成。这个过程其实和基于优化（Optimization）的图像生成算法（即固定网络，直接对图片本身进行优化）有很大的相似之处。这就意味着之前很多基于优化的图像生成算法都可以迁移到扩散模型上。换一句话说，我们可以轻易地通过修改Guided Diffusion中的条件类型，来实现更加丰富、有趣的扩散生成效果。在Semantic Guidence Diffusion （SGD）[5] 中，作者就将类别引导改成了基于参考图引导以及基于文本引导两种形式，通过设计对应的梯度项，实现对应的引导效果，实现了不错的效果。

![alt text](assets/outpaint/v2-7071fbfc940ea88ca9da1efe550ab370_720w.webp)

![alt text](assets/outpaint/image-8.png)

![alt text](assets/outpaint/image-9.png)


![alt text](assets/outpaint/image-10.png)


![alt text](assets/outpaint/image-11.png)



### 四、Classifier-Free Diffusion Guidence - 无分类器的扩散引导

上述的各种引导函数，基本都是额外的网络前向 + 梯度计算的形式，这种形式虽然有着成本低，见效快的优点。也存在着一些问题：（1）额外的计算量比较多；（2）引导函数和扩散模型分别进行训练，不利于进一步扩增模型规模，不能够通过联合训练获得更好的效果。DDPM的作者，谷歌的Jonathan Ho等人在今年NIPS 的workshop 上对Guided Diffusion 进行了一波改进，提出了无需额外分类器的扩散引导方法 [6]。


![alt text](assets/outpaint/image-12.png)


### 五、GLIDE - 基于扩散模型的文本图像生成大模型


GLIDE(Guided Language to Image Diffusion for Generation and Editing)       

GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models

https://github.com/openai/glide-text2im


[Submitted on 20 Dec 2021 (v1), last revised 8 Mar 2022 (this version, v3)]     
GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models



上一节说到no-classifer guidence 可以更好的将条件信息加入到扩散模型的训练中去以得到更好的训练效果，但同时也会增加训练成本。财大气粗的OpenAI 就基于no-classifier guidence 的思想，整了一个超大规模的基于扩散模型的文本图像生成模型GLIDE。其中算法的核心即将前面的类别条件更新为了文本条件：

![alt text](assets/outpaint/image-13.png)

其余部分在方法上并没有什么特别新的东西，说的上是大力出奇迹了。这里简单介绍一些重要的点

    更大的模型：算法采用了Guided Diffusion方法中相同的Autoencoder结构，但是进一步扩大了通道数量，使得最终的网络参数数量达到了3.5 billion；
    更多的数据：采用了和DALLE [7]相同的大规模文本-图像对数据集
    很高的训练成本：这里作者没有细说，只说了采用2048batch size，训练了250万轮，总体成本接近Dalle。

在2020年Google 发表DDPM后，这两年扩散模型有成为一个新的研究热点的趋势，除了上面介绍的几篇论文之外，还有不少基于扩散模型所设计的优秀的生成模型，应用于多种不同的任务，比如超分、inpainting等。除了在视觉任务上的应用，也有工作针对DDPM的速度进行优化[8]，加速生成时的采样过程。此外，也有将扩散模型与VQ-VAE结合起来实现文本图像生成的算法[9]。其实在七八月份的时候，就已经看了一些DDPM的相关工作，不过因为种种原因当时没有follow下去，还是比较可惜。













# 结尾