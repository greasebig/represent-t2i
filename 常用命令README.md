# 命令

## find
find /home/WujieAITeam/private -type d -name "stable-diffusion-xl-base-1.0"

find /home/WujieAITeam/private -name "mjhq30k_imgs.zip"



   

## tmux
tmux kill-session -t edm2  
tmux attach -t edm2   
tmux new-session -s edm2   




## kill 显存
通过以下命令查看僵尸进程    
sudo fuser -v /dev/nvidia*  
找到COMMAND=python的，然后通过以下命令逐一kill僵尸进程    
sudo kill -9 进程q

进程查看 ps -ef


## 镜像hug下载

export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --repo-type dataset --resume-download playgroundai/MJHQ-30K --local-dir playgroundai/MJHQ-30K

huggingface-cli download --resume-download gpt2 --local-dir gpt2


### 单文件
wget https://hf-mirror.com/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors?download=true

这样一般下载得，需要重命名 'yoso_lora.safetensors?download=true'

而不是  
wget https://hf-mirror.com/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors




## 删除设置的环境变量

unset HF_ENDPOINT   
echo $HF_ENDPOINT    






## Linux统计文件夹下的文件数目
统计当前目录下文件的个数（不包括目录）  
ls -l | grep "^-" | wc -l   
统计当前目录下文件的个数（包括子目录）   
ls -lR| grep "^-" | wc -l   
查看某目录下文件夹(目录)的个数（包括子目录）  
ls -lR | grep "^d" | wc -l   

ls -l   
长列表输出该目录下文件信息(注意这里的文件是指目录、链接、设备文件等)，每一行对应一个文件或目录，ls -lR是列出所有文件，包括子目录。

grep "^-"   
过滤ls的输出信息，只保留一般文件，只保留目录是grep "^d"。

wc -l   
统计输出信息的行数，统计结果就是输出信息的行数，一行信息对应一个文件，所以就是文件的个数。



## 端口转发
此外，如果我们开发的是 WEB 应用，为了能够浏览到远程主机上的应用，我们可以利用另一个端口转发的功能来实现。


## 远程服务器vscode debug
必须先在服务器上装python和python debugger扩展   
没有就是vscode版本太久，某些方面不兼容了    



## diffusers转ckpt safetensors
python scripts/convert_diffusers_to_original_stable_diffusion.py --model_path model_dir --checkpoint_path path_to_ckpt.ckpt




## github镜像
github在国内会碰到下载不稳定的情况，推荐使用镜像

https://bgithub.xyz/
将前缀更换即可，例如：

https://github.com/OpenGVLab/CaFo
https://bgithub.xyz/OpenGVLab/CaFo



## conda复制环境和删除环境

conda create --name <new_environment_name> --clone <existing_environment_name>

conda remove --name <environment_name> --all



## 终端上网
export http_proxy=127.0.0.1:7890
export https_proxy=127.0.0.1:7890

export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"


## 终端查找历史命令

使用 Ctrl + R 进行反向搜索：
您可以按下 Ctrl + R 键，然后开始输入部分命令。终端会自动显示最接近的匹配项。继续按 Ctrl + R 将继续在历史记录中搜索更早的命令。


## torch将变量从cpu转到cuda，相同浮点数
noise = noise.to('cuda')

mat1 and mat2 must have the same dtype
tensor = torch.randn(3, 3)
tensor_float16 = tensor.to(torch.float16)


















# 报错



## 安装xformers报错
ERROR: Could not build wheels for xformers, which is required to install pyproject.toml-based projects    
这个问题是由于cuda版本、nvcc版本、Pytorch版本不一致所导致的。    


## safetensor header too large
加载lora失败
Name: safetensors
Version: 0.4.2
Name: diffusers
Version: 0.28.0.dev0

换机器加载成功
都是a800
但是担心比较时显存和速度会有所不同
Name: safetensors
Version: 0.4.2
Name: diffusers
Version: 0.25.0


旧机器报错

    Traceback (most recent call last):
    File "/root/miniconda3/envs/emd-new/lib/python3.8/site-packages/diffusers/models/modeling_utils.py", line 109, in load_state_dict
    return safetensors.torch.load_file(checkpoint_file, device="cpu")
    File "/root/miniconda3/envs/emd-new/lib/python3.8/site-packages/safetensors/torch.py", line 308, in load_file
    with safe_open(filename, framework="pt", device=device) as f:
    safetensors_rust.SafetensorError: Error while deserializing header: HeaderTooLarge

    During handling of the above exception, another exception occurred:

    Traceback (most recent call last):
    File "/root/miniconda3/envs/emd-new/lib/python3.8/site-packages/diffusers/models/modeling_utils.py", line 120, in load_state_dict
    if f.read().startswith("version"):
    File "/root/miniconda3/envs/emd-new/lib/python3.8/codecs.py", line 322, in decode
    (result, consumed) = self._buffer_decode(data, self.errors, final)
    UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 64: invalid start byte

    During handling of the above exception, another exception occurred:

    Traceback (most recent call last):
    File "/lujunda/diffusers-main/examples/inference/yoso-infer-1step.py", line 23, in
    pipeline.load_lora_weights('/lujunda/diffusers-main/examples/inference/yoso_lora.safetensors')
    File "/root/miniconda3/envs/emd-new/lib/python3.8/site-packages/diffusers/loaders/lora.py", line 114, in load_lora_weights
    state_dict, network_alphas = self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)
    File "/root/miniconda3/envs/emd-new/lib/python3.8/site-packages/huggingface_hub/utils/_validators.py", line 119, in _inner_fn
    return fn(*args, **kwargs)
    File "/root/miniconda3/envs/emd-new/lib/python3.8/site-packages/diffusers/loaders/lora.py", line 284, in lora_state_dict
    state_dict = load_state_dict(model_file)
    File "/root/miniconda3/envs/emd-new/lib/python3.8/site-packages/diffusers/models/modeling_utils.py", line 132, in load_state_dict
    raise OSError(
    OSError: Unable to load weights from checkpoint file for '/lujunda/diffusers-main/examples/inference/yoso_lora.safetensors' at '/lujunda/diffusers-main/examples/inference/yoso_lora.safetensors'.






# 结尾




