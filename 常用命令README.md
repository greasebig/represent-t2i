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










# 结尾




