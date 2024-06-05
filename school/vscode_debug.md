# 原始配置
launch.json

{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            //"cwd": "${workspaceFolder}/${fileDirname}",
            "cwd": "${fileDirname}",
            //"cwd": "/data/lujunda/new-diffuser/diffusers-main/examples/controlnet",
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}




## 原始配置2
launch.json

{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Attach",
            "type": "python",
            "request": "attach",
            "connect": {
				"host": "localhost",
				"port": 7890
			},
            "justMyCode": false
        }
    ]
}

# debugpy配置
## 使用方式
  






服务器环境 pip install debugpy 

本地终端netstat -a        
找到一个未占用的端口号     
State显示为LISTENing即为未占用         
135端口一般用不了，权限不够       
用7890        

### 配置

修改launch.json中内容为：

    {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Python: Attach",
                "type": "python",
                "request": "attach",
                "connect": {
                    "host": "localhost",
                    "port": xxxx本地终端端口号
                }
            }
        ]
    } 


Step4：假设run 该Python脚本的命令为：python xxx.py -arg1 ARG1 -arg2 ARG2。即脚本有两个参数输入。在进行调试之前，在VSCode终端命令行中键入：

    python -m debugpy --listen xxxx本地终端端口号 --wait-for-client xxx.py -arg1 ARG1 -arg2 ARG2

执行上述命令，终端处于执行中，没有任何返回。接下来在程序中设置断点，按下F5键，即可进入VSCode的调试模式。调试方式与不带参数的情况相同。


win7 vscode 连断点都打不了，即使使用这个也没办法调试






## 使用案例   

短暂声明环境变量

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
ft_norm False --lr_anneal_steps 15000













