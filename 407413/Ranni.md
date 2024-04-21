Ranni: Taming Text-to-Image Diffusion for Accurate Prompt Following    
Ranni：驯服文本到图像的扩散以实现准确的提示跟随   



## TODO:
模型控制效果测试，五六个控制方法     
对比两个其他模型    
related_work????     


## 基本信息：  
机构：    
Alibaba Group   |   Ant Group     

仓库：     
https://github.com/ali-vilab/Ranni   

[Submitted on 28 Nov 2023 (v1), last revised 9 Apr 2024 (this version, v3)]    
https://arxiv.org/abs/2311.17002    


This repository is based on the following codebases:

https://github.com/Stability-AI/stablediffusion   
https://github.com/lllyasviel/ControlNet/

权重：    
https://modelscope.cn/models/yutong/Ranni/files   

2024 4.8 : Ranni 被接受为 CVPR 2024 口头论文 🎉    
2024年4.3：我们发布了Ranni的v1代码。    

待办事项列表：    
支持更多条件。    
基于聊天的编辑。    
具有 ID 一致性的连续生成。     





   
## 原理：     
该存储库是 CVPR 2024 论文“Ranni: Taming Text-to-Image Diffusion for Accurate instructions Follow”的官方实现。它包含两个主要组件：1）基于LLM的规划模型，将文本指令映射到图像中的视觉元素，2）基于扩散的绘画模型，在第一阶段按照视觉元素绘制图像。得益于LLM的强大能力，Ranni获得了更好的语义理解。目前，我们发布的模型权重包括 LoRA 微调的 LLaMa-2-7B 和完全微调的 SDv2.1 模型。     


![alt text](assets/Ranni/image.png)    
![alt text](assets/Ranni/image-1.png)     


现有的文本到图像（T2I）扩散模型通常难以解释复杂的提示，尤其是那些具有数量、对象属性绑定和多主题描述的提示。在这项工作中，我们引入了语义面板作为将文本解码为图像的中间件，支持生成器更好地遵循指令。该面板是通过借助大型语言模型对从输入文本中解析出的视觉概念进行排列而获得的，然后将其作为详细的控制信号注入到去噪网络中以补充文本条件。为了促进文本到面板的学习，我们提出了精心设计的语义格式化协议，并配有全自动数据准备管道。得益于这样的设计，我们称之为 `Ranni 的方法能够增强预训练的 T2I 生成器的文本可控性`。更重要的是，`生成中间件的引入带来了更便捷的交互形式`（即直接调整面板中的元素或使用语言指令），并进一步允许用户精细地定制他们的生成，在此基础上我们开发了实用的系统和展示其在连续生成和基于聊天的编辑方面的潜力。    

具有不同的交互方式，包括（a）直接生成，准确提示跟随， （b）连续生成，逐步细化，以及（c）基于聊天的文本指令生成。   


提出了数量意识提示     
![alt text](assets/Ranni/image-2.png)     
解决customnet的问题     

对空间关系进行提示。   
可能解决cosxl的问题    
但pixart sigma已经解决   


对属性绑定进行了提示，包括(a)颜色绑定和(b)纹理绑定。为了清楚比较，随机种子被固定以保留一行中的空间排列。    

颜色编辑      
可能解决cosxl的问题    



shape editing   
![alt text](assets/Ranni/image-3.png)    

### 实践

模型架构     
- base llama model   
meta-llama/Llama-2-7b-chat-hf    

- lora    
lora_weight_ele = torch.load('models/llama2_7b_lora_element.pth', map_location='cpu')   # load an empty lora here   
lora_weight_box = torch.load('models/llama2_7b_lora_bbox.pth', map_location='cpu')    

- panel2img   
model.load_state_dict(torch.load('models/ranni_sdv21_v1.pth', map_location='cpu'), strict=False)     
ddim_sampler = DDIMSampler(model)    

官网下载Llama-2-7b-chat   
转成huggingface格式     

    mv tokenizer.model llama
    mv tokenizer_checklist.chk consolidated.00.pth params.json llama/7B

    cd /root/autodl-fs/transformers
 ​
    python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /root/autodl-fs/Chinese-LLaMA-Alpaca/model/llama/ \
    --model_size 7B \
    --output_dir /root/autodl-fs/Chinese-LLaMA-Alpaca/model/output

python demo_gradio.py


AssertionError: Torch not compiled with CUDA enabled   
torch 1.13    
nvcc 11.8      

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

需要额外安装  
basicsr   
但依赖tb-nightly    
清华源不存在tb-nightly    
pip install tb-nightly -i https://mirrors.aliyun.com/pypi/simple    



#### 参数
    control_max_t = gr.Slider(label="Control start", minimum=0, maximum=1000, value=1000, step=0)
    control_min_t = gr.Slider(label="Control stop", minimum=0, maximum=1000, value=600, step=0)
    panel_control_scale = gr.Slider(label="Control scale", minimum=0, maximum=5.0, value=0.6, step=0.1)

函数   

    run_button.click(
        fn=stage1_process, 
        inputs=[prompt, seed], 
        outputs=[box_answer, mid_result_gallery])

    run_button2.click(
        fn=stage2_process, 
        inputs=[box_answer, prompt, postfix, seed, n_prompt, guide_scale, steps, control_max_t, control_min_t, panel_control_scale, with_memory], 
        outputs=[result_gallery])

    refresh_button.click(
        fn=refresh_condition,
        inputs=[box_answer, postfix],
        outputs=[mid_result_gallery]
        )

### Box Answer如何生成？



### 报错

with memory 

    Traceback (most recent call last):
    File "/root/miniconda3/envs/ranni/lib/python3.10/site-packages/gradio/queueing.py", line 388, in call_prediction
        output = await route_utils.call_process_api(
    File "/root/miniconda3/envs/ranni/lib/python3.10/site-packages/gradio/route_utils.py", line 219, in call_process_api
        output = await app.get_blocks().process_api(
    File "/root/miniconda3/envs/ranni/lib/python3.10/site-packages/gradio/blocks.py", line 1437, in process_api
        result = await self.call_function(
    File "/root/miniconda3/envs/ranni/lib/python3.10/site-packages/gradio/blocks.py", line 1109, in call_function
        prediction = await anyio.to_thread.run_sync(
    File "/root/miniconda3/envs/ranni/lib/python3.10/site-packages/anyio/to_thread.py", line 56, in run_sync
        return await get_async_backend().run_sync_in_worker_thread(
    File "/root/miniconda3/envs/ranni/lib/python3.10/site-packages/anyio/_backends/_asyncio.py", line 2144, in run_sync_in_worker_thread
        return await future
    File "/root/miniconda3/envs/ranni/lib/python3.10/site-packages/anyio/_backends/_asyncio.py", line 851, in run
        result = context.run(func, *args)
    File "/root/miniconda3/envs/ranni/lib/python3.10/site-packages/gradio/utils.py", line 641, in wrapper
        response = f(*args, **kwargs)
    File "/home/WujieAITeam/private/lujunda/newtest/Ranni/demo_gradio.py", line 232, in stage2_process
        edit_mask=resized_edit_mask if with_memory else None, edit_intermediates=intermediates)
    UnboundLocalError: local variable 'resized_edit_mask' referenced before assignment




 
## 语言模型 llama-2-7b-chat



### 语言模型训练方式
AI模型的训练训练过程分为如下三个阶段

第一个阶段叫做无监督学习（PreTraining），就是输入大量的文本语料让GPT自己寻找语言的规律， 这样一个巨大的词向量空间就形成了，但是话说的漂亮并不一定正确。

第二个阶段叫做监督学习(Supervised Fine-Tuning,也叫微调)，就是人工标注一些语料，教会GPT什 么该说，什么不该说。（训练数据集）

第三个阶段叫做强化学习（RM，也叫奖励模型训练），就是给GPT的回答进行打分，告诉他在他 的一众回答中，哪些回答更好。（验证数据集）

第一个阶段（无监督学习），又分为了底座模型预训练，及增量预训练，它们都属于无监督学习，基座模型预训练可以查看上篇文章：使用数据预训练一个AI语言模型

本文主要来聊聊有了一个底座模型之后，如何继续使用大量文本进行增量预训练。


#### 增量预训练

增量预训练也叫领域自适应预训练（domain-adapter pretraining），即在所属领域数据上继续预训练。

主要问题是在增量预训练后可能发生灾难性遗忘。

避免灾难性遗忘主要从以下几个方面入手：

1 领域相关性

增量数据与所选基座模型的原始训练数据尽量一定的相关性。

2 新数据分布与原始数据尽量相似

领域数据和通用数据的比率，结合具体数据：10%，15%，20%的都有。

方案之一是：让无监督数据和指令数据混合，合并增量预训练和微调两个阶段。

3 降低学习率

增量预训练2e-5；指令微调需要更低1e-6；但是得多跑几轮不然学不到领域知识

4 进行warm up，

在第一轮训练的时候，每个数据点对模型来说都是新的，模型会很快地进行数据分布修正，如果这时候学习率就很大，极有可能导致开始的时候就对该数据“过拟合”，后面要通过多轮训练才能拉回来，浪费时间。当训练了一段时间（比如两轮、三轮）后，模型已经对每个数据点看过几遍了，或者说对当前的batch而言有了一些正确的先验，较大的学习率就不那么容易会使模型学偏，所以可以适当调大学习率。这个过程就可以看做是warmup。那么为什么之后还要decay呢？当模型训到一定阶段后（比如十个epoch），模型的分布就已经比较固定了，或者说能学到的新东西就比较少了。如果还沿用较大的学习率，就会破坏这种稳定性，用我们通常的话说，就是已经接近loss的local optimal了，为了靠近这个point，我们就要慢慢来。

5 对新任务中参数的变化施加惩罚

6 知识蒸馏（KD），使微调模型的预测结果接近旧模型的预测结果。


本文主要来聊聊有了一个底座模型之后，如何继续使用大量文本进行增量预训练。   
##### 合并模型
1、llama模型转换(pytorch格式转换为HuggingFace格式)

由于使用的底座模型是llama，官方公布的是PyTorch版本，为了方便后续使用，需转换为HuggingFace格式

2.合并模型   
由于原始llama模型对中文的支持不是很优秀，所以需合并一个Chinese-LLaMA-Plus-7B模型和chinese_llama_plus_lora_7b模型​


2.1、下载Chinese-LLaMA-Plus-7B模型    
 unzip chinese_llama_plus_lora_7b.zip chinese_llama_plus_lora_7b

2.2、下载chinese_alpaca_plus_lora_7b模型   
 unzip chinese_alpaca_plus_lora_7b.zip chinese_alpaca_plus_lora_7b

 这是挂两个lora

    python scripts/merge_llama_with_chinese_lora.py \
    --base_model /root/autodl-fs/Chinese-LLaMA-Alpaca/model/output \
    --lora_model /root/autodl-fs/llama_7b/chinese_llama_plus_lora_7b,/root/autodl-fs/llama_7b/chinese_alpaca_plus_lora_7b \
    --output_type huggingface \
    --output_dir /root/autodl-fs/Chinese-LLaMA-Alpaca/model/firstmergemodels


4.2、训练后文件整理

训练后的LoRA权重和配置存放于

/root/autodl-fs/Chinese-LLaMA-Alpaca/model/pt_output/pt_lora_model，可用于后续的合并流程。


合并模型

    python scripts/merge_llama_with_chinese_lora.py \
    --base_model /root/autodl-fs/Chinese-LLaMA-Alpaca/model/firstmergemodels \
    --lora_model /root/autodl-fs/Chinese-LLaMA-Alpaca/model/pt_output/pt_lora_model/lora_model \
    --output_type huggingface \
    --output_dir /root/autodl-fs/Chinese-LLaMA-Alpaca/model/ptmerge_model



## 局限
after having tried many times with different prompts and seeds, most of results coming out are not desirable,
generally speaking they still got issues with colours and numbers, I feel like the default demo examples are kinda of cherry picking, hopefully they can release the version they used in the original paper

The current released version is based on the pure SDv2.1 and it not quite stable especially for local attribute binding. We improve it by ignoring the global <eos> token for better local control, but it is still worse than the paper version. The paper version is a larger private 3B diffusion model with better image quality and local sensibility (with different text conditioning). We are still working to develop and release better version of the panel-to-image model.   



# 结尾
