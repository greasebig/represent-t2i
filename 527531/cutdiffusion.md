
# webui环境运行报错
1 机       

    File "/root/miniconda3/envs/webui310/lib/python3.10/site-packages/transformers/models/clip/modeling_clip.py", line 229, in forward
        position_embeddings = self.position_embedding(position_ids)

    File "/root/miniconda3/envs/webui310/lib/python3.10/site-packages/torch/nn/modules/sparse.py", line 162, in forward
        return F.embedding(
    File "/root/miniconda3/envs/webui310/lib/python3.10/site-packages/torch/nn/functional.py", line 2233, in embedding

    RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.HalfTensor instead (while checking arguments for embedding)

网上说需要改变量数值类型，太蠢了     
转成long类型才能作为nn.embedding的输入      


3 机



    RuntimeError: Failed to import transformers.models.clip.modeling_clip because of the following error (look up to see its traceback):

        CUDA Setup failed despite GPU being available. Please run the following command to get more information:

        python -m bitsandbytes

        Inspect the output of the command and see if you can locate CUDA libraries. You might need to add them
        to your LD_LIBRARY_PATH. If you suspect a bug, please take the information from python -m bitsandbytes
        and open an issue at: https://github.com/TimDettmers/bitsandbytes/issues

网上解决方法：

pip install transformers -U


依旧

RuntimeError: Failed to import transformers.models.clip.modeling_clip because of the following error (look up to see its traceback):
Failed to import transformers.generation.utils because of the following error (look up to see its traceback):




 pip install bitsandbytes -U


# 论文信息

[提交日期：2024 年 4 月 23 日]     
CutDiffusion：一种简单、快速、廉价且强大的扩散外推方法

CutDiffusion: A Simple, Fast, Cheap, and Strong Diffusion Extrapolation Method

将大型预训练的低分辨率扩散模型转换为更高分辨率的需求，即扩散外推，可显著提高扩散的适应性。我们提出了无需调整的 CutDiffusion，旨在简化和加速扩散外推过程，使其更经济实惠并提高性能。CutDiffusion 遵循现有的逐块外推，但将标准块扩散过程分为专注于全面结构去噪的初始阶段和专注于特定细节细化的后续阶段。全面的实验凸显了 CutDiffusion 的众多强大优势：（1）简单的方法构建，无需第三方参与即可实现简洁的高分辨率扩散过程；（2）通过单步高分辨率扩散过程实现快速的推理速度，并且需要更少的推理块；（3）由于逐块推理和全面结构去噪期间的块数更少，GPU 成本低廉；（4）强大的生成性能，源于对特定细节细化


林明宝，林志航，詹文懿，曹柳娟，季蓉蓉


https://github.com/lmbxmu/CutDiffusion

1Skywork AI
2Xiamen University

昆仑天工的


![alt text](assets_picture/cutdiffusion/image.png)

1024         

整个十秒钟     


![alt text](assets_picture/cutdiffusion/image-1.png)

还是不太懂      
还打乱顺序      
从demo看确实都是1024倍率    







# 按照readme重装环境

diffusers                 0.21.4


    Traceback (most recent call last):
    File "/teams/ai_model_1667305326/WujieAITeam/private/lujunda/newlytest/CutDiffusion/cutdiffusion.py", line 1304, in <module>
        pipe = CutDiffusionSDXLPipeline.from_single_file(args.model_ckpt, torch_dtype=torch.float16).to("cuda")
    File "/root/miniconda3/envs/CutDiffusion/lib/python3.9/site-packages/diffusers/loaders.py", line 2268, in from_single_file
        raise ValueError(f"Unhandled pipeline class: {pipeline_name}")
    ValueError: Unhandled pipeline class: CutDiffusionSDXLPipeline



Successfully installed diffusers-0.28.2 huggingface-hub-0.23.3


tokenizers 0.14.1 requires huggingface_hub<0.18,>=0.16.4, but you have huggingface-hub 0.23.3 which is incompatible.

transformers 4.34.1 requires tokenizers<0.15,>=0.14, but you have tokenizers 0.19.1 which is incompatible.

全部升级


    deprecate("Transformer2DModelOutput", "1.0.0", deprecation_message)
    text_encoder/config.json: 100%|████████████████████████████████████████| 565/565 [00:00<00:00, 26.1kB/s]
    scheduler/scheduler_config.json: 100%|█████████████████████████████████| 479/479 [00:00<00:00, 39.2kB/s]
    tokenizer/tokenizer_config.json: 100%|█████████████████████████████████| 737/737 [00:00<00:00, 20.8kB/s]
    model_index.json: 100%|████████████████████████████████████████████████| 609/609 [00:00<00:00, 22.7kB/s]
    text_encoder_2/config.json: 100%|██████████████████████████████████████| 575/575 [00:00<00:00, 24.6kB/s]
    tokenizer/special_tokens_map.json: 100%|███████████████████████████████| 472/472 [00:00<00:00, 37.5kB/s]
    tokenizer/vocab.json: 100%|████████████████████████████████████████| 1.06M/1.06M [00:00<00:00, 2.68MB/s]
    tokenizer/merges.txt: 100%|███████████████████████████████████████████| 525k/525k [00:00<00:00, 961kB/s]
    tokenizer_2/special_tokens_map.json: 100%|██████████████████████████████| 460/460 [00:00<00:00, 274kB/s]
    tokenizer_2/tokenizer_config.json: 100%|████████████████████████████████| 725/725 [00:00<00:00, 342kB/s]
    vae/config.json: 100%|██████████████████████████████████████████████████| 642/642 [00:00<00:00, 338kB/s]
    unet/config.json: 100%|█████████████████████████████████████████████| 1.68k/1.68k [00:00<00:00, 841kB/s]
    vae_1_0/config.json: 100%|██████████████████████████████████████████████| 607/607 [00:00<00:00, 278kB/s]
    Fetching 17 files: 100%|████████████████████████████████████████████████| 17/17 [00:01<00:00,  9.21it/s]
    Loading pipeline components...:   0%|                                             | 0/7 [00:00<?, ?it/s]Some weights of the model checkpoint were not used when initializing CLIPTextModel: 
    ['text_model.embeddings.position_ids']
    Loading pipeline components...: 100%|█████████████████████████████████████| 7/7 [00:04<00:




FutureWarning: `Transformer2DModelOutput` is deprecated and will be removed in version 1.0.0. Importing `Transformer2DModelOutput` from `diffusers.models.transformer_2d` is deprecated and this will be removed in a future version. Please use `from diffusers.models.modeling_outputs import Transformer2DModelOutput`, instead.


终于可以了       



    #pipe = CutDiffusionSDXLPipeline.from_pretrained(args.model_ckpt, torch_dtype=torch.float16).to("cuda")
    pipe = CutDiffusionSDXLPipeline.from_single_file(args.model_ckpt, torch_dtype=torch.float16).to("cuda")




scheduler

    EulerDiscreteScheduler {
    "_class_name": "EulerDiscreteScheduler",
    "_diffusers_version": "0.28.2",
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "clip_sample": false,
    "final_sigmas_type": "zero",
    "interpolation_type": "linear",
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "rescale_betas_zero_snr": false,
    "sample_max_value": 1.0,
    "set_alpha_to_one": false,
    "sigma_max": null,
    "sigma_min": null,
    "skip_prk_steps": true,
    "steps_offset": 1,
    "timestep_spacing": "leading",
    "timestep_type": "discrete",
    "trained_betas": null,
    "use_karras_sigmas": false
    }





# gradio demo编写


    model_type = gr.Dropdown(
        label="Model",
        choices=model_type_choices,
        value=ModelType.FC.value,
        interactive=True,
    )


类

    from enum import Enum
    class ModelType(Enum):
        FC = ""
        FBC = ""

        @property
        def model_name(self) -> str:
            if self == ModelType.FC:
                return ".safetensors"
            else:
                assert self == ModelType.FBC
                return ".safetensors"
            
    model_type_choices = [ModelType.FC.value, ModelType.FBC.value]





调用设置

    selected_model_type = ModelType(model_type.value)
            print(selected_model_type.model_name)
            #ips = [prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, selected_model_type.model_name]

            不能直接字符串


            ips = [prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, model_type]
            relight_button.click(fn=process_cutdiffusion, inputs=ips, outputs=[result_gallery])


            input 放入这种类型

这个错误发生时,Gradio库(用于在Stable Diffusion网络UI中创建用户界面)遇到了输入类型的问题。错误消息表示Gradio期望一个带有_id属性的对象,但实际上收到了一个字符串。     
gradio/blocks.py文件中的这一行"inputs": [block._id for block in inputs],试图迭代inputs列表并收集列表中每个对象的_id属性。然而,它似乎inputs列表中有一个或多个元素是字符串,而不是带有_id属性的对象。


模型传参时候再转字符串输入模型


    #selected_model_type = ModelType(model_type.value)
    这个报错是str


    selected_model_type = ModelType(model_type)
    print(selected_model_type.model_name)

    results = process(prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, selected_model_type.model_name)




或者使用arg传入     
类似iclight

    args = ICLightArgs.fetch_from(p)
    if not args.enabled:
        return

    if isinstance(p, StableDiffusionProcessingTxt2Img) and p.enable_hr:
        raise NotImplementedError("Hires-fix is not yet supported in A1111.")

    self.apply_ic_light(p, args)




    def apply_ic_light(
        p: StableDiffusionProcessing,
        args: ICLightArgs,
    ):
        device = devices.get_device_for("ic_light")
        dtype = devices.dtype_unet

        # Load model
        unet_path = os.path.join(models_path, "unet", args.model_type.model_name)

## 清除ui_config

File "/teams/ai_model_1667305326/WujieAITeam/private/lujunda/newlytest/a1111webui193/stable-diffusion-webui/extensions/IC-Light-sd-webui/cutdiffusion.py", line 499, in check_inputs
    raise ValueError(f"the larger one of `height` and `width` has to be divisible by 1024 but are {height} and {width}.")
ValueError: the larger one of `height` and `width` has to be divisible by 1024 but are 640 and 512.

同时只能大分辨率生成



你可以像这样使用它：

    obj = gr.Checkbox(label="Some label",value=True)
    setattr(obj,"do_not_save_to_config",True)
唯一需要注意的是，您需要手动删除 ui-config.json 中该元素的任何现有条目。但以后不会再写入该条目。

 ui-config.json 在根目录




## webui环境报错
逐个对py包     

pip install omegaconf~=2.3.0


    Attempting uninstall: omegaconf
        Found existing installation: omegaconf 2.2.3
        Uninstalling omegaconf-2.2.3:
        Successfully uninstalled omegaconf-2.2.3
    Successfully installed omegaconf-2.3.0


    Installing collected packages: accelerate
    Attempting uninstall: accelerate
        Found existing installation: accelerate 0.21.0
        Uninstalling accelerate-0.21.0:
        Successfully uninstalled accelerate-0.21.0
    Successfully installed accelerate-0.23.0


pip install transformers~=4.34.0


diffusers 0.27.2 requires huggingface-hub>=0.20.2, but you have huggingface-hub 0.17.3 which is incompatible.

tokenizers 0.14.1 requires huggingface_hub<0.18,>=0.16.4, but you have huggingface-hub 0.23.3 which is incompatible.


pip install tokenizers -U

pip install transformers -U

后面这些没对



    tqdm
    einops
    matplotlib
    gradio
    gradio_imageslider
    opencv-python


运行到一半报错

    noise_pred = self.unet(

    File "/root/miniconda3/envs/webui310/lib/python3.10/site-packages/diffusers/models/unets/unet_2d_condition.py", line 1159, in forward
        emb = emb + aug_emb if aug_emb is not None else emb
    RuntimeError: The size of tensor a (8) must match the size of tensor b (2) at non-singleton dimension 0



## 从webui搬到原生gradio

    gradio/queueing.py", line 161, in attach_data
        raise ValueError("Event not found", event_id)
    ValueError: ('Event not found', '8a56c3f2dbf646c2ae2b7dd8aeaf9104')


    blocks._queue.attach_data(body)
    Fsite-packages/gradio/queueing.py", line 161, in attach_data
        raise ValueError("Event not found", event_id)
    ValueError: ('Event not found', '4fbbf417b6be478286ac55239c868dfd')
    ERROR:    Exception in ASGI application

@AntroSafin我降级到“gradio<4.0”（最新版本是 3.50.2）并且我不再看到这个问题，即使没有通过enable_queue=False。

After hours of debug, it turns out I was just missing proxy_buffering off; in location /

设置share并不行       


unset http_proxy     
unset https_proxy

unset all_proxy



降级？    

gradio                    4.8.0

pip install gradio==3.50.2

IMPORTANT: You are using gradio version 3.50.2, however version 4.29.0 is available, please upgrade.----------



终于运行进去了     



把queue去掉？？？



## 还是运行到一半报错



noise_pred = self.unet(

emb = emb + aug_emb if aug_emb is not None else emb
RuntimeError: The size of tensor a (8) must match the size of tensor b (2) at non-singleton dimension 0


和webui环境一样的错误     



尝试终端测试    

不会有问题    


返回运行gradio

image[0].save(f'{result_path}/{prompt}_{0}.png')
AttributeError: 'numpy.ndarray' object has no attribute 'save'

推理完了20步


有些奇怪，运行了一次源码就能用了    

emb = emb + aug_emb if aug_emb is not None else emb
RuntimeError: The size of tensor a (8) must match the size of tensor b (2) at non-singleton dimension 0

又有错？


cfg小数，step20         


好像是因为multiple cfg的特殊设置

    Image.fromarray(image[0]).save(f'{result_path}/{prompt}_{0}.png')
    File "/root/miniconda3/envs/CutDiffusion/lib/python3.9/site-packages/PIL/Image.py", line 3134, in fromarray
        raise TypeError(msg) from e
    TypeError: Cannot handle this data type: (1, 1, 3), <f4


该成pil     


 data = self.postprocess_data(fn_index, result["prediction"], state)
  File "/root/miniconda3/envs/CutDiffusion/lib/python3.9/site-packages/gradio/blocks.py", line 1447, in postprocess_data
    prediction_value = block.postprocess(prediction_value)
  File "/root/miniconda3/envs/CutDiffusion/lib/python3.9/site-packages/gradio/components/gallery.py", line 181, in postprocess
    for img in y:
TypeError: 'Image' object is not iterable


修改返回的np变量的维度

否则生成多张空白图


改batch view???





## 实现
参考webui的iclight diffusers加载版本    
再参考iclight源码，

显存缓存机制被claude优化了一点


    if not 'global_ckpt' in globals():
        global global_ckpt
        global_ckpt = {}
    if model_name in global_ckpt:  # 如果保存有，而且同名，复用
        pipe = global_ckpt[model_name]
    else: # 加载新的，或者不存在
        global_ckpt = {}
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        global_ckpt[model_name] = CutDiffusionSDXLPipeline.from_single_file(model_name, torch_dtype=torch.float16).to("cuda")
        pipe = global_ckpt[model_name]




## 写文件



    class StreamTee(object):
        """
        将输出重定向到多个流,例如sys.stdout到终端和日志文件。
        """
        def __init__(self, *streams):
            self.streams = streams
            
        def write(self, data):
            for stream in self.streams:
                stream.write(data)
                
        def flush(self):
            for stream in self.streams:
                stream.flush()


    log_file = open("output.txt", "a+")
    # 重定向标准输出到两个地方
    sys.stdout = StreamTee(sys.stdout, log_file)

    selected_model_type = ModelType(model_type)
    print(selected_model_type.model_name)
    print(scheduler_type)

    results = process(prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, float(cfg), selected_model_type.model_name,scheduler_type)
    # 关闭日志文件
    log_file.close()


stream.write(data)
ValueError: I/O operation on closed file.

    # StreamTee类的定义
    class StreamTee(object):
        """
        将输出重定向到多个流,例如sys.stdout到终端和日志文件。
        """
        def __init__(self, *streams):
            self.streams = streams
            
        def write(self, data):
            for stream in self.streams:
                stream.write(data)
                stream.flush()  # 及时将缓冲区数据写入文件
                
        def flush(self):
            pass  # 不需要执行任何操作

在这个版本中,我们没有显式地关闭日志文件,而是依赖Python在程序退出时自动关闭所有打开的文件。另外,我们在StreamTee的write方法中添加了stream.flush()语句,这样可以确保每次写入操作后,缓冲区的数据都会及时刷新到文件中,避免数据丢失。

通过这种方式,我们可以避免在关闭文件后继续向其写入数据,从而规避ValueError: I/O operation on closed file.错误的发生。



刚才你写的程序执行后，比如，
当前时间:当前时间:  2024年06月07日 14时51分40秒2024年06月07日 14时51分40秒

Edit
看起来是由于多线程或多进程同时向文件写入导致的乱序问题。在你之前的代码中,我们重定向了sys.stdout,这意味着所有的print语句都会通过多个流(终端和文件)输出。由于多个线程或进程同时向文件写入,可能会导致写入操作交错,从而出现乱序的情况。

为了解决这个问题,我们可以使用线程安全的队列来确保记录的有序性。下面是修改后的代码:

    python

    Copy code
    import sys
    import threading
    import queue

    # 创建一个线程安全的队列
    log_queue = queue.Queue()

    # 以追加模式打开文件,如果文件不存在则创建
    log_file = open("output.txt", "a+")

    # 定义一个线程,从队列中取出消息并写入文件
    def log_worker():
        while True:
            record = log_queue.get()
            if record is None:
                break
            log_file.write(record + '\n')
            log_file.flush()
            log_queue.task_done()

    # 启动日志记录线程
    log_thread = threading.Thread(target=log_worker, daemon=True)
    log_thread.start()

    # 重定向标准输出到队列
    sys.stdout = StreamTeeQueue(sys.stdout, log_queue)

    # 现在所有的print语句都会输出到终端和文件
    print("Hello, World!")
    print(f"当前时间: {datetime.datetime.now()}")

    # 等待队列中的消息全部写入文件
    log_queue.join()
    # 发送结束信号
    log_queue.put(None)
    # 等待日志线程结束
    log_thread.join()

    # 关闭日志文件
    log_file.close()

    # StreamTeeQueue类的定义
    class StreamTeeQueue(object):
        """
        将输出重定向到多个流,包括sys.stdout到终端和日志队列。
        """
        def __init__(self, stream, queue):
            self.stream = stream
            self.queue = queue

        def write(self, data):
            self.stream.write(data)
            self.queue.put(data)

        def flush(self):
            self.stream.flush()
在这个版本中,我们创建了一个线程安全的队列log_queue。所有的print语句将通过StreamTeeQueue类的write方法将消息写入终端和队列。我们启动了一个单独的线程log_worker,它会从队列中取出消息并顺序地写入文件。

在程序退出之前,我们需要等待队列中的所有消息都被写入文件。我们首先调用log_queue.join()等待队列中的所有任务完成,然后发送None作为结束信号,最后等待日志线程结束。

通过这种方式,我们将写入文件的操作从主线程中分离出来,由单独的线程顺序处理,从而避免了多线程同时写入导致的乱序问题。

执行这个程序,你应该能看到日志文件中的记录是有序的,即使在终端上可能由于多线程输出而出现乱序。

### 解决
不如写个logger

或者就是终端输入命令时候做下改变



## 关于deepcopy和全局变量的正确赋值和正确释放

原来是切换是命名错误没跳进if    

依然是正确的    

    if not 'global_ckpt' in globals():
        global global_ckpt
        global_ckpt = {}
        global global_scheduler
        global_scheduler = {}
    if model_name in global_ckpt:  # 如果保存有，而且同名，复用
        pipe = global_ckpt[model_name]
    else: # 加载新的，或者不存在
        global_ckpt = {}
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        global_ckpt[model_name] = CutDiffusionSDXLPipeline.from_single_file(model_name, torch_dtype=torch.float16).to("cuda")
        pipe = global_ckpt[model_name]
        global_scheduler[model_name] = copy.deepcopy(pipe.scheduler)


果然啊        







# 结尾
