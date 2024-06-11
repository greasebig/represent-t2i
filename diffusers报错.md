# 使用 StableDiffusionXLPipeline.from_single_file 读取原始civitai文件常常报错
    LocalEntryNotFoundError                   Traceback (most recent call last)
    File ~/miniconda3/envs/pixart/lib/python3.9/site-packages/transformers/utils/hub.py:389, in cached_file(path_or_repo_id, filename, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
        387 try:
        388     # Load from URL or cache if already cached
    --> 389     resolved_file = hf_hub_download(
        390         path_or_repo_id,
        391         filename,
        392         subfolder=None if len(subfolder) == 0 else subfolder,
        393         repo_type=repo_type,
        394         revision=revision,
        395         cache_dir=cache_dir,
        396         user_agent=user_agent,
        397         force_download=force_download,
        398         proxies=proxies,
        399         resume_download=resume_download,
        400         token=token,
        401         local_files_only=local_files_only,
        402     )
        403 except GatedRepoError as e:

    File ~/miniconda3/envs/pixart/lib/python3.9/site-packages/huggingface_hub/utils/_validators.py:119, in validate_hf_hub_args.<locals>._inner_fn(*args, **kwargs)
        117     kwargs = smoothly_deprecate_use_auth_token(fn_name=fn.__name__, has_token=has_token, kwargs=kwargs)
    --> 119 return fn(*args, **kwargs)
    ...
    1504         "text_encoder_2": text_encoder_2,
    1505     }
    1507 return

    ValueError: With local_files_only set to True, you must first locally save the text_encoder_2 and tokenizer_2 in the following path: laion/CLIP-ViT-bigG-14-laion2B-39B-b160k with `pad_token` set to '!'.

# 使用模型转换脚本    
    python scripts/convert_original_stable_diffusion_to_diffusers.py \
    --checkpoint_path \
    --original_config_file  \
    --dump_path  \
    --from_safetensors    

不开代理用不了   

    OSError: We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like laion/CLIP-ViT-bigG-14-laion2B-39B-b160k is not the path to a directory containing a file named config.json.
    Checkout your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

    During handling of the above exception, another exception occurred:

    Traceback (most recent call last):
    File "/teams/ai_model_1667305326/WujieAITeam/private/lujunda/newlytest/diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py", line 160, in <module>
        pipe = download_from_original_stable_diffusion_ckpt(
    File "/root/miniconda3/envs/pixart/lib/python3.9/site-packages/diffusers/pipelines/stable_diffusion/convert_from_ckpt.py", line 1748, in download_from_original_stable_diffusion_ckpt
        text_encoder_2 = convert_open_clip_checkpoint(
    File "/root/miniconda3/envs/pixart/lib/python3.9/site-packages/diffusers/pipelines/stable_diffusion/convert_from_ckpt.py", line 939, in convert_open_clip_checkpoint
        raise ValueError(
    ValueError: With local_files_only set to False, you must first locally save the configuration in the following path: 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'.

开代理才能转，几乎和from_single_file一样    
都要使用'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'.    

但是在学校4090代理会报错    

This program can only be run on AMD64 processors with v3 microarchitecture support.

Welcome to Ubuntu 20.04.5 LTS (GNU/Linux 5.15.0-107-generic x86_64)

直接转发

Downloading (…)_encoder/config.json: 100%|█████████████████| 633/633 [00:00<00:00, 150kB/s]
Downloading (…)okenizer_config.json: 100%|█████████████████| 824/824 [00:00<00:00, 211kB/s]
Downloading tokenizer/vocab.json: 100%|████████████████| 1.06M/1.06M [00:01<00:00, 713kB/s]
Downloading tokenizer/merges.txt: 100%|█████████████████| 525k/525k [00:00<00:00, 2.55MB/s]
Downloading (…)cial_tokens_map.json: 100%|█████████████████| 460/460 [00:00<00:00, 457kB/s]
You have disabled the safety checker for <class 'diffusers.pipel

这几个只要缓存有，可以不开代理      

 deprecate("direct config name access", "1.0.0", deprecation_message, standard_warn=False)

 pip install diffusers -U



    FutureWarning: `Transformer2DModelOutput` is deprecated and will be removed in version 1.0.0. Importing `Transformer2DModelOutput` from `diffusers.models.transformer_2d` is deprecated and this will be removed in a future version. Please use `from diffusers.models.modeling_outputs import Transformer2DModelOutput`, instead.
    deprecate("Transformer2DModelOutput", "1.0.0", deprecation_message)
    /home/kongzhi/miniconda3/envs/diffuserslowacce/lib/python3.9/site-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
    warnings.warn(
    You have disabled the safety checker for <class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline'> by passing `safety_checker=None`. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .
    /home/kongzhi/miniconda3/envs/diffuserslowacce/lib/python3.9/site-packages/diffusers/configuration_utils.py:140: FutureWarning: Accessing config attribute `requires_safety_checker` directly via 'StableDiffusionPipeline' object attribute is deprecated. Please access 'requires_safety_checker' over 'StableDiffusionPipeline's config object instead, e.g. 'scheduler.config.requires_safety_checker'.
    deprecate("direct config name access", "1.0.0", deprecation_message, standard_warn=False)


可能是diffusers的本地代码版本落后    
这个不是


也可能是自己修改的safe导致冲突   
这个可能是我改了conda里面      

或者是yaml版本不匹配 ？？？？     


我服气

这次是转原始inpaint模型      

直接拉最新diffusers重装环境 py310      


    /home/kongzhi/miniconda3/envs/diffusers/lib/python3.10/site-packages/diffusers/models/transformers/transformer_2d.py:34: FutureWarning: `Transformer2DModelOutput` is deprecated and will be removed in version 1.0.0. Importing `Transformer2DModelOutput` from `diffusers.models.transformer_2d` is deprecated and this will be removed in a future version. Please use `from diffusers.models.modeling_outputs import Transformer2DModelOutput`, instead.
    deprecate("Transformer2DModelOutput", "1.0.0", deprecation_message)
    Traceback (most recent call last):
    File "/data/master/lujunda/207/diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py", line 160, in <module>
        pipe = download_from_original_stable_diffusion_ckpt(
    File "/home/kongzhi/miniconda3/envs/diffusers/lib/python3.10/site-packages/diffusers/pipelines/stable_diffusion/convert_from_ckpt.py", line 1480, in download_from_original_stable_diffusion_ckpt
        set_module_tensor_to_device(unet, param_name, "cpu", value=param)
    File "/home/kongzhi/miniconda3/envs/diffusers/lib/python3.10/site-packages/accelerate/utils/modeling.py", line 358, in set_module_tensor_to_device
        raise ValueError(
    ValueError: Trying to set a tensor of shape torch.Size([320, 9, 3, 3]) in "weight" (which has shape torch.Size([320, 4, 3, 3])), this look incorrect.


难道这个脚本无法转inpaint???        

发现最新环境无法转

旧环境转成功

一下这个不是报错


    python scripts/convert_original_stable_diffusion_to_diffusers.py     --checkpoint_path /data/master/lujunda/207/stable-diffusion-2-inpainting/512-inpainting-ema.safetensors    --original_config_file  /data/master/lujunda/207/stable-diffusion-2-inpainting/512-inpainting-ema.yaml    --dump_path  /data/master/lujunda/207/stable-diffusion-2-inpainting-diffusers2    --from_safetensors  --image_size 512
    /home/kongzhi/miniconda3/envs/diffuserslowacce/lib/python3.9/site-packages/diffusers/models/transformers/transformer_2d.py:34: FutureWarning: `Transformer2DModelOutput` is deprecated and will be removed in version 1.0.0. Importing `Transformer2DModelOutput` from `diffusers.models.transformer_2d` is deprecated and this will be removed in a future version. Please use `from diffusers.models.modeling_outputs import Transformer2DModelOutput`, instead.
    deprecate("Transformer2DModelOutput", "1.0.0", deprecation_message)
    /home/kongzhi/miniconda3/envs/diffuserslowacce/lib/python3.9/site-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
    warnings.warn(
    You have disabled the safety checker for <class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline'> by passing `safety_checker=None`. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .
    /home/kongzhi/miniconda3/envs/diffuserslowacce/lib/python3.9/site-packages/diffusers/configuration_utils.py:140: FutureWarning: Accessing config attribute `requires_safety_checker` directly via 'StableDiffusionPipeline' object attribute is deprecated. Please access 'requires_safety_checker' over 'StableDiffusionPipeline's config object instead, e.g. 'scheduler.config.requires_safety_checker'.
    deprecate("direct config name access", "1.0.0", deprecation_message, standard_warn=False)
    (diffuserslowacce) kongzhi@ncms-kongzhi-02:/data/master/lujunda/diffusers-main$ 


因为禁用safe 转出来的没有









# 结尾




