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

使用    
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