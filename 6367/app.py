import gradio as gr
import numpy as np
import random
import torch
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler,DPMSolverMultistepScheduler
from enum import Enum
import time
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16


class ModelType(Enum):
    FC = "FlowMatchEulerDiscreteScheduler"
    FBC = "dpmpp_2m"

        
model_type_choices = [ModelType.FC.value, ModelType.FBC.value]



repo = "stabilityai/stable-diffusion-3-medium-diffusers"
pipe = StableDiffusion3Pipeline.from_pretrained(repo, torch_dtype=torch.float16, token='').to(device)

flow_scheduler = pipe.scheduler
dpmpp_2m_scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        use_karras_sigmas=False,
        steps_offset=1
    )







MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1344

def infer(prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps,scheduler_type, progress=gr.Progress(track_tqdm=True)):

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
        
    generator = torch.Generator().manual_seed(seed)
    if scheduler_type == "FlowMatchEulerDiscreteScheduler":
        pipe.scheduler = flow_scheduler ## 第三次肯定进入这里了，但是为什么赋值不对？ 这个global 的值好像还是euler_a_scheduler
    elif scheduler_type == "dpmpp_2m":
        pipe.scheduler = dpmpp_2m_scheduler
    # 获取当前时间戳
    current_timestamp = time.time()
    # 添加8小时的秒数(8 * 60 * 60)
    new_timestamp = current_timestamp + (8 * 60 * 60)
    # 自定义格式化字符串
    custom_format = "%Y年%m月%d日 %H时%M分%S秒"
    # 格式化修改后的时间戳
    current_time = time.strftime(custom_format, time.localtime(new_timestamp))
    print("当前时间:", current_time)

    print(f"Processing prompt: {prompt},n_prompt:{negative_prompt},width:{width},height:{height},multi_guidance_scale:{guidance_scale},scheduler_type:{scheduler_type},seed:{seed}")

    image = pipe(
        prompt = prompt, 
        negative_prompt = negative_prompt,
        guidance_scale = guidance_scale, 
        num_inference_steps = num_inference_steps, 
        width = width, 
        height = height,
        generator = generator
    ).images[0] 
    
    return image, seed

examples = [
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "An astronaut riding a green horse",
    "A delicious ceviche cheesecake slice",
]

css="""
#col-container {
    margin: 0 auto;
    max-width: 580px;
}
"""

with gr.Blocks(css=css) as demo:
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""
        # Demo [Stable Diffusion 3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium)
        Learn more about the [Stable Diffusion 3 series](https://stability.ai/news/stable-diffusion-3). Try on [Stability AI API](https://platform.stability.ai/docs/api-reference#tag/Generate/paths/~1v2beta~1stable-image~1generate~1sd3/post), [Stable Assistant](https://stability.ai/stable-assistant), or on Discord via [Stable Artisan](https://stability.ai/stable-artisan). Run locally with [ComfyUI](https://github.com/comfyanonymous/ComfyUI) or [diffusers](https://github.com/huggingface/diffusers)
        """)
        
        with gr.Row():
            
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            
            run_button = gr.Button("Run", scale=0)
        
        result = gr.Image(label="Result", show_label=False)

        with gr.Accordion("Advanced Settings", open=True):
            
            negative_prompt = gr.Text(
                label="Negative prompt",
                max_lines=1,
                placeholder="Enter a negative prompt",
            )

            scheduler_type = gr.Dropdown(
                label="采样器",
                choices=model_type_choices,
                value=ModelType.FC.value,
                interactive=True,
            )
            
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            
            with gr.Row():
                
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=64,
                    value=1024,
                )
                
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=64,
                    value=1024,
                )
            
            with gr.Row():
                
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=5.0,
                )
                
                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=28,
                )
        
        gr.Examples(
            examples = examples,
            inputs = [prompt]
        )
    gr.on(
        triggers=[run_button.click, prompt.submit, negative_prompt.submit],
        fn = infer,
        inputs = [prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps,scheduler_type],
        outputs = [result, seed]
    )

demo.launch(server_name="0.0.0.0", server_port=8895)