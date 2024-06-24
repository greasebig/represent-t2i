import gradio as gr
from enum import Enum
from cutdiffusion2 import process_cutdiffusion,ModelType

model_type_choices = [ModelType.yinglou.value, 
        ]


class schedulerType(Enum):
    ori = "original_come_from_Model"
    yinglou = "dpmpp_2m_sde_karras_scheduler"
    InterDesign = "euler_a_scheduler"
    dpmpp2msdekarras = "dpmpp_2m_sde_karras_scheduler2"
    dpmpp_2m_karras_scheduler2 = "dpmpp_2m_karras_scheduler2"
    dpmpp_2m_karras_scheduler = "dpmpp_2m_karras_scheduler"
    
scheduler_type_choices = [schedulerType.ori.value, 
                           schedulerType.InterDesign.value,
                             ] #

# schedulerType.dpmpp_2m_karras_scheduler.value,
#schedulerType.yinglou.value,schedulerType.dpmpp2msdekarras.value,
#                             schedulerType.dpmpp_2m_karras_scheduler2.value,
block = gr.Blocks().queue()

with block:
        
    with gr.Row():
        with gr.Column():
            #if_used=gr.Checkbox(label="Use cutdiffusion", value=True) # show_label=True,
            #if_used.change(change_state1,[if_used],[])
            
            # 待修改 
            '''

            with gr.Row():
                input_fg = gr.Image(source='upload', type="numpy", label="Image", height=480)
                output_bg = gr.Image(type="numpy", label="Preprocessed Foreground", height=480)
            
            with gr.Row():
                upload_button_text=gr.Button(value="from txt_to_img")
                upload_button_img=gr.Button(value="from img_to_img")

                upload_button_text.click(fn=get_img_from_txt2img, inputs=[], outputs=[input_fg])
                upload_button_img.click(fn=get_img_from_img2img, inputs=[], outputs=[input_fg])
            '''
            

            model_type = gr.Dropdown(
                label="Model",
                choices=model_type_choices,
                value=ModelType.HistFilm.value,
                interactive=True,
            )

            prompt = gr.Textbox(label="Prompt",
                                value="(8k, RAW photo,masterpiece),(realistic, photo-realistic:1.37),ID photo,close up,white background,woman,1girl, solo, long hair, simple background, white background, formal, suit,  front view portrait",
                                )
            setattr(prompt,"do_not_save_to_config",True)


            scheduler_type = gr.Dropdown(
                label="采样器",
                choices=scheduler_type_choices,
                value=schedulerType.ori.value,
                interactive=True,
            )

            #待修改
            '''
            bg_source = gr.Radio(choices=[e.value for e in BGSource_1],
                                value=BGSource_1.NONE.value,
                                label="Lighting Preference (Initial Latent)", type='value')
            '''
            
            #relight_button = gr.Button(value="Relight")
            relight_button = gr.Button(value="run")

            with gr.Group():
                with gr.Row():
                    num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                    seed = gr.Number(label="Seed", value=0, precision=0)

                with gr.Row():
                    image_width = gr.Slider(label="Image Width", minimum=256, maximum=4096, value=2048, step=64)
                    image_height = gr.Slider(label="Image Height", minimum=256, maximum=4096, value=2048, step=64)
                    setattr(image_width,"do_not_save_to_config",True)
                    setattr(image_height,"do_not_save_to_config",True)

            with gr.Accordion("Advanced options", open=True):
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                setattr(steps,"do_not_save_to_config",True)
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=7.5, step=0.01)
                setattr(cfg,"do_not_save_to_config",True)
                '''
                lowres_denoise = gr.Slider(label="Lowres Denoise (for initial latent)", minimum=0.1, maximum=1.0, value=0.9, step=0.01)
                highres_scale = gr.Slider(label="Highres Scale", minimum=1.0, maximum=3.0, value=1.5, step=0.01)
                highres_denoise = gr.Slider(label="Highres Denoise", minimum=0.1, maximum=1.0, value=0.5, step=0.01)
                '''
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality')
                n_prompt = gr.Textbox(label="Negative Prompt", value='worst quality,low quality,flag,bad eye,panorama,wrong hand,bad anatomy,wrong anatomy,three people,lipstick,open mouth,deformed,distorted, disfigured,cgi,illustration, cartoon, poorly drawn,watermark,tooth,black people,landscape,bad hand,bad fingers,distorted,twisted,')
                setattr(n_prompt,"do_not_save_to_config",True)
        with gr.Column():
            result_gallery = gr.Gallery(height=832, object_fit='contain', label='Outputs')
            
    #ips = [prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, selected_model_type.model_name]
    ips = [prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, model_type,scheduler_type]
    relight_button.click(fn=process_cutdiffusion, inputs=ips, outputs=[result_gallery])
        
block.launch(server_name="0.0.0.0", server_port=8895)