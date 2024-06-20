import cv2
import numpy as np
import gradio as gr
import torch
import onnxruntime as rt
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor
from PIL import Image
from optimum.pipelines import pipeline


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
score= None


def predict(img,balance):
    score2=predict2(img)
    img = Image.open(img)
    img = np.array(img)
    img = img.astype(np.float32) / 255
    s = 1024
    h, w = img.shape[:-1]
    h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
    img_input = cv2.resize(img, (w, h))
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input[np.newaxis, :]
    pred = model.run(None, {"img": img_input})[0].item()
    pred -= (score2*balance)
    if pred >= 0.9:
        _image_class = "masterpiece"
    elif pred < 0.9 and pred>=0.8:
       _image_class = "high quality"
    elif pred < 0.8 and pred>=0.5:   
        _image_class = "normal quality"
    elif pred < 0.5 and pred>=0.2:
        _image_class = "low quality"
    elif pred < 0.2:
        _image_class = "worst quality" 
    return pred,_image_class

def predict2(img):
    result = pipe(images=[img])
    prediction_single = result[0]
    return round([p for p in prediction_single if p['label'] == 'low'][0]['score'], 2)


css = """
#run-button {
  background: coral;
  color: white;
}
"""

with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
            <div><h1 >Anime Aesthetic Variant</h1></div>
        </div>
        """
    )
    with gr.Row():
        input_image = gr.Image(
            label='Image',
            sources='upload',
            type='filepath',
        )
        with gr.Column():
            output_score = gr.Number(
                label="score"
            )
            image_class = gr.Text(
                label='image-class',
                interactive=False  ,    
            )
            score_balance = gr.Slider(
                label='score-balance',
                minimum = 0,
                maximum = 1,
                value=1,
                info='Composition confidence'
            )
       
    
    with gr.Column():
        run_button = gr.Button(value='Run',elem_id='run-button')

 

    with gr.Column():
        masterspace = gr.Examples(
            label='masterpiece',
            examples = [[f"test/masterspace/{x:02d}.jpg"] for x in range(0, 7)],
            inputs=input_image,
            outputs=output_score,
            cache_examples=False
        )
        high = gr.Examples(
            label='high quality',
            examples = [[f"test/high quality/{x:02d}.jpg"] for x in range(0, 7)],
            inputs=input_image,
            outputs=output_score,
            cache_examples=False
        )
        normal = gr.Examples(
            label='normal quality',
            examples = [[f"test/normal quality/{x:02d}.jpg"] for x in range(0, 7)],
            inputs=input_image,
            outputs=output_score,
            cache_examples=False
        )
        low = gr.Examples(
            label='low quality',
            examples = [[f"test/low quality/{x:02d}.jpg"] for x in range(0, 7)],
            inputs=input_image,
            outputs=output_score,
            cache_examples=False
        )
        worst = gr.Examples(
            label='worst quality',
            examples = [[f"test/worst quality/{x:02d}.jpg"] for x in range(1, 7)],
            inputs=input_image,
            outputs=output_score,
            cache_examples=False
        )
    

    run_button.click(predict,inputs=[input_image,score_balance],outputs=[output_score,image_class])
  
if __name__ == "__main__":
    pipe = pipeline("image-classification", model="Laxhar/anime_aesthetic_variant",device='cpu',accelerator="ort")
    model_path = hf_hub_download(repo_id="Laxhar/anime_model", filename="anime_aesthetic.onnx")
    model = rt.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    demo.launch()