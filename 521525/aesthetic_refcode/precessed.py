import cv2
import numpy as np
import torch
import onnxruntime as rt
import transformers
from optimum.pipelines import pipeline
from huggingface_hub import hf_hub_download
from PIL import Image
import argparse
import os
import time
import shutil
from tqdm import tqdm
import threading
import queue

score= None
processed_images = []
 
def predict(images):
    processed_images = []
    for img in images:
        img = np.array(Image.open(img))
        channels = cv2.split(img)
        num_channels = len(channels)
        if num_channels == 4:   
            img = cv2.merge([channels[2], channels[1], channels[0]])
        elif num_channels == 3:
            img = cv2.merge([channels[2], channels[1], channels[0], np.ones_like(channels[0])])
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif num_channels == 1:
            img = np.expand_dims(img, axis=-1)
            img = np.repeat(img,3,axis=-1)
        img = img.astype(np.float32) / 255
        s = 1024 
        h, w = img.shape[:-1]     
        h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
        ph, pw = s - h, s - w
        img_input = np.zeros([s, s, 3], dtype=np.float32)
        img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(img, (w, h))
        img_input = np.transpose(img_input, (2, 0, 1))
        img_input = img_input[np.newaxis, :]
        processed_images.append(img_input)
    if len(processed_images) == 0: 
        return np.array([])
    img_batch = np.stack(processed_images, axis=0) 
    img_batch = np.squeeze(img_batch, axis=1)
    pred = model.run(None, {"img": img_batch})
    return pred    

def predict2(img):
    result = pipe(images=img)   
    result = np.array([d[1]['score'] for d in result],dtype=np.float32)
    result = result.reshape(-1,1)
    return result

def processing(args):
    image_files = [os.path.join(args.file_path, f) for f in os.listdir(args.file_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for i in tqdm(range(0, len(image_files), args.batch_size)):
        batch = image_files[i:i + args.batch_size]       
        results = predict(batch)
        results2 = predict2(batch)
        results = results[0] - results2
        for idx, result in enumerate(results):      
            predictions = result
            if predictions[0] >= 0.9:
                try:
                    shutil.copy(batch[idx],args.output_path + "/masterpiece/"+os.path.basename(batch[idx]))
                except FileNotFoundError:
                    os.makedirs(args.output_path + "/masterpiece")
                    shutil.copy(batch[idx],args.output_path + "/masterpiece/"+os.path.basename(batch[idx]))
            elif predictions[0] < 0.9 and predictions[0] >=0.8:
                try:
                    shutil.copy(batch[idx],args.output_path + "/high quality/"+os.path.basename(batch[idx]))
                except FileNotFoundError:
                    os.makedirs(args.output_path + "/high quality")
                    shutil.copy(batch[idx],args.output_path + "/high quality/"+os.path.basename(batch[idx]))
            elif predictions[0] < 0.8 and predictions[0] >=0.5:   
                try:
                    shutil.copy(batch[idx],args.output_path + "/normal quality/"+os.path.basename(batch[idx]))
                except FileNotFoundError:
                    os.makedirs(args.output_path + "/normal quality")
                    shutil.copy(batch[idx],args.output_path + "/normal quality/"+os.path.basename(batch[idx]))
            elif predictions[0] < 0.5 and predictions[0] >=0.2:
                try:
                    shutil.copy(batch[idx],args.output_path + "/low quality/"+os.path.basename(batch[idx]))
                except FileNotFoundError:
                    os.makedirs(args.output_path + "/low quality")
                    shutil.copy(batch[idx],args.output_path + "/low quality/"+os.path.basename(batch[idx]))
            elif predictions[0] < 0.2:
                try:
                    shutil.copy(batch[idx],args.output_path + "/worst quality/"+os.path.basename(batch[idx]))
                except FileNotFoundError:
                    os.makedirs(args.output_path + "/worst quality")
                    shutil.copy(batch[idx],args.output_path + "/worst quality/"+os.path.basename(batch[idx])) 
    print("Classification and sorting complete.")    
   


if __name__ == "__main__":
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path",type=str)
    parser.add_argument("--output_path",type=str)
    parser.add_argument("--batch_size",type=int,default=4)
    parser.add_argument("--device_number",type=int,default=1)
    args = parser.parse_args()
    pipe = pipeline("image-classification", model="Laxhar/anime_aesthetic_variant" ,device='cuda')
    model_path = hf_hub_download(repo_id="Laxhar/anime_model", filename="anime_aesthetic.onnx")
    model = rt.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    processing(args)
    end_time = time.perf_counter()
    print(f"程序运行所用时间: {end_time - start_time} 秒")