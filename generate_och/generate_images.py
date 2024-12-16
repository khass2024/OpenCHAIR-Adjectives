import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionAttendAndExcitePipeline
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
import torch
import os
from datasets import load_dataset
import argparse
import json

SD_KWARGS = {
    'guidance_scale': 10,
    'num_inference_steps': 40,
    'negative_prompt': "unclear, deformed, out of image, disfiguired, body out of frame"

}

def generate_dataset(args):
    data = pd.read_csv(args.data_file)
    
    #pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16, cache_dir=args.cache_dir)
    #pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16", cache_dir=args.cache_dir)
    pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)
    BS = args.batch_size
    data_ = data.set_index(data.index // BS).copy() 
    j = 0
    for i in tqdm(range(data_.index.max()+1)):
        #image_ids = data_.loc[i].filename.tolist()
        #prompts = data_.loc[i].altered_caption.to_list()
        prompts = data_.loc[i].cap.tolist()
        out = pipe(
            prompts,
            **SD_KWARGS,
            seed=args.seed,
        )
        for img in out.images:
        #for img, img_id in zip(out.images, image_ids):
            fn = args.output_dir + str(j).rjust(6, '0') + '.jpg'
            img.save(fn)
            j += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cache-dir", type=str, default="/data/hf_cache")
    parser.add_argument("--data-file", type=str, default="/home/assaf/projects/image_captioning_with_rlhf_2/openChair/generate/captions_och_extension.csv")
    parser.add_argument("--output-dir", type=str, default="/data2/assaf/mocha/och_extension/openChair_images_random_captions_sd_xl_40_steps_10_guidance_neg_prompt/")
    args = parser.parse_args()

    generate_dataset(args)