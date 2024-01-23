import os
import argparse

from diffusers import IFImg2ImgPipeline, IFImg2ImgSuperResolutionPipeline, DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch
from PIL import Image
import requests
from io import BytesIO


def enhance(image_dir, save_dir, prompt):

    images = os.listdir(image_dir)

    K = 4
    iters = len(images) // K

    prompt = [prompt] * K

    os.makedirs(save_dir, exist_ok=True)


    pipe = IFImg2ImgPipeline.from_pretrained(
        "DeepFloyd/IF-I-XL-v1.0",
        repo_type="model",
        variant="fp16",
        torch_dtype=torch.float16,
    )
    pipe.enable_model_cpu_offload()

    prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)

    super_res_1_pipe = IFImg2ImgSuperResolutionPipeline.from_pretrained(
        "DeepFloyd/IF-II-L-v1.0",
        repo_type="model",
        text_encoder=None,
        variant="fp16",
        watermarker=None,
        torch_dtype=torch.float16,
    )
    super_res_1_pipe.enable_model_cpu_offload()

    for i in range(iters):
        image_names = images[K*i:K*(i+1)]

        image_paths = [os.path.join(image_dir, image_name) for image_name in image_names]

        original_images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        original_images = [original_image.resize((512, 512)) for original_image in original_images]

        image = [original_image.resize((64, 64)) for original_image in original_images]
        
        image = super_res_1_pipe(
            image=image,
            original_image=original_images,
            num_inference_steps=50,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            strength=0.35,
        ).images

        for image_name, item in zip(image_names, image):
            save_path = os.path.join(save_dir, image_name)
            item.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='split files.')
    parser.add_argument('--image_dir', type=str, help='save name.')
    parser.add_argument('--save_dir', type=str, help='The name of the second level folder.')
    parser.add_argument('--prompt', type=str, help='The name of the second level folder.')

    args = parser.parse_args()

    enhance(args.image_dir, args.save_dir, args.prompt)
