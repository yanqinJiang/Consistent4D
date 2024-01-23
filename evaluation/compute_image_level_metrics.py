import sys
import torch
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
import os
from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import peak_signal_noise_ratio
from torchmetrics.functional import structural_similarity_index_measure

import argparse

import numpy as np
import json
# perceptual loss
lpip = LearnedPerceptualImagePatchSimilarity(net_type='vgg', reduction='mean').to('cuda')

# Load the CLIP model
# model_ID = "/home/yqjiang/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/e6a30b603a447e251fdaca1c3056b2a16cdfebeb/"
model_ID = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_ID)

preprocess = CLIPImageProcessor.from_pretrained(model_ID)

# Define a function to load an image and preprocess it for CLIP
def load_and_preprocess_image(image_path, rgba=False):
    # Load the image from the specified path
    image = Image.open(image_path)
    if rgba:
        tmp = np.array(image)
        rgb = tmp[..., :3]
        mask = tmp[..., -1]
        rgb[mask==0] = 255
        image = Image.fromarray(rgb)

    image = image.resize((512, 512))
    image_rgb = torch.from_numpy(np.array(image)).float().permute(2, 0, 1).unsqueeze(0)
    # Apply the CLIP preprocessing to the image
    image = preprocess(image, return_tensors="pt")

    # Return the preprocessed image
    return image, image_rgb

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='evaluate FVD') 
    parser.add_argument('--gt_root', type=str, default='./gt', help='gt_folder')
    parser.add_argument('--pred_root', type=str, default='./pred', help='result_folder')
    parser.add_argument('--eval_views', type=int, default=4, help='view num')
    parser.add_argument('--output_file', type=str, default='./result_image_level.json', help='output')

    args = parser.parse_args()
    gt_root = args.gt_root
    pred_root = args.pred_root
    eval_views = args.eval_views
    output_file = args.output_file

    gt_objects = os.listdir(gt_root)
    pred_objects = os.listdir(pred_root)

    assert len(gt_objects) == len(pred_objects), "gt and result do not match!"

    result_dict = dict(
        clip=dict(),
        lpips=dict(),
        psnr=dict(),
        ssim=dict(),
    )

    print("start evaluation...")
    for eval_object in gt_objects:
        object_clip = 0.
        object_lpip = 0.
        object_psnr = 0.
        object_ssim = 0.
        for view_id in range(eval_views):

            pred_folder = os.path.join(pred_root, eval_object, f'eval_{view_id}')
            gt_folder = os.path.join(gt_root, eval_object, f'eval_{view_id}')

            pred_imgs = os.listdir(pred_folder)
            gt_imgs = os.listdir(gt_folder)
    
            pred_imgs = sorted(pred_imgs, key=lambda i: int(i[:-4]))
            gt_imgs = sorted(gt_imgs, key=lambda i: int(i[:-4]))

            clip_scores = []
            ssims = []
            psnrs = []
            lpip_scores = []
            num_images = len(pred_imgs)
            for idx in range(num_images):
        
                pred_path = os.path.join(pred_folder, pred_imgs[idx])
                gt_path = os.path.join(gt_folder, gt_imgs[idx])

                pred_img, pred_img_rgb = load_and_preprocess_image(pred_path)
                gt_img, gt_img_rgb = load_and_preprocess_image(gt_path)

                # Calculate the embeddings for the images using the CLIP model
                pred_img = pred_img["pixel_values"]
                gt_img = gt_img["pixel_values"]
                with torch.no_grad():
                    embedding_pred = model.get_image_features(pred_img)
                    embedding_gt = model.get_image_features(gt_img)

                # Calculate the cosine similarity between the embeddings
                clip_score = torch.nn.functional.cosine_similarity(embedding_pred, embedding_gt)


                clip_scores.append(clip_score.item())
                
                # calculate ssim and psnr
                ssim = structural_similarity_index_measure(pred_img_rgb, gt_img_rgb)
                psnr = peak_signal_noise_ratio(pred_img_rgb, gt_img_rgb)

                ssims.append(ssim)
                psnrs.append(psnr)

                # calculate lpip
                lpip_score = lpip((pred_img_rgb/127.5-1).cuda(), (gt_img_rgb/127.5-1).cuda())
                lpip_scores.append(lpip_score.item())

            clip_score = np.mean(clip_scores)
            lpip_score = np.mean(lpip_scores)
            ssim = np.mean(ssims)
            psnr = np.mean(psnrs)

            object_clip += clip_score
            object_lpip += lpip_score
            object_ssim += ssim
            object_psnr += psnr

        object_clip = object_clip / 4
        object_lpip = object_lpip / 4
        object_ssim = object_ssim / 4
        object_psnr = object_psnr / 4

        print(eval_object)
        print(f"clip: {object_clip}")
        print(f"lpips: {object_lpip}")
        print(f"ssim: {object_ssim}")
        print(f"psnr: {object_psnr}")

        result_dict["clip"][eval_object] = object_clip
        result_dict["lpips"][eval_object] = object_lpip
        result_dict["ssim"][eval_object] = object_ssim
        result_dict["psnr"][eval_object] = object_psnr
    
    result_dict["clip"]["average"] = sum(v for k, v in result_dict["clip"].items()) / len(gt_objects)
    result_dict["lpips"]["average"] = sum(v for k, v in result_dict["lpips"].items()) / len(gt_objects)
    result_dict["ssim"]["average"] = sum(v for k, v in result_dict["ssim"].items()) / len(gt_objects)
    result_dict["psnr"]["average"] = sum(v for k, v in result_dict["psnr"].items()) / len(gt_objects)
    
    with open(output_file, 'w+') as file_to_write:
        json.dump(result_dict, file_to_write, indent=3)