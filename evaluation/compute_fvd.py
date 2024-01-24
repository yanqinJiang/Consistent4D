# This code is reorganized from DisCo (https://github.com/Wangt-CN/DisCo?tab=readme-ov-file)
# The difference between the result computed by this code and the original one lies in the two decimal places. (误差在小数点后两位）
# And this difference might be caused by the following problem:
# https://github.com/pytorch/pytorch/issues/75363 
# conv3d has numerical issue where same input produces output that are not bit-wise identical
# Generally, we think it is okay to use this code for evaluation.

import torch
import os

import numpy as np

from PIL import Image

from inception3d import InceptionI3d

from scipy import linalg

import json

import argparse

def extract_id(image_path):
    filename = os.path.basename(image_path)  # 获取文件名（包含扩展名）
    id_str = os.path.splitext(filename)[0]  # 去除扩展名，得到纯文件名
    return int(id_str)  # 将 id 号转换为整数

def resize_single_channel(x_np, output_size):
    s1, s2 = output_size

    img = Image.fromarray(x_np.astype(np.float32), mode='F')
    img = img.resize(output_size, resample=3)
    return np.asarray(img).reshape(s1, s2, 1)

def resize_func(x, output_size):
    x = [resize_single_channel(x[:, :, idx], output_size) for idx in range(3)]
    x = np.concatenate(x, axis=2).astype(np.float32)
    return x

def read_video_frames(frame_paths):
    frame_list = []
    for frame_path in frame_paths:
        frame = Image.open(frame_path).convert('RGB')
        frame_list.append(np.array(frame))
    video = np.stack(frame_list, axis=0)

    output_size= (224, 224)
    video_resize = []
    for vim in video:
        vim_resize = resize_func(vim, output_size)
        video_resize.append(vim_resize)
    
    video = np.stack(video_resize, axis=0)
    # import ipdb; ipdb.set_trace()
    video = torch.as_tensor(video.copy()).float()
    
    video = video / 127.5 - 1
    video = video.unsqueeze(0).permute(0, 4, 1, 2, 3).float()  # num_seg, 3, sample_during h, w

    return video


"""
Compute the FID score given the mu, sigma of two sets
"""


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Danica J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def fid_from_feats(feats1, feats2):
    mu1, sig1 = np.mean(feats1, axis=0), np.cov(feats1, rowvar=False)
    mu2, sig2 = np.mean(feats2, axis=0), np.cov(feats2, rowvar=False)
    return frechet_distance(mu1, sig1, mu2, sig2)

def extract_feature(model, batch):
    with torch.no_grad():
        feat = model(batch.to(device))
    return feat.detach().cpu().numpy()

device = "cuda"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='evaluate FVD') 
    parser.add_argument('--gt_root', type=str, default='./gt', help='gt_folder')
    parser.add_argument('--pred_root', type=str, default='./pred', help='result_folder')
    parser.add_argument('--eval_views', type=int, default=4, help='view num')
    parser.add_argument('--model_path', type=str, default='./i3d_pretrained_400.pt', help='view num')
    parser.add_argument('--output_file', type=str, default='./result_video_level.json', help='output')


    args = parser.parse_args()
    gt_root = args.gt_root
    pred_root = args.pred_root
    eval_views = args.eval_views
    model_path = args.model_path
    output_file = args.output_file

    gt_objects = os.listdir(gt_root)
    pred_objects = os.listdir(pred_root)

    assert len(gt_objects) == len(pred_objects), "gt and result do not match!"

    # load model
    feat_model = InceptionI3d(400, in_channels=3)
    feat_model.load_state_dict(torch.load(model_path))
    feat_model = feat_model.to(device).eval()

    result_dict = dict()
    print("start evalution...")
    for eval_object in gt_objects:
        gt_folder = os.path.join(gt_root, eval_object)
        pred_folder = os.path.join(pred_root, eval_object)
        views = [f'eval_{view_id}' for view_id in range(eval_views)]

        eval_fvd = 0.
        for view in views:
            gt_view = os.path.join(gt_folder, view)
            pred_view = os.path.join(pred_folder, view)

            gt_files = os.listdir(gt_view)
            gt_files = [os.path.join(gt_view, filename) for filename in gt_files]
            gt_files = sorted(gt_files, key=extract_id)

            pred_files = os.listdir(pred_view)
            pred_files = [os.path.join(pred_view, filename) for filename in pred_files]
            pred_files = sorted(pred_files, key=extract_id)

            pred_tensor = read_video_frames(pred_files).cuda()
            gt_tensor = read_video_frames(gt_files).cuda()

            pred_feat = extract_feature(feat_model, pred_tensor)
            gt_feat = extract_feature(feat_model, gt_tensor)

            fvd_score = fid_from_feats(feats1=pred_feat, feats2=gt_feat)
            
            eval_fvd += fvd_score
        
        eval_fvd /= eval_views
        result_dict[eval_object] = eval_fvd
        print(eval_object, eval_fvd)

    fvd_avg = sum(v for k, v in result_dict.items()) / len(gt_objects)
    result_dict["average"] = fvd_avg
    print("average", fvd_avg)

    with open(output_file, 'w') as file_to_write:
        json.dump(result_dict, file_to_write, indent=3)




