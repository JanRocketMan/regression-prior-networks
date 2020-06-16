import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
import glob

import torch
from torchvision import transforms as ts

from utils.depth_utils import predict_distributions, get_uncertainty_measure


def load_ood_data(files_path, itype='kitti'):
    """Rescale & center-crop out-of-domain image to match nyu dims"""
    fileend = '.png' if itype == 'kitti' else '.jpg'
    assert len(glob.glob(files_path + '/*' + fileend)) == 654
    all_files = np.zeros((654, 480, 640, 3))
    if itype == 'kitti':
        crop_transform = ts.Compose([
            ts.Resize((480, 1588)),
            ts.CenterCrop((480, 640))
        ])
    else:
        crop_transform = ts.Compose([
            ts.CenterCrop(256),
            ts.Resize((480, 640))
        ])
    for i, path in enumerate(glob.glob(files_path + '/*' + fileend)):
        img = Image.open(path)
        all_files[i] = np.array(crop_transform(img))
    return all_files


def nyu_evaluate_ood_auc_scores(
    model, rgb_test, rgb_ood, unc_measures,
    transform_type='scaled', device='cuda:0'
):
    """Compute roc-auc of ood detection based on averaged image uncertainty"""
    nyu_inputs = torch.FloatTensor(rgb_test / 255).permute(0, 3, 1, 2)
    ood_inputs = torch.FloatTensor(rgb_ood / 255).permute(0, 3, 1, 2)

    nyu_dists, ood_dists = [
        predict_distributions(
            model, inputs,
            transform_type=transform_type, device=device, posterior=False
        ) for inputs in [nyu_inputs, ood_inputs]
    ]

    measure_auc_dict = {}
    for measure in unc_measures:
        # When dealing with variance, take logarithm to estimate the volume
        if 'variance' in measure:
            avg_img_unc_fn = lambda dist: get_uncertainty_measure(
                dist, measure
            ).log().mean().item()
        else:
            avg_img_unc_fn = lambda dist: get_uncertainty_measure(
                dist, measure
            ).mean().item()

        nyu_stats, ood_stats = [np.array([
            avg_img_unc_fn(dist)
            for dist in dists
        ]) for dists in [nyu_dists, ood_dists]]

        joint_scores = np.concatenate((nyu_stats, ood_stats), axis=0)
        true_labels = np.concatenate((np.zeros(654), np.ones(654)), axis=0)

        measure_auc_dict[measure] = roc_auc_score(true_labels, joint_scores)

    return measure_auc_dict
