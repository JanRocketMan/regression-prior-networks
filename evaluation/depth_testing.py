"""Evaluate test metrics of trained model.
The code is adapted from
https://github.com/ialhashim/DenseDepth/blob/master/evaluate.py
"""
import time
import numpy as np

import torch

from utils.nyuv2_loading import load_test_data
from utils.depth_utils import scale_up, predict_targets
from utils.model_utils import load_unet_model_from_checkpoint


def compute_rel_metrics(gt, pred):
    """Computes \\delta, rel, rms and log_10 scores"""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return a1, a2, a3, abs_rel, rmse, log_10


def nyu_evaluate_performance_metrics(
    model, rgb, depth, crop,
    batch_size=6, verbose=False, transform_type='scaled', device='cuda:0'
):
    """
    Evaluates common performance metrics (\\delta_1, \\delta_2, etc)
    of MDE trained model on Nyu v2
    """
    N = len(rgb)
    bs = batch_size

    depth_scores = np.zeros((6, len(rgb)))  # six metrics

    for i in range(N//bs):
        x = rgb[(i)*bs:(i+1)*bs, :, :, :]
        with torch.no_grad():
            # Compute results
            true_y = depth[(i)*bs:(i+1)*bs, :, :]
            pred_y = scale_up(
                2, predict_targets(
                    model, x/255, minDepth=10, maxDepth=1000,
                    transform_type=transform_type,
                    device=device
                )[:, :, :, 0]
            ) * 10.0
            # Test time augmentation: mirror image estimate
            pred_y_flip = scale_up(
                2, predict_targets(
                    model, x[..., ::-1, :]/255, minDepth=10, maxDepth=1000,
                    transform_type=transform_type,
                    device=device
                )[:, :, :, 0]
            ) * 10.0

        # Crop based on Eigen et al. crop
        true_y = true_y[:, crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        pred_y = pred_y[:, crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        pred_y_flip = pred_y_flip[:, crop[0]:crop[1]+1, crop[2]:crop[3]+1]

        # Compute errors per image in batch
        for j in range(len(true_y)):
            errors = compute_rel_metrics(
                true_y[j],
                (0.5 * pred_y[j]) + (0.5 * np.fliplr(pred_y_flip[j]))
            )

            for k in range(len(errors)):
                depth_scores[k][(i*bs)+j] = errors[k]

    e = depth_scores.mean(axis=1)
    return e
