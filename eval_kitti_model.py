import json
from argparse import ArgumentParser
from tqdm import tqdm
from numpy import nan
import numpy as np
import pandas as pd
import torch

from evaluation.depth_testing import compute_rel_metrics

from utils.data_loading import load_test_data, getTrainingEvalDataKITTI
from utils.model_utils import load_unet_model_from_checkpoint

from utils.depth_utils import predict_targets, scale_up
from distributions.distribution_wrappers import ProbabilisticWrapper
from torch.distributions import Normal


def get_test_metrics(model, test_loader, target_transform, device):
    model.eval()
    metric_names = [
        'delta_1', 'delta_2', 'delta_3', 'rel',
        'rms', 'log10', 'rmse_log'
    ]
    all_metrics_buffer = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            sample_img, sample_depth = batch['image'].to(device), \
                batch['depth'].to(device)

            prediction = predict_targets(
                model, sample_img.permute(0, 2, 3, 1).cpu().numpy(),
                minDepth=1e-2, maxDepth=85.0,
                transform_type=target_transform,
                device=device, clip=True, no_renorm=True
            )[:, :, :, 0]

            prediction_np = scale_up(
                2, prediction
            )

            sample_depth = np.clip(sample_depth.cpu().numpy(), 0, 80)
            prediction_np = np.clip(prediction_np[:, np.newaxis, :, :], 0, 80)
            valid_mask = np.logical_and(
                sample_depth > 0.0, sample_depth < 80.0
            )

            # Crop by Eigen et al.
            eval_mask = np.zeros(valid_mask.shape)
            gt_height, gt_width = sample_depth[0, 0].shape
            eval_mask[
                :, :,
                int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                int(0.0359477 * gt_width):int(0.96405229 * gt_width)
            ] = 1
            valid_mask = np.logical_and(valid_mask, eval_mask)

            sample_crop = sample_depth[valid_mask]
            prediction_crop = prediction_np[valid_mask]

            all_metrics = compute_rel_metrics(
                sample_crop, prediction_crop, rmse_log=True
            )
            all_metrics_buffer.append(all_metrics)

    all_metrics_buffer = np.array(all_metrics_buffer).mean(axis=0)
    print("Eval scores", end='\t', flush=True)
    for mname, metric in zip(metric_names, all_metrics_buffer):
        print(mname, ': %.3f' % metric, end=' ', flush=True)
    print("")


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Evaluation of trained KITTI'
    )
    parser.add_argument(
        '--checkpoint',
        default=['checkpoints/dense_depth_gaussian/1/19.ckpt'],
        nargs='+'
    )
    parser.add_argument('--backbone', default='densenet169', choices=[
        'resnet18', 'densenet169'
    ])
    parser.add_argument('--path_to_kitti', type=str)
    parser.add_argument('--path_to_csv_test', type=str)
    parser.add_argument('--bs', type=int, default=6)
    parser.add_argument('--model_type', default='gaussian', choices=[
        'gaussian', 'gaussian-ensemble', 'nw_prior', 'l1-ssim', 'hydra'
    ])
    parser.add_argument('--targets_transform', default='scaled', choices=[
        'inverse', 'scaled'
    ])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--verbose', dest='verbose', action='store_true', default=False
    )
    args = parser.parse_args()

    model = load_unet_model_from_checkpoint(
        args.checkpoint, args.model_type, args.backbone, args.device
    )
    model.eval()
    _, test_loader = getTrainingEvalDataKITTI(
        path_to_kitti=args.path_to_kitti,
        path_to_csv_train=args.path_to_csv_test,
        path_to_csv_val=args.path_to_csv_test,
        batch_size=args.bs,
        resize_depth=False
    )

    get_test_metrics(
        model, test_loader, target_transform=args.targets_transform,
        device=args.device
    )
