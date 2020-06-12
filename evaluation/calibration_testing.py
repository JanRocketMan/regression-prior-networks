import numpy as np
from scipy.stats import norm, t
from tqdm import tqdm

import torch
from torch.distributions.normal import Normal

from distributions import GaussianDiagonalMixture, NormalWishartPrior
from utils.nyuv2_loading import load_test_data
from utils.model_utils import load_unet_model_from_checkpoint
from utils.depth_utils import renorm_distribution


def evaluate_distributions(
    model, inputs, device='cuda:0'
):
    """Returns list of predicted distributions"""
    dists_list = []

    for i in range(len(inputs)):
        with torch.no_grad():
            # Compute results
            pred_dist = model(inputs[i].unsqueeze(0).to(device))

            if isinstance(pred_dist, NormalWishartPrior):
                # Infer posterior t-distribution from a prior prediction
                pred_dist = pred_dist.forward()
                pred_dist = renorm_distribution(pred_dist)
                pred_dist = Normal(
                    pred_dist.mean,
                    pred_dist.variance.pow(0.5)
                )
            elif isinstance(pred_dist, GaussianDiagonalMixture):
                # Approximate ensemble with a single Gaussian
                pred_dist = renorm_distribution(pred_dist)
                pred_dist = Normal(
                    pred_dist.expected_mean(),
                    pred_dist.total_variance().pow(0.5)
                )
            else:
                pred_dist = renorm_distribution(pred_dist)

        dists_list.append(pred_dist)

    return dists_list


def calculate_diff_auc(pred, step=0.05):
    """Calculates area to diagonal (ideal calibration). Lower - better"""
    return np.sum(
        np.abs(
            np.array(pred) - np.arange(0.0, 1.0 + step, step)
        ) * step
    )


def calculate_calibration_intervals(
    targets, est_dist_params, step=0.05, quantile_distribution='Normal'
):
    diff_y = np.abs(targets - est_dist_params['mean'])

    if quantile_distribution == 'Normal':
        mean, std = est_dist_params['mean'], est_dist_params['scale']
        th_delta_fn = lambda q: -mean + norm.interval(q, mean, std)[1]
    else:
        df, loc = est_dist_params['df'], est_dist_params['mean']
        scale = est_dist_params['scale']
        th_delta_fn = lambda q: -loc + t.interval(q, df, loc=loc, scale=scale)[1]
    all_fractions = []

    for q in tqdm(np.arange(0.0, 1.0 + step, step)):
        deltas = th_delta_fn(q)
        emp_fraction = np.mean((
            diff_y <= deltas
        ).astype(float))
        all_fractions.append(emp_fraction)
    return all_fractions


def get_calibration_metrics(args):
    print('Loading trained model...')
    model = load_unet_model_from_checkpoint(
        args.checkpoint, args.model_type, args.backbone, args.device
    )
    print("Trained model loaded.\n")

    print('Loading test data...', end='')
    rgb, depth, _ = load_test_data(args.zip_folder)
    print('Test data loaded.\n')

    print("Calculating NLLs...\n")

    rgb = torch.FloatTensor(rgb / 255).permute(0, 3, 1, 2)
    targets = torch.FloatTensor(depth) / 10.0

    dists = evaluate_distributions(
        model, rgb, device=args.device
    )

    targets = torch.nn.functional.interpolate(
        targets.unsqueeze(1), scale_factor=0.5
    )

    image_nlls = [
        -dists[i].log_prob(
            targets[i].view(dists[i].mean.shape)
        ).mean().item() for i in range(len(dists))
    ]

    mean_nll, std_nll = np.mean(image_nlls), np.std(image_nlls)
    print('NLL: %.4f +- %.4f' % (mean_nll, std_nll))

    print("Estimating C-AUC...")

    if isinstance(dists[0], Normal):
        est_dist_params = {
            'mean': np.concatenate(
                [dist.mean.reshape(-1) for dist in dists]
            ),
            'scale': np.concatenate(
                [dist.stddev.reshape(-1) for dist in dists]
            )
        }
    else:
        est_dist_params = {
            'mean': np.concatenate(
                [dist.loc.reshape(-1) for dist in dists]
            ),
            'df': np.concatenate(
                [dist.df.reshape(-1) for dist in dists]
            ),
            'scale': np.concatenate(
                [dist.variance.pow(0.5).reshape(-1) for dist in dists]
            )
        }

    calibr_curve = calculate_calibration_intervals(
        targets.reshape(-1).numpy(), est_dist_params
    )

    print("C-AUC score: %.3f" % calculate_diff_auc(calibr_curve))

    return image_nlls, calibr_curve
