import numpy as np
from scipy.stats import norm, t
from tqdm import tqdm

import torch
from torch.distributions.normal import Normal

from distributions import GaussianDiagonalMixture, NormalWishartPrior
from utils.depth_utils import DepthNorm, predict_distributions


def calculate_calibration_intervals(
    targets, predicted_means, predicted_stddevs,
    step=0.05, verbose=False
):
    """
    Computes calibration curve - how theoretical
    conf intervals correlate wti practical (assuming prediction is Normal)
    """
    real_errors = np.abs(targets - predicted_means)

    all_fractions = []

    q_list = np.arange(0.0, 1.0 + step, step)
    if verbose:
        q_list = tqdm(q_list)

    for q in q_list:
        predicted_error_bound = -predicted_means + \
            norm.interval(q, predicted_means, predicted_stddevs)[1]

        emp_fraction = np.mean((
            real_errors <= predicted_error_bound
        ).astype(float))
        all_fractions.append(emp_fraction)

    # Calculates area to diagonal (ideal calibration).
    c_auc_score = np.abs(
        np.array(all_fractions) - np.arange(0.0, 1.0 + step, step)
    ).sum() * step

    return all_fractions, c_auc_score


def nyu_evaluate_calibration_metrics(
    model, rgb, depth, args
):
    """Evaluates quality of errors detection via NLL and C-AUC scores"""
    inputs = torch.FloatTensor(rgb / 255).permute(0, 3, 1, 2)

    # Rescale & downsample targets
    targets = torch.FloatTensor(depth)
    targets = DepthNorm(targets, transform_type=args.targets_transform)
    targets = torch.nn.functional.interpolate(
        targets.unsqueeze(1), scale_factor=0.5
    )

    dists = predict_distributions(
        model, inputs, transform_type=args.targets_transform,
        device=args.device
    )

    image_nlls = [
        -dists[i].log_prob(
            targets[i].view(dists[i].mean.shape)
        ).mean().item() for i in range(len(dists))
    ]

    mean_nll, std_nll = np.mean(image_nlls), np.std(image_nlls)

    if args.verbose:
        print('NLL: %.4f +- %.4f' % (mean_nll, std_nll))

        print("Estimating C-AUC...")

    calibr_curve, c_auc_score = calculate_calibration_intervals(
        targets.reshape(-1).numpy(),
        np.concatenate([dist.mean.reshape(-1) for dist in dists]),
        np.concatenate([dist.variance.reshape(-1) ** 0.5 for dist in dists]),
        verbose=args.verbose
    )

    return mean_nll, c_auc_score, image_nlls, calibr_curve
