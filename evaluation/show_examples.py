import numpy as np
import matplotlib.pyplot as plt

import torch

from utils.depth_utils import predict_distributions, get_uncertainty_measure
from utils.depth_utils import DepthNorm
from utils.turbo_cmap import turbo_cm


def standartize_array(arr, arr_min, arr_max):
    return np.clip((arr - arr_min) / (arr_max - arr_min), 0.0, 1.0)


def show_model_examples(
    model, rgb_images, depth, indices, unc_measures,
    transform_type='scaled', device='cuda:0', max_limits=None, trainmodel='nyu'
):
    inputs = torch.FloatTensor(rgb_images[indices] / 255).permute(0, 3, 1, 2)

    pred_dists = predict_distributions(
        model, inputs,
        transform_type=transform_type, device=device, posterior=False,
        renorm=(trainmodel == 'nyu')
    )

    # Rescale & downsample targets
    if depth is not None:
        targets = torch.FloatTensor(depth[indices])
        if trainmodel == 'nyu':
            targets = DepthNorm(targets, transform_type=transform_type)
        targets = torch.nn.functional.interpolate(
            targets.unsqueeze(1), scale_factor=0.5
        )
    # Compute differences to ground truth & measures of uncertainty
    # If targets not available, show predictions only
    if depth is not None:
        diffs = [
            np.abs(targets[idx] - pred_dists[idx].mean)
            for idx in range(targets.shape[0])
        ]
    else:
        diffs = [
            pred_dists[idx].mean
            for idx in range(len(pred_dists))
        ]

    measure_maps = {}
    for measure in unc_measures:
        all_measures = [
            get_uncertainty_measure(pred_dists[idx], measure)
            for idx in range(len(diffs))
        ]
        measure_maps[measure] = all_measures
        # Transform all variances to standard deviations
        if 'variance' in measure:
            measure_maps[measure] = [
                np.sqrt(2) * m.pow(0.5) for m in all_measures
            ]

    # Save difference & unc measures stats to plot histograms
    all_hists = {}
    all_hists['diffs'] = torch.cat([
        diffs[idx].reshape(-1).unsqueeze(0) for idx in range(len(diffs))
    ])
    for measure in unc_measures:
        all_hists[measure] = torch.cat(
            [m.reshape(-1).unsqueeze(0) for m in measure_maps[measure]]
        )

    # Rescale & clip diffs and uncertainties and visualize them with colormaps
    if (max_limits is None) or (depth is None):
        max_limits = [
            np.percentile(diffs[i], 99) for i in range(len(diffs))
        ]

    plot_diffs = torch.from_numpy(np.array([
        turbo_cm(
            standartize_array(
                diffs[i].squeeze(),
                np.percentile(diffs[i], 0),
                max_limits[i]
            )
        )[..., :3]
        for i in range(len(diffs))
    ])).permute(0, 3, 1, 2)

    plot_measures = {}
    for measure in unc_measures:
        if 'variance' in measure:
            normalized_measures = [
                standartize_array(
                    measure_maps[measure][i].squeeze(),
                    np.percentile(measure_maps[measure][i], 0),
                    max_limits[i]
                )
                for i in range(len(diffs))
            ]
        else:
            normalized_measures = [
                standartize_array(
                    measure_maps[measure][i].squeeze(),
                    np.percentile(measure_maps[measure][i], 0),
                    np.percentile(measure_maps[measure][i], 99)
                )
                for i in range(len(diffs))
            ]

        c_plot = torch.cat([torch.from_numpy(
            np.array(turbo_cm(meas)[..., :3])
        ).permute(2, 0, 1).unsqueeze(0) for meas in normalized_measures])

        plot_measures[measure] = c_plot

    downscaled_inputs = torch.nn.functional.interpolate(
        inputs, scale_factor=0.5
    )
    return downscaled_inputs, all_hists, plot_diffs, plot_measures, max_limits
