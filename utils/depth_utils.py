import matplotlib
import matplotlib.cm
import numpy as np
import torch
from torch.distributions.normal import Normal
import re

from skimage.transform import resize

from distributions import GaussianDiagonalMixture, NormalWishartPrior


def reshape_images(images):
    # Support multiple RGBs, one RGB image, even grayscale
    if len(images.shape) < 3:
        images = np.stack((images, images, images), axis=2)
    if len(images.shape) < 4:
        images = images.reshape(
            (1, images.shape[0], images.shape[1], images.shape[2])
        )
    return images


def scale_up(scale, images):
    """Rescales images by :scale factor"""
    scaled = []
    for i in range(len(images)):
        img = images[i]
        output_shape = (scale * img.shape[0], scale * img.shape[1])
        scaled.append(resize(
            img, output_shape, order=1,
            preserve_range=True, mode='reflect', anti_aliasing=True
        ))
    return np.stack(scaled)


def DepthNorm(depth, maxDepth=1000.0, minDepth=10, transform_type='inverse'):
    """Normalizes targets"""
    if transform_type == 'inverse':
        return maxDepth / depth
    elif transform_type == 'scaled':
        return depth / minDepth
    elif transform_type == 'log':
        return torch.log10(depth / minDepth)


def InvertDepthNorm(
    depth, maxDepth=1000.0, minDepth=10, transform_type='inverse'
):
    """Renormalizes predictions back to targets space"""
    if transform_type == 'inverse':
        return maxDepth / depth
    elif transform_type == 'scaled':
        return depth * minDepth
    elif transform_type == 'log':
        return (10 ** depth) * minDepth


def renorm_param(
    param, maxDepth=1000.0, minDepth=10,
    transform_type='scaled', clip=False
):
    renormed_param = InvertDepthNorm(
        param, maxDepth=maxDepth, minDepth=minDepth,
        transform_type=transform_type
    ) / maxDepth
    if clip:
        renormed_param = torch.clamp(renormed_param, minDepth / maxDepth, 1.0)
    return renormed_param


def renorm_distribution(
    dist, maxDepth=1000.0, minDepth=10, transform_type='scaled'
):
    """Renormalizes distribution parameters back to targets space."""
    if transform_type != 'scaled':
        raise Exception("Only scaling transformations are supported")

    if isinstance(dist, GaussianDiagonalMixture):
        for i in range(len(dist.distributions)):
            dist.distributions[i] = renorm_distribution(
                dist.distributions[i], maxDepth=maxDepth, minDepth=minDepth,
                transform_type=transform_type
            )
    else:
        dist.loc = renorm_param(
            dist.loc, maxDepth, minDepth, transform_type, True
        )

        if isinstance(dist, NormalWishartPrior):
            dist.precision_diag = dist.precision_diag * (maxDepth / minDepth) ** 2
        else:
            dist.scale = renorm_param(
                dist.scale, maxDepth, minDepth, transform_type, False
            )

    return dist


def predict_targets(
    model, images, minDepth=10, maxDepth=1000,
    transform_type='scaled', device='cuda:0', clip=True, no_renorm=False
):
    """Use trained model to predict depths"""
    images = reshape_images(images)

    # Compute predictions
    if hasattr(model, 'distribution_cls'):
        predictions = model(
            torch.FloatTensor(images).permute(0, 3, 1, 2).to(device)
        ).mean.cpu().permute(0, 2, 3, 1)
    else:
        predictions = model(
            torch.FloatTensor(images).permute(0, 3, 1, 2).to(device)
        ).cpu().permute(0, 2, 3, 1)
    if no_renorm:
        if clip:
            predictions = np.clip(predictions, minDepth, maxDepth)
        return predictions
    # Put in expected range
    return renorm_param(
        predictions, maxDepth=maxDepth, minDepth=minDepth,
        transform_type=transform_type, clip=clip
    )


def predict_distributions(
    model, images, minDepth=10, maxDepth=1000,
    transform_type='scaled', device='cuda:0', renorm=True, posterior=True,
    valid_masks=None
):
    """Returns list of predicted distributions for a given inputs"""
    dists_list = []

    for idx in range(len(images)):
        with torch.no_grad():
            # Get mask to filter some outputs (if required)
            if valid_masks is not None:
                c_mask = valid_masks[idx].unsqueeze(0)
            else:
                c_mask = None

            # Compute results
            pred_dist = model(images[idx].to(device).unsqueeze(0), mask=c_mask)

            if posterior and isinstance(pred_dist, NormalWishartPrior):
                # Infer posterior t-distribution from a prior prediction
                pred_dist = pred_dist.forward()
            elif posterior and isinstance(pred_dist, GaussianDiagonalMixture):
                # Approximate ensemble with a single Gaussian
                pred_dist = Normal(
                    pred_dist.expected_mean(),
                    pred_dist.total_variance().pow(0.5)
                )

        # Renormalize distribution parameters (if required)
        if renorm:
            pred_dist = renorm_distribution(pred_dist)

        dists_list.append(pred_dist)

    return dists_list


def get_uncertainty_measure(dist, unc_name):
    unc_measure = getattr(dist, unc_name)
    if hasattr(unc_measure, '__call__'):
        unc_measure = unc_measure().squeeze()
    else:
        unc_measure = unc_measure.squeeze()
    return unc_measure
