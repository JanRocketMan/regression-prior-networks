import matplotlib
import matplotlib.cm
import numpy as np
import torch
from torch.distributions.normal import Normal
import re

from skimage.transform import resize

from distributions import GaussianDiagonalMixture, NormalWishartPrior


def _load_densenet_dict(model, load_path):
    """
    Load trained model from provided checkpoint file -
    this is used when no internet on device is available"""
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(\
            ?:weight|bias|running_mean|running_var))$'
    )

    state_dict = torch.load(load_path)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


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
        renormed_param = np.clip(renormed_param, minDepth / maxDepth, 1.0)
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
    transform_type='scaled', device='cuda:0'
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
    # Put in expected range
    return renorm_param(
        predictions, maxDepth=maxDepth, minDepth=minDepth,
        transform_type=transform_type, clip=True
    )


def predict_distributions(
    model, images, minDepth=10, maxDepth=1000,
    transform_type='scaled', device='cuda:0', renorm=True, posterior=True
):
    """Returns list of predicted distributions for a given inputs"""
    dists_list = []

    for idx in range(len(images)):
        with torch.no_grad():
            # Compute results
            pred_dist = model(images[idx].to(device).unsqueeze(0))

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
