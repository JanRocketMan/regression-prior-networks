import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from distributions import NormalWishartPrior
from distributions.distribution_wrappers import ProbabilisticWrapper
from distributions.distribution_wrappers import GaussianEnsembleWrapper
from models.unet_model import UNetModel


def load_unet_model_from_checkpoint(
    checkpoints, model_type, backbone='densenet169', device='cuda:0'
):
    if model_type in ['gaussian', 'nw_prior', 'l1-ssim']:
        channels = {
            'l1-ssim': 1,
            'gaussian': 2, 'nw_prior': 3
        }
        model = UNetModel(
            backbone,
            pretrained=False, out_channels=channels[model_type]
        ).to(device)
        model = nn.DataParallel(model)

        model.load_state_dict(torch.load(checkpoints[0]))
        model = model.module.eval().to(device)

        if model_type == 'gaussian':
            model = ProbabilisticWrapper(
                Normal, model
            )
        elif model_type == 'nw_prior':
            model = ProbabilisticWrapper(
                NormalWishartPrior, model
            )
    elif model_type == 'gaussian-ensemble':
        models = []
        for checkpoint in checkpoints:
            cur_model = UNetModel(
                backbone, pretrained=False, out_channels=2
            ).to(device)
            cur_model = nn.DataParallel(cur_model)
            cur_model.load_state_dict(torch.load(checkpoint))
            cur_model = cur_model.module.eval().to(device)
            models.append(cur_model)
        model = GaussianEnsembleWrapper(models)
    else:
        raise ValueError("Provided model_type is not supported")
    return model.eval()
