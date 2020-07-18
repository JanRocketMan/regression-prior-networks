import torch
import torch.nn as nn
import re
from torch.distributions.normal import Normal

from distributions import NormalWishartPrior
from distributions.distribution_wrappers import ProbabilisticWrapper
from distributions.distribution_wrappers import GaussianEnsembleWrapper
from models.unet_model import UNetModel


def _load_densenet_dict(model, load_path):
    """Load trained model from provided checkpoint file - this is used when no internet on device is available"""
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = torch.load(load_path)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


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


def switch_bn_updates(model, mode):
    """Disables/unables accumulation of statistics in BN layers"""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            if mode == 'train':
                m.train()
            elif mode == 'eval':
                m.eval()
