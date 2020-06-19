from argparse import ArgumentParser

import numpy as np
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import ToTensor, Resize, ToPILImage

from evaluation.show_examples import show_model_examples
from utils.nyuv2_loading import load_test_data
from evaluation.ood_testing import load_ood_data
from utils.model_utils import load_unet_model_from_checkpoint
from utils.viz_utils import get_example_figure, get_tensor_with_histograms


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Evaluation of trained \
            Monocular Depth Estimation model on Nyu Depth v2'
    )
    parser.add_argument('--data_path', default='data')
    parser.add_argument(
        '--data_type', default='nyu', choices=['nyu', 'kitti', 'lsun']
    )
    parser.add_argument(
        '--gaussian_checkpoints',
        default=[
            'checkpoints/dense_depth_gaussian/' + str(i + 1) + '/19.ckpt'
            for i in range(5)
        ],
        nargs='+'
    )
    parser.add_argument(
        '--endd_checkpoint',
        default='checkpoints/dense_depth_endd/1/21.ckpt'
    )
    parser.add_argument(
        '--indices',
        default=[20, 57, 71, 102, 106, 121, 270, 435, 466, 491],
        nargs='+'
    )
    parser.add_argument('--backbone', default='densenet169', choices=[
        'resnet18', 'densenet169'
    ])
    parser.add_argument('--bs', type=int, default=6)
    parser.add_argument(
        '--models', default=['gaussian', 'gaussian-ensemble', 'nw_prior'],
        nargs='+'
    )
    parser.add_argument('--targets_transform', default='scaled', choices=[
        'inverse', 'scaled'
    ])
    parser.add_argument(
        '--measures', default=['total_variance', 'expected_pairwise_kl'],
        nargs='+', help='Uncertainty measures to visualize'
    )
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    all_indices = [int(idx) for idx in args.indices]
    if args.data_type == 'nyu':
        rgb, depth, crop = load_test_data(args.data_path)
    else:
        rgb = load_ood_data(args.data_path, args.data_type)
        depth, crop = None, None

    max_limits = None
    all_results = {}
    checkpoints = {
        'gaussian': [args.gaussian_checkpoints[2]],
        'gaussian-ensemble': args.gaussian_checkpoints,
        'nw_prior': [args.endd_checkpoint]
    }
    for model_type in args.models:
        model = load_unet_model_from_checkpoint(
            checkpoints[model_type], model_type, args.backbone, args.device
        )
        if model_type == 'gaussian':
            results = show_model_examples(
                model, rgb, depth, all_indices,
                ['variance'], args.targets_transform, args.device
            )
            max_limits = results[-1]
        else:
            results = show_model_examples(
                model, rgb, depth, all_indices,
                args.measures, args.targets_transform, args.device,
                (None if 'gaussian' in model_type else max_limits)
            )
            if 'gaussian' in model_type:
                max_limits = results[-1]
            all_results[model_type] = results

    model_names = ['gaussian', 'gaussian-ensemble', 'nw_prior']
    plot_names = ['Single', 'ENSM', 'EnD$^2$']
    all_hists = get_tensor_with_histograms(
        all_results, plot_names, model_names
    )

    for i, idx in enumerate(args.indices):
        ensm_data, endd_data = [[
            all_results[model_n][0][i],
            all_results[model_n][2][i]
        ] + [
            all_results[model_n][3][measure][i]
            for measure in args.measures
        ] for model_n in ['gaussian-ensemble', 'nw_prior']]
        figure = get_example_figure(
            ensm_data, endd_data, all_hists[i*4:i*4+4]
        )
        figure.savefig(
            'temp_plots/example_' + args.data_type + '_' + str(idx) + '.png',
            dpi=300, bbox_inches='tight'
        )
        plt.clf()
