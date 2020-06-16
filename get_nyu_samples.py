from argparse import ArgumentParser

import torch
import torchvision.utils as vutils

from evaluation.show_examples import show_model_examples
from utils.nyuv2_loading import load_test_data
from utils.model_utils import load_unet_model_from_checkpoint


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Evaluation of trained \
            Monocular Depth Estimation model on Nyu Depth v2'
    )
    parser.add_argument('--zip_folder', default='data')
    parser.add_argument(
        '--checkpoint',
        default=['checkpoints/dense_depth_gaussian/1/19.ckpt'],
        nargs='+'
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

    rgb, depth, crop = load_test_data(args.zip_folder)

    max_limits = None
    all_results = {}
    checkpoints = {
        'gaussian': ['checkpoints/dense_depth_gaussian/1/19.ckpt'],
        'gaussian-ensemble': [
            'checkpoints/dense_depth_gaussian/1/19.ckpt', 
            'checkpoints/dense_depth_gaussian/2/19.ckpt', 
            'checkpoints/dense_depth_gaussian/3/19.ckpt',
            'checkpoints/dense_depth_gaussian/4/19.ckpt',
            'checkpoints/dense_depth_gaussian/5/19.ckpt'
        ],
        'nw_prior': ['checkpoints/dense_depth_endd/1/21.ckpt']
    }
    for model_type in args.models:
        print(model_type)
        model = load_unet_model_from_checkpoint(
            checkpoints[model_type], model_type, args.backbone, args.device
        )
        if model_type == 'gaussian':
            results = show_model_examples(
                model, rgb, depth, [int(idx) for idx in args.indices],
                ['variance'], args.targets_transform, args.device
            )
            max_limits = results[-1]
        else:
            results = show_model_examples(
                model, rgb, depth, [int(idx) for idx in args.indices],
                args.measures, args.targets_transform, args.device, max_limits
            )
            all_results[model_type] = results

    output_res = torch.zeros(len(args.indices) * 8, 3, 240, 320)
    output_res[0::8] = all_results["gaussian-ensemble"][0]
    output_res[1::8] = all_results["gaussian-ensemble"][3]
    output_res[2::8] = all_results["gaussian-ensemble"][4]['total_variance']
    output_res[3::8] = all_results["gaussian-ensemble"][4]['expected_pairwise_kl']

    output_res[4::8] = all_results["nw_prior"][0]
    output_res[5::8] = all_results["nw_prior"][3]
    output_res[6::8] = all_results["nw_prior"][4]['total_variance']
    output_res[7::8] = all_results["nw_prior"][4]['expected_pairwise_kl']

    vutils.save_image(
        output_res, "plots/nyu_result.png", nrow=4, pad_value=255
    )
