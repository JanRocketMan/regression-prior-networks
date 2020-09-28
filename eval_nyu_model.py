import json
from argparse import ArgumentParser

from numpy import nan

from evaluation.depth_testing import nyu_evaluate_performance_metrics
from evaluation.calibration_testing import nyu_evaluate_calibration_metrics

from utils.data_loading import load_test_data
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
    parser.add_argument('--backbone', default='densenet169', choices=[
        'resnet18', 'densenet169'
    ])
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

    rgb, depth, crop = load_test_data(args.zip_folder)

    e = nyu_evaluate_performance_metrics(
        model, rgb, depth, crop, args.bs, args.verbose,
        args.targets_transform, args.device
    )

    if args.model_type != 'l1-ssim':
        out_results = nyu_evaluate_calibration_metrics(
            model, rgb, depth, args
        )

        m_nll, c_auc, image_nlls, calibr_curve = out_results
        store_beg = "logs/" + args.checkpoint[0].replace("/", "_")
        store_beg += "_" + args.model_type

        json.dump(
            image_nlls,
            open(store_beg + '-nll.json', 'w')
        )
        json.dump(
            calibr_curve,
            open(store_beg + '-calibration.json', 'w')
        )
    else:
        m_nll, c_auc = nan, nan

    print(
        "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(
            'a1', 'a2', 'a3', 'rel', 'rms', 'log_10', 'NLL', 'C-AUC'
        )
    )
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(
            e[0], e[1], e[2], e[3], e[4], e[5], m_nll, c_auc
        )
    )
