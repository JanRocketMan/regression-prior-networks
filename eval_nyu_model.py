import json
from argparse import ArgumentParser

from evaluation.depth_testing import get_test_metrics
from evaluation.calibration_testing import get_calibration_metrics


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Evaluation of trained \
            Monocular Depth Estimation model on Nyu Depth v2'
    )
    parser.add_argument('--zip_folder', default='data')
    parser.add_argument(
        '--checkpoint',
        default=['checkpoints/dense_depth_l1-ssim/1/19.ckpt'],
        nargs='+'
    )
    parser.add_argument('--backbone', default='densenet169', choices=[
        'resnet18', 'densenet169'
    ])
    parser.add_argument('--bs', type=int, default=6)
    parser.add_argument('--model_type', default='l1-ssim', choices=[
        'gaussian', 'gaussian-ensemble', 'nw_prior', 'l1-ssim'
    ])
    parser.add_argument('--targets_transform', default='inverse', choices=[
        'inverse', 'scaled', 'log'
    ])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--calibration', dest='calibration', action='store_true', default=False
    )
    args = parser.parse_args()

    get_test_metrics(args)

    if args.calibration:
        if args.model_type not in [
            'gaussian', 'gaussian-ensemble', 'nw_prior'
        ]:
            raise Exception(
                "Cannot compute calibration / NLL for non-probabilistic model"
            )
        image_nlls, calibr_curve = get_calibration_metrics(args)

        store_beg = "logs/" + args.checkpoint[0].replace("/", "_")

        json.dump(
            image_nlls,
            open(store_beg + '-nll.json', 'w')
        )
        json.dump(
            calibr_curve,
            open(store_beg + '-calibration.json', 'w')
        )
