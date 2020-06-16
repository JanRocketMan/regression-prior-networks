from argparse import ArgumentParser

from evaluation.ood_testing import load_ood_data, nyu_evaluate_ood_auc_scores

from utils.nyuv2_loading import load_test_data
from utils.model_utils import load_unet_model_from_checkpoint


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Evaluation of trained \
            Monocular Depth Estimation model on Nyu Depth v2'
    )
    parser.add_argument('--zip_folder', default='data')
    parser.add_argument('--ood_folder', default='data/kitti_subset')
    parser.add_argument(
        '--ood_type', default='kitti',
        choices=['kitti', 'lsun-church', 'lsun-bedroom']
    )
    parser.add_argument(
        '--checkpoint',
        default=['checkpoints/dense_depth_gaussian/1/19.ckpt'],
        nargs='+'
    )
    parser.add_argument('--backbone', default='densenet169', choices=[
        'resnet18', 'densenet169'
    ])
    parser.add_argument('--model_type', default='gaussian', choices=[
        'gaussian', 'gaussian-ensemble', 'nw_prior', 'l1-ssim'
    ])
    parser.add_argument('--targets_transform', default='scaled', choices=[
        'inverse', 'scaled'
    ])
    parser.add_argument(
        '--measures', default=['total_variance'], nargs='+',
        help='Uncertainty measures to use for ood detection'
    )
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    model = load_unet_model_from_checkpoint(
        args.checkpoint, args.model_type, args.backbone, args.device
    )

    rgb, _, _ = load_test_data(args.zip_folder)
    ood_rgb = load_ood_data(args.ood_folder, itype=args.ood_type)

    all_aucs = nyu_evaluate_ood_auc_scores(
        model, rgb, ood_rgb, args.measures,
        transform_type=args.targets_transform,
        device=args.device
    )

    print("OOD detection results vs", args.ood_type, ":\n")
    for measure in args.measures:
        print(measure, "%.3f" % all_aucs[measure])
