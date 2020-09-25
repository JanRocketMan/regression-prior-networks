from argparse import ArgumentParser
import os

import torch
from torchvision.models import densenet169
from tensorboardX import SummaryWriter
from torch.distributions import Normal

from distributions import NormalWishartPrior, GaussianDiagonalMixture

from utils.data_loading import getTrainingEvalData
from distributions.distribution_wrappers import ProbabilisticWrapper
from models.unet_model import UNetModel
from training.nyu_trainers import NyuNLLDistributionTrainer
from training.nyu_trainers import NyuDistillationTrainer, NyuRKLTrainer
from utils.model_utils import load_unet_model_from_checkpoint
from utils.model_utils import _load_densenet_dict


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Probabilistic Monocular Depth Estimation on Nyu v2'
    )
    parser.add_argument(
        '--backbone', default='densenet169',
        choices=['resnet18', 'densenet169']
    )
    parser.add_argument('--zip_folder', default='data', type=str)
    parser.add_argument(
        '--checkpoint', required=True, type=str,
        help="Name of the folder to save model/trainer states to"
    )
    parser.add_argument('--pretrained_path', default=None, type=str)
    parser.add_argument('--teacher_checkpoints', default=None, nargs='+')
    parser.add_argument(
        '--epochs', default=20, type=int,
        help='number of total epochs to run'
    )
    parser.add_argument('--model_type', default='gaussian', choices=[
        'gaussian', 'nw_prior', 'l1-ssim', 'nw_prior_rkl', 'nw_end', 'hydra'
    ])
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--warmup_steps', default=1000)
    parser.add_argument('--bs', default=8, type=int, help='batch size')
    parser.add_argument(
        '--log_dir', default="", type=str,
        help='Directory to save tensorboard logs'
    )
    parser.add_argument(
        '--state_dict', default=None, type=str,
        help='Continue training from a given state dict (if it exists)'
    )
    parser.add_argument(
        '--targets_transform', type=str, default='scaled',
        choices=['inverse', 'scaled', 'log'],
        help="Type of transformation to perform with targets"
    )
    parser.add_argument(
        '--overfit_check', dest='overfit', action='store_true', default=False,
        help="If true, uses a tiny subset of the whole train"
    )
    parser.add_argument('--max_temperature', default=10.0, type=float)
    parser.add_argument('--rkl_inv_beta', default=1e-2, type=float)
    parser.add_argument('--rkl_ood_coeff', default=1.0, type=float)
    parser.add_argument('--rkl_warmup_steps', default=30000, type=int)
    parser.add_argument('--rkl_prior_beta', default=1e-2, type=float)
    args = parser.parse_args()

    for path in [args.checkpoint, args.zip_folder]:
        if not os.path.isdir(path):
            raise ValueError(
                "Incorrect path to folder:" + path
            )
    # Load model
    channels = {
        'l1-ssim': 1,
        'gaussian': 2, 'nw_prior': 3, 'nw_prior_rkl': 3, 'nw_end': 2,
        'hydra': len(args.teacher_checkpoints) * 2
    }[args.model_type]
    if args.pretrained_path is None:
        model = UNetModel(args.backbone, out_channels=channels).cuda()
    else:
        model = UNetModel(
            args.backbone, pretrained=False, out_channels=channels
        ).cuda()

        loaded_densenet = densenet169(pretrained=False)
        _load_densenet_dict(loaded_densenet, args.pretrained_path)
        model.encoder.original_model = loaded_densenet.features.cuda()
        if args.model_type == 'nw_prior_rkl':
            # Adjust L and \beta initialization for RKL
            model.decoder.conv3.weight[0].data.mul_(10)
            model.decoder.conv3.weight[1].data.mul_(0.001)
    model = torch.nn.DataParallel(model)
    if args.model_type == 'gaussian' or args.model_type == 'nw_end':
        model = ProbabilisticWrapper(Normal, model)
    elif 'nw' in args.model_type:
        model = ProbabilisticWrapper(
            NormalWishartPrior, model
        )
    elif args.model_type == 'hydra':
        model = ProbabilisticWrapper(
            GaussianDiagonalMixture, model
        )
    print("Model created")

    if args.teacher_checkpoints is not None:
        teacher_model = load_unet_model_from_checkpoint(
            args.teacher_checkpoints, "gaussian-ensemble", args.backbone
        )

    logdir = args.log_dir
    if logdir == '':
        logdir = 'logs/' + '{}-lr{}-e{}-bs{}'.format(
            args.backbone, args.lr, args.epochs, args.bs
        )

    # Create trainer
    if args.model_type == 'gaussian':
        print("Training with NLL objective")
        trainer_cls = NyuNLLDistributionTrainer(
            model, torch.optim.Adam, SummaryWriter, logdir,
            epochs=args.epochs, optimizer_args={
                'lr': args.lr, 'amsgrad': True, 'warmup_steps': args.warmup_steps
            },
            additional_params={'targets_transform': args.targets_transform}
        )
    elif args.teacher_checkpoints is not None:
        if args.model_type == 'nw_end':
            print("Distilling with pairwise kl divergence")
        elif args.model_type == 'hydra':
            print("Distilling into multiple heads simultaneously")
        else:
            print("Distilling with log prob")
        max_T = args.max_temperature
        trainer_cls = NyuDistillationTrainer(
            teacher_model, max_T,
            model, torch.optim.Adam, SummaryWriter, logdir,
            args.epochs, {'lr': args.lr, 'amsgrad': True, 'warmup_steps': args.warmup_steps},
            additional_params={'targets_transform': args.targets_transform}
        )
    elif args.model_type == 'nw_prior_rkl':
        print("Performing RKL training with custom OOD data")
        ood_loader, _ = getTrainingEvalData(
            path=args.zip_folder + '/nyu_ood_church.zip', batch_size=args.bs,
            sanity_check=args.overfit, is_ood=True
        )
        trainer_cls = NyuRKLTrainer(
            model, torch.optim.Adam, SummaryWriter, logdir,
            epochs=args.epochs, optimizer_args={
                'lr': args.lr, 'amsgrad': True, 'warmup_steps': args.warmup_steps
            },
            additional_params={
                'targets_transform': args.targets_transform,
                'inv_real_beta': args.rkl_inv_beta,
                'ood_coeff': args.rkl_ood_coeff,
                'prior_beta': args.rkl_prior_beta,
                "ood_coeff_warmup": args.rkl_warmup_steps
            }
        )
    else:
        raise Exception("RKL training for Nyu currently not supported")
    print("Trainer created")

    # Load data
    train_loader, val_loader = getTrainingEvalData(
        path=args.zip_folder + '/nyu_data.zip', batch_size=args.bs,
        sanity_check=args.overfit
    )
    print("Data loaded")

    print("Training...")
    if args.model_type == 'nw_prior_rkl':
        trainer_cls.train(
            train_loader, val_loader, ood_loader,
            args.checkpoint, args.state_dict
        )
    else:
        trainer_cls.train(
            train_loader, val_loader, args.checkpoint, args.state_dict
        )
