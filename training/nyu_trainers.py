"""Train a single model on NyuV2 dataset."""
from datetime import timedelta

import numpy as np
import torchvision.utils as vutils

from distributions.distribution_wrappers import ProbabilisticWrapper
from distributions.distribution_wrappers import GaussianEnsembleWrapper
from training.distribution_trainer import NLLSingleDistributionTrainer, SingleDistributionTrainer
from training.distillation_trainer import DistillationTrainer
from training.rkl_nwp_trainer import NWPriorRKLTrainer

from evaluation.depth_testing import compute_rel_metrics
from utils.depth_utils import DepthNorm, predict_targets
from utils.viz_utils import colorize


class NyuNLLDistributionTrainer(NLLSingleDistributionTrainer):
    def __init__(
        self, model: ProbabilisticWrapper, optimizer_cls, logger_cls,
        logger_logdir='.',
        epochs=20,
        optimizer_args={'lr': 1e-4, 'amsgrad': True},
        additional_params={'targets_transform': 'scaled'}
    ):
        """NyuV2-specialized single-model trainer"""
        super(NyuNLLDistributionTrainer, self).__init__(
            model, optimizer_cls, logger_cls, logger_logdir,
            epochs, optimizer_args
        )
        self.additional_params = additional_params

    def preprocess_batch(self, batch):
        inputs, targets = batch['image'].to(self.device), \
            batch['depth'].to(self.device)
        # Normalize depth
        targets_n = DepthNorm(
            targets,
            transform_type=self.additional_params['targets_transform']
        )
        return inputs, targets_n

    def logging_step(
        self, val_loader, current_step,
        current_epoch, step_idx, steps_per_epoch
    ):
        self.logger.add_scalar('Train/Loss', self.loss_stats.val, current_step)
        if current_step % 300 == 0:
            eta = str(timedelta(
                seconds=int(
                    self.timing_stats.val * (steps_per_epoch - step_idx)
                )
            ))
            # Print to console and log val
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                'ETA {eta}\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    current_epoch, step_idx, steps_per_epoch,
                    batch_time=self.timing_stats,
                    loss=self.loss_stats, eta=eta
                )
            )
        if current_step % 6336 == 0:
            # Record intermediate results
            self.show_examples_and_get_val_metrics(val_loader, current_step)
            self.model.train()

    def show_examples_and_get_val_metrics(self, val_loader, current_step):
        self.model.eval()
        batch = next(iter(val_loader))
        sample_img, sample_depth = batch['image'].to(self.device), \
            batch['depth'].to(self.device) / 1000.0
        if current_step == 0:
            self.logger.add_image(
                'Train.1.Image',
                vutils.make_grid(sample_img, nrow=6, normalize=True),
                0
            )
            self.logger.add_image(
                'Train.2.Depth',
                colorize(
                    vutils.make_grid(sample_depth, nrow=6, normalize=False)
                ),
                0
            )

        prediction = predict_targets(
            self.model, sample_img.permute(0, 2, 3, 1).cpu().numpy(),
            transform_type=self.additional_params['targets_transform'],
            device=self.device
        ).permute(0, 3, 1, 2)

        self.logger.add_image(
            'Train.3.Prediction',
            colorize(
                vutils.make_grid(prediction, nrow=6, normalize=False)
            ), current_step
        )
        self.logger.add_image(
            'Train.3.Diff',
            colorize(
                vutils.make_grid(
                    (prediction - sample_depth.cpu()).abs(),
                    nrow=6, normalize=False
                )
            ), current_step
        )
        
        all_metrics_buffer = []
        for i, batch in enumerate(val_loader):
            sample_img, sample_depth = batch['image'].to(self.device), \
                batch['depth'].to(self.device)
            prediction = predict_targets(
                self.model, sample_img.permute(0, 2, 3, 1).cpu().numpy(),
                minDepth=1e-2, maxDepth=80.0,
                transform_type=self.additional_params['targets_transform'],
                device=self.device, clip=True, no_renorm=True
            ).permute(0, 3, 1, 2)
            all_metrics = compute_rel_metrics(
                sample_depth.cpu().numpy(), prediction.cpu().numpy()
            )
            all_metrics_buffer.append(all_metrics)
        metric_names = ['delta_1', 'delta_2', 'delta_3', 'rel', 'rms', 'log10']
        all_metrics_buffer = np.array(all_metrics_buffer).mean(axis=0)
        for mname, metric in zip(metric_names, all_metrics_buffer):
            self.logger.add_scalar("Val/" + mname, metric, current_step)

        print("Eval scores", end='\t', flush=True)
        for mname, metric in zip(metric_names, all_metrics_buffer):
            print(mname, ': %.3f' % metric, end=' ', flush=True)
        print("")


class NyuDistillationTrainer(DistillationTrainer, NyuNLLDistributionTrainer):
    def __init__(
        self, teacher_model: GaussianEnsembleWrapper, T, *args, **kwargs
    ):
        self.additional_params = kwargs.pop('additional_params', None)
        super(NyuDistillationTrainer, self).__init__(
            teacher_model, T, *args, **kwargs
            )

    def preprocess_batch(self, batch):
        return NyuNLLDistributionTrainer.preprocess_batch(self, batch)

    def logging_step(
        self, val_loader, current_step,
        current_epoch, step_idx, steps_per_epoch
    ):
        return NyuNLLDistributionTrainer.logging_step(
            self,
            val_loader, current_step, current_epoch, step_idx, steps_per_epoch
        )

    def show_examples_and_get_val_metrics(self, val_loader, current_step):
        return NyuNLLDistributionTrainer.show_examples_and_get_val_metrics(
            self, val_loader, current_step
        )


class NyuRKLTrainer(NWPriorRKLTrainer, NyuNLLDistributionTrainer):
    def preprocess_batch(self, batch):
        return NyuNLLDistributionTrainer.preprocess_batch(self, batch)

    def preprocess_ood_batch(self, batch):
        return batch['image'].to(self.device)

    def logging_step(
        self, val_loader, current_step,
        current_epoch, step_idx, steps_per_epoch
    ):
        return NyuNLLDistributionTrainer.logging_step(
            self,
            val_loader, current_step, current_epoch, step_idx, steps_per_epoch
        )

    def show_examples_and_get_val_metrics(self, val_loader, current_step):
        return NyuNLLDistributionTrainer.show_examples_and_get_val_metrics(
            self, val_loader, current_step
        )
