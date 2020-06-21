"""
Train a single model that was wrapped with our distribution wrappers.
Use either NLL or RKL-based objectives
"""
from time import time
import datetime

import torch
import torch.nn as nn
import torchvision.utils as vutils

from distributions.distribution_wrappers import ProbabilisticWrapper
from evaluation.depth_testing import compute_rel_metrics
from utils.func_utils import AverageMeter
from utils.depth_utils import DepthNorm, predict_targets
from utils.viz_utils import colorize


class SingleDistributionTrainer:
    def __init__(
        self, model: ProbabilisticWrapper, optimizer_cls, logger_cls,
        logger_logdir='.',
        epochs=20,
        optimizer_args={'lr': 1e-4, 'amsgrad': True},
        additional_params={'targets_transform': 'scaled'}
    ):
        self.model = model
        self.device = 'cuda:0'
        self.optimizer = optimizer_cls(
            self.model.parameters(), **optimizer_args
        )
        self.epochs = epochs
        self.logger_cls, self.logger = logger_cls, None
        self.logger_logdir = logger_logdir
        self.additional_params = additional_params
        self.loss_stats, self.timing_stats = [AverageMeter() for _ in range(2)]

    def train_step(self, batch):
        inputs, targets = batch['image'].to(self.device), \
            batch['depth'].to(self.device)
        batch_size = inputs.size(0)
        # Normalize depth
        targets_n = DepthNorm(
            targets,
            transform_type=self.additional_params['targets_transform']
        )
        # Predict & get loss
        output_distr = self.model(inputs)
        loss = self.loss_fn(output_distr, targets_n)
        return loss, batch_size

    def loss_fn(self, output_distr, targets):
        raise NotImplementedError

    def update_step(self, loss):
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

    def train(self, train_loader, val_loader, save_path=None, load_path=None):

        if load_path is not None:
            init_epoch, global_step = self.load_current_state(load_path)
        else:
            init_epoch, global_step = 0, 0
            self.logger = self.logger_cls(logdir=self.logger_logdir)

        for epoch in range(init_epoch, self.epochs):
            self.loss_stats.reset()
            self.timing_stats.reset()

            start_time = time()
            for i, batch in enumerate(train_loader):
                current_step = global_step + i

                self.optimizer.zero_grad()
                loss, batch_size = self.train_step(batch)
                self.update_step(loss)

                with torch.no_grad():
                    self.logging_step(
                        val_loader, current_step,
                        epoch, i, len(train_loader)
                    )

                elapsed_time = time() - start_time
                self.timing_stats.update(elapsed_time)
                self.loss_stats.update(loss.item(), batch_size)
                start_time = time()

            global_step += len(train_loader)

            with torch.no_grad():
                self.eval_step(val_loader, global_step, epoch)
            if save_path is not None:
                self.save_current_state(save_path, global_step, epoch)

    def logging_step(
        self, val_loader, current_step,
        current_epoch, step_idx, steps_per_epoch
    ):
        self.logger.add_scalar('Train/Loss', self.loss_stats.val, current_step)
        if current_step % 300 == 0:
            eta = str(datetime.timedelta(
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
            # Record intermediate results
            self.show_examples_and_get_metrics(val_loader, current_step)
            self.model.train()

    def eval_step(self, val_loader, current_step, current_epoch):
        self.model.eval()
        self.show_examples_and_get_metrics(val_loader, current_step)
        self.logger.add_scalar(
            'Train/Loss.avg', self.loss_stats.avg, current_epoch
        )

    def show_examples_and_get_metrics(
        self, val_loader, current_step
    ):
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

        all_metrics = compute_rel_metrics(
            sample_depth.cpu().numpy(), prediction.cpu().numpy()
        )
        metric_names = ['delta_1', 'delta_2', 'delta_3', 'rel', 'rms', 'log10']

        for mname, metric in zip(metric_names, all_metrics):
            self.logger.add_scalar("Val/" + mname, metric, current_step)

    def save_current_state(self, save_path, global_step, epoch):
        train_state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "optimizer": self.optimizer.state_dict(),
            "tensorboard_logdir": self.logger.logdir,
            "tensorboard_purge_step": global_step
        }
        end_str = '/' + str(epoch)
        torch.save(train_state, save_path + '/traindata.ckpt')
        torch.save(self.model.state_dict(), save_path + end_str + '.ckpt')

    def load_current_state(self, load_path):
        train_state = torch.load(load_path + '/traindata.ckpt')
        end_str = '/' + str(train_state["epoch"])
        self.model.load_state_dict(torch.load(load_path + end_str + '.ckpt'))
        self.optimizer.load_state_dict(train_state["optimizer"])
        self.logger = self.logger_cls(
            logdir=train_state["tensorboard_logdir"],
            purge_step=train_state["tensorboard_purge_step"]
        )
        return train_state["epoch"] + 1, train_state["global_step"]


class NLLDistributionTrainer(SingleDistributionTrainer):
    def loss_fn(self, output_distr, targets):
        return -output_distr.log_prob(targets).mean()
