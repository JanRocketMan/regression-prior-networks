"""
Train a single model that was wrapped with our distribution wrappers.
Use either NLL or RKL-based objectives
"""
from time import time
import datetime

import torch
import torch.nn as nn

from distributions.distribution_wrappers import ProbabilisticWrapper
from utils.func_utils import AverageMeter


class SingleDistributionTrainer:
    def __init__(
        self, model: ProbabilisticWrapper, optimizer, logger,
        optimizer_args={'lr': 1e-4, 'amsgrad': True},
        epochs=20, additional_params={'targets_transform': 'scaled'}
    ):
        self.model = model
        self.device = self.model.model.device
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), *optimizer_args
        )
        self.logger = logger
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
        output_distr = model(inputs)
        loss = self.loss_fn(targets_n).mean()
        return loss, batch_size

    def update_step(self, loss):
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        self.optimizer.step()

    def train(self, train_loader, val_loader, save_path=None, load_path=None):
        global_step = 0

        if load_path is not None:
            init_epoch = self.load_current_state(load_path)

        for epoch in range(init_epoch, self.epochs):
            self.loss_stats.reset()
            self.timing_stats.reset()

            start_time = time()
            for i, batch in enumerate(train_loader):
                current_step = global_step + i

                self.optimizer.zero_grad()
                loss, batch_size = self.train_step(batch)
                self.update_step(loss)

                elapsed_time = time() - start_time
                start_time = time()
                self.timing_stats.update(elapsed_time)
                self.loss_stats.update(loss.item(), batch_size)

                self.logging_step(val_loader, current_step, epoch, i)

            global_step += len(train_loader)

            self.eval_step(val_loader, global_step, epoch)
            if save_path is not None:
                self.save_current_state(save_path, global_step)

    def logging_step(self, val_loader, current_step, epoch, step_idx):
        self.logger.add_scalar('Train/Loss', self.loss_stats.val, current_step)
        if current_step % 300 == 0:
            steps_per_epoch = int((current_step - step_idx) / epoch)
            eta = str(datetime.timedelta(
                seconds=int(
                    self.timing_stats.val * steps_per_epoch - step_idx
                )
            ))
            # Print to console and log val
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                'ETA {eta}\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch, step_idx, steps_per_epoch,
                    batch_time=self.timing_stats,
                    loss=self.loss_stats, eta=eta
                )
            )
            # Record intermediate results
            self.show_examples_and_get_metrics(val_loader, current_step)
            self.model.train()

    def eval_step(self, val_loader, current_step, epoch):
        self.model.eval()
        self.show_examples_and_get_metrics(val_loader, current_step)
        self.logger.add_scalar('Train/Loss.avg', self.loss_stats.avg, epoch)

    def show_examples_and_get_metrics(self, val_loader, current_step):
        raise NotImplementedError

    def save_current_state(self, save_path, global_step):
        raise NotImplementedError

    def load_current_state(self, load_path):
        raise NotImplementedError
