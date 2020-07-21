"""
Train a single model that was wrapped with our distribution wrappers.
Use either NLL or RKL-based objectives
"""
from time import time
import os

import torch
import torch.nn as nn

from distributions.distribution_wrappers import ProbabilisticWrapper
from utils.func_utils import AverageMeter

import torchvision.utils as vutils

class SingleDistributionTrainer:
    def __init__(
        self, model: ProbabilisticWrapper, optimizer_cls, logger_cls,
        logger_logdir: str,
        epochs: int,
        optimizer_args: dict
    ):
        """General single-model trainer"""
        self.model = model
        self.device = next(self.model.parameters()).device
        self.optimizer = optimizer_cls(
            self.model.parameters(), **optimizer_args
        )
        self.epochs = epochs
        self.logger_cls, self.logger = logger_cls, None
        self.logger_logdir = logger_logdir
        self.loss_stats, self.timing_stats = [AverageMeter() for _ in range(2)]

    def preprocess_batch(self, batch):
        return batch[0].to(self.device), batch[1].to(self.device)

    def train_step(self, batch):
        inputs, targets = self.preprocess_batch(batch)

        # Predict & get loss
        output_distr = self.model(inputs)
        """
        vutils.save_image(inputs,'/home/iv-provilkov/data/uncertainty/kitti/inputs.png', normalize=True)
        vutils.save_image(targets, '/home/iv-provilkov/data/uncertainty/kitti/targets.png', normalize=True)
        vutils.save_image(output_distr.mean, '/home/iv-provilkov/data/uncertainty/kitti/output.png', normalize=True)
        print("INSIDE BATCH")
        print("OSHAPE", output_distr.mean.shape)
        print(inputs.min(), inputs.max())
        print(targets.min(), targets.max())
        print(output_distr.mean.min(), output_distr.mean.max())
        """

        loss = self.loss_fn(output_distr, targets)
        return loss

    def loss_fn(self, output_distr, targets):
        raise NotImplementedError

    def update_step(self, loss):
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

    def train(self, train_loader, val_loader, save_path=None, load_path=None):
        if load_path is not None and os.path.isfile(load_path + '/traindata.ckpt') :
            init_epoch, global_step = self.load_current_state(load_path)
        else:
            init_epoch, global_step = 0, 0
            self.logger = self.logger_cls(logdir=self.logger_logdir)

        self.max_steps = len(train_loader) * self.epochs

        for epoch in range(init_epoch, self.epochs):
            self.loss_stats.reset()
            self.timing_stats.reset()

            start_time = time()
            self.model.train()
            for i, batch in enumerate(train_loader):
                current_step = global_step + i
                self.current_step = current_step

                self.optimizer.zero_grad()
                loss = self.train_step(batch)
                self.update_step(loss)

                with torch.no_grad():
                    self.logging_step(
                        val_loader, current_step,
                        epoch, i, len(train_loader)
                    )

                elapsed_time = time() - start_time
                self.timing_stats.update(elapsed_time)
                self.loss_stats.update(loss.item())
                start_time = time()

            global_step += len(train_loader)

            self.model.eval()
            with torch.no_grad():
                self.eval_step(val_loader, global_step, epoch)
            if save_path is not None:
                self.save_current_state(save_path, global_step, epoch)

    def logging_step(
        self, val_loader, current_step,
        current_epoch, step_idx, steps_per_epoch
    ):
        self.logger.add_scalar('Train/Loss', self.loss_stats.val, current_step)

    def eval_step(self, val_loader, current_step, current_epoch):
        self.model.eval()
        self.show_examples_and_get_val_metrics(val_loader, current_step)
        self.logger.add_scalar(
            'Train/Loss.avg', self.loss_stats.avg, current_epoch
        )

    def show_examples_and_get_val_metrics(self, val_loader, current_step):
        raise NotImplementedError

    def save_current_state(self, save_path, global_step, epoch):
        train_state = {
            "epoch": epoch,
            "global_step": global_step,
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


class NLLSingleDistributionTrainer(SingleDistributionTrainer):
    def loss_fn(self, output_distr, targets):
        return -output_distr.log_prob(targets).mean()
