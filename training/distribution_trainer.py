"""
Train a single model that was wrapped with our distribution wrappers.
Use either NLL or RKL-based objectives
"""
from time import time
import os

import torch
import torch.nn as nn
import numpy as np

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
        self.warmup_steps = optimizer_args.pop("warmup_steps")
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

        loss = self.loss_fn(output_distr, targets)
        return loss

    def loss_fn(self, output_distr, targets):
        raise NotImplementedError

    def update_step(self, loss):
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

    def warmup_lr(self, step):
        for g in self.optimizer.param_groups:
            if not hasattr(self, 'default_lr'):
                self.default_lr = g['lr']
            g['lr'] = self.default_lr * float(step) / self.warmup_steps

    def train(self, train_loader, val_loader, save_path=None, load_path=None):
        if load_path is not None and os.path.isfile(load_path + '/traindata.ckpt'):
            init_epoch, global_step = self.load_current_state(load_path)
        else:
            init_epoch, global_step = 0, 0
            self.logger = self.logger_cls(logdir=self.logger_logdir)

        if not hasattr(self, 'max_steps'):
            self.max_steps = len(train_loader) * self.epochs
            self.len_train_loader = len(train_loader)

        print("Starting training for %d steps" % self.max_steps)

        for epoch in range(init_epoch, self.epochs):
            self.loss_stats.reset()
            self.timing_stats.reset()

            start_time = time()
            self.model.train()
            for i, batch in enumerate(train_loader):
                if epoch == 0 and i < self.warmup_steps:
                    self.warmup_lr(i)
                current_step = global_step + i
                self.current_step = current_step

                self.optimizer.zero_grad()
                loss = self.train_step(batch)
                self.update_step(loss)

                with torch.no_grad():
                    self.logging_step(
                        val_loader, current_step,
                        epoch, i, self.len_train_loader
                    )

                elapsed_time = time() - start_time
                self.timing_stats.update(elapsed_time)
                self.loss_stats.update(loss.item())
                start_time = time()

            global_step += self.len_train_loader

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


from torch.distributions.normal import Normal
from torch.distributions import StudentT

class NLLSingleDistributionTrainer(SingleDistributionTrainer):
    def loss_fn(self, output_distr, targets):
        if isinstance(output_distr, Normal):
            return -output_distr.log_prob(targets).mean()
        else:
            # Sergey wrote:
            #m, L, beta = output_distr.loc, output_distr.precision_diag, output_distr.belief.unsqueeze(-1)
            #sos_terms = (torch.log((1 + beta) / (beta * L) + beta * (targets.unsqueeze(-1) - m).pow(2)) 
            #              + torch.lgamma(beta / 2) - torch.lgamma(beta / 2 - 1))
            #reg_terms = 2 * (beta + 1) * (targets.unsqueeze(-1) - m).pow(2)
            #return (sos_terms + reg_terms).mean()

            # New version
            m, L, kappa, nu = output_distr.loc, output_distr.precision_diag, output_distr.belief.unsqueeze(-1), output_distr.df.unsqueeze(-1)
            nll_terms = ( 0.5 * (-torch.log(kappa/np.pi)) - nu/2 * (torch.log(1 + kappa)-torch.log(L)) 
                        + (nu/2 + 0.5) * ( -torch.log(L) + torch.log((targets.unsqueeze(-1) - m) ** 2 * kappa * L + 1 + kappa) )
                        + torch.lgamma(nu/2) - torch.lgamma(nu/2 + 0.5) 
                        )
            # + (nu/2 + 0.5) * torch.log((targets.unsqueeze(-1) - m) ** 2 * kappa +  (1 + kappa) / L)
            reg_terms = torch.abs(targets.unsqueeze(-1) - m) * (2 * kappa + nu/2)
            
            return (nll_terms + 0.1 * reg_terms).mean()

            #Student
            #precision_coeff = (output_distr.belief + 1) / (
            #output_distr.belief * (output_distr.df - output_distr.dimensionality + 1))
            #return -StudentT((output_distr.df - output_distr.dimensionality + 1).unsqueeze(-1), loc=output_distr.loc,
            #        scale=(
            #            precision_coeff.unsqueeze(-1) / output_distr.precision_diag
            #        ).pow(0.5),
            #        ).log_prob(targets.unsqueeze(-1)).mean()
