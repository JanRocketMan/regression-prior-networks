"""
Train a single model that is wrapped into NormalWishartPrior distribution.
Uses RKL-based objective (see formula 11 in our paper).
"""
from time import time
import os

import torch
from torch.distributions.kl import kl_divergence

from itertools import cycle
from distributions.distribution_wrappers import ProbabilisticWrapper
from distributions import NormalWishartPrior
from training.distribution_trainer import SingleDistributionTrainer
from utils.model_utils import switch_bn_updates


class NWPriorRKLTrainer(SingleDistributionTrainer):
    def __init__(
        self, model: ProbabilisticWrapper, optimizer_cls, logger_cls,
        logger_logdir,
        epochs,
        optimizer_args,
        additional_params
    ):
        super(NWPriorRKLTrainer, self).__init__(
            model, optimizer_cls, logger_cls, logger_logdir,
            epochs, optimizer_args
        )
        self.loss_params = {}
        self.loss_params["inv_real_beta"] = additional_params["inv_real_beta"]
        self.loss_params["ood_coeff"] = additional_params["ood_coeff"]
        self.loss_params["prior_beta"] = additional_params["prior_beta"]
        self.loss_params["ood_coeff_warmup"] = additional_params["ood_coeff_warmup"]

    def warmup_ood_coeff(self, step):
        maxstep = self.loss_params["ood_coeff_warmup"]
        if maxstep > 0 and step < maxstep:
            self.ood_coeff = self.loss_params["ood_coeff"] * (step / maxstep)
        else:
            self.ood_coeff = self.loss_params["ood_coeff"]

    def train(
        self, train_loader, val_loader,
        ood_loader=None, save_path=None, load_path=None
    ):
        if ood_loader is None:
            raise Exception("Training with RKL objective requires OOD data.")
        self.estimate_targets_avg_mean_var(train_loader)  # Required for Prior

        if load_path is not None and os.path.isfile(load_path + '/traindata.ckpt'):
            print("Loading model from", load_path)
            init_epoch, global_step = self.load_current_state(load_path)
        else:
            print("No model found at", load_path, "starting from beginning")
            init_epoch, global_step = 0, 0
            self.logger = self.logger_cls(logdir=self.logger_logdir)

        if not hasattr(self, 'max_steps'):
            self.max_steps = len(train_loader) * self.epochs
            self.len_train_loader = len(train_loader)

        print("Starting training for %d steps" % self.max_steps)

        # Memory-efficient cycling of ood dataset with trick by
        # https://github.com/pytorch/pytorch/issues/1917#issuecomment-433698337
        ood_iterator = iter(ood_loader)

        for epoch in range(init_epoch, self.epochs):
            self.loss_stats.reset()
            self.timing_stats.reset()

            start_time = time()
            self.model.train()
            for i, batch in enumerate(train_loader):
                # Warmup learning rate
                if epoch == 0 and i < self.warmup_steps:
                    self.warmup_lr(i)
                # Warmup ood coefficient
                self.warmup_ood_coeff(i + epoch * len(train_loader))
                try:
                    batch_ood = next(ood_iterator)
                except StopIteration:
                    ood_iterator = iter(ood_loader)
                    batch_ood = next(ood_iterator)
                batch = (batch, batch_ood)

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

    def preprocess_ood_batch(self, batch):
        return batch.to(self.device)

    def train_step(self, batch):
        id_inputs, id_targets = self.preprocess_batch(batch[0])
        ood_inputs = self.preprocess_ood_batch(batch[1])

        # Predict & get loss on out-of-domain data
        switch_bn_updates(self.model, "eval")  # Disable BN acc on OOD
        ood_output_distr = self.model(ood_inputs)
        ood_prior_distr = self.get_prior_distribution(ood_inputs)
        ood_loss = self.loss_fn(
            ood_output_distr, ood_prior_distr, None
        )
        for param_n in ['loc', 'precision_diag', 'belief']:
            self.logger.add_scalar(
                'Prior/OOD_' + param_n + '_mean', getattr(ood_prior_distr, param_n).mean().item(),
                self.current_step
            )
            self.logger.add_scalar(
                'Prior/OOD_' + param_n + '_std', getattr(ood_prior_distr, param_n).std().item(),
                self.current_step
            )
        for param_n in ['loc', 'precision_diag', 'belief']:
            self.logger.add_scalar(
                'Output/OOD_' + param_n + '_mean', getattr(ood_output_distr, param_n).mean().item(),
                self.current_step
            )
            self.logger.add_scalar(
                'Output/OOD_' + param_n + '_std', getattr(ood_output_distr, param_n).std().item(),
                self.current_step
            )

        # Predict & get loss on in-domain data
        switch_bn_updates(self.model, "train")
        id_output_distr = self.model(id_inputs)
        id_prior_distr = self.get_prior_distribution(id_inputs)
        id_loss = self.loss_fn(
            id_output_distr, id_prior_distr, id_targets
        )
        if self.current_step == 1:
            import torchvision.utils as vutils
            vutils.save_image(id_inputs.cpu(), "id_ex.png", nrow=3)
            vutils.save_image(ood_inputs.cpu(), "ood_ex.png", nrow=3)
            print("ID:", id_inputs.shape, id_inputs.min(), id_inputs.max(), id_inputs.mean(), id_inputs.std())
            print("OOD:", ood_inputs.shape, ood_inputs.min(), ood_inputs.max(), ood_inputs.mean(), ood_inputs.std())
            print("ID targets:", id_targets.shape, id_targets.min(), id_targets.max(), id_targets.mean(), id_targets.std())
            vutils.save_image(id_targets.cpu().repeat(1, 3, 1, 1), "id_targets.png", nrow=3, normalize=True)

        for param_n in ['loc', 'precision_diag', 'belief']:
            self.logger.add_scalar(
                'Prior/ID_' + param_n + '_mean', getattr(id_prior_distr, param_n).mean().item(),
                self.current_step
            )
            self.logger.add_scalar(
                'Prior/ID_' + param_n + '_std', getattr(id_prior_distr, param_n).std().item(),
                self.current_step
            )
        for param_n in ['loc', 'precision_diag', 'belief']:
            self.logger.add_scalar(
                'Output/ID_' + param_n + '_mean', getattr(id_output_distr, param_n).mean().item(),
                self.current_step
            )
            self.logger.add_scalar(
                'Output/ID_' + param_n + '_std', getattr(id_output_distr, param_n).std().item(),
                self.current_step
            )

        return id_loss + self.ood_coeff * ood_loss

    def loss_fn(self, predicted_distr, prior_distr, targets=None):
        if targets is not None:
            n_exp_log_prob = -predicted_distr.expected_log_prob(
                targets.unsqueeze(-1)
            ).mean()
            weighted_kl = self.loss_params["inv_real_beta"] * kl_divergence(
                predicted_distr, prior_distr
            ).mean()
            self.logger.add_scalar(
                "Train/ID_neg_exp_log_prob", n_exp_log_prob.item(),
                self.current_step
            )
            self.logger.add_scalar(
                "Train/ID_weighted_kl", weighted_kl.item(),
                self.current_step 
            )

            return n_exp_log_prob + weighted_kl
        else:
            weighted_kl = self.loss_params["inv_real_beta"] * kl_divergence(
                predicted_distr, prior_distr
            ).mean()
            self.logger.add_scalar(
                "Train/OOD_weighted_kl", weighted_kl.item(),
                self.current_step 
            )
            return weighted_kl

    def get_prior_distribution(self, inputs):
        prior_mean = torch.repeat_interleave(
            self.avg_mean.unsqueeze(0), repeats=inputs.size(0), dim=0
        ).to(inputs.device)
        prior_beta = self.loss_params['prior_beta'] * torch.ones_like(
            prior_mean
        ).to(self.device)
        prior_kappa, prior_nu = prior_beta, prior_beta + prior_mean.size(1) + 2 #!
        prior_precision = (
            1 / (prior_nu * self.avg_scatter.unsqueeze(0))
        ).to(inputs.device)

        all_params = [
            prior_mean.unsqueeze(-1), prior_precision.unsqueeze(-1),
            prior_kappa, prior_nu
        ]
        return NormalWishartPrior(*all_params)

    def estimate_targets_avg_mean_var(self, dataloader):
        """Computes average statistics of a given dataset"""
        print("Computing average stats for prior distr...")
        self.avg_mean = None
        for i, batch in enumerate(dataloader):
            _, y = self.preprocess_batch(batch)
            if self.avg_mean is None:
                self.avg_mean = y.mean(dim=0) / min(200, len(dataloader))
            else:
                self.avg_mean += y.mean(dim=0) / min(200, len(dataloader))
            if i > 200:
                break
        sum_var = torch.zeros_like(self.avg_mean)
        num_samples = 0
        for i, batch in enumerate(dataloader):
            _, y = self.preprocess_batch(batch)
            avg_mean = torch.repeat_interleave(
                self.avg_mean.unsqueeze(0), repeats=y.size(0), dim=0
            )
            sum_var += (y - avg_mean).pow(2).sum(dim=0)
            num_samples += y.size(0)
            if i > 200:
                break
        self.avg_scatter = sum_var / num_samples
        print("Finished, starting training...")
