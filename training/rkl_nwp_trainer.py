"""
Train a single model that is wrapped into NormalWishartPrior distribution.
Uses RKL-based objective (see formula 11 in our paper).
"""
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

    def train(
        self, train_loader, val_loader,
        ood_loader=None, save_path=None, load_path=None
    ):
        if ood_loader is None:
            raise Exception("Training with RKL objectives requires OOD data.")
        self.estimate_targets_avg_mean_var(train_loader)  # Required for Prior
        mixed_loader = zip(dataloader, cycle(oodloader))
        """
        With only zip() the iterator will be exhausted when the length
        is equal to that of the smallest dataset.
        But with the use of cycle(), we will repeat the smallest dataset again
        unless our iterator looks at all the samples from the largest dataset.
        """
        super(NWPriorRKLTrainer, self).train(
            mixed_loader, val_loader, save_path, load_path
        )

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

        # Predict & get loss on in-domain data
        switch_bn_updates(self.model, "train")
        id_output_distr = self.model(id_inputs)
        id_prior_distr = self.get_prior_distribution(id_inputs)
        id_loss = self.loss_fn(
            id_output_distr, id_prior_distr, id_targets
        )
        return id_loss + self.loss_params["ood_coeff"] * ood_loss

    def loss_fn(self, predicted_distr, prior_distr, targets=None):
        if targets is not None:
            return -predicted_distr.expected_log_prob(
                targets
            ).mean() + self.loss_params["inv_real_beta"] * kl_divergence(
                predicted_distr, prior_distr
            ).mean()
        else:
            return self.loss_params["inv_real_beta"] * kl_divergence(
                predicted_distr, prior_distr
            ).mean()

    def get_prior_distribution(self, inputs):
        prior_mean = torch.repeat_interleave(
            self.avg_mean.unsqueeze(0), repeats=inputs.size(0), dim=0
        ).to(inputs.device)
        prior_precision = torch.repeat_interleave(
            (1 / (prior_nu * self.avg_scatter.unsqueeze(0))),
            repeats=inputs.size(0), dim=0
        ).to(inputs.device)
        prior_beta = self.loss_params['prior_beta'] * torch.ones_like(
            prior_mean
        ).to(self.device)
        prior_kappa, prior_nu = prior_beta, prior_beta + prior_mean.size(1) + 1

        all_params = [
            prior_mean.unsqueeze(-1), prior_precision.unsqueeze(-1),
            prior_kappa, prior_nu
        ]
        return NormalWishartPrior(*all_params)

    def estimate_targets_avg_mean_var(self, dataloader):
        """Computes average statistics of a given dataset"""
        self.avg_mean = None
        for batch in dataloader:
            _, y = self.preprocess_batch(batch)
            if self.avg_mean is None:
                self.avg_mean = y.mean(dim=0) / len(dataloader)
            else:
                self.avg_mean += y.mean(dim=0) / len(dataloader)
        sum_var = torch.zeros_like(self.avg_mean)
        num_samples = 0
        for _, batch in dataloader:
            _, y = self.preprocess_batch(batch)
            avg_mean = torch.repeat_interleave(
                self.avg_mean.unsqueeze(0), repeats=y.size(0), dim=0
            )
            sum_var += (y - avg_mean).pow(2).sum(dim=0)
            num_samples += y.size(0)
        self.avg_scatter = sum_var / num_samples
