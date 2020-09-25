"""Distill an ensemble into a single model"""
import torch
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from distributions import GaussianDiagonalMixture
from distributions.distribution_wrappers import GaussianEnsembleWrapper
from training.distribution_trainer import SingleDistributionTrainer


def smoothed_kl_normal_normal(p, q, temperature):
    # Adjust KL divergence by multiplying MSE term by 2T^2
    var_ratio = (p.scale / q.scale).pow(2)
    t1 = ((p.loc - q.loc) / q.scale).pow(2)
    return 0.5 * (var_ratio + 2 * (temperature ** 2) * t1 - 1 - var_ratio.log())


class DistillationTrainer(SingleDistributionTrainer):
    def __init__(
        self, teacher_model: GaussianEnsembleWrapper, T, *args, **kwargs
    ):
        super(DistillationTrainer, self).__init__(
            *args, **kwargs
        )
        self.max_temperature = T

        self.teacher_model = teacher_model
        self.teacher_model.eval()

    def train_step(self, batch):
        inputs, targets = self.preprocess_batch(batch)

        # Get teacher prediction
        with torch.no_grad():
            teacher_distr = self.teacher_model(inputs)
        # Predict & get loss
        output_distr = self.model(inputs)
        loss = self.distill_loss_fn(output_distr, teacher_distr, targets)

        return loss

    def get_current_temperature(self):
        # Use linear schedule for temperature decay
        T = 1
        T_0 = self.max_temperature
        first_part = float(0.2 * self.max_steps)
        third_part = float(0.6 * self.max_steps)
        if self.current_step < first_part:
            T = T_0
        elif self.current_step < third_part:
            T = T_0 - (T_0 - 1) * min(
                float(self.current_step - first_part) /
                (third_part - first_part),
                1.0
            )
        return T

    def smooth_params(self, all_means, all_vars, temperature):
        return all_means, [var * (temperature ** 2) for var in all_vars]

    def smooth_predicted_params(self, output_distr, temperature):
        min_df = output_distr.dimensionality + 2  # !
        output_distr.belief *= temperature
        output_distr.df = (output_distr.df - min_df) * temperature + min_df
        return output_distr

    def distill_loss_fn(self, output_distr, teacher_distr, targets):
        T = self.get_current_temperature()

        all_means, all_vars = [
            [
                getattr(dist, attr_n) for dist in teacher_distr.distributions
            ]
            for attr_n in ['loc', 'variance']
        ]

        smoothed_means, smoothed_vars = self.smooth_params(
            all_means, all_vars, T
        )

        all_losses = []
        if isinstance(output_distr, Normal):
            output_distr.scale *= T
            for i in range(len(smoothed_means)):
                all_losses.append(
                    smoothed_kl_normal_normal(
                        output_distr,
                        Normal(smoothed_means[i], smoothed_vars[i].pow(0.5)),
                        T
                    ).mean()
                )
        elif isinstance(output_distr, GaussianDiagonalMixture):
            assert len(smoothed_means) == len(output_distr.distributions)
            for dist in output_distr.distributions:
                dist.scale *= T
            for i in range(len(smoothed_means)):
                all_losses.append(
                    smoothed_kl_normal_normal(
                        output_distr.distributions[i],
                        Normal(smoothed_means[i], smoothed_vars[i].pow(0.5)),
                        T
                    ).mean()
                )
        else:
            output_distr = self.smooth_predicted_params(output_distr, T)

            for i in range(len(smoothed_means)):
                all_losses.append(
                    -output_distr.log_prob(
                        smoothed_means[i].unsqueeze(-1).to(self.device),
                        1.0 / (
                            smoothed_vars[i].unsqueeze(-1).to(self.device) +
                            1e-5
                        )
                    ).mean()
                )
        return sum(all_losses) / (T * len(all_losses))
