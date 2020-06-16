import math
import torch

from torch.distributions import StudentT
from distributions.diagonal_normal_wishart import NormalDiagonalWishart
from utils.func_utils import mvdigamma


class NormalWishartPrior(NormalDiagonalWishart):
    r"""
    A Normal-Wishart prior distribution to emulate ensemble
    of Gaussians
    """

    def forward(self):
        """Returns predictive posterior distribution"""
        self.precision_coeff = (self.belief + 1) / (
            self.belief * (self.df - self.dimensionality + 1)
        )
        return StudentT(
            (self.df - self.dimensionality + 1).unsqueeze(-1),
            loc=self.loc,
            scale=(
                self.precision_coeff.unsqueeze(-1) / self.precision_diag
            ).pow(0.5),
        )

    @property
    def mean(self):
        """Returns predictive posterior mean"""
        ppe_mean = self.forward().mean
        if ppe_mean.size(-1) == 1:
            return ppe_mean[..., 0]
        return ppe_mean

    def predictive_posterior_log_prob(self, value):
        return self.forward().log_prob(value)

    def predictive_posterior_variance(self):
        variance_res = self.forward().variance
        if variance_res.size(-1) != 1:
            raise ValueError(
                "Predictive posterior returned entropy with incorrect shapes"
            )
        return variance_res[..., 0]

    def predictive_posterior_entropy(self):
        entropy_res = self.forward().entropy()
        if entropy_res.size(-1) != 1:
            raise ValueError(
                "Predictive posterior returned entropy with incorrect shapes"
            )
        return entropy_res[..., 0]

    def expected_entropy(self):
        mvdigamma_term = mvdigamma(0.5 * self.df, self.dimensionality)
        return 0.5 * (
            self.dimensionality * (1 + math.log(2 * math.pi)) -
            (2 * self.precision_diag).log().sum(dim=-1) -
            mvdigamma_term
        )

    def expected_log_prob(self, value):
        neg_mse_term = -torch.sum(
            (self.loc - value).pow(2) * self.precision_diag *
            self.df.unsqueeze(-1),
            dim=-1
        )
        mvdigamma_term = mvdigamma(0.5 * self.df, self.dimensionality)

        reg_terms = (
            2 * self.precision_diag
        ).log().sum(dim=-1) + mvdigamma_term
        conf_term = -self.dimensionality * self.belief.pow(-1)
        return 0.5 * (neg_mse_term + reg_terms + conf_term)

    def mutual_information(self):
        predictive_posterior_entropy = self.predictive_posterior_entropy()
        expected_entropy = self.expected_entropy()
        return predictive_posterior_entropy - expected_entropy

    def expected_pairwise_kl(self):
        term1 = 0.5 * (
            self.df * self.dimensionality / (
                self.df - self.dimensionality - 1
            ) - self.dimensionality
        )
        term2 = 0.5 * (
            self.df * self.dimensionality / (
                self.df - self.dimensionality - 1
            ) + self.dimensionality
        ) / self.belief
        return term1 + term2

    def variance_of_expected(self):
        return self.expected_variance() / self.belief

    def expected_variance(self):
        result = 1 / (
            self.precision_diag * (
                self.df.unsqueeze(-1) - self.dimensionality - 1
            )
        )
        if result.size(-1) != 1:
            raise ValueError(
                "Expected variance currently supports\
                    only one-dimensional targets"
            )

        return result[..., 0]

    def total_variance(self):
        tv = self.variance_of_expected() + self.expected_variance()
        ppv = self.predictive_posterior_variance()

        rel_diff = (tv - ppv).abs() / tv.abs().pow(0.5) / ppv.abs().pow(0.5)
        assert (rel_diff < 1e-6).all()
        return tv


def test_nw_prior():
    print("Testing Normal-Wishart Prior...")
    ex_mean = torch.zeros(32, 1, 200, 400, 1)
    ex_var = torch.ones(32, 1, 200, 400, 1)
    ex_belief = torch.ones(32, 1, 200, 400)
    ex_df = 10 * torch.ones(32, 1, 200, 400)

    ex_dist = NormalWishartPrior(ex_mean, ex_var, ex_belief, ex_df)
    assert ex_dist.predictive_posterior_log_prob(
        2 * torch.ones(32, 1, 200, 400, 1)
    ).shape == (32, 1, 200, 400, 1)
    assert ex_dist.log_prob(
        2 * torch.ones(32, 1, 200, 400, 1),
        2 * torch.ones(32, 1, 200, 400, 1),
    ).shape == (32, 1, 200, 400)

    for method in [
        'predictive_posterior_entropy', 'mutual_information',
        'expected_entropy', 'expected_pairwise_kl',
        'variance_of_expected', 'expected_variance', 'total_variance',
        'predictive_posterior_variance'
    ]:
        assert getattr(ex_dist, method)().shape == (32, 1, 200, 400)

    print("Passed")
