"""Implements diagonal Normal-Wishart distribution
(see https://en.wikipedia.org/wiki/Normal-Wishart_distribution)"""

import math
import torch

from torch.distributions import constraints, Distribution, Normal
from torch.distributions import register_kl
from torch.distributions.kl import kl_divergence
from torch.distributions.independent import Independent

from utils.func_utils import mvdigamma, rel_error


class DiagonalWishart(Distribution):
    r"""
    Creates a diagonal version of Wishart distribution parameterized
    by its scale :attr:`scale_diag` and degrees of freedom :attr:`df`.

    Args:
        scale_diag (Tensor) (or L): positive scale of the distribution
            with shapes (bs, ..., p), where p is the dimensionality.
        df (Tensor) (or \nu): degrees of freedom with shapes (bs, ...).
            Should have the same shape as :attr:`scale_diag`,
            but without last dim.
    """
    arg_constraints = {
        'scale_diag': constraints.positive,
        'df': constraints.positive
    }
    support = constraints.positive
    has_rsample = False

    def __init__(self, scale_diag, df, validate_args=True):
        if scale_diag.dim() < 1 or df.dim() < 1:
            raise ValueError(
                "scale_diag or df must be at least one-dimensional."
            )
        if df.size(-1) == 1 and scale_diag.size(-1) != 1:
            raise ValueError(
                "df shouldn't end with dimensionality 1 if scale_diag doesn't."
            )
        df_ = df.unsqueeze(-1)  # add dim on right
        self.scale_diag, df_ = torch.broadcast_tensors(scale_diag, df_)
        self.df = df_[..., 0]  # drop rightmost dim

        batch_shape = self.scale_diag.shape[:-1]
        event_shape = self.scale_diag.shape[-1:]
        self.dimensionality = event_shape.numel()
        if (self.df <= (self.dimensionality - 1)).any():
            raise ValueError("df must be greater than dimensionality - 1")
        super(DiagonalWishart, self).__init__(
            batch_shape, event_shape,
            validate_args=validate_args
        )

    @property
    def mean(self):
        return self.df.unsqueeze(-1) * self.scale_diag

    @property
    def variance(self):
        return 2 * self.df.unsqueeze(-1) * self.scale_diag.pow(2)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        tr_term = -0.5 * torch.div(value, self.scale_diag).sum(dim=-1)
        norm_term = 0.5 * (
            self.df - self.dimensionality - 1
        ) * torch.log(value).sum(dim=-1)

        return -self.log_normalizer() + norm_term + tr_term

    def log_normalizer(self):
        log_normalizer_1 = 0.5 * self.df * self.dimensionality * math.log(2)
        log_normalizer_2 = 0.5 * self.df * self.scale_diag.log().sum(dim=-1)
        log_normalizer_3 = torch.mvlgamma(0.5 * self.df, self.dimensionality)
        return log_normalizer_1 + log_normalizer_2 + log_normalizer_3

    def log_expectation(self):
        mvdigamma_term = mvdigamma(0.5 * self.df, self.dimensionality)
        other_terms = self.dimensionality * math.log(2) + torch.log(
            self.scale_diag
        ).sum(dim=-1)
        return mvdigamma_term + other_terms

    def entropy(self):
        return self.log_normalizer() - 0.5 * (
                self.df - self.dimensionality - 1
            ) * self.log_expectation() + 0.5 * self.df * self.dimensionality


class NormalDiagonalWishart(Distribution):
    r"""
    Creates a diagonal version of Normal-Wishart distribution parameterized
    by its mean :attr:`mean`, diagonal precision :attr:`precision_diag`,
    degrees of freedom :attr:`df` and belief in mean :attr:`belief`

    Args:
        loc (Tensor) (or m): location of the distribution
            with shapes (bs, ..., p), where p is dimensionality.
        precision_diag (Tensor or float) (or L): precision of the distribution
            with shapes (bs, ..., p). It should have the same shape
            as :attr:`mean`.
        belief (Tensor or float) (or \kappa): confidence of belief in mean
            with shapes (bs, ...). It should have the same shape
            as :attr:`mean`, but without last dim.
        df (Tensor or float) (or \nu): degrees of freedom with shapes (bs, ...)
            It should have the same shape as :attr:`mean`,
            but without last dim.
    """
    arg_constraints = {
        'precision_diag': constraints.positive,
        'belief': constraints.positive,
        'df': constraints.positive,
    }
    support = constraints.real
    has_rsample = False

    def __init__(self, loc, precision_diag, belief, df, validate_args=True):
        precision_diag, belief, df = self.convert_float_params_to_tensor(
            loc, precision_diag, belief, df
        )
        if (loc.dim() < 1 or precision_diag.dim() < 1) or (
            df.dim() < 1 or belief.dim() < 1
        ):
            raise ValueError(
                "loc, precision_diag, df, belief must have at least one dim."
            )
        if belief.size(-1) == 1 and precision_diag.size(-1) != 1:
            raise ValueError(
                "belief shouldn't end with dim 1 if precision_diag doesn't"
            )
        if df.size(-1) == 1 and precision_diag.size(-1) != 1:
            raise ValueError(
                "df shouldn't end with dim 1 if precision_diag doesn't"
            )
        # add dim on right
        df_, belief_ = df.unsqueeze(-1), belief.unsqueeze(-1)
        self.loc, self.precision_diag, df_, belief_ = torch.broadcast_tensors(
            loc, precision_diag, df_, belief_
        )
        # drop rightmost dim
        self.df, self.belief = df_[..., 0], belief_[..., 0]

        batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]
        self.dimensionality = event_shape.numel()
        if (self.df <= (self.dimensionality + 1)).any():
            raise ValueError(
                "df must be greater than dimensionality + 1\
                    to have expectation"
            )
        super(NormalDiagonalWishart, self).__init__(
            batch_shape, event_shape, validate_args=validate_args
        )

    def log_prob(self, value_mean, value_precision):
        if self._validate_args:
            self._validate_sample(value_mean)
            self._validate_sample(value_precision)
        if (value_precision <= 0).any():
            raise ValueError("desired precision must be greater that 0")
        wishart_log_prob = DiagonalWishart(
            self.precision_diag, self.df
        ).log_prob(value_precision)
        normal_log_prob = Independent(
            Normal(
                self.loc, (
                    1 / (self.belief.unsqueeze(-1) * value_precision)
                ).pow(0.5)
            ), 1
        ).log_prob(value_mean)
        return normal_log_prob + wishart_log_prob

    def expectation_entropy_normal(self):
        return 0.5 * (
            self.dimensionality * (
                1 + math.log(2 * math.pi)
            ) - torch.log(
                2 * self.precision_diag * self.belief.unsqueeze(-1)
            ).sum(dim=-1) - mvdigamma(0.5 * self.df, self.dimensionality)
        )

    def entropy(self):
        wishart_entropy = DiagonalWishart(
            self.precision_diag, self.df
        ).entropy()
        expectation_entropy_normal = self.expectation_entropy_normal()
        return wishart_entropy + expectation_entropy_normal

    def convert_float_params_to_tensor(self, loc, precision_diag, belief, df):
        if isinstance(precision_diag, float):
            precision_diag = precision_diag * torch.ones_like(loc).to(
                loc.device
            )
        if isinstance(belief, float):
            belief = belief * torch.ones_like(loc).to(loc.device)[..., 0]
        if isinstance(df, float):
            df = df * torch.ones_like(loc).to(loc.device)[..., 0]
        return precision_diag, belief, df


@register_kl(DiagonalWishart, DiagonalWishart)
def kl_diag_wishart(p: DiagonalWishart, q: DiagonalWishart):
    if p.event_shape != q.event_shape:
        raise ValueError("KL-divergence between two Diagonal Wisharts with\
                          different event shapes cannot be computed")
    log_det_term = -(0.5 * q.df) * torch.div(
        p.scale_diag, q.scale_diag
    ).log().sum(dim=-1)
    tr_term = (0.5 * p.df) * (
        torch.div(p.scale_diag, q.scale_diag).sum(dim=-1) - p.dimensionality
    )
    mvlgamma_term = torch.mvlgamma(
        0.5 * q.df, q.dimensionality
    ) - torch.mvlgamma(
        0.5 * p.df, p.dimensionality
    )
    digamma_term = 0.5 * (p.df - q.df) * mvdigamma(
        0.5 * p.df, p.dimensionality
    )
    return log_det_term + tr_term + mvlgamma_term + digamma_term


@register_kl(NormalDiagonalWishart, NormalDiagonalWishart)
def kl_normal_diag_wishart(p: NormalDiagonalWishart, q: NormalDiagonalWishart):
    if p.event_shape != q.event_shape:
        raise ValueError(
            "KL-divergence between two Normal Diagonal Wisharts with\
                different event shapes cannot be computed")

    wishart_KL = kl_divergence(
        DiagonalWishart(p.precision_diag, p.df),
        DiagonalWishart(q.precision_diag, q.df)
    )
    weighted_mse_term = torch.sum(
        0.5 * q.belief.unsqueeze(-1) * (
            p.loc - q.loc
        ).pow(2) * p.precision_diag * p.df.unsqueeze(-1),
        dim=-1
    )
    expected_conditioned_normal_KL = (
        weighted_mse_term + (0.5 * p.dimensionality) * (
            torch.div(q.belief, p.belief) -
            torch.div(q.belief, p.belief).log() - 1
        )
    )

    return expected_conditioned_normal_KL + wishart_KL


def test_diagonal_normal_wishart():
    import numpy as np
    from scipy.stats import wishart
    x = np.linspace(1e-6, 20, 100)

    print("Testing wishart entropy/logprob vs scipy implementation...")
    for k in range(1000):
        df_val = torch.randn(1).exp() + 2
        scale_val = torch.randn(1).exp()

        scipy_dist = wishart(df=df_val.item(), scale=scale_val.item())
        torch_dist = DiagonalWishart(
            scale_val.unsqueeze(-1),
            df_val
        )

        torch_ent = torch_dist.entropy()[0]
        scipy_ent = torch.FloatTensor([scipy_dist.entropy()])
        if (rel_error(torch_ent, scipy_ent) > 1e-3).any():
            raise ValueError(
                "Entropies of torch and scipy versions doesn't match"
            )

        scipy_w = torch.FloatTensor(scipy_dist.logpdf(x))
        torch_w = torch_dist.log_prob(torch.FloatTensor(x).unsqueeze(-1))

        if (rel_error(torch_w, scipy_w) > 1e-6).any():
            raise ValueError(
                "Log pdf of torch and scipy versions doesn't match"
            )
    print("Passed")

    print("Testing wishart KL divergence...")
    df1, scale1 = torch.randn(32).exp() + 2, torch.randn(32).exp() + 1e-5
    df2, scale2 = torch.randn(32).exp() + 2, torch.randn(32).exp() + 1e-5
    init_df1, init_scale1 = df1[0].clone(), scale1[0].clone()
    dist2 = DiagonalWishart(scale2.unsqueeze(-1), df2)
    df1.requires_grad, scale1.requires_grad = True, True
    gamma = 0.1
    for k in range(10000):
        dist1 = DiagonalWishart(scale1.unsqueeze(-1), df1)
        loss = kl_divergence(dist1, dist2).mean()
        loss.backward()
        with torch.no_grad():
            scale1 = scale1 - gamma * scale1.grad
            df1 = df1 - gamma * df1.grad
        scale1.requires_grad, df1.requires_grad = True, True

    print('Distribution 1 - initial df %.3f and scale %.3f' % (
        init_df1, init_scale1
    ))
    print('Distribution 1 - trained df %.3f and scale %.3f' % (
        df1[0], scale1[0]
    ))
    print('Distribution 2 - target df %.3f and scale %.3f' % (
        df2[0], scale2[0]
    ))
    print("Passed")

    print("Testing Normal Wishart Distribution...")

    torch_dist = NormalDiagonalWishart(
        torch.ones((100, 1, 32, 32, 1)),
        torch.ones((100, 1, 32, 32, 1)),
        3 * torch.ones((100, 1, 32, 32)),
        3 * torch.ones((100, 1, 32, 32)),
    )

    ex_w = torch_dist.log_prob(
        torch.ones(100, 1, 32, 32, 1),
        torch.ones(100, 1, 32, 32, 1),
    )
    assert ex_w.shape == (100, 1, 32, 32)
    print("Passed")
