import torch
from itertools import combinations
from torch.distributions import Distribution, Normal, kl_divergence


class GaussianDiagonalMixture(Distribution):
    r"""
    Creates a mixture of diagonal Normal distributions parameterized
    by their means :attr:`means` and scales :attr:`scales`.
    Note that the first dim corresponds to the ensemble size.
    """
    def __init__(self, means, scales):
        assert means.shape == scales.shape

        self.distributions = []
        for i in range(len(means)):
            self.distributions.append(
                Normal(means[i], scales[i], validate_args=True)
            )

    @property
    def mean(self):
        return self.expected_mean()

    def expected_mean(self):
        return sum(
            [dist.mean for dist in self.distributions]
        ) / len(self.distributions)

    def expected_entropy(self):
        return sum(
            [dist.entropy() for dist in self.distributions]
        ) / len(self.distributions)

    def expected_pairwise_kl(self):
        curr_sum_pairwise_kl = None
        num_pairs = 0

        for dist1, dist2 in combinations(self.distributions, r=2):
            num_pairs += 1
            if curr_sum_pairwise_kl is None:
                curr_sum_pairwise_kl = kl_divergence(dist1, dist2)
            else:
                curr_sum_pairwise_kl += kl_divergence(dist1, dist2)
        return curr_sum_pairwise_kl / num_pairs

    def variance_of_expected(self):
        avg_mean = self.expected_mean()
        return sum(
            [
                (dist.mean.pow(2) - avg_mean.pow(2))
                for dist in self.distributions
            ]
        ) / len(self.distributions)

    def expected_variance(self):
        return sum(
            [dist.variance for dist in self.distributions]
        ) / len(self.distributions)

    def total_variance(self):
        return self.variance_of_expected() + self.expected_variance()

    def estimated_total_entropy(self):
        return self.expected_entropy() + self.expected_pairwise_kl()

    def log_prob(self, value):
        mean = self.expected_mean()
        var = self.total_variance()
        return Normal(mean, var.pow(0.5)).log_prob(value)


def test_gaussian_mixture():
    print("Testing Gaussian Mixture...")
    ex_means = torch.ones(5, 8, 1, 32, 32)
    ex_vars = 2 * torch.ones(5, 8, 1, 32, 32)
    mixture_dis = GaussianDiagonalMixture(ex_means, ex_vars)

    assert mixture_dis.log_prob(
        torch.zeros((8, 1, 32, 32))
    ).shape == (8, 1, 32, 32)

    for method in [
        'expected_mean', 'expected_entropy', 'expected_pairwise_kl',
        'variance_of_expected', 'expected_variance', 'total_variance'
    ]:
        assert getattr(mixture_dis, method)().shape == (8, 1, 32, 32)

    print("Passed")
