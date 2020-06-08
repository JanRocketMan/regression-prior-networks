import torch
from torch.nn import Softplus, Module

from torch.distributions.normal import Normal
from .mixture_gaussian import GaussianDiagonalMixture


def transform_to_distribution_params(params, distr_dim=1, eps=1e-6):
    """Apply nonlinearities to unconstrained model outputs so
    they can be represented as parameters of either
    Normal or Normal-Wishart distributions"""
    mean = params[0]
    std = Softplus()(params[1]) + eps
    if len(params) == 2:
        return [mean, std]
    elif len(params) == 3:
        beta = Softplus()(params[2]) + eps
        kappa, nu = beta, beta + params[0].size(distr_dim) + 1
        return [mean.unsqueeze(-1), std.unsqueeze(-1), kappa, nu]


class ProbabilisticWrapper(Module):
    def __init__(self, distribution_cls, model, distr_dim=1):
        super(ProbabilisticWrapper, self).__init__()
        self.distribution_cls = distribution_cls
        self.model = model
        self.distr_dim = distr_dim

    def forward(self, x):
        out_params = self.model(x)
        assert len(out_params) in [2, 3]
        predicted_params = transform_to_distribution_params(
            out_params, self.distr_dim
        )
        return self.distribution_cls(*predicted_params)


class GaussianEnsembleWrapper(Module):
    """Wraps list of models to a Gaussian Mixture"""
    def __init__(self, models):
        super(GaussianEnsembleWrapper, self).__init__()
        self.models = models

    def forward(self, x):
        agg = []
        for model in self.models:
            agg.append(model(x))
        assert len(agg[0]) == 2
        all_predicted_params = [
            transform_to_distribution_params(p) for p in agg
        ]
        return GaussianDiagonalMixture(
            torch.cat(
                [params[0].unsqueeze(0) for params in all_predicted_params]
            ),
            torch.cat(
                [params[1].unsqueeze(0) for params in all_predicted_params]
            )
        )

    def load_state_dict(self, list_of_state_dicts):
        assert len(list_of_state_dicts) == len(self.models)
        for i, model in enumerate(self.models):
            model.load_state_dict(list_of_state_dicts[i])

    def eval(self):
        for model in self.models:
            model.eval()

    def cuda(self):
        return self.to('cuda')

    def to(self, device):
        for model in self.models:
            model.to(device)
        return self

    def make_dataparallel(self):
        for i, model in enumerate(self.models):
            self.models[i] = nn.DataParallel(model)
