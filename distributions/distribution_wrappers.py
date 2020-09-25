import torch
from torch.nn import Softplus, Module

from torch.distributions.normal import Normal
from .mixture_gaussian import GaussianDiagonalMixture


def transform_to_distribution_params(params, distr_dim=1, eps=1e-6):
    """Apply nonlinearities to unconstrained model outputs so
    they can be represented as parameters of either
    Normal or Normal-Wishart distributions"""
    if len(params) > 3:
        all_means, all_stds = [], []
        for i in range(len(params) // 2):
            all_means.append(params[i * 2])
            all_stds.append(Softplus()(params[i * 2 + 1]) + eps)
        return torch.cat(all_means, dim=1), torch.cat(all_stds, dim=1)

    mean = params[0]
    std = Softplus()(params[1]) + eps
    if len(params) == 2:
        return [mean, std]
    elif len(params) == 3:
        beta = Softplus()(params[2]) + eps
        min_df = 3
        #min_df = params[0].size(distr_dim) + 2  # !!!
        kappa, nu = beta, beta + min_df
        return [mean.unsqueeze(-1), std.unsqueeze(-1), kappa, nu]


class ProbabilisticWrapper(Module):
    def __init__(self, distribution_cls, model, distr_dim=1):
        super(ProbabilisticWrapper, self).__init__()
        self.distribution_cls = distribution_cls
        self.model = model
        self.distr_dim = distr_dim

    def forward(self, x, mask=None):
        out_params = self.model(x)
        assert (len(out_params) in [2, 3]) or isinstance(
            self.distribution_cls, GaussianDiagonalMixture
        )
        if mask is not None:
            predicted_params = transform_to_distribution_params(
                [p[mask] for p in out_params], self.distr_dim
            )
        else:
            predicted_params = transform_to_distribution_params(
                out_params, self.distr_dim
            )
        if not self.training:
            predicted_params = [param.cpu() for param in predicted_params]
        return self.distribution_cls(*predicted_params)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


class GaussianEnsembleWrapper(Module):
    """Wraps list of models to a Gaussian Mixture"""
    def __init__(self, models):
        super(GaussianEnsembleWrapper, self).__init__()
        self.models = models
        self.distribution_cls = GaussianDiagonalMixture

    def forward(self, x, mask=None):
        agg = []
        for model in self.models:
            agg.append(model(x))
        assert len(agg[0]) == 2
        if mask is not None:
            all_predicted_params = [
                transform_to_distribution_params([ps[mask] for ps in p])
                for p in agg
            ]
        else:
            all_predicted_params = [
                transform_to_distribution_params(p) for p in agg
            ]
        if not self.training:
            all_predicted_params = [
                [p for p in internal_params]
                for internal_params in all_predicted_params
            ]
        return self.distribution_cls(
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
        super(GaussianEnsembleWrapper, self).eval()
        for model in self.models:
            model.eval()
        return self

    def cuda(self):
        return self.to('cuda')

    def cpu(self):
        return self.to('cpu')

    def to(self, device):
        for model in self.models:
            model.to(device)
        return self

    def make_dataparallel(self):
        for i, model in enumerate(self.models):
            self.models[i] = nn.DataParallel(model)
