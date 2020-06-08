import torch
import torch.nn as nn


class GaussianNoise(nn.Module):
    def __init__(self, mean=0.0, sigma=0.05):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.sigma = sigma

    def forward(self, input):
        if not self.training:
            return input
        noise = input.clone().normal_(self.mean, self.sigma)
        return input + noise


class SimpleModel(nn.Module):
    """Returns predicted parameters of either Normal
    or Normal-Wishart distribution"""
    def __init__(
        self, input_dim, output_dim, num_units,
        out_channels=2, num_hidden=1,
        activation=nn.ReLU, drop_rate=0.0, noise_level=0.05,
        eps=1e-6
    ):
        super(SimpleModel, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden = num_hidden
        self.out_channels = out_channels
        self.eps = eps

        # dense network with %num_units hidden layers
        self.features, curr_dim = [], input_dim
        self.features.append(
            GaussianNoise(sigma=noise_level)
        )
        for _ in range(num_hidden):
            self.features.append(nn.Linear(curr_dim, num_units))
            self.features.append(activation())
            if drop_rate > 0.0:
                self.features.append(nn.Dropout(drop_rate))
            curr_dim = num_units
        self.features = nn.Sequential(*self.features)

        # generate stats of output distribution
        self.out_params = nn.Linear(num_units, output_dim * out_channels)

        self._initialize_weights()

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.features(x)

        params = self.out_params(x).split(self.output_dim, dim=1)
        return params

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1.0 / (m.weight.size(1) + 1))
                nn.init.constant_(m.bias, 0)


def test_simple_model():
    print("Testing SimpleModel...")
    s_model = SimpleModel(
        input_dim=11, output_dim=8,
        num_units=32, out_channels=3
    )
    ex_inp = torch.randn((32, 11))
    ex_out = s_model(ex_inp)
    assert ex_out[0].shape == (32, 8)

    from distributions.dist_wrappers import ProbabilisticWrapper
    from distributions.nw_prior import NormalWishartPrior

    wrapper = ProbabilisticWrapper(NormalWishartPrior, s_model)
    assert wrapper(ex_inp).expected_variance().shape == (32, 8)
    print("Passed")
