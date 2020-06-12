from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()
        self.convA = nn.Conv2d(
            skip_input, output_features,
            kernel_size=3, stride=1, padding=1
        )
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(
            output_features, output_features,
            kernel_size=3, stride=1, padding=1
        )
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(
            x,
            size=[concat_with.size(2), concat_with.size(3)],
            mode='bilinear', align_corners=True
        )
        return self.leakyreluB(
            self.convB(
               self.leakyreluA(
                   self.convA(torch.cat([up_x, concat_with], dim=1))
               )
            )
        )


class Decoder(nn.Module):
    def __init__(
        self, backbone: str, decoder_width=1.0, out_channels=1
    ):
        super(Decoder, self).__init__()
        self.backbone = backbone
        self.out_channels = out_channels
        in_features = {
            'resnet18': 512,
            'densenet169': 1664
        }[self.backbone]
        features = int(in_features * decoder_width)

        self.conv2 = nn.Conv2d(in_features, features, kernel_size=1, stride=1)

        ac1, ac2, ac3, ac4 = 256, 128, 64, 64
        self.up1 = UpSample(
            skip_input=features//1 + ac1,
            output_features=features//2
        )
        self.up2 = UpSample(
            skip_input=features//2 + ac2,
            output_features=features//4
        )
        self.up3 = UpSample(
            skip_input=features//4 + ac3,
            output_features=features//8
        )
        self.up4 = UpSample(
            skip_input=features//8 + ac4,
            output_features=features//16
        )

        self.conv3 = nn.Conv2d(
            features//16, out_channels,
            kernel_size=3, stride=1, padding=1
        )

    def forward(self, features):
        if self.backbone == 'densenet169':
            x_block0, x_block1 = features[3], features[4]
            x_block2, x_block3 = features[6], features[8]
            x_block4 = features[11]
        elif self.backbone == 'resnet18':
            x_block0, x_block1 = features[3], features[4]
            x_block2, x_block3 = features[6], features[7]
            x_block4 = features[8]
        x_d0 = self.conv2(x_block4)
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        x_d5 = self.conv3(x_d4)
        if self.out_channels > 1:
            params = x_d5.split(1, dim=1)
            return params
        else:
            return x_d5


class Encoder(nn.Module):
    def __init__(self, backbone: str, pretrained=True):
        super(Encoder, self).__init__()
        assert backbone in ['densenet169', 'resnet18']
        self.backbone = backbone

        if backbone == 'densenet169':
            self.original_model = models.densenet169(
                pretrained=pretrained
            ).features
        elif backbone == 'resnet18':
            self.original_model = nn.Sequential(
                *list(models.resnet18(pretrained=pretrained).children())[:-1]
            )

    def forward(self, x):
        features = [x]
        for _, v in self.original_model._modules.items():
            features.append(v(features[-1]))
        return features


class UNetModel(nn.Module):
    def __init__(
        self, backbone: str, pretrained=True,
        out_channels=1
    ):
        """UNet-like model for MDE based on DenseDepth architecture
        (https://arxiv.org/abs/1812.11941)
        """
        super(UNetModel, self).__init__()
        self.encoder = Encoder(backbone, pretrained=pretrained)
        self.decoder = Decoder(
            backbone, out_channels=out_channels
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def test_densedepth():
    print("Testing DenseDepth Normal-Wishart model...")
    denseDepth_model = UNetModel(
        'densenet169', out_channels=3
    )
    ex_inp = torch.randn(8, 3, 480, 640)

    from distributions.distribution_wrappers import ProbabilisticWrapper
    from distributions.nw_prior import NormalWishartPrior

    wrapper = ProbabilisticWrapper(NormalWishartPrior, denseDepth_model)
    assert wrapper(ex_inp).expected_variance().shape == (8, 1, 240, 320)
    print("Num params", sum(p.numel() for p in denseDepth_model.parameters()))
    print("Passed")

    print("Testing DenseDepth ensemble of Gaussian models...")
    from distributions.distribution_wrappers import GaussianEnsembleWrapper
    wrapper = GaussianEnsembleWrapper(
        [UNetModel('densenet169', out_channels=2) for _ in range(3)]
    )
    assert wrapper(ex_inp).expected_variance().shape == (8, 1, 240, 320)
    print("Passed")
