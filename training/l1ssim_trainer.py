from torch import clamp as t_clamp
from torch.nn import L1Loss
from training.distribution_trainer import SingleDistributionTrainer
from utils.ssim_gradient_losses import ssim, image_gradient_loss


class L1SSIMTrainer(SingleDistributionTrainer):
    def loss_fn(self, output_distr, targets):
        output = output_distr
        l_depth = L1Loss()(output, targets)
        l_ssim = t_clamp(
            (1 - ssim(output, targets, val_range=80.0)) * 0.5,
            0, 1
        )
        l_grad = image_gradient_loss(output, targets)
        return (1.0 * l_ssim) + (1.0 * l_grad) + (0.1 * l_depth)
