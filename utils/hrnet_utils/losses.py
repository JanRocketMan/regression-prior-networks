import torch
import torch.nn as nn
from hrnet.lib.core.inference import get_max_preds
from torch.distributions.normal import Normal

class JointsNLLLoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsNLLLoss, self).__init__()
        #self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output_distr, target, target_weight, step=None):
        #batch_size = output_distr.size(0)
        #num_joints = output_distr.size(1)
        #heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        #heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        #loss = 0
        #print("OUTPUT shape", output_distr.mean.shape)
        #print("TARGET shape", target.shape)
        #print("TARGET WEIGHT", target_weight.shape)

        with torch.no_grad():
            raw_indices = target.view(target.size(0), target.size(1), -1).argmax(-1)
            target = torch.cat([(raw_indices / target.size(2)).unsqueeze(2), (raw_indices % target.size(2)).unsqueeze(2)], dim=2).float() # bs x joint x 2
        #print("TARGET SHAPE", target.shape)
        #if self.use_target_weight:
            #mask = target_weight.unsqueeze(-1)
            #mask = mask.repeat((1,1,target.shape[2], target.shape[3]))
            #target = target * mask
            #new_mean = output_distr.mean * mask

        #new_distribution = Normal(new_mean, output_distr.stddev)
        loss = -output_distr.log_prob(target)
        #if step is None:
        #    reg_coeff = 0
        #elif step <= 3480:
        #    reg_coeff = 1
        #else:
        #    reg_coeff = 3480 / step
        #loss = -new_distribution.log_prob(target) + reg_coeff * 10 * (new_distribution.stddev - 1)**2
        loss = loss.mean(dim=2)

        #return loss.mean()
        #print("BATCH LOSS", loss.shape)
        if self.use_target_weight:
            assert target_weight.shape[2] == 1
            loss = loss * target_weight[:,:,0]
            return loss.mean()
        else:
            return loss.mean()

        #for idx in range(num_joints):
        #    heatmap_pred = heatmaps_pred[idx].squeeze()
        #    heatmap_gt = heatmaps_gt[idx].squeeze()
        #    if self.use_target_weight:
        #        loss += 0.5 * self.criterion(
        #            heatmap_pred.mul(target_weight[:, idx]),
        #            heatmap_gt.mul(target_weight[:, idx])
        #        )
        #    else:
        #        loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        #return loss / num_joints