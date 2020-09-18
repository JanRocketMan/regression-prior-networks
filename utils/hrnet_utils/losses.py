import torch
import torch.nn as nn


class JointsNLLLoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsNLLLoss, self).__init__()
        #self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output_distr, target, target_weight):
        #batch_size = output_distr.size(0)
        #num_joints = output_distr.size(1)
        #heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        #heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        #loss = 0
        loss = -output_distr.log_prob(target)
        loss = loss.mean(dim=(2,3))
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