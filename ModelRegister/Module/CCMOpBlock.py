import torch
from torch import nn


class ChannelClusterLoss(nn.Module):
    """
    通过输入序列进行通道聚类, 达成损失优化（类似于图像识别里多个损失加权相加）, 间接的提升模型的泛化能力
    """
    def __init__(self, device, Alpha=0.1):
        super(ChannelClusterLoss, self).__init__()
        self.__Alpha = Alpha
        self.device = device

    # @property
    # def Alpha(self):
    #     return self.__Alpha

    def get_similarity_matrix(self, batch_x):
        sample = batch_x.squeeze(-1)  # [bsz, in_len]
        diff = sample.unsqueeze(1) - sample.unsqueeze(0)
        # Compute the Euclidean distance (squared)
        dist_squared = torch.sum(diff ** 2, dim=-1)  # [bsz, bsz]
        param = torch.max(dist_squared)
        euc_similarity = torch.exp(-5 * dist_squared / param)
        return euc_similarity

    def similarity_loss_batch(self, prob, simMatrix):
        def concrete_bern(prob, temp=0.07):
            random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(self.device)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
            prob_bern = ((prob + random_noise) / temp).sigmoid()
            return prob_bern * self.__Alpha

        membership = concrete_bern(prob)  # [n_vars, n_clusters]
        # membership = prob
        temp_1 = torch.mm(membership.t(), simMatrix)
        SAS = torch.mm(temp_1, membership)
        _SS = 1 - torch.mm(membership, membership.t())
        loss = -torch.trace(SAS) + torch.trace(torch.mm(_SS, simMatrix)) + membership.shape[0]
        ent_loss = (-prob * torch.log(prob + 1e-15)).sum(dim=-1).mean()
        return loss + ent_loss