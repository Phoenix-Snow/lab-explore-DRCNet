import math

import torch
from torch import nn
import torch.nn.functional as F

from Module.Layer.Attention import CrossAttention


class Cluster_Assigner(nn.Module):
    def __init__(self, n_vars, n_cluster, seq_len, d_model, device, epsilon=0.05):
        super(Cluster_Assigner, self).__init__()
        self.n_vars = n_vars
        self.n_cluster = n_cluster
        self.d_model = d_model
        self.epsilon = epsilon
        self.linear = nn.Linear(seq_len, d_model)
        self.cluster_emb = torch.empty(self.n_cluster, self.d_model).to(device)
        # nn.Parameter(torch.rand(n_cluster, in_dim * out_dim), requires_grad=True)
        nn.init.kaiming_uniform_(self.cluster_emb, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        self.cross_attention = CrossAttention(n_heads=1)

    def forward(self, input_map, cluster_emb):
        """
        input_map: [batch_size, seq_len, n_vars]
        cluster_emb: [n_cluster, d_model]
        """
        n_vars = input_map.shape[-1]

        # input_map: [batch_size, n_vars, seq_len]
        input_map = input_map.permute(0, 2, 1)
        # input_emb: [batch_size, n_vars, d_model]
        input_emb = self.linear(input_map)
        # input_emb: [batch_size * n_vars, d_model]
        input_emb = input_emb.reshape(-1, self.d_model)

        batchSize_nVars = input_emb.shape[0]
        prob_batchSize = max(int(batchSize_nVars / n_vars), 1)
        # prob_batchSize = batch_size * n_vars / n_vars = batch_size

        # prob: [batch_size * n_vars, d_model] * [n_cluster, d_model].T -> [batch_size * n_vars, n_cluster]
        prob = torch.mm(self.l2norm(input_emb), self.l2norm(cluster_emb).t())
        # [prob_batchSize, n_vars, n_cluster] = [batch_size, n_vars, n_cluster]
        prob = prob.reshape(prob_batchSize, n_vars, -1)

        # prob_avg: [n_vars, n_cluster]
        prob_avg = torch.mean(prob, dim=0)
        prob_avg = self.sinkhorn(prob_avg, epsilon=self.epsilon)
        # mask: [batch_size, n_vars, n_cluster]
        mask = self.concrete_bern(prob_avg)

        # input_emb: [batch_size * n_vars, d_model] -> input_emb_: [batch_size, n_vars, d_model]
        input_emb_ = input_emb.reshape(prob_batchSize, n_vars, -1)
        cluster_emb_ = cluster_emb.repeat(prob_batchSize, 1, 1).transpose(0, 1)
        cluster_emb = self.cross_attention(cluster_emb_, input_emb_, input_emb_, mask=mask)

        return prob_avg, cluster_emb

    @staticmethod
    def concrete_bern(prob, temp=0.07):
        random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(prob.device)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
        prob_bern = ((prob + random_noise) / temp).sigmoid()
        return prob_bern

    @staticmethod
    def sinkhorn(prob_out, epsilon=0.05):  # [n_vars, n_cluster]
        Q = torch.exp(prob_out / epsilon)
        Q = Q / torch.sum(Q, dim=1, keepdim=True)
        return Q


class Cluster_wise_linear(nn.Module):
    def __init__(self, n_cluster, in_dim, out_dim):
        super().__init__()
        self.n_cluster = n_cluster
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear_layers = nn.ModuleList()
        for i in range(n_cluster):  # n_cluster次 操作完全相同 但 参数未必相同 的线性层
            self.linear_layers.append(nn.Linear(in_dim, out_dim))

    def forward(self, x, prob):
        """
        input_map: [batch_size, n_vars, in_dim]
        prob: [n_vars, n_cluster]
        return: [batch_size, n_vars, out_dim]
        """
        output = []
        for layer in self.linear_layers:
            output.append(layer(x))
        output = torch.stack(output, dim=-1).to(x.device)  # output转换为张量  [batch_size, n_vars, out_dim, n_cluster]
        prob = prob.unsqueeze(0).unsqueeze(-1)  #  -> [1, n_vars, n_cluster, 1]
        output = torch.matmul(output, prob).squeeze(-1)  # [batch_size, n_vars, out_dim, 1]  -> [batch_size, n_vars, out_dim]
        return output


class CCM_Args:
    def __init__(self, n_vars, n_cluster, in_len, d_model, out_dim):
        self.__n_vars = n_vars
        self.__n_cluster = n_cluster
        self.__in_len = in_len
        self.__d_model = d_model
        self.__out_dim = out_dim

    @property
    def n_vars(self):
        return self.__n_vars

    @n_vars.setter
    def n_vars(self, value):
        self.__n_vars = value

    @property
    def n_cluster(self):
        return self.__n_cluster

    @n_cluster.setter
    def n_cluster(self, value):
        self.__n_cluster = value

    @property
    def in_len(self):
        return self.__in_len

    @in_len.setter
    def in_len(self, value):
        self.__in_len = value

    @property
    def d_model(self):
        return self.__d_model

    @d_model.setter
    def d_model(self, value):
        self.__d_model = value

    @property
    def out_dim(self):
        return self.__out_dim

    @out_dim.setter
    def out_dim(self, value):
        self.__out_dim = value

