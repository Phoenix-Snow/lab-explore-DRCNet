import torch
from torch import nn

from Layer.CCM import CCM_Args, Cluster_Assigner, Cluster_wise_linear
from Layer.Embedding_Family import DataEmbedding, DataEmbedding_Args


# 父类
class CCM_Module(nn.Module):
    def __init__(self, ccm_args: CCM_Args, embed_args: DataEmbedding_Args, device):
        super().__init__()
        self.device = device
        self.cluster_assigner = Cluster_Assigner(
            ccm_args.n_vars, ccm_args.n_cluster, ccm_args.in_len, ccm_args.d_model,
            device
        )
        self.cluster_emb = self.cluster_assigner.cluster_emb
        self.predict_linear = Cluster_wise_linear(ccm_args.n_cluster, ccm_args.n_vars, ccm_args.out_dim)
        self.enc_embedding = DataEmbedding(embed_args.input_channels_num, embed_args.d_model,
                                           frequency_type=embed_args.frequency_type, dropout=embed_args.dropout)
