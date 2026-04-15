import torch
from torch import nn

from ModelRegister.Frequency_Encoder import T_Model, T_Model_trans_tMLP2Linear, T_Model_new


class Competition_Model(nn.Module):
    def __init__(self, t_model: T_Model, normal_feature_num: int, device, Linear_Normal_Func=nn.BatchNorm1d):
        super().__init__()
        self.feature_num = t_model.feature_num
        self.normal_feature_num = normal_feature_num  # 地理
        self.__temporal_block = t_model
        neueon_options = [
            [200, 30],
            [200, 40],
            [200, 45],
            [200, 50],
            [200, 55],
            [200, 60],
            [200, 70],
            [200, 80],
        ]
        option = 4

        neueon = neueon_options[option - 1]
        if Linear_Normal_Func == nn.LayerNorm:
            self.__spatial_linear_layer = nn.Sequential(
                nn.Linear(self.feature_num + self.normal_feature_num, self.h1),
                Linear_Normal_Func(self.h1),
                nn.PReLU(),
                # nn.Dropout(0.2),
                nn.Linear(self.h1, self.h2),
                Linear_Normal_Func(self.h2),
                nn.PReLU(),
                # nn.Dropout(0.2),
                nn.Linear(self.h2, 2),
                # nn.PReLU()
                nn.Sigmoid()
            ).to(device)
        elif Linear_Normal_Func == nn.BatchNorm1d:
            self.__spatial_linear_layer = nn.Sequential(
                nn.Linear(self.feature_num + self.normal_feature_num, self.h1),
                Linear_Normal_Func(t_model.series_len),
                nn.PReLU(),
                # nn.Dropout(0.2),
                nn.Linear(self.h1, self.h2),
                Linear_Normal_Func(t_model.series_len),
                nn.PReLU(),
                # nn.Dropout(0.2),
                nn.Linear(self.h2, 2),
                # nn.PReLU()
                nn.Sigmoid()
            ).to(device)

        del neueon, neueon_options

    def forward(self, temporal_map: torch.Tensor, spatial_id_list: list, normal_data: torch.Tensor):
        if list(temporal_map.shape) != [temporal_map.shape[0], 365, self.feature_num]:
            raise ValueError(f"temporal_map shape should be "
                             f"torch.Size([{temporal_map.shape[0]}, 365, {self.feature_num}]),"
                             f"actual shape is {temporal_map.shape}")
        # 预处理
        temporal_detach = temporal_map.detach()

        # 分离时序空序
        temporal_output, temporal_feature = self.__temporal_block.forward_4_TS(temporal_detach)  # 时间特征强化, 使周期性更加明显
        spatial_feature = temporal_map  # 先复制一份
        for i in range(9):  # f * 3 * 3  9层感受野
            spatial_feature = spatial_feature - temporal_feature[:, :, i: i + self.feature_num]  # 剩余特征应该大半都是空间特征了

        B, T, F = spatial_feature.shape
        spatial_feature = torch.cat((spatial_feature, normal_data), dim=-1).reshape(B * T, -1)
        # 2. 预测w和b
        spatial_weight_bias = self.__spatial_linear_layer(spatial_feature).reshape(B, T, 2)
        spatial_weight = spatial_weight_bias[:, :, 0]
        spatial_bias = spatial_weight_bias[:, :, 1]
        # 3. 加权
        temporal_output = temporal_output * spatial_weight + spatial_bias

        return temporal_output

    def hidden_params(self, temporal_map: torch.Tensor, spatial_id_list: list, normal_data: torch.Tensor):
        if list(temporal_map.shape) != [temporal_map.shape[0], 365, self.feature_num]:
            raise ValueError(f"temporal_map shape should be "
                             f"torch.Size([{temporal_map.shape[0]}, 365, {self.feature_num}]),"
                             f"actual shape is {temporal_map.shape}")
        temporal_detach = temporal_map.detach()
        temporal_output, temporal_feature = self.__temporal_block.forward_4_TS(temporal_detach)
        spatial_feature = temporal_map
        for i in range(9):
            spatial_feature = spatial_feature - temporal_feature[:, :, i: i + self.feature_num]
        B, T, F = spatial_feature.shape
        spatial_feature = torch.cat((spatial_feature, normal_data), dim=-1).reshape(B * T, -1)
        spatial_weight_bias = self.__spatial_linear_layer(spatial_feature).reshape(B, T, 2)
        spatial_weight = spatial_weight_bias[:, :, 0]
        spatial_bias = spatial_weight_bias[:, :, 1]
        temporal_output = temporal_output * spatial_weight + spatial_bias

        return (temporal_feature.reshape(B, T, -1), temporal_output.reshape(B, T, -1),
                spatial_feature.reshape(B, T, -1), normal_data.reshape(B, T, -1),
                spatial_weight.reshape(B, T, -1), spatial_bias.reshape(B, T, -1),
                temporal_output.reshape(B, T, -1), B, T)


class TS_Model(nn.Module):
    def __init__(self, t_model: T_Model_new | T_Model_trans_tMLP2Linear, normal_feature_num: int, model_save_path, device,
                 Linear_Normal_Func=nn.BatchNorm1d):
        super().__init__()
        self.feature_num = t_model.feature_num
        self.normal_feature_num = normal_feature_num  # 地理
        self.__temporal_block = t_model  # None  |Over: 3.40
        # self.__temporal_block.requires_grad_(False)
        # 显存消耗: 6.6G
        # 超参对比:
        # 线性对比  (~<2.7)
        # neueon_options = [
        #     # BEST 200-50-v1-1k
        #     [200, 30],  # -v1-  noDropout
        #     #             -v2-  Dropout   Y  2.72
        #     [200, 40],  # -v1-  noDropout X  2.88
        #     #             -v2-  Dropout   X  2.92
        #     [200, 45],  # -v1-  noDropout X  3.17
        #     # -v2-  Dropout
        #     [200, 50],
        #     # -v1-  Sigmoid + noDropout Y  2.48  |Over: 2.2    -again- Y 2.47   -memory2k- 2.48   -memory4k- 2.6061
        #     #             -v2-  Sigmoid + Dropout   X  2.78  |Over: 2.73   -again- X 2.90
        #     #             -v3-  PRelu + noDropout   X  2.96  |Over: 2.92
        #     [200, 55],  # -v1-  noDropout X  2.9
        #     #             -v2-  Dropout   X  2.86
        #     [200, 60],  # -v1-  noDropout X  2.85
        #     #             -v2-  Dropout   Y  2.68
        #     [200, 70],  # -v1-
        #     [200, 80],  # -v2-  Dropout   X  2.83
        #     # [450, 50],  # X  3.13
        #     # [450, 80],  # X  2.85
        #     # [900, 80],
        #     # [1350, 80], # X  2.92
        # ]
        # option = 4
        # memory容量对比   1000 2000 4000
        self.memory_capacity = 1000

        # neueon = neueon_options[option - 1]
        self.h1 = 200
        self.h2 = 50
        self.__spatial_memory = nn.Embedding(self.memory_capacity, self.feature_num).to(device)  # （词向量个数, 特征数）
        if Linear_Normal_Func == nn.LayerNorm:
            self.__spatial_linear_layer = nn.Sequential(
                nn.Linear(self.feature_num + self.normal_feature_num, self.h1),
                Linear_Normal_Func(self.h1),
                nn.PReLU(),
                # nn.Dropout(0.2),
                nn.Linear(self.h1, self.h2),
                Linear_Normal_Func(self.h2),
                nn.PReLU(),
                # nn.Dropout(0.2),
                nn.Linear(self.h2, 2),
                # nn.PReLU()
                nn.Sigmoid()
            ).to(device)
        elif Linear_Normal_Func == nn.BatchNorm1d:
            self.__spatial_linear_layer = nn.Sequential(
                nn.Linear(self.feature_num + self.normal_feature_num, self.h1),
                Linear_Normal_Func(t_model.series_len),
                nn.PReLU(),
                # nn.Dropout(0.2),
                nn.Linear(self.h1, self.h2),
                Linear_Normal_Func(t_model.series_len),
                nn.PReLU(),
                # nn.Dropout(0.2),
                nn.Linear(self.h2, 2),
                # nn.PReLU()
                nn.Sigmoid()
            ).to(device)

        # os.makedirs(model_save_path, exist_ok=True)
        # with open(os.path.join(model_save_path, f"s_model_params{self.memory_capacity}_{self.h1}_{self.h2}.json"),
        #           'r') as f:
        #     json.dump(self.__dict__, f)

        

    def forward(self, temporal_map: torch.Tensor, normal_data: torch.Tensor):
        # if list(temporal_map.shape) != [temporal_map.shape[0], 365, self.feature_num]:
        #     raise ValueError(f"temporal_map shape should be "
        #                      f"torch.Size([{temporal_map.shape[0]}, 365, {self.feature_num}]),"
        #                      f"actual shape is {temporal_map.shape}")
        # 预处理
        temporal_detach = temporal_map.detach()

        # 分离时序空序
        temporal_output, temporal_feature = self.__temporal_block.forward_4_TS(temporal_detach)  # 时间特征强化, 使周期性更加明显
        spatial_feature = temporal_map  # 先复制一份
        for i in range(9):  # f * 3 * 3  9层感受野
            spatial_feature = spatial_feature - temporal_feature[:, :, i: i + self.feature_num]  # 剩余特征应该大半都是空间特征了

        # 时序预测
        # temporal_feature = self.__temporal_linear_layer_pre(temporal_feature)
        # temporal_feature = temporal_feature.reshape(-1, temporal_feature.size(2))
        # temporal_feature = self.__temporal_transformer(temporal_feature)
        # temporal_output = self.__temporal_linear_layers(temporal_feature)
        # temporal_output = temporal_output.reshape(temporal_output.size(0) // 365, 365)  # 时间序列得到输出, 是目标序列的时序情况

        B, T, F = spatial_feature.shape
        spatial_feature = spatial_feature.reshape(B * T, F)
        # 1. 空序优化
        ## 平方欧氏距离计算 (通过词向量方式, 更新出最完整最贴切的空间特征)
        d = torch.sum(spatial_feature.square(), dim=1, keepdim=True) \
            + torch.sum(self.__spatial_memory.weight.data.square(), dim=1, keepdim=False)
        ## addmm_(): d_no_grad = d_no_grad + (-2)*A@B
        d.addmm_(spatial_feature, self.__spatial_memory.weight.data.T, beta=1, alpha=-2)
        vocab_idx_list_N = torch.argmin(d, dim=1)  # (N, vocab_size)
        vocab_idx_list_BTF = vocab_idx_list_N.reshape(B, T, 1)
        memory_feature = self.__spatial_memory(vocab_idx_list_BTF.reshape(-1, 1)).reshape(B * T, F)
        # print(spatial_feature.shape, normal_data.shape)
        spatial_feature = torch.cat((memory_feature.reshape(B, T, F), normal_data), dim=-1).reshape(B * T, -1)
        # 2. 预测w和b
        spatial_weight_bias = self.__spatial_linear_layer(spatial_feature).reshape(B, T, 2)
        spatial_weight = spatial_weight_bias[:, :, 0].reshape(B, T, 1)
        spatial_bias = spatial_weight_bias[:, :, 1].reshape(B, T, 1)
        # 3. 加权
        temporal_output = temporal_output * spatial_weight + spatial_bias

        return temporal_output

    def hidden_params(self, temporal_map: torch.Tensor, normal_data: torch.Tensor):
        # if list(temporal_map.shape) != [temporal_map.shape[0], 365, self.feature_num]:
        #     raise ValueError(f"temporal_map shape should be "
        #                      f"torch.Size([{temporal_map.shape[0]}, 365, {self.feature_num}]),"
        #                      f"actual shape is {temporal_map.shape}")
        temporal_detach = temporal_map.detach()
        temporal_output, temporal_feature = self.__temporal_block.forward_4_TS(temporal_detach)
        spatial_feature = temporal_map
        for i in range(9):
            spatial_feature = spatial_feature - temporal_feature[:, :, i: i + self.feature_num]
        B, T, F = spatial_feature.shape
        spatial_feature = spatial_feature.reshape(B * T, F)
        d = torch.sum(spatial_feature.square(), dim=1, keepdim=True) \
            + torch.sum(self.__spatial_memory.weight.data.square(), dim=1, keepdim=False)
        d.addmm_(spatial_feature, self.__spatial_memory.weight.data.T, beta=1, alpha=-2)
        vocab_idx_list_N = torch.argmin(d, dim=1)
        vocab_idx_list_BTF = vocab_idx_list_N.reshape(B, T, 1)
        memory_feature = self.__spatial_memory(vocab_idx_list_BTF.reshape(-1, 1)).reshape(B * T, F)
        spatial_feature = torch.cat((memory_feature.reshape(B, T, F), normal_data), dim=-1).reshape(B * T, -1)
        spatial_weight_bias = self.__spatial_linear_layer(spatial_feature).reshape(B, T, 2)
        spatial_weight = spatial_weight_bias[:, :, 0].reshape(B, T, 1)
        spatial_bias = spatial_weight_bias[:, :, 1].reshape(B, T, 1)
        output = temporal_output * spatial_weight + spatial_bias

        return (temporal_feature.reshape(B, T, -1), temporal_output.reshape(B, T, -1),
                spatial_feature.reshape(B, T, -1), normal_data.reshape(B, T, -1), memory_feature.reshape(B, T, -1),
                spatial_weight.reshape(B, T, -1), spatial_bias.reshape(B, T, -1),
                output.reshape(B, T, -1), B, T)

    @property
    def NAME(self) -> str:
        return 'DRCNet'

    def PARAMS(self):
        return f'baseTS{str(self.h1)}_{str(self.h2)}_{str(self.memory_capacity)}'


# class TS_Model_times(nn.Module):
#     def __init__(self, t_model: T_Model, normal_feature_num: int, device, Linear_Normal_Func=nn.BatchNorm1d):
#         super().__init__()
#         self.feature_num = t_model.feature_num
#         self.normal_feature_num = normal_feature_num
#         self.__temporal_block = t_model  # None  |Over: 3.40
#         # self.__temporal_block.requires_grad_(False)
#         # 显存消耗: 6.6G
#         # 超参对比:
#         # 线性对比  (~<2.7)
#         neueon_options = [
#             [200, 30],
#             [200, 40],
#             [200, 50],
#             [200, 55],
#             [200, 60],
#             [200, 70],
#         ]
#         option = 3
#         # memory容量对比   1000 2000 4000  200_50_
#         self.memory_capacity = 2000
#         # 执行次数
#         self.k_times = 2
#
#         # 空间模块层List
#         self.__spatial_module = nn.ModuleList()
#         neueon = neueon_options[option - 1]
#         for _ in range(self.k_times - 1):
#             spatial_memory = nn.Embedding(self.memory_capacity, self.feature_num).to(device)  # （词向量个数, 特征数）
#             self.__spatial_module.append(spatial_memory)
#             if Linear_Normal_Func == nn.LayerNorm:
#                 spatial_linear_layer = nn.Sequential(
#                     nn.Linear(self.feature_num + self.normal_feature_num, self.h1),
#                     Linear_Normal_Func(self.h1),
#                     nn.PReLU(),
#                     # nn.Dropout(0.2),
#                     nn.Linear(self.h1, self.h2),
#                     Linear_Normal_Func(self.h2),
#                     nn.PReLU(),
#                     # nn.Dropout(0.2),
#                     nn.Linear(self.h2, 2),
#                     # nn.PReLU()
#                     nn.Sigmoid()
#                 ).to(device)
#             elif Linear_Normal_Func == nn.BatchNorm1d:
#                 spatial_linear_layer = nn.Sequential(
#                     nn.Linear(self.feature_num + self.normal_feature_num, self.h1),
#                     Linear_Normal_Func(t_model.series_len),
#                     nn.PReLU(),
#                     # nn.Dropout(0.2),
#                     nn.Linear(self.h1, self.h2),
#                     Linear_Normal_Func(t_model.series_len),
#                     nn.PReLU(),
#                     # nn.Dropout(0.2),
#                     nn.Linear(self.h2, 2),
#                     # nn.PReLU()
#                     nn.Sigmoid()
#                 ).to(device)
#             self.__spatial_module.append(spatial_linear_layer)
#         spatial_memory = nn.Embedding(self.memory_capacity, self.feature_num).to(device)  # （词向量个数, 特征数）
#         self.__spatial_module.append(spatial_memory)
#         if Linear_Normal_Func == nn.LayerNorm:
#             spatial_linear_layer = nn.Sequential(
#                 nn.Linear(self.feature_num + self.normal_feature_num, self.h1),
#                 Linear_Normal_Func(self.h1),
#                 nn.PReLU(),
#                 # nn.Dropout(0.2),
#                 nn.Linear(self.h1, self.h2),
#                 Linear_Normal_Func(self.h2),
#                 nn.PReLU(),
#                 # nn.Dropout(0.2),
#                 nn.Linear(self.h2, 1),
#                 # nn.PReLU()
#                 nn.Sigmoid()
#             ).to(device)
#         elif Linear_Normal_Func == nn.BatchNorm1d:
#             spatial_linear_layer = nn.Sequential(
#                 nn.Linear(self.feature_num + self.normal_feature_num, self.h1),
#                 Linear_Normal_Func(t_model.series_len),
#                 nn.PReLU(),
#                 # nn.Dropout(0.2),
#                 nn.Linear(self.h1, self.h2),
#                 Linear_Normal_Func(t_model.series_len),
#                 nn.PReLU(),
#                 # nn.Dropout(0.2),
#                 nn.Linear(self.h2, 1),
#                 # nn.PReLU()
#                 nn.Sigmoid()
#             ).to(device)
#         self.__spatial_module.append(spatial_linear_layer)
#
#         , neueon, neueon_options
#
#     def forward(self, temporal_map: torch.Tensor, spatial_id_list: list, normal_data: torch.Tensor):
#         if list(temporal_map.shape) != [temporal_map.shape[0], 365, self.feature_num]:
#             raise ValueError(f"temporal_map shape should be "
#                              f"torch.Size([{temporal_map.shape[0]}, 365, {self.feature_num}]),"
#                              f"actual shape is {temporal_map.shape}")
#         # 预处理
#         temporal_detach = temporal_map.detach()
#
#         # 分离时序空序
#         temporal_output, temporal_feature = self.__temporal_block.forward_4_TS(temporal_detach)  # 时间特征强化, 使周期性更加明显
#         spatial_feature = temporal_map  # 先复制一份
#         for i in range(9):  # f * 3 * 3  9层感受野
#             spatial_feature = spatial_feature - temporal_feature[:, :, i: i + self.feature_num]  # 剩余特征应该大半都是空间特征了
#
#         # 时序预测
#         # temporal_feature = self.__temporal_linear_layer_pre(temporal_feature)
#         # temporal_feature = temporal_feature.reshape(-1, temporal_feature.size(2))
#         # temporal_feature = self.__temporal_transformer(temporal_feature)
#         # temporal_output = self.__temporal_linear_layers(temporal_feature)
#         # temporal_output = temporal_output.reshape(temporal_output.size(0) // 365, 365)  # 时间序列得到输出, 是目标序列的时序情况
#
#         B, T, F = spatial_feature.shape
#         spatial_feature = spatial_feature.reshape(B * T, F)
#         spatial_feature_residual = spatial_feature.clone()
#         for i in range(self.k_times - 1):
#             # 提 网络层
#             spatial_memory = self.__spatial_module[2 * i]
#             spatial_linear_layer = self.__spatial_module[2 * i + 1]
#             # 1. 空序优化
#             ## 平方欧氏距离计算 (通过词向量方式, 更新出最完整最贴切的空间特征)
#             d_no_grad = torch.sum(spatial_feature.square(), dim=1, keepdim=True) \
#                         + torch.sum(spatial_memory.weight.data.square(), dim=1, keepdim=False)
#             ## addmm_(): d_no_grad = d_no_grad + (-2)*A@B
#             d_no_grad.addmm_(spatial_feature, spatial_memory.weight.data.T, beta=1, alpha=-2)
#             vocab_idx_list_N = torch.argmin(d_no_grad, dim=1)  # (N, vocab_size)
#             vocab_idx_list_BTF = vocab_idx_list_N.reshape(B, T, 1)
#             spatial_feature = spatial_memory(vocab_idx_list_BTF.reshape(-1, 1)).reshape(B * T, F)
#             # print(spatial_feature.shape, normal_data.shape)
#             spatial_feature = torch.cat((spatial_feature.reshape(B, T, F), normal_data), dim=-1).reshape(B * T, -1)
#             # 2. 预测w和b
#             spatial_weight_bias = spatial_linear_layer(spatial_feature).reshape(B, T, 2)
#             spatial_weight = spatial_weight_bias[:, :, 0]
#             spatial_bias = spatial_weight_bias[:, :, 1]
#             # 3. 加权
#             temporal_output = temporal_output * spatial_weight + spatial_bias
#             # 去 特征
#             spatial_feature_residual = spatial_feature_residual - spatial_feature
#         i = self.k_times - 1
#         spatial_memory = self.__spatial_module[2 * i]
#         spatial_linear_layer = self.__spatial_module[2 * i + 1]
#         d_no_grad = torch.sum(spatial_feature.square(), dim=1, keepdim=True) \
#                     + torch.sum(spatial_memory.weight.data.square(), dim=1, keepdim=False)
#         d_no_grad.addmm_(spatial_feature, spatial_memory.weight.data.T, beta=1, alpha=-2)
#         vocab_idx_list_N = torch.argmin(d_no_grad, dim=1)  # (N, vocab_size)
#         vocab_idx_list_BTF = vocab_idx_list_N.reshape(B, T, 1)
#         spatial_feature = spatial_memory(vocab_idx_list_BTF.reshape(-1, 1)).reshape(B * T, F)
#         spatial_feature = torch.cat((spatial_feature.reshape(B, T, F), normal_data), dim=-1).reshape(B * T, -1)
#         spatial_bias = spatial_linear_layer(spatial_feature).reshape(B, T, 1)
#         # 3. 加权
#         temporal_output = temporal_output + spatial_bias
#
#         return temporal_output
#
#
# class TS_Model_melt_emb(nn.Module):
#     def __init__(self, t_model: T_Model, normal_feature_num: int, device, Linear_Normal_Func=nn.BatchNorm1d):
#         super().__init__()
#         self.feature_num = t_model.feature_num
#         self.normal_feature_num = normal_feature_num
#         self.__temporal_block = t_model
#         neueon = [200, 50]
#         if Linear_Normal_Func == nn.LayerNorm:
#             self.__spatial_linear_layer = nn.Sequential(
#                 nn.Linear(self.feature_num + self.normal_feature_num, self.h1),
#                 Linear_Normal_Func(self.h1),
#                 nn.PReLU(),
#                 # nn.Dropout(0.2),
#                 nn.Linear(self.h1, self.h2),
#                 Linear_Normal_Func(self.h2),
#                 nn.PReLU(),
#                 # nn.Dropout(0.2),
#                 nn.Linear(self.h2, 2),
#                 # nn.PReLU()
#                 nn.Sigmoid()
#             ).to(device)
#         elif Linear_Normal_Func == nn.BatchNorm1d:
#             self.__spatial_linear_layer = nn.Sequential(
#                 nn.Linear(self.feature_num + self.normal_feature_num, self.h1),
#                 Linear_Normal_Func(t_model.series_len),
#                 nn.PReLU(),
#                 # nn.Dropout(0.2),
#                 nn.Linear(self.h1, self.h2),
#                 Linear_Normal_Func(t_model.series_len),
#                 nn.PReLU(),
#                 # nn.Dropout(0.2),
#                 nn.Linear(self.h2, 2),
#                 # nn.PReLU()
#                 nn.Sigmoid()
#             ).to(device)
#
#     def forward(self, temporal_map: torch.Tensor, spatial_id_list: list, normal_data: torch.Tensor):
#         if list(temporal_map.shape) != [temporal_map.shape[0], 365, self.feature_num]:
#             raise ValueError(f"temporal_map shape should be "
#                              f"torch.Size([{temporal_map.shape[0]}, 365, {self.feature_num}]),"
#                              f"actual shape is {temporal_map.shape}")
#         # 预处理
#         temporal_detach = temporal_map.detach()
#
#         # 分离时序空序
#         temporal_output, temporal_feature = self.__temporal_block.forward_4_TS(temporal_detach)  # 时间特征强化, 使周期性更加明显
#         spatial_feature = temporal_map  # 先复制一份
#         for i in range(9):  # f * 3 * 3  9层感受野
#             spatial_feature = spatial_feature - temporal_feature[:, :, i: i + self.feature_num]  # 剩余特征应该大半都是空间特征了
#
#         B, T, F = spatial_feature.shape
#         # 预测w和b
#         spatial_feature = torch.cat((spatial_feature, normal_data), dim=-1).reshape(B * T, -1)
#         spatial_weight_bias = self.__spatial_linear_layer(spatial_feature).reshape(B, T, 2)
#         spatial_weight = spatial_weight_bias[:, :, 0]
#         spatial_bias = spatial_weight_bias[:, :, 1]
#         # 加权
#         temporal_output = temporal_output * spatial_weight + spatial_bias
#
#         return temporal_output
#
#
# class TS_Model_melt_geo(nn.Module):
#     def __init__(self, t_model: T_Model, normal_feature_num: int, device, Linear_Normal_Func=nn.BatchNorm1d):
#         super().__init__()
#         self.feature_num = t_model.feature_num
#         self.normal_feature_num = normal_feature_num  # 地理
#         self.__temporal_block = t_model
#
#         self.__spatial_memory = nn.Embedding(1000, self.feature_num).to(device)  # （词向量个数, 特征数）
#         neueon = [200, 50]
#         if Linear_Normal_Func == nn.LayerNorm:
#             self.__spatial_linear_layer = nn.Sequential(
#                 nn.Linear(self.feature_num, self.h1),
#                 Linear_Normal_Func(self.h1),
#                 nn.PReLU(),
#                 # nn.Dropout(0.2),
#                 nn.Linear(self.h1, self.h2),
#                 Linear_Normal_Func(self.h2),
#                 nn.PReLU(),
#                 # nn.Dropout(0.2),
#                 nn.Linear(self.h2, 2),
#                 # nn.PReLU()
#                 nn.Sigmoid()
#             ).to(device)
#         elif Linear_Normal_Func == nn.BatchNorm1d:
#             self.__spatial_linear_layer = nn.Sequential(
#                 nn.Linear(self.feature_num, self.h1),
#                 Linear_Normal_Func(t_model.series_len),
#                 nn.PReLU(),
#                 # nn.Dropout(0.2),
#                 nn.Linear(self.h1, self.h2),
#                 Linear_Normal_Func(t_model.series_len),
#                 nn.PReLU(),
#                 # nn.Dropout(0.2),
#                 nn.Linear(self.h2, 2),
#                 # nn.PReLU()
#                 nn.Sigmoid()
#             ).to(device)
#
#         del neueon
#
#     def forward(self, temporal_map: torch.Tensor, spatial_id_list: list, normal_data: torch.Tensor):
#         if list(temporal_map.shape) != [temporal_map.shape[0], 365, self.feature_num]:
#             raise ValueError(f"temporal_map shape should be "
#                              f"torch.Size([{temporal_map.shape[0]}, 365, {self.feature_num}]),"
#                              f"actual shape is {temporal_map.shape}")
#         # 预处理
#         temporal_detach = temporal_map.detach()
#
#         # 分离时序空序
#         temporal_output, temporal_feature = self.__temporal_block.forward_4_TS(temporal_detach)  # 时间特征强化, 使周期性更加明显
#         spatial_feature = temporal_map  # 先复制一份
#         for i in range(9):  # f * 3 * 3  9层感受野
#             spatial_feature = spatial_feature - temporal_feature[:, :, i: i + self.feature_num]  # 剩余特征应该大半都是空间特征了
#
#         B, T, F = spatial_feature.shape
#         spatial_feature = spatial_feature.reshape(B * T, F)
#         # 1. 空序优化
#         ## 平方欧氏距离计算 (通过词向量方式, 更新出最完整最贴切的空间特征)
#         d_no_grad = torch.sum(spatial_feature.square(), dim=1, keepdim=True) \
#                     + torch.sum(self.__spatial_memory.weight.data.square(), dim=1, keepdim=False)
#         ## addmm_(): d_no_grad = d_no_grad + (-2)*A@B
#         d_no_grad.addmm_(spatial_feature, self.__spatial_memory.weight.data.T, beta=1, alpha=-2)
#         vocab_idx_list_N = torch.argmin(d_no_grad, dim=1)  # (N, vocab_size)
#         vocab_idx_list_BTF = vocab_idx_list_N.reshape(B, T, 1)
#         spatial_feature = self.__spatial_memory(vocab_idx_list_BTF.reshape(-1, 1)).reshape(B * T, F)
#         # print(spatial_feature.shape, normal_data.shape)
#         spatial_feature = spatial_feature.reshape(B * T, F)
#         # 2. 预测w和b
#         spatial_weight_bias = self.__spatial_linear_layer(spatial_feature).reshape(B, T, 2)
#         spatial_weight = spatial_weight_bias[:, :, 0]
#         spatial_bias = spatial_weight_bias[:, :, 1]
#         # 3. 加权
#         temporal_output = temporal_output * spatial_weight + spatial_bias
#
#         return temporal_output
#
#
# class TS_Model_melt_residual_spatial(nn.Module):
#     def __init__(self, t_model: T_Model, normal_feature_num: int, device, Linear_Normal_Func=nn.BatchNorm1d):
#         super().__init__()
#         self.feature_num = t_model.feature_num
#         self.normal_feature_num = normal_feature_num  # 地理
#         self.__temporal_block = t_model
#
#         neueon = [200, 50]
#         if Linear_Normal_Func == nn.LayerNorm:
#             self.__spatial_linear_layer = nn.Sequential(
#                 nn.Linear(self.normal_feature_num, self.h1),
#                 Linear_Normal_Func(self.h1),
#                 nn.PReLU(),
#                 # nn.Dropout(0.2),
#                 nn.Linear(self.h1, self.h2),
#                 Linear_Normal_Func(self.h2),
#                 nn.PReLU(),
#                 # nn.Dropout(0.2),
#                 nn.Linear(self.h2, 2),
#                 # nn.PReLU()
#                 nn.Sigmoid()
#             ).to(device)
#         elif Linear_Normal_Func == nn.BatchNorm1d:
#             self.__spatial_linear_layer = nn.Sequential(
#                 nn.Linear(self.normal_feature_num, self.h1),
#                 Linear_Normal_Func(t_model.series_len),
#                 nn.PReLU(),
#                 # nn.Dropout(0.2),
#                 nn.Linear(self.h1, self.h2),
#                 Linear_Normal_Func(t_model.series_len),
#                 nn.PReLU(),
#                 # nn.Dropout(0.2),
#                 nn.Linear(self.h2, 2),
#                 # nn.PReLU()
#                 nn.Sigmoid()
#             ).to(device)
#
#         del neueon
#
#     def forward(self, temporal_map: torch.Tensor, spatial_id_list: list, normal_data: torch.Tensor):
#         if list(temporal_map.shape) != [temporal_map.shape[0], 365, self.feature_num]:
#             raise ValueError(f"temporal_map shape should be "
#                              f"torch.Size([{temporal_map.shape[0]}, 365, {self.feature_num}]),"
#                              f"actual shape is {temporal_map.shape}")
#         # 预处理
#         temporal_detach = temporal_map.detach()
#
#         # 分离时序空序
#         temporal_output, temporal_feature = self.__temporal_block.forward_4_TS(temporal_detach)  # 时间特征强化, 使周期性更加明显
#         # 2. 预测w和b
#         B, T, F = normal_data.shape
#         normal_data = normal_data.reshape(B * T, F)
#         spatial_weight_bias = self.__spatial_linear_layer(normal_data).reshape(B, T, 2)
#         spatial_weight = spatial_weight_bias[:, :, 0]
#         spatial_bias = spatial_weight_bias[:, :, 1]
#         # 3. 加权
#         temporal_output = temporal_output * spatial_weight + spatial_bias
#
#         return temporal_output
#
#
# class TS_Model_trans_sMLP2Linear(nn.Module):
#     def __init__(self, t_model: T_Model, normal_feature_num: int, device, Linear_Normal_Func=nn.BatchNorm1d):
#         super().__init__()
#         self.feature_num = t_model.feature_num
#         self.normal_feature_num = normal_feature_num  # 地理
#         self.__temporal_block = t_model  # None  |Over: 3.40
#
#         self.__spatial_memory = nn.Embedding(1000, self.feature_num).to(device)  # （词向量个数, 特征数）
#         self.__spatial_linear_layer = nn.Sequential(
#             nn.Linear(self.feature_num + self.normal_feature_num, 2),
#             nn.Sigmoid()
#         ).to(device)
#
#     def forward(self, temporal_map: torch.Tensor, normal_data: torch.Tensor):
#         if list(temporal_map.shape) != [temporal_map.shape[0], 365, self.feature_num]:
#             raise ValueError(f"temporal_map shape should be "
#                              f"torch.Size([{temporal_map.shape[0]}, 365, {self.feature_num}]),"
#                              f"actual shape is {temporal_map.shape}")
#         # 预处理
#         temporal_detach = temporal_map.detach()
#
#         # 分离时序空序
#         temporal_output, temporal_feature = self.__temporal_block.forward_4_TS(temporal_detach)  # 时间特征强化, 使周期性更加明显
#         spatial_feature = temporal_map  # 先复制一份
#         for i in range(9):  # f * 3 * 3  9层感受野
#             spatial_feature = spatial_feature - temporal_feature[:, :, i: i + self.feature_num]  # 剩余特征应该大半都是空间特征了
#
#         B, T, F = spatial_feature.shape
#         spatial_feature = spatial_feature.reshape(B * T, F)
#         # 1. 空序优化
#         ## 平方欧氏距离计算 (通过词向量方式, 更新出最完整最贴切的空间特征)
#         d = torch.sum(spatial_feature.square(), dim=1, keepdim=True) \
#             + torch.sum(self.__spatial_memory.weight.data.square(), dim=1, keepdim=False)
#         ## addmm_(): d_no_grad = d_no_grad + (-2)*A@B
#         d.addmm_(spatial_feature, self.__spatial_memory.weight.data.T, beta=1, alpha=-2)
#         vocab_idx_list_N = torch.argmin(d, dim=1)  # (N, vocab_size)
#         vocab_idx_list_BTF = vocab_idx_list_N.reshape(B, T, 1)
#         memory_feature = self.__spatial_memory(vocab_idx_list_BTF.reshape(-1, 1)).reshape(B * T, F)
#         # print(spatial_feature.shape, normal_data.shape)
#         spatial_feature = torch.cat((memory_feature.reshape(B, T, F), normal_data), dim=-1).reshape(B * T, -1)
#         # 2. 预测w和b
#         spatial_weight_bias = self.__spatial_linear_layer(spatial_feature).reshape(B, T, 2)
#         spatial_weight = spatial_weight_bias[:, :, 0]
#         spatial_bias = spatial_weight_bias[:, :, 1]
#         # 3. 加权
#         temporal_output = temporal_output * spatial_weight + spatial_bias
#
#         return temporal_output
#
#     def hidden_params(self, temporal_map: torch.Tensor, spatial_id_list: list, normal_data: torch.Tensor):
#         if list(temporal_map.shape) != [temporal_map.shape[0], 365, self.feature_num]:
#             raise ValueError(f"temporal_map shape should be "
#                              f"torch.Size([{temporal_map.shape[0]}, 365, {self.feature_num}]),"
#                              f"actual shape is {temporal_map.shape}")
#         temporal_detach = temporal_map.detach()
#         temporal_output, temporal_feature = self.__temporal_block.forward_4_TS(temporal_detach)
#         spatial_feature = temporal_map
#         for i in range(9):
#             spatial_feature = spatial_feature - temporal_feature[:, :, i: i + self.feature_num]
#         B, T, F = spatial_feature.shape
#         spatial_feature = spatial_feature.reshape(B * T, F)
#         d = torch.sum(spatial_feature.square(), dim=1, keepdim=True) \
#             + torch.sum(self.__spatial_memory.weight.data.square(), dim=1, keepdim=False)
#         d.addmm_(spatial_feature, self.__spatial_memory.weight.data.T, beta=1, alpha=-2)
#         vocab_idx_list_N = torch.argmin(d, dim=1)
#         vocab_idx_list_BTF = vocab_idx_list_N.reshape(B, T, 1)
#         memory_feature = self.__spatial_memory(vocab_idx_list_BTF.reshape(-1, 1)).reshape(B * T, F)
#         spatial_feature = torch.cat((memory_feature.reshape(B, T, F), normal_data), dim=-1).reshape(B * T, -1)
#         spatial_weight_bias = self.__spatial_linear_layer(spatial_feature).reshape(B, T, 2)
#         spatial_weight = spatial_weight_bias[:, :, 0]
#         spatial_bias = spatial_weight_bias[:, :, 1]
#         temporal_output = temporal_output * spatial_weight + spatial_bias
#
#         return (temporal_feature.reshape(B, T, -1), temporal_output.reshape(B, T, -1),
#                 spatial_feature.reshape(B, T, -1), normal_data.reshape(B, T, -1), memory_feature.reshape(B, T, -1),
#                 spatial_weight.reshape(B, T, -1), spatial_bias.reshape(B, T, -1),
#                 temporal_output.reshape(B, T, -1), B, T)
#
#
# class TS_Model_hyper(nn.Module):
#     def __init__(self, t_model: T_Model, normal_feature_num: int, device, h1, h2, self.memory_capacity=1000,
#                  Linear_Normal_Func=nn.BatchNorm1d):
#         super().__init__()
#         self.feature_num = t_model.feature_num
#         self.normal_feature_num = normal_feature_num
#         self.__temporal_block = t_model
#         # neueon_options = [
#         #     [200, 50],  # 1 正常
#         #
#         #     # [200, 30],
#         #     # [200, 35],
#         #     [200, 40], # 2
#         #     [200, 45], # 3
#         #     [200, 55], # 4
#         #     [200, 60], # 5
#         #     # [200, 65],
#         #     # [200, 70],
#         #     [160, 50], # 6
#         #     [180, 50], # 7
#         #     [220, 50], # 8
#         #     [240, 50], # 9
#         #     #
#         #     [200, 35], # 10, 2Z
#         #     [200, 47], # 11, 3Y
#         #     [200, 53], # 12, 4Z
#         #     [200, 65], # 13, 5Y
#         #     #
#         #     [210, 50], # 14
#         #     [230, 50], # 15
#         #     #
#         #     [200, 30], # 16, 10Z
#         # ]
#         # option = option - 1
#         # neueon = neueon_options[option - 1]
#
#         self.__spatial_memory = nn.Embedding(self.memory_capacity, self.feature_num).to(device)  # （词向量个数, 特征数）
#         self.__spatial_linear_layer = nn.Sequential(
#             nn.Linear(self.feature_num + self.normal_feature_num, h1),
#             Linear_Normal_Func(h1),
#             nn.PReLU(),
#             nn.Linear(h1, h2),
#             Linear_Normal_Func(h2),
#             nn.PReLU(),
#             nn.Linear(h2, 2),
#             nn.Sigmoid()
#         ).to(device)
#         if Linear_Normal_Func == nn.LayerNorm:
#             self.__spatial_linear_layer = nn.Sequential(
#                 nn.Linear(self.feature_num + self.normal_feature_num, h1),
#                 Linear_Normal_Func(h1),
#                 nn.PReLU(),
#                 # nn.Dropout(0.2),
#                 nn.Linear(h1, h2),
#                 Linear_Normal_Func(h2),
#                 nn.PReLU(),
#                 # nn.Dropout(0.2),
#                 nn.Linear(h2, 2),
#                 # nn.PReLU()
#                 nn.Sigmoid()
#             ).to(device)
#         elif Linear_Normal_Func == nn.BatchNorm1d:
#             self.__spatial_linear_layer = nn.Sequential(
#                 nn.Linear(self.feature_num + self.normal_feature_num, h1),
#                 Linear_Normal_Func(t_model.series_len),
#                 nn.PReLU(),
#                 # nn.Dropout(0.2),
#                 nn.Linear(h1, h2),
#                 Linear_Normal_Func(t_model.series_len),
#                 nn.PReLU(),
#                 # nn.Dropout(0.2),
#                 nn.Linear(h2, 2),
#                 # nn.PReLU()
#                 nn.Sigmoid()
#             ).to(device)
#
#
#         # os.makedirs(model_save_path, exist_ok=True)
#         # with open(os.path.join(model_save_path, f"s_model_params{self.memory_capacity}_{self.h1}_{self.h2}.json"),
#         #           'r') as f:
#         #     json.dump(self.__dict__, f)
#
#         
#
#     def forward(self, temporal_map: torch.Tensor, normal_data: torch.Tensor):
#         # if list(temporal_map.shape) != [temporal_map.shape[0], 365, self.feature_num]:
#         #     raise ValueError(f"temporal_map shape should be "
#         #                      f"torch.Size([{temporal_map.shape[0]}, 365, {self.feature_num}]),"
#         #                      f"actual shape is {temporal_map.shape}")
#         # 预处理
#         temporal_detach = temporal_map.detach()
#
#         # 分离时序空序
#         temporal_output, temporal_feature = self.__temporal_block.forward_4_TS(temporal_detach)  # 时间特征强化, 使周期性更加明显
#         spatial_feature = temporal_map  # 先复制一份
#         for i in range(9):  # f * 3 * 3  9层感受野
#             spatial_feature = spatial_feature - temporal_feature[:, :, i: i + self.feature_num]  # 剩余特征应该大半都是空间特征了
#
#         # 时序预测
#         # temporal_feature = self.__temporal_linear_layer_pre(temporal_feature)
#         # temporal_feature = temporal_feature.reshape(-1, temporal_feature.size(2))
#         # temporal_feature = self.__temporal_transformer(temporal_feature)
#         # temporal_output = self.__temporal_linear_layers(temporal_feature)
#         # temporal_output = temporal_output.reshape(temporal_output.size(0) // 365, 365)  # 时间序列得到输出, 是目标序列的时序情况
#
#         B, T, F = spatial_feature.shape
#         spatial_feature = spatial_feature.reshape(B * T, F)
#         # 1. 空序优化
#         ## 平方欧氏距离计算 (通过词向量方式, 更新出最完整最贴切的空间特征)
#         d = torch.sum(spatial_feature.square(), dim=1, keepdim=True) \
#             + torch.sum(self.__spatial_memory.weight.data.square(), dim=1, keepdim=False)
#         ## addmm_(): d_no_grad = d_no_grad + (-2)*A@B
#         d.addmm_(spatial_feature, self.__spatial_memory.weight.data.T, beta=1, alpha=-2)
#         vocab_idx_list_N = torch.argmin(d, dim=1)  # (N, vocab_size)
#         vocab_idx_list_BTF = vocab_idx_list_N.reshape(B, T, 1)
#         memory_feature = self.__spatial_memory(vocab_idx_list_BTF.reshape(-1, 1)).reshape(B * T, F)
#         # print(spatial_feature.shape, normal_data.shape)
#         spatial_feature = torch.cat((memory_feature.reshape(B, T, F), normal_data), dim=-1).reshape(B * T, -1)
#         # 2. 预测w和b
#         spatial_weight_bias = self.__spatial_linear_layer(spatial_feature).reshape(B, T, 2)
#         spatial_weight = spatial_weight_bias[:, :, 0]
#         spatial_bias = spatial_weight_bias[:, :, 1]
#         # 3. 加权
#         temporal_output = temporal_output * spatial_weight + spatial_bias
#
#         return temporal_output
#
#     def hidden_params(self, temporal_map: torch.Tensor, normal_data: torch.Tensor):
#         # if list(temporal_map.shape) != [temporal_map.shape[0], 365, self.feature_num]:
#         #     raise ValueError(f"temporal_map shape should be "
#         #                      f"torch.Size([{temporal_map.shape[0]}, 365, {self.feature_num}]),"
#         #                      f"actual shape is {temporal_map.shape}")
#         temporal_detach = temporal_map.detach()
#         temporal_output, temporal_feature = self.__temporal_block.forward_4_TS(temporal_detach)
#         spatial_feature = temporal_map
#         for i in range(9):
#             spatial_feature = spatial_feature - temporal_feature[:, :, i: i + self.feature_num]
#         B, T, F = spatial_feature.shape
#         spatial_feature = spatial_feature.reshape(B * T, F)
#         d = torch.sum(spatial_feature.square(), dim=1, keepdim=True) \
#             + torch.sum(self.__spatial_memory.weight.data.square(), dim=1, keepdim=False)
#         d.addmm_(spatial_feature, self.__spatial_memory.weight.data.T, beta=1, alpha=-2)
#         vocab_idx_list_N = torch.argmin(d, dim=1)
#         vocab_idx_list_BTF = vocab_idx_list_N.reshape(B, T, 1)
#         memory_feature = self.__spatial_memory(vocab_idx_list_BTF.reshape(-1, 1)).reshape(B * T, F)
#         spatial_feature = torch.cat((memory_feature.reshape(B, T, F), normal_data), dim=-1).reshape(B * T, -1)
#         spatial_weight_bias = self.__spatial_linear_layer(spatial_feature).reshape(B, T, 2)
#         spatial_weight = spatial_weight_bias[:, :, 0]
#         spatial_bias = spatial_weight_bias[:, :, 1]
#         temporal_output = temporal_output * spatial_weight + spatial_bias
#
#         return (temporal_feature.reshape(B, T, -1), temporal_output.reshape(B, T, -1),
#                 spatial_feature.reshape(B, T, -1), normal_data.reshape(B, T, -1), memory_feature.reshape(B, T, -1),
#                 spatial_weight.reshape(B, T, -1), spatial_bias.reshape(B, T, -1),
#                 temporal_output.reshape(B, T, -1), B, T)


class TS_Model_ENHANCE(nn.Module):
    def __init__(self, t_model: T_Model_new | T_Model_trans_tMLP2Linear, normal_feature_num: int, model_save_path, device,
                 Linear_Normal_Func=nn.BatchNorm1d):
        super().__init__()
        self.feature_num = t_model.feature_num
        self.normal_feature_num = normal_feature_num  # 地理
        self.__temporal_block = t_model  # None  |Over: 3.40
        # self.__temporal_block.requires_grad_(False)
        # 显存消耗: 6.6G
        # 超参对比:
        # 线性对比  (~<2.7)
        # neueon_options = [
        #     # BEST 200-50-v1-1k
        #     [200, 30],  # -v1-  noDropout
        #     #             -v2-  Dropout   Y  2.72
        #     [200, 40],  # -v1-  noDropout X  2.88
        #     #             -v2-  Dropout   X  2.92
        #     [200, 45],  # -v1-  noDropout X  3.17
        #     # -v2-  Dropout
        #     [200, 50],
        #     # -v1-  Sigmoid + noDropout Y  2.48  |Over: 2.2    -again- Y 2.47   -memory2k- 2.48   -memory4k- 2.6061
        #     #             -v2-  Sigmoid + Dropout   X  2.78  |Over: 2.73   -again- X 2.90
        #     #             -v3-  PRelu + noDropout   X  2.96  |Over: 2.92
        #     [200, 55],  # -v1-  noDropout X  2.9
        #     #             -v2-  Dropout   X  2.86
        #     [200, 60],  # -v1-  noDropout X  2.85
        #     #             -v2-  Dropout   Y  2.68
        #     [200, 70],  # -v1-
        #     [200, 80],  # -v2-  Dropout   X  2.83
        #     # [450, 50],  # X  3.13
        #     # [450, 80],  # X  2.85
        #     # [900, 80],
        #     # [1350, 80], # X  2.92
        # ]
        # option = 4
        # memory容量对比   1000 2000 4000
        self.memory_capacity = 1000

        # neueon = neueon_options[option - 1]
        self.h1 = 200
        self.h2 = 50
        self.h3 = 150
        self.h4 = 16
        self.__residual_memory = nn.Embedding(self.memory_capacity, self.feature_num).to(device)  # （词向量个数, 特征数）
        if Linear_Normal_Func == nn.LayerNorm:
            self.__residual_linear_layer = nn.Sequential(
                nn.Linear(self.feature_num, self.h1),
                Linear_Normal_Func(self.h1),
                nn.PReLU(),
                # nn.Dropout(0.2),
                nn.Linear(self.h1, self.h2),
                Linear_Normal_Func(self.h2),
                nn.PReLU(),
                # nn.Dropout(0.2),
                nn.Linear(self.h2, 2),
                # nn.PReLU()
                nn.Sigmoid()
            ).to(device)
        elif Linear_Normal_Func == nn.BatchNorm1d:
            self.__residual_linear_layer = nn.Sequential(
                nn.Linear(self.feature_num, self.h1),
                Linear_Normal_Func(t_model.series_len),
                nn.PReLU(),
                # nn.Dropout(0.2),
                nn.Linear(self.h1, self.h2),
                Linear_Normal_Func(t_model.series_len),
                nn.PReLU(),
                # nn.Dropout(0.2),
                nn.Linear(self.h2, 2),
                # nn.PReLU()
                nn.Sigmoid()
            ).to(device)

        if Linear_Normal_Func == nn.LayerNorm:
            self.__direction_linear_layer = nn.Sequential(
                nn.Linear(self.normal_feature_num, 200),
                Linear_Normal_Func(200),
                nn.PReLU(),
                # nn.Dropout(0.2),
                nn.Linear(200, 16),
                Linear_Normal_Func(16),
                nn.PReLU(),
                # nn.Dropout(0.2),
                nn.Linear(16, 2),
                # nn.PReLU()
                nn.Sigmoid()
            ).to(device)
        elif Linear_Normal_Func == nn.BatchNorm1d:
            self.__direction_linear_layer = nn.Sequential(
                nn.Linear(self.normal_feature_num, 200),
                Linear_Normal_Func(t_model.series_len),
                nn.PReLU(),
                # nn.Dropout(0.2),
                nn.Linear(200, 16),
                Linear_Normal_Func(t_model.series_len),
                nn.PReLU(),
                # nn.Dropout(0.2),
                nn.Linear(16, 2),
                # nn.PReLU()
                nn.Sigmoid()
            ).to(device)

        

    def forward(self, temporal_map: torch.Tensor, normal_data: torch.Tensor):
        # 预处理
        temporal_detach = temporal_map.detach()

        # 分离时序空序
        temporal_output, residual_feature = self.__temporal_block.forward_4_TS(temporal_detach)  # 时间特征强化, 使周期性更加明显

        B, T, *_ = temporal_output.shape
        spatial_weight_bias = self.__direction_linear_layer(normal_data).reshape(B, T, 2)
        spatial_weight = spatial_weight_bias[:, :, 0].reshape(B, T, -1)
        spatial_bias = spatial_weight_bias[:, :, 1].reshape(B, T, -1)
        temporal_output = temporal_output * spatial_weight + spatial_bias

        spatial_feature = temporal_map  # 先复制一份
        for i in range(9):  # f * 3 * 3  9层感受野
            spatial_feature = spatial_feature - residual_feature[:, :, i: i + self.feature_num]  # 剩余特征应该大半都是空间特征了

        # 时序预测
        # residual_feature = self.__temporal_linear_layer_pre(residual_feature)
        # residual_feature = residual_feature.reshape(-1, residual_feature.size(2))
        # residual_feature = self.__temporal_transformer(residual_feature)
        # temporal_output = self.__temporal_linear_layers(residual_feature)
        # temporal_output = temporal_output.reshape(temporal_output.size(0) // 365, 365)  # 时间序列得到输出, 是目标序列的时序情况

        B, T, F = spatial_feature.shape
        spatial_feature = spatial_feature.reshape(B * T, F)
        # 1. 空序优化
        ## 平方欧氏距离计算 (通过词向量方式, 更新出最完整最贴切的空间特征)
        d = torch.sum(spatial_feature.square(), dim=1, keepdim=True) \
            + torch.sum(self.__residual_memory.weight.data.square(), dim=1, keepdim=False)
        ## addmm_(): d_no_grad = d_no_grad + (-2)*A@B
        d.addmm_(spatial_feature, self.__residual_memory.weight.data.T, beta=1, alpha=-2)
        vocab_idx_list_N = torch.argmin(d, dim=1)  # (N, vocab_size)
        vocab_idx_list_BTF = vocab_idx_list_N.reshape(B, T, 1)
        memory_feature = self.__residual_memory(vocab_idx_list_BTF.reshape(-1, 1)).reshape(B * T, F)
        # print(spatial_feature.shape, normal_data.shape)
        spatial_feature = memory_feature.reshape(B * T, -1)
        # 2. 预测w和b
        spatial_weight_bias = self.__residual_linear_layer(spatial_feature).reshape(B, T, 2)
        spatial_weight = spatial_weight_bias[:, :, 0].reshape(B, T, 1)
        spatial_bias = spatial_weight_bias[:, :, 1].reshape(B, T, 1)
        # 3. 加权
        temporal_output = temporal_output * spatial_weight + spatial_bias

        return temporal_output

    def hidden_params(self, temporal_map: torch.Tensor, normal_data: torch.Tensor):
        # if list(temporal_map.shape) != [temporal_map.shape[0], 365, self.feature_num]:
        #     raise ValueError(f"temporal_map shape should be "
        #                      f"torch.Size([{temporal_map.shape[0]}, 365, {self.feature_num}]),"
        #                      f"actual shape is {temporal_map.shape}")
        temporal_detach = temporal_map.detach()

        temporal_output, temporal_feature = self.__temporal_block.forward_4_TS(temporal_detach)

        B, T, *_ = temporal_output.shape
        spatial_weight_bias = self.__direction_linear_layer(normal_data).reshape(B, T, 2)
        spatial_weight = spatial_weight_bias[:, :, 0].reshape(B, T, -1)
        spatial_bias = spatial_weight_bias[:, :, 1].reshape(B, T, -1)
        temporal_output = temporal_output * spatial_weight + spatial_bias

        spatial_feature = temporal_map
        for i in range(9):
            spatial_feature = spatial_feature - temporal_feature[:, :, i: i + self.feature_num]

        B, T, F = spatial_feature.shape
        spatial_feature = spatial_feature.reshape(B * T, F)
        d = torch.sum(spatial_feature.square(), dim=1, keepdim=True) \
            + torch.sum(self.__residual_memory.weight.data.square(), dim=1, keepdim=False)
        d.addmm_(spatial_feature, self.__residual_memory.weight.data.T, beta=1, alpha=-2)
        vocab_idx_list_N = torch.argmin(d, dim=1)
        vocab_idx_list_BTF = vocab_idx_list_N.reshape(B, T, 1)
        memory_feature = self.__residual_memory(vocab_idx_list_BTF.reshape(-1, 1)).reshape(B * T, F)
        spatial_feature = memory_feature.reshape(B * T, -1)
        spatial_weight_bias = self.__residual_linear_layer(spatial_feature).reshape(B, T, 2)
        spatial_weight = spatial_weight_bias[:, :, 0].reshape(B, T, 1)
        spatial_bias = spatial_weight_bias[:, :, 1].reshape(B, T, 1)
        output = temporal_output * spatial_weight + spatial_bias

        return (temporal_feature.reshape(B, T, -1), temporal_output.reshape(B, T, -1),
                spatial_feature.reshape(B, T, -1), normal_data.reshape(B, T, -1), memory_feature.reshape(B, T, -1),
                spatial_weight.reshape(B, T, -1), spatial_bias.reshape(B, T, -1),
                output.reshape(B, T, -1), B, T)

    @property
    def NAME(self) -> str:
        return 'DRCNet'

    def PARAMS(self):
        return f'enhanceTS{str(self.h1)}_{str(self.h2)}_{str(self.h3)}_{str(self.h4)}_{str(self.memory_capacity)}'
