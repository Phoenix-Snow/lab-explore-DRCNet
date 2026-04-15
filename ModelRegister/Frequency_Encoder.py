import torch
from torch import nn

from ModelRegister.Module.TimesBlock import T


class T_Model(nn.Module):
    def __init__(self, feature_num: int, normal_feature_num: int, model_save_path, series_len: int,
                 device: str = 'cuda', Linear_Normal_Func=nn.BatchNorm1d):
        super().__init__()
        # self.hidden_layers = 45
        self.feature_num = feature_num
        self.temporal_net = T(series_len, feature_num, 128, 3, 3).to(device)  # f * 3 * 3
        self.t_feature_num = feature_num * 3 * 3
        # self.temporal_linear_layer_pre = nn.Linear(self.t_feature_num, self.hidden_layers).to(device)
        # self.temporal_encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_layers, nhead=3).to(device)
        # self.temporal_transformer = torch.nn.TransformerEncoder(self.temporal_encoder_layer, num_layers=24).to(device)
        self.h1 = 1350  # 1350 | 1215  1080 | 1485  1620
        self.h2 = 80  # 80 | 72  64 | 88  96
        if Linear_Normal_Func == nn.LayerNorm:
            self.temporal_linear_layers = nn.Sequential(
                nn.Linear(self.t_feature_num, self.h1),
                Linear_Normal_Func(self.h1),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.h1, self.h2),
                Linear_Normal_Func(self.h2),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.h2, 1),
                nn.Sigmoid()
            ).to(device)
        elif Linear_Normal_Func == nn.BatchNorm1d:
            self.temporal_linear_layers = nn.Sequential(
                nn.Linear(self.t_feature_num, self.h1),
                Linear_Normal_Func(series_len),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.h1, self.h2),
                Linear_Normal_Func(series_len),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.h2, 1),
                nn.Sigmoid()
            ).to(device)
        self.device = device
        self.series_len = series_len

        # os.makedirs(model_save_path, exist_ok=True)
        # with open(os.path.join(model_save_path, f't_model_params{self.h1}_{self.h2}.json'), 'r') as f:
        #     json.dump(self.__dict__, f)

    # def linear_init_cpu(self):
    # self.temporal_linear_layer_pre = self.temporal_linear_layer_pre.cpu()
    # self.temporal_linear_layers = self.temporal_linear_layers.cpu()

    def forward(self, temporal_map: torch.Tensor):
        if list(temporal_map.shape) != [temporal_map.shape[0], self.series_len, self.feature_num]:
            raise ValueError(f"temporal_map shape should be "
                             f"torch.Size([{temporal_map.shape[0]}, {self.series_len}, {self.feature_num}]),"
                             f"actual shape is {temporal_map.shape}")

        temporal_feature = self.temporal_net.TimesBlock(temporal_map.to(self.device))  # 时间特征强化, 使周期性更加明显

        # 时序预测
        # temporal_feature = torch.cat((temporal_feature.to(self.device), normal_data.to(self.device)), dim=-1)  # 拼接地理信息
        # temporal_feature = self.temporal_linear_layer_pre(temporal_feature)
        # temporal_feature = temporal_feature.reshape(-1, self.hidden_layers)
        # temporal_feature = self.temporal_transformer(temporal_feature)
        # temporal_feature = temporal_feature.reshape(-1, self.t_feature_num)

        temporal_output = self.temporal_linear_layers(temporal_feature)
        # temporal_output = temporal_output.reshape(temporal_output.size(0) // self.series_len,
        #                                           self.series_len)  # 时间序列得到输出, 是目标序列的时序情况

        return temporal_output

    def forward_4_TS(self, temporal_map: torch.Tensor):
        # if list(temporal_map.shape) != [temporal_map.shape[0], self.series_len, self.feature_num]:
        #     raise ValueError(f"temporal_map shape should be "
        #                      f"torch.Size([{temporal_map.shape[0]}, {self.series_len}, {self.feature_num}]),"
        #                      f"actual shape is {temporal_map.shape}")

        temporal_feature = self.temporal_net.TimesBlock(temporal_map.to(self.device))  # 时间特征强化, 使周期性更加明显

        # 时序预测
        # temporal_feature = torch.cat((temporal_feature.to(self.device), normal_data.to(self.device)), dim=-1)  # 拼接地理信息
        # temporal_feature = self.temporal_linear_layer_pre(temporal_feature)
        # temporal_feature = temporal_feature.reshape(-1, self.hidden_layers)
        # temporal_feature = self.temporal_transformer(temporal_feature)

        # temporal_feature = temporal_feature.reshape(-1, self.t_feature_num)

        temporal_output = self.temporal_linear_layers(temporal_feature)
        # temporal_output = temporal_output.reshape(temporal_output.size(0) // self.series_len,
        #                                           self.series_len)  # 时间序列得到输出, 是目标序列的时序情况

        return temporal_output, temporal_feature


class T_Model_trans_tMLP2Linear(nn.Module):
    def __init__(self, feature_num: int, normal_feature_num: int, series_len, device: str = 'cuda'):
        super().__init__()
        self.feature_num = feature_num
        self.temporal_net = T(series_len, feature_num, 128, 3, 3).to(device)  # f * 3 * 3
        self.t_feature_num = feature_num * 3 * 3
        self.temporal_linear_layers = nn.Sequential(
            nn.Linear(self.t_feature_num, 1),
            nn.Sigmoid()
        ).to(device)
        self.device = device

    def forward(self, temporal_map: torch.Tensor, spatial_id_list: list, normal_data: torch.Tensor):
        if list(temporal_map.shape) != [temporal_map.shape[0], self.series_len, self.feature_num]:
            raise ValueError(f"temporal_map shape should be "
                             f"torch.Size([{temporal_map.shape[0]}, {self.series_len}, {self.feature_num}]),"
                             f"actual shape is {temporal_map.shape}")

        temporal_feature = self.temporal_net.TimesBlock(temporal_map.to(self.device))  # 时间特征强化, 使周期性更加明显

        # 时序预测
        temporal_feature = temporal_feature.reshape(-1, self.t_feature_num)

        temporal_output = self.temporal_linear_layers(temporal_feature)
        temporal_output = temporal_output.reshape(temporal_output.size(0) // self.series_len,
                                                  self.series_len)  # 时间序列得到输出, 是目标序列的时序情况

        return temporal_output

    def forward_4_TS(self, temporal_map: torch.Tensor, spatial_id_list: list, normal_data: torch.Tensor):
        if list(temporal_map.shape) != [temporal_map.shape[0], self.series_len, self.feature_num]:
            raise ValueError(f"temporal_map shape should be "
                             f"torch.Size([{temporal_map.shape[0]}, {self.series_len}, {self.feature_num}]),"
                             f"actual shape is {temporal_map.shape}")

        temporal_feature = self.temporal_net.TimesBlock(temporal_map.to(self.device))  # 时间特征强化, 使周期性更加明显

        # 时序预测
        feature = temporal_feature.reshape(-1, self.t_feature_num)

        temporal_output = self.temporal_linear_layers(feature)
        temporal_output = temporal_output.reshape(temporal_output.size(0) // self.series_len,
                                                  self.series_len)  # 时间序列得到输出, 是目标序列的时序情况

        return temporal_output, temporal_feature


class T_Model_hyper(nn.Module):
    def __init__(self, feature_num: int, series_len: int, device: str = 'cuda', hyper=1,
                 Linear_Normal_Func=nn.BatchNorm1d):
        super().__init__()
        # self.hidden_layers = 45
        self.feature_num = feature_num
        self.temporal_net = T(series_len, feature_num, 128, 3, 3).to(device)  # f * 3 * 3
        self.t_feature_num = feature_num * 3 * 3
        # self.temporal_linear_layer_pre = nn.Linear(self.t_feature_num, self.hidden_layers).to(device)
        # self.temporal_encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_layers, nhead=3).to(device)
        # self.temporal_transformer = torch.nn.TransformerEncoder(self.temporal_encoder_layer, num_layers=24).to(device)
        neueon_options = [
            [1350, 80],  # 正常
            [1080, 80],
            [1215, 80],
            [1485, 80],
            [1620, 80],
            [1350, 64],
            [1350, 72],
            [1350, 88],
            [1350, 96],
        ]
        hyper = hyper - 1
        option = neueon_options[hyper]
        self.h1 = option[0]
        self.h2 = option[1]
        if Linear_Normal_Func == nn.LayerNorm:
            self.temporal_linear_layers = nn.Sequential(
                nn.Linear(self.t_feature_num, self.h1),
                Linear_Normal_Func(self.h1),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.h1, self.h2),
                Linear_Normal_Func(self.h2),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.h2, 1),
                nn.Sigmoid()
            ).to(device)
        elif Linear_Normal_Func == nn.BatchNorm1d:
            self.temporal_linear_layers = nn.Sequential(
                nn.Linear(self.t_feature_num, self.h1),
                Linear_Normal_Func(series_len),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.h1, self.h2),
                Linear_Normal_Func(series_len),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.h2, 1),
                nn.Sigmoid()
            ).to(device)
        self.device = device
        self.series_len = series_len

        # os.makedirs(model_save_path, exist_ok=True)
        # with open(os.path.join(model_save_path, f't_model_params{self.h1}_{self.h2}.json'), 'r') as f:
        #     json.dump(self.__dict__, f)

    # def linear_init_cpu(self):
    # self.temporal_linear_layer_pre = self.temporal_linear_layer_pre.cpu()
    # self.temporal_linear_layers = self.temporal_linear_layers.cpu()

    def forward(self, temporal_map: torch.Tensor):
        if list(temporal_map.shape) != [temporal_map.shape[0], self.series_len, self.feature_num]:
            raise ValueError(f"temporal_map shape should be "
                             f"torch.Size([{temporal_map.shape[0]}, {self.series_len}, {self.feature_num}]),"
                             f"actual shape is {temporal_map.shape}")

        temporal_feature = self.temporal_net.TimesBlock(temporal_map.to(self.device))  # 时间特征强化, 使周期性更加明显

        # 时序预测
        # temporal_feature = torch.cat((temporal_feature.to(self.device), normal_data.to(self.device)), dim=-1)  # 拼接地理信息
        # temporal_feature = self.temporal_linear_layer_pre(temporal_feature)
        # temporal_feature = temporal_feature.reshape(-1, self.hidden_layers)
        # temporal_feature = self.temporal_transformer(temporal_feature)
        temporal_feature = temporal_feature.reshape(-1, self.t_feature_num)

        temporal_output = self.temporal_linear_layers(temporal_feature)
        temporal_output = temporal_output.reshape(temporal_output.size(0) // self.series_len,
                                                  self.series_len)  # 时间序列得到输出, 是目标序列的时序情况

        return temporal_output

    def forward_4_TS(self, temporal_map: torch.Tensor):
        # if list(temporal_map.shape) != [temporal_map.shape[0], self.series_len, self.feature_num]:
        #     raise ValueError(f"temporal_map shape should be "
        #                      f"torch.Size([{temporal_map.shape[0]}, {self.series_len}, {self.feature_num}]),"
        #                      f"actual shape is {temporal_map.shape}")

        temporal_feature = self.temporal_net.TimesBlock(temporal_map.to(self.device))  # 时间特征强化, 使周期性更加明显

        # 时序预测
        # temporal_feature = torch.cat((temporal_feature.to(self.device), normal_data.to(self.device)), dim=-1)  # 拼接地理信息
        # temporal_feature = self.temporal_linear_layer_pre(temporal_feature)
        # temporal_feature = temporal_feature.reshape(-1, self.hidden_layers)
        # temporal_feature = self.temporal_transformer(temporal_feature)
        feature = temporal_feature.reshape(-1, self.t_feature_num)

        temporal_output = self.temporal_linear_layers(feature)
        temporal_output = temporal_output.reshape(temporal_output.size(0) // self.series_len,
                                                  self.series_len)  # 时间序列得到输出, 是目标序列的时序情况

        return temporal_output, temporal_feature


class T_Model_new(nn.Module):
    def __init__(self, feature_num: int, normal_feature_num: int, model_save_path, series_len: int,
                 device: str = 'cuda', Linear_Normal_Func=nn.BatchNorm1d):
        super().__init__()
        self.feature_num = feature_num
        self.series_len = series_len
        self.temporal_net = T(series_len, feature_num, 128, 3, 3).to(device)  # f * 3 * 3
        self.t_feature_num = feature_num * 3 * 3
        self.h1 = 1350  # 1350 | 1215  1080 | 1485  1620
        self.h2 = 80  # 80 | 72  64 | 88  96
        if Linear_Normal_Func == nn.LayerNorm:
            self.temporal_linear_layers = nn.Sequential(
                nn.Linear(self.t_feature_num, self.h1),
                Linear_Normal_Func(self.h1),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.h1, self.h2),
                Linear_Normal_Func(self.h2),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.h2, 1),
                nn.Sigmoid()
            ).to(device)
            self.func = 0
        elif Linear_Normal_Func == nn.BatchNorm1d:
            self.temporal_linear_layers = nn.Sequential(
                nn.Linear(self.t_feature_num, self.h1),
                Linear_Normal_Func(self.series_len),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.h1, self.h2),
                Linear_Normal_Func(self.series_len),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.h2, 1),
                nn.Sigmoid()
            ).to(device)
            self.func = 1
        self.device = device
        self.series_len = series_len

    def forward(self, temporal_map: torch.Tensor):
        if list(temporal_map.shape) != [temporal_map.shape[0], self.series_len, self.feature_num]:
            raise ValueError(f"temporal_map shape should be "
                             f"torch.Size([{temporal_map.shape[0]}, {self.series_len}, {self.feature_num}]),"
                             f"actual shape is {temporal_map.shape}")

        temporal_feature = self.temporal_net.TimesBlock(temporal_map.to(self.device))  # 时间特征强化, 使周期性更加明显

        if self.func == 0:
            temporal_output = self.temporal_linear_layers(temporal_feature)
        elif self.func == 1:
            temporal_feature = temporal_feature.transpose(1, 2).reshape(-1, self.series_len)
            temporal_output = self.temporal_linear_layers(temporal_feature)
            temporal_output = temporal_output.reshape(-1, self.series_len).transpose(1, 2)
            # 时间序列得到输出, 是目标序列的时序情况
        else:
            temporal_output = None

        return temporal_output

    def forward_4_TS(self, temporal_map: torch.Tensor):
        if list(temporal_map.shape) != [temporal_map.shape[0], self.series_len, self.feature_num]:
            raise ValueError(f"temporal_map shape should be "
                             f"torch.Size([{temporal_map.shape[0]}, {self.series_len}, {self.feature_num}]),"
                             f"actual shape is {temporal_map.shape}")

        temporal_feature = self.temporal_net.TimesBlock(temporal_map.to(self.device))  # 时间特征强化, 使周期性更加明显

        if self.func == 0:
            temporal_output = self.temporal_linear_layers(temporal_feature)
        elif self.func == 1:
            temporal_feature = temporal_feature.transpose(1, 2).reshape(-1, self.series_len)
            temporal_output = self.temporal_linear_layers(temporal_feature)
            temporal_output = temporal_output.reshape(-1, self.series_len).transpose(1, 2)
            # 时间序列得到输出, 是目标序列的时序情况
        else:
            temporal_output = None

        return temporal_output, temporal_feature

    @property
    def NAME(self):
        return "Frequency-Encoder"

    @property
    def PARAMS(self):
        return f'T{str(self.h1)}_{str(self.h2)}'
