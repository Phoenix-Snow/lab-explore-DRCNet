import torch
from torch import nn


class Inception_Conv_Layer(nn.Module):
    """
    并行卷积层, 每层都是相同输入的卷积层, 卷积核大小不同, 分别为 1x1, 3x3, 5x5, 7x7, 9x9, 11x11 ...
    """
    def __init__(self, in_dim, out_dim, n_kernels=6, init_weight=True):
        super(Inception_Conv_Layer, self).__init__()
        self.conv_layers = nn.ModuleList()
        for i in range(n_kernels):
            self.conv_layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=2 * i + 1, padding=i))  # 1, 3, 5
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_map: torch.Tensor):
        output = []
        for layer in self.conv_layers:
            output_map = layer(input_map).to('cpu')
            output.append(output_map)
        return torch.stack(output, dim=-1).mean(-1).to(input_map.device)
        # [B, F, T_Len // Seq_Len, Seq_Len]
        # (n times conv for input and concat)-> [B, F, T_Len // Seq_Len, Seq_Len, n_layers]
        # (mean for n_layers)-> [B, F, T_Len // Seq_Len, Seq_Len]


class Inception_Same_Linear_Layer(nn.Module):
    """
    并行线性层, 每层都是相同结构和输入的线性层, 唯一线性参数可能不同（in_dim * out_dim）
    """
    def __init__(self, in_dim, out_dim, n_cluster, init_weight=True):
        super().__init__()
        self.linear_layers = nn.ModuleList()
        for i in range(n_cluster):
            self.linear_layers.append(nn.Linear(in_dim, out_dim))
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_map: torch.Tensor):
        """
        input_map: [batch_size, n_vars, in_dim]
        prob: [n_vars, n_cluster]
        return: [batch_size, n_vars, out_dim, n_cluster]
        """
        output = []
        for layer in self.linear_layers:
            output.append(layer(input_map))
        return torch.stack(output, dim=-1).mean(-1).to(input_map.device)  # output转换为张量  [batch_size, n_vars, out_dim, n_cluster]


class Inception_Same_Channel_Conv_Layer(nn.Module):
    """
    并行卷积层, 每层都是相同输入的卷积层, 卷积核大小不同, 分别为 1x1, 3x3, 5x5, 7x7, 9x9, 11x11 ...
    """
    def __init__(self, dim, n_kernels=6, init_weight=True):
        super(Inception_Conv_Layer, self).__init__()
        self.conv_layers = nn.ModuleList()
        for i in range(n_kernels):
            self.conv_layers.append(nn.Conv2d(dim, dim, kernel_size=2 * i + 1, padding=i))  # 1, 3, 5
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_map: torch.Tensor):
        output = []
        for layer in self.conv_layers:
            output_map = layer(input_map)
            output.append(output_map)
        return torch.stack(output, dim=-1).mean(-1).to(input_map.device)
        # [B, F, T_Len // Seq_Len, Seq_Len]
        # (n times conv for input and concat)-> [B, F, T_Len // Seq_Len, Seq_Len, n_layers]
        # (mean for n_layers)-> [B, F, T_Len // Seq_Len, Seq_Len]


class Inception_Conv_Cat_Layer(nn.Module):
    """
    并行卷积层, 每层都是相同输入的卷积层, 卷积核大小不同, 分别为 1x1, 3x3, 5x5, 7x7, 9x9, 11x11 ...
    """
    def __init__(self, dim, n_kernels=6, init_weight=True):
        super(Inception_Conv_Cat_Layer, self).__init__()
        self.conv_layers = nn.ModuleList()
        for i in range(n_kernels):
            self.conv_layers.append(nn.Conv2d(dim, dim, kernel_size=2 * i + 1, padding=i))  # 1, 3, 5
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_map: torch.Tensor):
        output = torch.Tensor([]).to(input_map.device)
        for layer in self.conv_layers:
            output_map = layer(input_map)
            output = torch.cat((output, output_map), dim=1)
        return output
        # [B, F, T_Len // Seq_Len, Seq_Len]
        # (n times conv for input and concat)-> [B, F, T_Len // Seq_Len, Seq_Len, n_layers]
        # (mean for n_layers)-> [B, F, T_Len // Seq_Len, Seq_Len]
