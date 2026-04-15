import torch
from torch import nn


class MatrixFusion(nn.Module):
    def __init__(self, return_state='X1PY2'):
        """
        默认情况X1PY2下融合时, X的每行和Y的每行融合, X在左, Y在右, 下面以X为时序表为例
        :input_map: B * T * F（Batch * TimeSeriesLength * FeatureNum）, 保持TimeSeries不变, 选择X1【目的是使得训练和预测时可以使用不同的时序长度】
        :y: 该例子是 100 * 100 的地理特征图, 若是选择保留纬度信息, 就需要选择Y2
        :param return_state: 1是行2是列, 例如PXYB-X1PY2-B是X的每行和Y的每行融合。P_B: 是否对目标进行预处理, B: 是否对融合后的结果进行偏置
        :return: 融合后得到的Z, 默认情况下应该是B * T * 100
        """
        super(MatrixFusion, self).__init__()
        self.z = None
        self.bias = None
        self.param = None
        self.bias_y = None
        self.param_y = None
        self.bias_x = None
        self.param_x = None
        self.x = None
        self.y = None
        # 如果不以P(X/Y)B,PXY,X(1/2)P开头, 报错
        if return_state[0:3] not in ['PXY', 'PXB', 'PYB', 'X1P', 'X2P']:
            raise ValueError("the main part of return_state must be like 'X1PY1', and the possible added part is 'P_B' and 'B', like 'PXYB-X1PY2-B'")
        self.return_state = return_state

    def forward(self, x, y):
        if len(x.shape) != 3 and len(y.shape) != 3:
            raise ValueError("input_map and y must be 3D Matrices")
        if x.device!= y.device:
            raise ValueError("input_map and y must be on the same device")
        device = x.device
        # 处理x和y
        if 'X1' in self.return_state:
            self.x = x
        elif 'X2' in self.return_state:
            self.x = x.transpose(1, 2)
        if 'Y1' in self.return_state:
            self.y = y
        elif 'Y2' in self.return_state:
            self.y = y.transpose(1, 2)
        # return_state开头三个字符, 可能：P(X/Y)B,PXY,X(1/2)P,2和3为X或Y的仅前面两种
        if self.return_state[1] == 'X':
            self.param_x = nn.Parameter(torch.randn(1, self.x.shape[2])).reshape(1, 1, self.x.shape[2]).to(device)
            self.bias_x = nn.Parameter(torch.randn(1, self.x.shape[2])).reshape(1, 1, self.x.shape[2]).to(device)
            self.x = self.x * self.param_x + self.bias_x
        if self.return_state[1] == 'Y' or self.return_state[2] == 'Y':
            self.param_y = nn.Parameter(torch.randn(self.y.shape[1], self.y.shape[2])).reshape(1, self.y.shape[1], self.y.shape[2]).to(device)
            self.bias_y = nn.Parameter(torch.randn(self.y.shape[1], self.y.shape[2])).reshape(1, self.y.shape[1], self.y.shape[2]).to(device)
            self.y = self.y * self.param_y + self.bias_y

        # 融合
        self.param = nn.Parameter(torch.randn(self.x.shape[2], self.y.shape[1])).reshape(1, self.x.shape[2], self.y.shape[1]).to(device)
        if self.return_state.endswith('B'):
            self.bias = nn.Parameter(torch.randn(self.x.shape[1], self.y.shape[2])).reshape(1, self.x.shape[1], self.y.shape[2]).to(device)
            self.z = self.x @ self.param @ self.y + self.bias
        else:
            self.z = self.x @ self.param @ self.y
        # print(self.z.shape)
        return self.z


