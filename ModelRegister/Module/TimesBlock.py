import torch
import torch.nn as nn
import torch.nn.functional as F

from ModelRegister.Module.Layer.InceptionLayer import Inception_Conv_Layer, Inception_Conv_Cat_Layer


class TimesNet(nn.Module):
    def __init__(self, seq_len=28, feature_num=14, hidden_size=32, FFT_out_num=5, kernels_num=3):
        super(TimesNet, self).__init__()
        self._middle_size = hidden_size
        self._T = seq_len
        self._f = feature_num
        # self._t = cycle_len
        self._n = FFT_out_num
        self._kernels_num = kernels_num
        self.__Inception_pro = Inception_Conv_Layer(self._f, self._middle_size, n_kernels=self._kernels_num)
        self.__Inception_last = Inception_Conv_Layer(self._middle_size, self._f, n_kernels=self._kernels_num)

    def TimesBlock(self, x):
        # reshape：处理输入数据中如果batch为1的情况
        x = x.reshape(-1, self._T, self._f)
        batch = x.shape[0]

        x_clone = x.clone()

        # 傅里叶变换 B T f
        xf = torch.fft.rfft(x, dim=1)
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, self._n + 2)  # 选取频率最快的前n个波段
        top_list = top_list.detach().cpu().numpy()  # 处理成numpy格式, 方便矩阵运算

        # 获取最终输入波谱
        period_list = self._T // top_list  # 周期列表
        period_weight = abs(xf).mean(-1)[:, top_list]  # 周期权重（振幅）

        # 对self._n个波段挨个进行卷积conv2d
        temp = []
        pass_time = 0
        for i in range(self._n + 2):
            # 拿到第i+1个周期的波
            period = period_list[i]
            if period == 1 or period == self._T or pass_time + i + 1 == self._n:
                pass_time += 1
                # 删掉该周期的波段
                if i == 0:
                    period_weight = period_weight[..., 1:]
                elif i == self._n + 2:
                    period_weight = period_weight[..., :-1]
                else:
                    period_weight = torch.cat((period_weight[..., :i], period_weight[..., i + 1:]), dim=-1)
                continue
            # 如果不够整除, 则补0
            if self._T % period != 0:
                length = ((self._T // period) + 1) * period
                padding = torch.zeros([batch, (length - self._T), self._f]).to(x.device)
                out = torch.cat([padding, x_clone], dim=1)
            else:
                length = self._T
                out = x_clone
            # 对第i+1个周期的波进行分段分离
            out = out.reshape(batch, length // period, period, self._f)

            out = out.permute(0, 3, 1, 2).contiguous()
            # 对分离后的波段2D conv
            out = self.__Inception_pro(out)
            out = nn.GELU()(out)
            out = self.__Inception_last(out)
            # out = Inception_Conv_Layer(self._f, self._middle_size, n_kernels=self._kernels_num)(out)
            # out = nn.GELU()(out)
            # out = Inception_Conv_Layer(self._middle_size, self._f, n_kernels=self._kernels_num)(out)
            out = out
            # 对完成特征提取的波段进行拼接
            out = out.permute(0, 2, 3, 1)

            out = out.reshape(batch, -1, self._f)

            temp.append(out[:, :self._T, :])  # 切除多余的数据

        out = torch.stack(temp, dim=-1)

        # 根据权重（振幅）合并全部周期
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, self._T, self._f, 1)
        out = torch.sum(out * period_weight, -1)

        return out + x

    def forward(self, x, block_times=3):
        device = x.device
        for i in range(block_times):
            x = self.TimesBlock(x)
        return x.to(device)


class T(nn.Module):
    def __init__(self, seq_len=28, feature_num=14, hidden_size=32, FFT_out_num=5, kernels_num=3):
        super(T, self).__init__()
        self._middle_size = hidden_size
        self._T = seq_len
        self._f = feature_num
        # self._t = cycle_len
        self._n = FFT_out_num
        self._kernels_num = kernels_num
        self.__Inception_pro = Inception_Conv_Cat_Layer(self._f, n_kernels=self._kernels_num)
        self.__Inception_last = Inception_Conv_Cat_Layer(self._f * self._kernels_num, n_kernels=self._kernels_num)


    def TimesBlock(self, x):
        # reshape：处理输入数据中如果batch为1的情况
        x = x.reshape(-1, self._T, self._f)
        device = x.device
        batch = x.shape[0]

        x_clone = x.clone()

        # 傅里叶变换 B T f
        xf = torch.fft.rfft(x, dim=1)
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, self._n + 2)  # 选取频率最快的前n个波段
        top_list = top_list.detach().cpu().numpy()  # 处理成numpy格式, 方便矩阵运算

        # 获取最终输入波谱
        period_list = self._T // top_list  # 周期列表
        period_weight = abs(xf).mean(-1)[:, top_list]  # 周期权重（振幅）

        # 对self._n个波段挨个进行卷积conv2d
        temp = []
        pass_time = 0
        for i in range(self._n + 2):
            # 拿到第i+1个周期的波
            period = period_list[i]
            if period == 1 or period == self._T or pass_time + i + 1 == self._n:
                pass_time += 1
                # 删掉该周期的波段
                if i == 0:
                    period_weight = period_weight[..., 1:]
                elif i == self._n + 1:  # 最后一个索引应该是self._n + 1，不是self._n + 2
                    period_weight = period_weight[..., :-1]
                else:
                    period_weight = torch.cat((period_weight[..., :i], period_weight[..., i + 1:]), dim=-1)
                continue
            # 如果不够整除, 则补0
            if self._T % period != 0:
                length = ((self._T // period) + 1) * period
                padding = torch.zeros([batch, (length - self._T), self._f]).to(device)
                out = torch.cat([padding, x_clone], dim=1)
            else:
                length = self._T
                out = x_clone
            # 对第i+1个周期的波进行分段分离
            out = out.reshape(batch, length // period, period, self._f)

            out = out.permute(0, 3, 1, 2).contiguous()
            # 对分离后的波段2D conv
            out = self.__Inception_pro(out)
            out = nn.GELU()(out)
            out = self.__Inception_last(out)
            out = out.to(device)
            # 对完成特征提取的波段进行拼接
            out = out.permute(0, 2, 3, 1)

            out = out.reshape(batch, -1, self._f * 3 * 3)

            temp.append(out[:, :self._T, :])  # 切除多余的数据

        out = torch.stack(temp, dim=-1)

        # 根据权重（振幅）合并全部周期
        period_weight = F.softmax(period_weight.to(device), dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, self._T, self._f * 3 * 3, 1)
        try:
            out = torch.sum(out * period_weight, -1)
        except Exception as e:
            print(out.shape, period_weight.shape)
            raise e

        return out

    def forward(self, x, block_times=3):
        for i in range(block_times):
            x = self.TimesBlock(x)
        return x
