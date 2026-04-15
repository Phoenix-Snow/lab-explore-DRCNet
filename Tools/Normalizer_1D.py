import torch
from torch import nn


class Normalizer:
    def __init__(self, normal_params: dict[str, list[float]], normalize_type: str = 'MinMaxNormalize'):
        if normalize_type == 'StdMeanNormalize':
            self.normalization_input = StdMeanNormalize(normal_params['mean_values_input'],
                                                        normal_params['std_values_input'])
            self.normalization_label = StdMeanNormalize(normal_params['mean_values_output'],
                                                        normal_params['std_values_output'])
            self.antiNormalization = StdMeanAntiNormalize(normal_params['mean_values_output'],
                                                          normal_params['std_values_output'])
        elif normalize_type == 'MinMaxNormalize':
            self.normalization_input = MinMaxNormalize(normal_params['min_values_input'],
                                                       normal_params['max_values_input'])
            self.normalization_label = MinMaxNormalize(normal_params['min_values_output'],
                                                       normal_params['max_values_output'])
            self.antiNormalization = MinMaxAntiNormalize(normal_params['min_values_output'],
                                                         normal_params['max_values_output'])
        else:
            raise ValueError(f'[ERROR] Normalization Type Error: normalize_type - StdMeanNormalize / MinMaxNormalize '
                             f'(actual error input: {normalize_type})')


class StdMeanAntiNormalize(nn.Module):
    def __init__(self, mean_params, std_params):
        super(StdMeanAntiNormalize, self).__init__()
        self.__mean = torch.tensor(mean_params, dtype=torch.float32)
        self.__std = torch.tensor(std_params, dtype=torch.float32)
        self.__len = len(self.__mean)
        self.device = "cpu"

    def location(self, location_index_list: list):
        self.__mean = self.__mean[location_index_list]
        self.__std = self.__std[location_index_list]

    def cuda_device(self, device: str):
        self.__mean = self.__mean.to(device)
        self.__std = self.__std.to(device)
        self.device = device

    @property
    def normalize_params(self):
        return {'mean': self.__mean, 'std': self.__std}

    @property
    def len(self):
        return self.__len

    def work(self, data):
        if not isinstance(data, torch.Tensor):
            if isinstance(data, float):
                data = torch.tensor([data], dtype=torch.float32)
            else:
                data = torch.tensor(data, dtype=torch.float32)
        data = data.to(self.device)
        # print(data.shape, self.__mean.shape, self.__std.shape)
        return (data * self.__std) + self.__mean


class StdMeanNormalize(nn.Module):
    def __init__(self, mean_params, std_params):
        super(StdMeanNormalize, self).__init__()
        self.__mean = torch.tensor(mean_params, dtype=torch.float32)
        self.__std = torch.tensor(std_params, dtype=torch.float32)
        self.__len = len(self.__mean)
        self.device = "cpu"

    def cuda_device(self, device: str):
        self.__mean = self.__mean.to(device)
        self.__std = self.__std.to(device)
        self.device = device

    @property
    def normalize_params(self):
        return {'mean': self.__mean, 'std': self.__std}

    @property
    def len(self):
        return self.__len

    def work(self, data):
        if not isinstance(data, torch.Tensor):
            if isinstance(data, float):
                data = torch.tensor([data], dtype=torch.float32)
            else:
                data = torch.tensor(data, dtype=torch.float32)
        data = data.to(self.device)
        return (data - self.__mean) / self.__std


class MinMaxAntiNormalize(nn.Module):
    def __init__(self, min_params, max_params):
        super(MinMaxAntiNormalize, self).__init__()
        self.__max = torch.tensor(max_params, dtype=torch.float32)
        self.__min = torch.tensor(min_params, dtype=torch.float32)
        self.__len = len(self.__max)
        self.device = 'cpu'

    def location(self, location_index_list: list):
        self.__max = self.__max[location_index_list]
        self.__min = self.__min[location_index_list]

    def cuda_device(self, device: str):
        self.__max = self.__max.to(device)
        self.__min = self.__min.to(device)
        self.device = device

    @property
    def normalize_params(self):
        return {'max': self.__max, 'min': self.__min}

    @property
    def len(self):
        return self.__len

    def work(self, data):
        if not isinstance(data, torch.Tensor):
            if isinstance(data, float):
                data = torch.tensor([data], dtype=torch.float32)
            else:
                data = torch.tensor(data, dtype=torch.float32)
        data = data.to(self.device)
        return (data * (self.__max - self.__min)) + self.__min


class MinMaxNormalize(nn.Module):
    def __init__(self, min_params, max_params):
        super(MinMaxNormalize, self).__init__()
        self.__max = torch.tensor(max_params, dtype=torch.float32)
        self.__min = torch.tensor(min_params, dtype=torch.float32)
        self.__len = len(self.__max)
        self.device = 'cpu'

    def cuda_device(self, device: str):
        self.__max = self.__max.to(device)
        self.__min = self.__min.to(device)
        self.device = device

    @property
    def normalize_params(self):
        return {'max': self.__max, 'min': self.__min}

    @property
    def len(self):
        return self.__len

    def work(self, data):
        if not isinstance(data, torch.Tensor):
            if isinstance(data, float):
                data = torch.tensor([data], dtype=torch.float32)
            else:
                data = torch.tensor(data, dtype=torch.float32)
        data = data.to(self.device)
        # return (data - self.__min) / (self.__max - self.__min)
        # print("Max:", self.__max)
        # print("Min:", self.__min)
        # print("Max - Min:", self.__max - self.__min)
        denom = self.__max - self.__min
        # 将零或极小值替换为1，避免除零和溢出
        denom = torch.clamp(denom, min=1e-8)
        return (data - self.__min) / denom


class NoAnti(nn.Module):
    def work(self, data):
        return data
