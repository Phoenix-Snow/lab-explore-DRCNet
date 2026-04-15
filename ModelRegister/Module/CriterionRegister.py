import torch
from torch import nn


class Criterion:
    def __init__(self, name='LossFunction', criterion_function='MSE', device='cuda', training=True):
        self.__name = name
        self.__criterion_function_name = criterion_function
        if self.__criterion_function_name == 'MSE':
            self.__criterion = MSE().to(device)
        elif self.__criterion_function_name == 'MAE':
            self.__criterion = MAE().to(device)
        else:
            raise ValueError(f"Invalid criterion function name: {self.__criterion_function_name}")
        self.__training = training

    @property
    def name(self):
        return self.__name

    @property
    def criterion_function(self):
        return self.__criterion_function_name

    @property
    def description(self):
        return f"{self.__name} -> Loss Function: {self.__criterion_function_name} ! ! !"

    def __str__(self):
        return self.description

    def calculate(self, model_output, label):
        return self.__criterion.forward(model_output, label)

    def __call__(self, *input, **kwargs):
        result = self.calculate(*input, **kwargs)
        if self.__training:
            # 注册反向传播的钩子
            result = result.requires_grad_()
        return result


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, model_output, label):
        return torch.mean((model_output - label) ** 2)


class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()

    def forward(self, model_output, label):
        return torch.mean(torch.abs(model_output - label))



