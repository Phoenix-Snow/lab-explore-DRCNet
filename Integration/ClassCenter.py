from typing import Optional

import numpy as np
# import tensorflow as tf
import torch



def convert_to_float(value):
    """
    将特定类型转换为 float 类型
    支持的类型：list、numpy.ndarray、torch.Tensor、tf.Tensor
    """
    if isinstance(value, float):
        return value

    # 如果是 list 且长度为 1, 则转换为 float
    elif isinstance(value, list) and len(value) == 1:
        return float(value[0])

    # 如果是 numpy 数组且元素个数为 1, 则转换为 float
    elif isinstance(value, np.ndarray) and value.size == 1:
        return float(value.item())

    # 如果是 PyTorch 张量且元素个数为 1, 则转换为 float
    elif isinstance(value, torch.Tensor) and value.numel() == 1:
        return float(value.item())

    # 如果是 TensorFlow 张量且元素个数为 1, 则转换为 float
    # elif isinstance(value, tf.Tensor) and tf.size(value).numpy() == 1:
    #     return float(value.numpy().item())

    # 如果类型不支持或不是单值, 则抛出异常
    else:
        raise ValueError("不支持的类型或值不是单个数值")


class TS_Output:
    def __init__(self):
        # 输出
        self.__temporal_output = None
        self.__spatial_output = None
        self.__output = None
        self.__avg_output = None
        # 损失
        self.__temporal_loss: float
        self.__spatial_loss: float
        self.__loss: float
        self.__avg_loss: float
        # 最小损失记录
        self.__temporal_min_loss: float = 10000000.
        self.__spatial_min_loss: float = 10000000.
        self.__min_loss: float = 10000000.
        self.__avg_min_loss: float = 10000000.

    @property
    def temporal_output(self):
        return self.__temporal_output

    @temporal_output.setter
    def temporal_output(self, value):
        self.__temporal_output = value

    @property
    def spatial_output(self):
        return self.__spatial_output

    @spatial_output.setter
    def spatial_output(self, value):
        self.__spatial_output = value

    @property
    def output(self):
        return self.__output

    @output.setter
    def output(self, value):
        self.__output = value

    @property
    def avg_output(self):
        return self.__avg_output

    @avg_output.setter
    def avg_output(self, value):
        self.__avg_output = value

    @property
    def temporal_loss(self):
        return self.__temporal_loss

    @temporal_loss.setter
    def temporal_loss(self, value):
        self.__temporal_loss = convert_to_float(value)

    @property
    def spatial_loss(self):
        return self.__spatial_loss

    @spatial_loss.setter
    def spatial_loss(self, value):
        self.__spatial_loss = convert_to_float(value)

    @property
    def loss(self):
        return self.__loss

    @loss.setter
    def loss(self, value):
        self.__loss = convert_to_float(value)

    @property
    def avg_loss(self):
        return self.__avg_loss

    @avg_loss.setter
    def avg_loss(self, value):
        self.__avg_loss = convert_to_float(value)

    @property
    def temporal_min_loss(self):
        return self.__temporal_min_loss

    @temporal_min_loss.setter
    def temporal_min_loss(self, value):
        self.__temporal_min_loss = convert_to_float(value)

    @property
    def spatial_min_loss(self):
        return self.__spatial_min_loss

    @spatial_min_loss.setter
    def spatial_min_loss(self, value):
        self.__spatial_min_loss = convert_to_float(value)

    @property
    def min_loss(self):
        return self.__min_loss

    @min_loss.setter
    def min_loss(self, value):
        self.__min_loss = convert_to_float(value)

    @property
    def avg_min_loss(self):
        return self.__avg_min_loss

    @avg_min_loss.setter
    def avg_min_loss(self, value):
        self.__avg_min_loss = convert_to_float(value)


class TS_Output_Pro:
    def __init__(self):
        self.temporal_output = TrainVal_Output()
        self.spatial_output = TrainVal_Output()
        self.output = TrainVal_Output()
        self.avg_output = TrainVal_Output()


class TrainVal_Output:
    def __init__(self):
        # 输出
        self.__pred = None
        # 损失
        self.__train_loss: Optional[float] = None
        self.__val_loss: Optional[float] = None
        self.__loss: Optional[float] = None
        # 最小损失记录
        self.__min_loss: float = 10000000.
        self.__train_loss_whenMIN: float = 10000000.
        self.__val_loss_whenMIN: float = 10000000.

    @property
    def pred(self):
        return self.__pred

    @pred.setter
    def pred(self, value):
        self.__pred = value

    @property
    def train_loss(self):
        return self.__train_loss

    @train_loss.setter
    def train_loss(self, value):
        self.__train_loss = convert_to_float(value)

    @property
    def val_loss(self):
        return self.__val_loss

    @val_loss.setter
    def val_loss(self, value):
        self.__val_loss = convert_to_float(value)

    @property
    def loss(self):
        return self.__loss

    @loss.setter
    def loss(self, value):
        self.__loss = convert_to_float(value)

    @property
    def min_loss(self):
        return self.__min_loss

    @property
    def train_loss_whenMIN(self):
        return self.__train_loss_whenMIN

    @property
    def val_loss_whenMIN(self):
        return self.__val_loss_whenMIN

    def update_min_loss(self):
        self.__min_loss = self.loss
        self.__train_loss_whenMIN = self.train_loss
        self.__val_loss_whenMIN = self.val_loss

    @min_loss.setter
    def min_loss(self, value):
        self.__min_loss = convert_to_float(value)


class Configs:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
