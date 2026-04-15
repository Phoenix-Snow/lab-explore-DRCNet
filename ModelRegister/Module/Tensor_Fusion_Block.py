#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : KittenSleepwalk
# @time    : 2025/1/18 上午11:49
# @function: 想法 - 提取x个空间特征, 与y个时序做向量带权融合为二维矩阵, 
# 【return_state='matrix'】 P1[m*1] input_map[m*1] B1[m*1] -> X[m*1], P2[1*n] y[1*n] B2[1*n] -> Y[1*n], 然后再做矩阵乘法得到最终的融合结果Z[m*n], 更新P1, P2, B1, B2
# 【return_state='number'】 提取x个空间特征, 与y个时序做二维矩阵权重融合, input_map[1*m] P1[m*n] y[n*1] = __out_linear[1*1], 更新P1
# @version : V1.0.0
import torch
from torch import nn


class TensorFusion(nn.Module):
    def __init__(self):
        super(TensorFusion, self).__init__()
        self.x = None
        self.y = None
        self.param1 = None
        self.bias1 = None
        self.param2 = None
        self.bias2 = None

    def forward(self, x, y, return_state='matrix'):
        if len(x.shape) == 3 and len(y.shape) == 3:
            self.x = x
            self.y = y.transpose(1, 2)
        elif len(x.shape) == 2 and len(y.shape) == 2:
            self.x = x.unsqueeze(0)
            self.y = y.transpose(0, 1).unsqueeze(0)
        else:
            raise ValueError("input_map and y must be 2D or 3D tensor and must be at the same time")
        if x.shape[1]!=1:
            self.x = x.reshape(x.shape[0], 1, x.shape[1]*x.shape[2])
        if y.shape[1]!=1:
            self.y = y.reshape(y.shape[0], 1, y.shape[1]*y.shape[2])
        if return_state =='matrix':
            # self.param = nn.Parameter(torch.randn(__out_linear-dim, in-dim))
            self.param1 = nn.Parameter(torch.randn(x.shape[1], 1)).unsqueeze(0)
            self.bias1 = nn.Parameter(torch.randn(x.shape[1], 1)).unsqueeze(0)
            self.param2 = nn.Parameter(torch.randn(1, y.shape[1])).unsqueeze(0)
            self.bias2 = nn.Parameter(torch.randn(1, y.shape[1])).unsqueeze(0)
            X = torch.mul(self.x, self.param1) + self.bias1
            Y = torch.mul(self.param2, self.y) + self.bias2
            return torch.matmul(X, Y)
        elif return_state == 'number':
            self.param1 = nn.Parameter(torch.randn(self.y.shape[1], self.x.shape[2]))
            return torch.matmul(self.y, torch.matmul(self.param, self.x))
        else:
            raise ValueError("return_state must be 'number' or 'matrix'")