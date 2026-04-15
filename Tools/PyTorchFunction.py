import math
from typing import Tuple, Optional, Union, List, Dict, Iterator

import torch
from torch import nn
from torch.nn import Parameter


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class AmpOptimizer:
    def __init__(self,  # 是否使用混合精度训练, 1 表示使用, 其他值可能表示不使用（具体取决于实现）
                 mixed_precision: int,
                 # 优化器对象, 用于更新模型参数
                 optimizer: torch.optim.Optimizer,
                 # 参数列表
                 op_name_params: Union[Dict[str, torch.nn.Parameter], Iterator[Parameter]],
                 # 梯度裁剪阈值, 用于防止梯度爆炸
                 grad_clip: float,
                 # 梯度累积步数, 默认为1（即每一步更新一次参数）
                 gradient_num: int = 1, ):
        # 是否启用自动混合精度训练
        is_enable_amp_op: bool = (mixed_precision > 0)
        # 是否使用FP16而不是BF16
        is_f16_rather_bf16: bool = (mixed_precision == 1)

        if is_enable_amp_op:
            # 创建自动混合精度上下文, 根据设置选择FP16或BF16
            self.__amp_ctx = torch.autocast('cuda', enabled=True,
                                            dtype=torch.float16 if is_f16_rather_bf16 else torch.bfloat16,
                                            cache_enabled=True)
            # 如果使用FP16, 创建梯度缩放器(用于防止梯度下溢), BF16不需要
            self.__scaler = torch.cuda.amp.GradScaler(init_scale=2. ** 11, growth_interval=1000) \
                if is_f16_rather_bf16 else None
        else:
            # 不启用AMP时使用空上下文
            self.__amp_ctx = NullCtx()
            self.__scaler = None

        # 保存优化器、参数名和参数列表(已过滤确保所有参数都需要梯度)
        self.__optimizer, self.__op_name_param = optimizer, op_name_params
        self.__grad_clip: float = grad_clip  # 梯度裁剪阈值
        # 如果有梯度裁剪阈值, 证明需要梯度裁剪
        if self.__grad_clip > 0:
            # 有global_grad_norm参数需要更新, 就梯度更新后裁剪；若没有, 则提前就进行梯度裁剪
            if hasattr(optimizer, 'global_grad_norm'):
                self.__early_clipping: bool = False
                self.__late_clipping: bool = True
            else:
                self.__early_clipping: bool = True
                self.__late_clipping: bool = False

        # 计算每个梯度累积步数的倒数
        self.__gradient_num_reciprocal: float = 1. / gradient_num  # 1.0 / n_gradient_accumulation

    def backward(self, stepping: bool, loss: torch.Tensor, ) \
            -> Tuple[Optional[Union[torch.Tensor, float]], Optional[float]]:
        """
        反向传播函数, 有scaler缩放器时, 就使用amp反向传播
        self.scaler.scale(loss).backward(retain_graph=False, create_graph=False);
        否则就正常反向传播
        loss.backward(retain_graph=False, create_graph=False);
        :param stepping: 是否执行优化器更新步骤的标志
        :param loss: 计算得到的损失值张量
        :return: 返回原始梯度范数和缩放器缩放值(可能为None)
        """
        # 根据梯度累积步数缩放损失值
        loss: torch.Tensor = loss.mul(self.__gradient_num_reciprocal)  # 1.0 / n_gradient_accumulation

        # 初始化cache
        orig_norm: torch.Tensor = torch.Tensor()  # 初始化 原始梯度范数
        scaler_sc: float = 9999.9999  # 初始化 缩放器缩放值

        # 计算梯度
        if self.__scaler is not None:
            # 使用梯度缩放器进行反向传播(用于FP16训练)
            self.__scaler.scale(loss).backward(retain_graph=False, create_graph=False)
        else:
            # 普通反向传播(无梯度缩放阈值或BF16训练)
            loss.backward(retain_graph=False, create_graph=False)

        # 检查是否有需要梯度但梯度为None的参数
        if self.__op_name_param is not None:
            for name, param in self.__op_name_param.items():
                if param.grad is None and param.requires_grad:
                    print(name)  # 打印参数名

        # 参数更新
        if stepping:
            # 如果需要进行优化器更新步骤
            if self.__scaler is not None:
                # 如果使用FP16, 需要先对优化器进行unscale操作(将梯度从缩放空间还原)
                self.__scaler.unscale_(self.__optimizer)

            # 参数 更新前 进行梯度裁剪
            if self.__early_clipping:
                if type(self.__op_name_param) == Iterator[Parameter]:
                    orig_norm: torch.Tensor = torch.nn.utils.clip_grad_norm_(self.__op_name_param,
                                                                             self.__grad_clip)
                else:
                    orig_norm: torch.Tensor = torch.nn.utils.clip_grad_norm_(list(self.__op_name_param.values()),
                                                                             self.__grad_clip)

            if self.__scaler is not None:
                # FP16训练时: 缩放空间进行更新
                self.__scaler.step(self.__optimizer)
                scaler_sc: float = self.__scaler.get_scale()  # 获取当前缩放因子
                # 防止缩放因子过大导致FP16溢出(>65536会溢出, 这里设置32768作为安全阈值)
                if scaler_sc > 32768.:
                    self.__scaler.update(new_scale=32768.)
                else:
                    self.__scaler.update()  # 正常更新缩放因子
                try:
                    # log2缩放因子
                    scaler_sc = float(math.log2(scaler_sc))
                except Exception as e:
                    # 如果转换失败, 打印错误信息并重新抛出异常
                    print(f'[scaler_sc = {scaler_sc}]\n' * 15, flush=True)
                    raise e
            else:
                # 正常进行更新
                self.__optimizer.step()

            # 参数 更新后 进行梯度裁剪
            if self.__late_clipping:
                orig_norm: torch.Tensor = self.__optimizer.global_grad_norm  # 获取全局梯度范数
                # # 计算缩放因子
                # clip_grad_norm: float = self.__grad_clip
                # scale: float = min(1.0, clip_grad_norm / (orig_norm + 1e-6))
                # # 梯度裁剪
                # for name, param in self.__op_name_param.items():
                #     param.grad.data.mul_(scale)

            # 清空梯度 节省内存
            self.__optimizer.zero_grad(set_to_none=True)

        return orig_norm, scaler_sc

    def state_dict(self):
        # 返回优化器的状态字典
        return {
            'scaler': self.__scaler.state_dict() if self.__scaler is not None else None,
            'optimizer': self.__optimizer.state_dict()
        }

    def load_state_dict(self, state, strict=True):
        # 加载优化器状态
        if self.__scaler is not None:
            try:
                # 如果使用FP16训练, 先加载缩放器状态
                self.__scaler.load_state_dict(state['scaler'])
            except Exception as e:
                # 如果加载失败, 打印错误信息
                print(f'[fp16 load_state_dict err] {e}')
        # 加载优化器状态
        self.__optimizer.load_state_dict(state['optimizer'])

    def zero_grad(self):
        self.__optimizer.zero_grad(set_to_none=True)

    @property
    def AMP_CTX(self):
        return self.__amp_ctx


class NullCtx:
    # 空上下文管理器, 用于在不启用AMP时替代autocast上下文
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
