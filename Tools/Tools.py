import msvcrt
import os
import re
import sys

import cpca
import json
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
# import tensorflow as tf
from thop import profile
from thop import clever_format


def convert_to_float(value) -> float:
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


def print_net_params_list(net):
    # 收集全部参数
    plist = [(n, p) for n, p in net.named_parameters()]

    # 计算最大宽度
    max_name = max(len(n) for n, _ in plist) if plist else 6
    head_name = 'net_param'
    max_name = max(max_name, len(head_name))

    # 表头
    header = f'{head_name:<{max_name}}   requires_grad   shape              mem(bytes)'
    print(header)
    print('-' * (len(header) + 12))

    # 数据行
    total = 0.
    for name, p in plist:
        # print(f'{name:<{max_name}}   {str(p.requires_grad):<12}    {list(p.shape)}')
        param_num = p.numel()
        bytes_num = param_num * (4 if p.dtype == torch.float32 else 2)  # fp16=2B
        bn = bytes_num
        total += bn
        print(f'{name:<{max_name}}   {str(p.requires_grad):<11}     {str(list(p.shape)):<16}   {bn:6.2f}')

    total = total / 1024 / 1024
    print('-' * (len(header) + 12))
    print(f'total param mem ≈ {total:.2f} MB  (fp32)')


def count_flops_params(model, input_shapes, device='cpu'):
    """
    计算 PyTorch 模型的一次前向 FLOPs
    :param model: nn.Module
    :param input_shapes: tuple | list, 例如 ((1,3,224,224),)
    :param device: cpu | cuda
    :return: total_flops (单位 GFlops), params
    """
    inputs = []
    for input_shape in input_shapes:
        input_tensor = torch.randn(input_shape).to(device)
        inputs.append(input_tensor)
    flops, params = profile(model.to(device), inputs=inputs)
    flops, params = clever_format([flops, params], "%.3f")
    print(f'total FLOPs: {flops}; params: {params};')
    return flops, params


def safe_name(name: str) -> str:
    # 1. 先把 Windows/Linux 都不允许的字符列出来
    #    <>:"/\|?* 以及 0x00-0x1f 控制字符
    unsafe = r'[<>:/\\|?*\x00-\x1f]'
    # 2. 连续多个替换成单个下划线；再去掉空格
    name = re.sub(unsafe, '_', name)  # 特殊字符 改 _
    name = name.replace('-', '_')  # - 改 _
    name = re.sub(r'_+', '_', name)  # 多个_ 改 单个_
    name = name.replace(' ', '')  # 去掉空格
    return name.strip('_')  # 首尾不要多余下划线


def ged_func(res_1d):
    # ---------- 1. 负对数似然 ----------
    def ged_nll(params, data):
        beta, loc, scale = params
        if scale <= 0 or beta <= 0:
            return 1e8
        return -stats.gennorm.logpdf(data, beta, loc, scale).sum()

    # ---------- 2. 拟合 GED ----------
    initial = [1.0, np.median(res_1d), np.std(res_1d)]
    bounds = [(1e-3, None), (None, None), (1e-6, None)]
    res = minimize(ged_nll, initial, args=(res_1d,), method='SLSQP', bounds=bounds)
    beta_ged, loc_ged, scale_ged = res.x

    # ---------- 3. 冻结分布 ----------
    ged_dist = stats.gennorm(beta_ged, loc=loc_ged, scale=scale_ged)

    return ged_dist


def calculate_percentiles(errors):
    """
    从CSV文件的最后一列读取数据，计算绝对误差的P50和P90

    参数:
        errors: 误差列表

    返回:
        dict: 包含P50和P90的字典
    """

    # 将列表转换为numpy数组以便计算
    errors_array = np.array(errors)

    # 计算P50和P90
    p50 = np.percentile(errors_array, 50)  # 中位数
    p90 = np.percentile(errors_array, 90)  # 第90百分位数

    return {
        'P50': p50,
        'P90': p90,
        'count': len(errors)
    }


def extract_coordinates(file_path, lat_col='LATITUDE', lon_col='LONGITUDE', ):
    if file_path.endswith(".csv"):
        df_head = pd.read_csv(file_path, nrows=1)
    elif file_path.endswith(".xlsx", ".xls"):
        df_head = pd.read_excel(file_path, nrows=1)
    else:
        raise Exception(
            '[❌ ERROR: Integration/DRCNet/Merge.py/func-extract_coordinates] file_path need to end with .csv, .xlsx or .xls')
    if lat_col not in df_head.columns or lon_col not in df_head.columns:
        raise Exception(
            '[❌ ERROR: Integration/DRCNet/Merge.py/func-extract_coordinates] lat_col or lon_col is error. (actual column name in origin tabel)')
    return df_head[lon_col].iloc[0], df_head[lat_col].iloc[0]


def extract_coordinates_plus(file_path, lat_col='LATITUDE', lon_col='LONGITUDE', ele_col='ELEVATION'):
    if file_path.endswith(".csv"):
        df_head = pd.read_csv(file_path, nrows=1)
    elif file_path.endswith(".xlsx", ".xls"):
        df_head = pd.read_excel(file_path, nrows=1)
    else:
        raise Exception(
            '[❌ ERROR: Integration/DRCNet/Merge.py/func-extract_coordinates] file_path need to end with .csv, .xlsx or .xls')
    if lat_col not in df_head.columns:
        raise Exception(f'[❌ ERROR] lat_col "{lat_col}" not found in columns: {df_head.columns.tolist()}')
    if lon_col not in df_head.columns:
        raise Exception(f'[❌ ERROR] lon_col "{lon_col}" not found in columns: {df_head.columns.tolist()}')
    if ele_col not in df_head.columns:
        raise Exception(f'[❌ ERROR] ele_col "{ele_col}" not found in columns: {df_head.columns.tolist()}')
    return df_head[lon_col].iloc[0], df_head[lat_col].iloc[0], df_head[ele_col].iloc[0]


def normal_province(address):
    df_func = cpca.transform([address])
    return df_func['省'][0]


def normal_city(address):
    df_func = cpca.transform([address])
    return df_func['市'][0]


class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，用于处理NumPy数据类型"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)  # 将 numpy.int64 等转换为 Python int
        elif isinstance(obj, np.floating):
            return float(obj)  # 将 numpy.float32 等转换为 Python float
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # 将 numpy 数组转换为 Python 列表
        else:
            return super(NumpyEncoder, self).default(obj)


def draw_menu(options, selected_index):
    print("请使用上下键选择，回车确认：")
    for i, option in enumerate(options):
        # 简单的样式：选中项前面加星号
        prefix = "★ " if i == selected_index else "  "
        # 打印整行，注意行尾加空格防止旧字符残留
        print(f"{prefix}{option}          ")


def clear_line():
    """清除当前光标所在行的剩余内容"""
    sys.stdout.write("\033[K")


def move_cursor_up(lines):
    """光标上移指定行数"""
    sys.stdout.write(f"\033[{lines}A")


def get_key():
    """获取按键，返回方向键的代号或字符"""
    if os.name == 'nt':  # Windows
        key = msvcrt.getch()
        # 方向键在Windows下通常是两个字符，第一个是 b'\xe0' 或 b'\x00'
        if key == b'\xe0' or key == b'\x00':
            key2 = msvcrt.getch()  # 读取第二个字符来判断具体方向
            if key2 == b'H': return 'UP'
            if key2 == b'P': return 'DOWN'
            if key2 == b'K': return 'LEFT'
            if key2 == b'M': return 'RIGHT'
            return 'OTHER'
        else:
            return key.decode('utf-8', errors='ignore')
    else:
        # 简单的非Windows兼容（Linux/Mac下通常读取 \033[A 等序列）
        # 这里为了演示简洁，主要适配Windows，Linux需配合 curses 或 readchar
        return ''


def jsonDumps(data: Dict[str, Any], indent: int = 2, current_level: int = 0) -> str:
    """
    格式化JSON，使字典的键值对换行缩进，但列表和字典值保持在一行内。

    Args:
        data: 要格式化的字典数据
        indent: 缩进空格数，默认为2
        current_level:

    Returns:
        格式化后的JSON字符串
    """
    # def _format_value(value: Any, current_indent: int) -> str:
    #     if isinstance(value, dict):
    #         # 对于嵌套的字典，递归处理，但保持其内容在一行
    #         return json.dumps(value, ensure_ascii=False, cls=NumpyEncoder)
    #     elif isinstance(value, list):
    #         # 对于列表，保持在一行
    #         return json.dumps(value, ensure_ascii=False, cls=NumpyEncoder)
    #     else:
    #         # 对于其他类型（字符串、数字等），直接返回JSON表示
    #         return json.dumps(value, ensure_ascii=False, cls=NumpyEncoder)
    #
    # lines = []
    # lines.append('{')
    #
    # items = list(data.items())
    # for i, (key, value) in enumerate(items):
    #     # 格式化值
    #     formatted_value = _format_value(value, indent)
    #
    #     # 添加逗号（最后一个元素不加）
    #     comma = ',' if i < len(items) - 1 else ''
    #
    #     # 构建行：缩进 + "key": value + 逗号
    #     line = ' ' * indent + f'"{key}": {formatted_value}{comma}'
    #     lines.append(line)
    #
    # lines.append('}')
    #
    # return '\n'.join(lines)
    # 1. 计算当前应该缩进多少空格
    # 第0层缩进0，第1层缩进2，第2层缩进4...
    current_indent_str = ' ' * (current_level * indent)
    next_indent_str = ' ' * ((current_level + 1) * indent)

    # 2. 如果是字典 -> 递归处理
    if isinstance(data, dict):
        if not data:
            return '{}'

        lines = ['{']
        items = list(data.items())

        for i, (key, value) in enumerate(items):
            # 【核心逻辑】：遇到 value 是字典，就递归调用自己！
            # 这里的 level 自动 +1，缩进自动加深
            formatted_value = jsonDumps(value, indent, current_level + 1)

            comma = ',' if i < len(items) - 1 else ''
            lines.append(f'{next_indent_str}"{key}": {formatted_value}{comma}')

        lines.append(f'{current_indent_str}}}')
        return '\n'.join(lines)

    # 3. 如果是列表
    elif isinstance(data, list):
        if not data:
            return '[]'
        formatted_value_list = []
        str_cache = ''
        for item in data:
            if isinstance(item, dict) or isinstance(item, list):
                if str_cache != '':
                    formatted_value_list.append(str_cache)
                    str_cache = ''  # 放入一行数据，重置缓存
                formatted_value_list.append(jsonDumps(item, indent, current_level + 1) + ', ')  # 放入当前行
            else:
                if isinstance(item, str):
                    str_cache = str_cache + '\"' + str(item) + '\"' + ', '  # 当前行追加
                else:
                    str_cache = str_cache + str(item) + ', '
        if str_cache != '':
            formatted_value_list.append(str_cache)
        formatted_value_list[-1] = formatted_value_list[-1][:-2]  # 去掉末尾逗号
        return '[' + '\n'.join(formatted_value_list) + ']'

    # 4. 如果是基础类型（字符串、数字等） -> 直接返回
    else:
        return json.dumps(data, ensure_ascii=False, cls=NumpyEncoder)
