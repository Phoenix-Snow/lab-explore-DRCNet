import json
import os
import time
from pathlib import Path

import numpy as np
import openpyxl
import csv

import pandas as pd
from scipy.io import loadmat


def excel2csv(excel_path, csv_path):
    # 使用openpyxl打开Excel文件
    wb = openpyxl.load_workbook(excel_path)
    # sheet = wb.active  # 选择活动的工作表
    #
    # # 使用open函数以UTF-8编码打开CSV文件
    # with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
    #     writer = csv.writer(file)
    #     for row in sheet.iter_rows(values_only=True):
    #         writer.writerow(row)  # 写入一行数据

    sheet_titles = None
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for sheet_name in wb.sheetnames:  # 遍历所有工作表
            sheet = wb[sheet_name]
            for i, row in enumerate(sheet.iter_rows(values_only=True)):
                if i == 0:  # 第一行是标题
                    if sheet_titles is None:
                        sheet_titles = row
                        # 初始的标题行要写入文件
                    else:
                        assert sheet_titles == row, '工作表标题不一致'  # 确保工作表标题一致
                        # 一致后跳过标题行
                        continue
                writer.writerow(row)

    print(f'Excel文件已成功转换为CSV文件：{csv_path}')


def mat2csv_with_json(mat_path, csv_dir):
    """
    把单个 .mat 文件里所有变量分别写成 csv, 
    并把元信息写进 csv_dir/info.json
    风格对齐 excel2csv 函数
    """
    mat_path = Path(mat_path)
    csv_dir  = Path(csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读 mat
    data = loadmat(mat_path, simplify_cells=True)
    data = {k: v for k, v in data.items() if not k.startswith('__')}  # 去系统字段

    meta_list = []          # 收集元信息
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    # 2. 遍历变量
    for var_name, var_value in data.items():
        csv_name = f'{mat_path.stem}__{var_name}.csv'
        csv_path = csv_dir / csv_name

        # 只处理能写成二维表的 numeric 数组
        if not isinstance(var_value, np.ndarray):
            continue
        if not np.issubdtype(var_value.dtype, np.number):
            continue

        # 高维数组拉成 2D
        if var_value.ndim == 1:
            var_value = var_value.reshape(-1, 1)
        elif var_value.ndim > 2:
            var_value = var_value.reshape(var_value.shape[0], -1)

        # 写 csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for row in var_value:
                writer.writerow(row)

        # 收集元信息
        meta_list.append({
            'mat_file': str(mat_path.absolute()),
            'variable': var_name,
            'csv_file': str(csv_path.absolute()),
            'shape': var_value.shape,
            'dtype': str(var_value.dtype),
            'timestamp': timestamp
        })

    # 3. 写 json
    json_path = csv_dir / 'info.json'
    # 如果 json 已存在则增量合并
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            old = json.load(f)
        meta_list.extend(old)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta_list, f, indent=2, ensure_ascii=False)

    print(f'MAT 文件已成功转换并记录：{mat_path} -> {csv_dir}')


def npy2csv(npy_path, csv_path, header=None):
    """
    将 .npy 文件转换为 .csv 文件. 如果文件格式高于2维, 则保留最后一维, 展平前面的维度

    参数:
        npy_file: str, 输入的 .npy 文件路径
        csv_file: str, 输出的 .csv 文件路径
        header: csv表头, 如果不设置默认None, 即不保存表头
    """
    # 1. 加载 .npy 文件
    data = np.load(npy_path)
    print(f"原始数组形状: {data.shape}")

    # 2. 重塑数组 (sample_num, timeseries_len, feature_num)
    #    -> (sample_num * timeseries_len, feature_num)
    if len(data.shape) > 2:
        feature_num = data.shape[-1]
        data = data.reshape(-1, feature_num)
        print(f"重塑后数组形状: {data.shape}")
    elif len(data.shape) == 1:  # 一维数据给第一个维度
        data = data.reshape(1, -1)

    # 3. 保存为 .csv 文件
    if not header:
        pd.DataFrame(data).to_csv(csv_path, index=False, header=False)
    else:
        pd.DataFrame(data, columns=header).to_csv(csv_path, index=False)
    print(f"已成功保存至: {csv_path}")


def npy2csv_with_autoHeader(npy_path, csv_path, results_num=1):
    """
    将 *.npy 文件转换为 *.csv 文件. 如果文件格式高于2维, 则保留最后一维, 展平前面的维度
    自动设置表头: 除最后 results_num 个使用 results, 前面皆为 feature_*

    参数:
        npy_file: str, 输入的 .npy 文件路径
        csv_file: str, 输出的 .csv 文件路径
    """
    # 1. 加载 .npy 文件
    data = np.load(npy_path)
    print(f"原始数组形状: {data.shape}")

    # 2. 重塑数组 (sample_num, timeseries_len, feature_num)
    #    -> (sample_num * timeseries_len, feature_num)
    if len(data.shape) > 2:
        sample_num, timeseries_len, feature_num = data.shape
        data = data.reshape(sample_num * timeseries_len, feature_num)
        print(f"重塑后数组形状: {data.shape}")
    elif len(data.shape) == 1:
        data = data.reshape(1, -1)

    # 3. 保存为 .csv 文件
    header = [f"feature_{i}" for i in range(data.shape[-1] - results_num)]
    header.extend([f"result_{i}" for i in range(results_num)])
    pd.DataFrame(data, columns=header).to_csv(csv_path, index=False)
    print(f"已成功保存至: {csv_path}")
