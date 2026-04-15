import csv
import pandas as pd
import numpy as np

def write_large_csv(file_path, fieldnames, data_generator, chunk_size=5000):
    """
    分批保存
    文件地址、表头、数据单例方法、一批多少单例
    """
    with open(file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        chunk = []
        for row in data_generator:
            chunk.append(row)
            if len(chunk) >= chunk_size:
                writer.writerows(chunk)
                chunk.clear()
        # 写入剩余部分
        if chunk:
            writer.writerows(chunk)

def results_generator(results_dict):
    """
    将数据分单个样本返回
    """
    for row in results_dict:
        yield row


def save_structured_npz(station_buffer, save_path):
    # 定义结构化dtype
    dtype_list = []
    for col in station_buffer.columns:
        if pd.api.types.is_numeric_dtype(station_buffer[col]):
            dtype_list.append((col, np.float32))
        else:
            # 字符串列使用固定长度Unicode
            max_len = station_buffer[col].str.len().max()
            dtype_list.append((col, f'U{max_len}'))

    dtype = np.dtype(dtype_list)

    # 创建结构化数组
    structured_data = np.array([
        tuple(row) for row in station_buffer.itertuples(index=False)
    ], dtype=dtype)

    # 保存为NPZ
    np.savez_compressed(save_path, data=structured_data)
