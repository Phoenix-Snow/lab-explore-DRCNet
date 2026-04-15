from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from Tools.Normalizer_1D import Normalizer


class Register:
    def __init__(self, station_path_dict: Dict[str, str], index_dict: Dict[str, List[int]] | None,
                 time_seq_length: int, normal_params, normalize_type, t_features_name, s_features_name,
                 label_features_name):
        self.time_seq_length = time_seq_length

        # 1. station_path_dict 地址库 / 完整{id-DataFrame}
        self.data_source = {}
        self.input_features_index = []
        self.label_features_index = []
        self.t_features_index, self.s_features_index = [], []
        station_path_dict = station_path_dict
        for station_id, path in station_path_dict.items():
            if path.endswith(".npz"):
                npz_file = np.load(path, mmap_mode='r')
            else:
                raise FileNotFoundError(f"File {path[:-1] + 'z'} not found. WARNING: only support .npz files.")
            self.data_source[station_id] = npz_file['data']
            # 3. columns index
            if self.input_features_index == [] or self.label_features_index == []:
                columns = npz_file['columns'].tolist()

                # 预计算列索引映射并缓存
                column_index_map = {col: i for i, col in enumerate(columns)}

                # 缓存输入和标签的列索引列表
                # self.t_features_index = [column_index_map[col] for col in columns_input_t]
                # self.s_features_index = [column_index_map[col] for col in columns_input_s]
                self.input_features_name = t_features_name + s_features_name
                self.label_features_name = label_features_name
                self.t_features_index = [i for i, col in enumerate(self.input_features_name) if col in t_features_name]
                self.s_features_index = [i for i, col in enumerate(self.input_features_name) if col in s_features_name]
                self.input_features_index = [column_index_map[col] for col in self.input_features_name]
                self.label_features_index = [column_index_map[col] for col in self.label_features_name]

        # 2. index_dict 构建全局样本库 [(station_id, start_row), ...]
        self.samples = []
        if index_dict is not None:
            for station_id, index_list in index_dict.items():
                # IF (起始位置起两年) 没有越界，保存进库
                split_indices = [idx for idx in index_list if
                                 idx + time_seq_length * 2 <= len(self.data_source[station_id])]
                self.samples.extend([(station_id, start_idx) for start_idx in split_indices])
        else:
            for station_id in self.data_source.keys():
                self.samples.extend([(station_id, start_idx) for start_idx in
                                     range(len(self.data_source[station_id]) - self.time_seq_length * 2)])

        print(f"Register dataset with {len(self.samples)} samples.")

        # 3. 输入和标签的列索引列表 self.input_features_index self.label_features_index，切 normal_params
        if normalize_type == 'StdMeanNormalize':
            origin_input_len = len(normal_params['mean_values_input'])
            normal_params['mean_values_input'] = [normal_params['mean_values_input'][i] for i in
                                                  self.input_features_index]
            normal_params['std_values_input'] = [normal_params['std_values_input'][i] for i in
                                                 self.input_features_index]
            normal_params['mean_values_output'] = [normal_params['mean_values_output'][i - origin_input_len] for i in
                                                   self.label_features_index]
            normal_params['std_values_output'] = [normal_params['std_values_output'][i - origin_input_len] for i in
                                                  self.label_features_index]
        elif normalize_type == 'MinMaxNormalize':
            origin_input_len = len(normal_params['min_values_input'])
            normal_params['min_values_input'] = [normal_params['min_values_input'][i] for i in
                                                 self.input_features_index]
            normal_params['max_values_input'] = [normal_params['max_values_input'][i] for i in
                                                 self.input_features_index]
            normal_params['min_values_output'] = [normal_params['min_values_output'][i - origin_input_len] for i in
                                                  self.label_features_index]
            normal_params['max_values_output'] = [normal_params['max_values_output'][i - origin_input_len] for i in
                                                  self.label_features_index]
        else:
            raise ValueError(f'[ERROR] Normalization Type Error: normalize_type - StdMeanNormalize / MinMaxNormalize '
                             f'(actual error input: {normalize_type})')
        normalizer = Normalizer(normal_params, normalize_type)
        self.normalizer_input, self.normalizer_label, self.antiNormalizer = normalizer.normalization_input, normalizer.normalization_label, normalizer.antiNormalization


class StreamForecastingDataset(Dataset):
    def __init__(self, register_class, desc='Hello'):
        self.time_seq_length = register_class.time_seq_length

        # 1. station_path_dict
        self.data_source = register_class.data_source
        self.columns_input = register_class.input_features_index
        self.columns_label = register_class.label_features_index

        # 2. index_dict 构建全局样本库 [(station_id, start_row), ...]
        self.samples = register_class.samples

        # 3. 输入和标签的列索引列表 self.input_features_index self.label_features_index，切 normal_params
        self.normalizer_input = register_class.normalizer_input
        self.normalizer_label = register_class.normalizer_label
        self.antiNormalizer = register_class.antiNormalizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        station_id, start_row = self.samples[idx]

        # 获取数据源
        data_handle = self.data_source[station_id]

        # 2. 动态读取 (流式核心)
        # 如果 data_handle 是字符串 (路径)，请在此处使用 np.load / pd.read_csv 等读取
        # 如果 data_handle 是 np.ndarray，直接切片
        # if isinstance(data_handle, str):  # (T*2, Features)
        #     data_array = pd.read_csv(data_handle, skiprows=start_row, nrows=self.time_seq_length * 2)
        # elif isinstance(data_handle, pd.DataFrame):
        #     data_array: pd.DataFrame = data_handle.iloc[start_row : start_row + self.time_seq_length * 2]  # 直接引用内存数组
        # else:
        #     raise TypeError("station_path_dict must be Dict[str, str | pd.DataFrame]")
        window_data = data_handle[start_row:start_row + self.time_seq_length * 2]

        # 分离输入输出
        input_rows = window_data[:self.time_seq_length]  # 前T行
        label_rows = window_data[self.time_seq_length:]  # 后T行

        input_rows = input_rows[:, self.columns_input]  # 输入特征
        label_rows = label_rows[:, self.columns_label]  # 标签

        # 应用 Normalizer (保持不变)
        input_normalized = self.normalizer_input.work(input_rows)
        label_normalized = self.normalizer_label.work(label_rows)

        # return torch.FloatTensor(input_normalized), torch.FloatTensor(label_normalized)
        return input_normalized, label_normalized
