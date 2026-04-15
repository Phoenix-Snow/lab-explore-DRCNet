from typing import Union, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from Tools.Normalizer_2D import MinMaxNormalize, StdMeanNormalize


def build_dataset4now(station_path_dict: Dict, train_random_index_dict: Dict, valid_random_index_dict: Dict,
                      time_seq_length: int):
    train_dataset = Dataset_Cache()
    valid_dataset = Dataset_Cache()
    for station_id in station_path_dict.keys():
        station_data = np.array(station_path_dict[station_id])  # (S*T, F)
        station_data = station_data.reshape(-1, time_seq_length, station_data.shape[-1])  # (station_sample_num, T, F)

        # 训练集
        train_idx_list = train_random_index_dict[station_id]
        train_data = station_data[train_idx_list]
        train_dataset.date_l.extend(train_data[:, :, 0])  # 时序
        train_dataset.input_t.extend(train_data[:, :, 1:-2])  # 输入特征（预测现在则目标特征不可输入）
        train_dataset.label.extend(train_data[:, :, -2:])  # 目标特征
        train_dataset.station_l.extend([station_id for _ in range(len(train_idx_list))])

        valid_idx_list = valid_random_index_dict[station_id]
        valid_data = station_data[valid_idx_list]
        valid_dataset.date_l.extend(valid_data[:, :, 0])
        valid_dataset.input_t.extend(valid_data[:, :, 1:-2])
        valid_dataset.label.extend(valid_data[:, :, -2:])
        valid_dataset.station_l.extend([station_id for _ in range(len(valid_idx_list))])

    return train_dataset, valid_dataset

def build_dataset4future(station_path_dict: Dict, train_random_index_dict: Dict, valid_random_index_dict: Dict,
                  time_seq_length: int, is_use_target_series: bool):
    train_dataset = Dataset_Cache()
    valid_dataset = Dataset_Cache()
    for station_id in station_path_dict.keys():
        station_data = np.array(station_path_dict[station_id])  # (S*T, F)
        station_data = station_data.reshape(-1, time_seq_length, station_data.shape[-1])  # (station_sample_num, T, F)  切好了数据集，方便后面直接根据第一个维度二次划分
        # 这里切好后，每个sample相互之间的第一行数据日期 理应是顺序的，即第一个sample的第一天是1-1，则第二个sample的第一天是1-2（前提是数据集没有删除一些数据导致sample中间有缺损）
        # Merge里有对删除过数据的相应处理操作，但只要merge过程中允许补全数据则无需考虑这些因素

        # 训练集
        train_idx_list = train_random_index_dict[station_id]
        # 清理越界索引
        max_valid_idx = max(train_idx_list) - time_seq_length
        train_idx_list_input = []
        train_idx_list_label = []
        for idx in train_idx_list:
            if idx <= max_valid_idx:
                train_idx_list_input.append(idx)
                train_idx_list_label.append(idx + time_seq_length)
        # 整合数据集
        train_input = station_data[train_idx_list_input]
        train_label = station_data[train_idx_list_label]
        # if is_use_target_series:
        #     train_dataset.input_t.extend(train_input[:, :, 1:])
        # else:
        #     train_dataset.input_t.extend(train_input[:, :, 1:-2])
        train_dataset.date_l.extend(train_input[:, :, 0])  # 时序
        train_dataset.input_t.extend(train_input[:, :, 1:-2])
        train_dataset.label.extend(train_label[:, :, -2:])  # 目标特征
        train_dataset.station_l.extend([station_id for _ in range(len(train_idx_list_label))])

        # 验证集
        valid_idx_list = valid_random_index_dict[station_id]
        # 清理越界索引
        max_valid_idx = max(valid_idx_list) - time_seq_length
        valid_idx_list_input = []
        valid_idx_list_label = []
        for idx in valid_idx_list:
            if idx <= max_valid_idx:
                valid_idx_list_input.append(idx)
                valid_idx_list_label.append(idx + time_seq_length)
        # 整合数据集
        valid_input = station_data[valid_idx_list_input]
        valid_label = station_data[valid_idx_list_label]
        # valid_dataset.input_t.extend(valid_input[:, :, 1:-2])
        # if is_use_target_series:
        #     valid_dataset.input_t.extend(valid_input[:, :, 1:])
        # else:
        #     valid_dataset.input_t.extend(valid_input[:, :, 1:-2])
        valid_dataset.date_l.extend(valid_input[:, :, 0])
        valid_dataset.input_t.extend(valid_input[:, :, 1:-2])
        valid_dataset.label.extend(valid_label[:, :, -2:])
        valid_dataset.station_l.extend([station_id for _ in range(len(valid_idx_list_label))])

    return train_dataset, valid_dataset

def build_test_dataset4now(station_path_dict: Dict, time_seq_length: int):
    dataset = Dataset_Cache()
    for station_id in station_path_dict.keys():
        station_data = np.array(station_path_dict[station_id])  # (S*T, F)
        station_data = station_data.reshape(-1, time_seq_length, station_data.shape[-1])  # (station_sample_num, T, F)

        dataset.date_l.extend(station_data[:, :, 0])
        dataset.input_t.extend(station_data[:, :, 1:-2])
        dataset.label.extend(station_data[:, :, -2:])
        dataset.station_l.extend([station_id for _ in range(station_data.shape[0])])
        # dataset.date.extend(np.concat([station_data[:, :, 0], station_data[:, :, 0]], axis=-1))  # 输入时间 -> 输出时间

    return dataset

def build_test_dataset4future(station_path_dict: Dict, time_seq_length: int, is_use_target_series: bool):
    dataset = Dataset_Cache()
    for station_id in station_path_dict.keys():
        station_data = np.array(station_path_dict[station_id])  # (S*T, F)
        station_data = station_data.reshape(-1, time_seq_length, station_data.shape[-1])  # (station_sample_num, T, F)

        dataset.date_l.extend(station_data[:-time_seq_length, :, 0])
        dataset.input_t.extend(station_data[:-time_seq_length, :, 1:-2])  # 没有乱序 - 顺序直接衔接365天
        dataset.label.extend(station_data[time_seq_length:, :, -2:])  # 365天后
        dataset.station_l.extend([station_id for _ in range(station_data.shape[0] - time_seq_length)])
        # dataset.date.extend(np.concat([station_data[:-365, :, 0], station_data[365:, :, 0]], axis=-1))  # 输入时间 -> 输出时间

    return dataset


class Dataset_Cache:
    def __init__(self):
        self.date_l = []
        self.input_t = []
        self.label = []
        self.station_l = []
        # self.date = []


class SimpleReaderTrain(Dataset):  # 单例数据读取器, 是BatchReader(使用DataLoader, 其中第一个参数引入SimpleReader)的基础
    def __init__(self, station_t_input, station_t_label, station_id_list,
                 normalize_input_func: Union[MinMaxNormalize, StdMeanNormalize],
                 normalize_label_func: Union[MinMaxNormalize, StdMeanNormalize], ):
        # print(torch.Tensor(station_t_input).shape)
        self.station_t_input = normalize_input_func.work(station_t_input)
        # print(torch.Tensor(self.station_t_input).shape)
        self.station_t_label = normalize_label_func.work(station_t_label)
        self.station_id_list = station_id_list
        assert len(station_t_input) == len(station_t_label) == len(station_id_list), \
            f"{len(station_t_input)=} {len(station_t_label)=} {len(station_id_list)=}"

    def __getitem__(self, idx) -> Tuple[torch.Tensor, str, torch.Tensor]:
        input_t = self.station_t_input[idx]
        label = self.station_t_label[idx]
        station_id = str(self.station_id_list[idx])
        # input_s = self.station_s_data[station_id]
        return torch.Tensor(input_t), station_id, torch.Tensor(label)  # 两个输入, 一个输出

    def __len__(self):
        return len(self.station_id_list)

class SimpleReaderPred(Dataset):  # 单例数据读取器, 是BatchReader(使用DataLoader, 其中第一个参数引入SimpleReader)的基础
    def __init__(self, station_t_input, station_t_label, station_id_list,
                 normalize_input_func: Union[MinMaxNormalize, StdMeanNormalize],
                 normalize_label_func: Union[MinMaxNormalize, StdMeanNormalize], ):
        self.station_t_input = normalize_input_func.work(station_t_input)
        self.station_t_label = normalize_label_func.work(station_t_label)
        # self.station_s_data = station_s_data
        # for station_id in station_s_data.keys():
        #     self.station_s_data[station_id] = (torch.tensor(station_s_data[station_id],
        #                                                     dtype=torch.float32) / 255. - 1.) * 2.
        self.station_id_list = station_id_list
        assert len(station_t_input) == len(station_t_label) == len(station_id_list), \
            f"{len(station_t_input)=} {len(station_t_label)=} {len(station_id_list)=}"

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]:
        input_t = self.station_t_input[idx]
        label = self.station_t_label[idx]
        station_id = str(self.station_id_list[idx])
        # input_s = self.station_s_data[station_id]
        return torch.Tensor(input_t), torch.Tensor(input_t), station_id, torch.Tensor(label) # 两个输入, 一个输出

    def __len__(self):
        return len(self.station_id_list)

class SpatialReader:
    def __init__(self, station_s_data):
        self.station_s_data = station_s_data
        for station_id in station_s_data.keys():
            transform = transforms.Compose([
                transforms.ToTensor(),  # → [C,H,W], float32, 0-1
            ])
            s = transform(station_s_data[station_id].convert('RGB'))
            self.station_s_data[station_id] = (torch.tensor(s, dtype=torch.float32) / 255. - 1.) * 2.

    def __getitem__(self, station_id):
        assert station_id in self.station_s_data.keys(), "station_id: {} not in station_s_data.keys()".format(
            station_id)
        input_s = self.station_s_data[station_id]
        return input_s

