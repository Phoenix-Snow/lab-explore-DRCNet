import copy
import csv
import json

import numpy as np
from openpyxl import Workbook
import os
import time

import pandas as pd
import torch
from torch import nn
from typing import Union, Any, Generator

from ModelRegister.Frequency_Encoder import T_Model, T_Model_trans_tMLP2Linear
from ModelRegister.DRCNet import TS_Model, TS_Model_melt_geo, TS_Model_melt_residual_spatial, \
    TS_Model_trans_sMLP2Linear, TS_Model_melt_emb
from Tools.Normalizer_1D import StdMeanAntiNormalize, MinMaxAntiNormalize
from Tools.Tools import print_net_params_list, count_flops_params
from Tools.draw import draw_tsne_by_ss, draw_residuals, draw_tsne_by_sss, draw_tsne_by_feature_sss, \
    draw_true1pred_img

terrain_labels = {
    '51828': 0,  # desert_oasis
    '51777': 0,
    '51709': 0,
    '51628': 1,  # north_slope_foothill
    '51567': 1,
    '52203': 1,
    '51573': 1,
    '52112': 1,
    '51463': 2,  # mountain_meadow
    '51431': 1,
    '51358': 1,
    '51133': 2,
    '51076': 2,
}


class Procedure(nn.Module):
    def __init__(self, series_len: int, device: str,
                 antiNormalize_function: Union[MinMaxAntiNormalize, StdMeanAntiNormalize],
                 reader, results_save_path: str, pretrained_model_path: str,
                 target_mode, task_desc: str):
        super().__init__()

        input_feature_list = [
            'T_ave', 'T_max', 'T_min', 'AtmospherePressure_avg', 'WindSpeed_avg', 'RelativeHumidity', 'SunDuration',
        ]
        normal_feature_list = [
            'theory_sunDuration', 'theory_DailySunRadiation', 'Longitude', 'Latitude', 'Elevation', 'slope', 'aspect',
        ]
        # 'DailySunRadiation', 'dailySunRadiation_theoryPlus'
        output_feature_list = ['DailySunRadiation']

        # dataset  不要动这些设置
        dataset_input_feature_list = [
            'T_ave', 'T_max', 'T_min', 'AtmospherePressure_avg', 'WindSpeed_avg',
            'RelativeHumidity', 'theory_sunDuration', 'SunDuration', 'sunDuration_theoryPlus',
            'theory_DailySunRadiation', 'Longitude', 'Latitude', 'Elevation',
            '3d_x', '3d_y', '3d_z', 'slope', 'aspect',
        ]  # 数据集中包含特征
        dataset_output_feature_list = [
            'DailySunRadiation', 'dailySunRadiation_theoryPlus'
        ]  # 数据集中包含特征

        self.__input_feature_list = [dataset_input_feature_list.index(f) for f in input_feature_list]
        self.__normal_feature_list = [dataset_input_feature_list.index(f) for f in normal_feature_list]
        self.__output_feature_list = [dataset_output_feature_list.index(f) for f in output_feature_list]
        self.__results_head = input_feature_list + normal_feature_list
        for idx, feature in enumerate(output_feature_list):
            self.__results_head += [feature, 'predict' + str(idx + 1)]
        self.__results_head += ['error']

        self.__device = device

        self.__results_save_path = results_save_path

        if 'timesnet_modified_' + task_desc in pretrained_model_path:  # 去空间优化模块的消融 (纯时序模型)
            self.__net = T_Model(len(self.__input_feature_list), len(self.__normal_feature_list),
                                 self.__results_save_path,
                                 series_len, device)
            self.__net.eval()
            self.__results_save_path = os.path.join(self.__results_save_path,
                                                    'timesnet_modified_' + task_desc + str(series_len))
        elif 'timesnet_modified_add_spatial_' + task_desc + '_melt_emb' in pretrained_model_path:  # 空间模块去向量空间记忆的消融
            print('[INFO] ')
            self.__t_net = T_Model(len(self.__input_feature_list), len(self.__normal_feature_list),
                                   self.__results_save_path,
                                   series_len, device)
            self.__net = TS_Model_melt_emb(self.__t_net, len(self.__normal_feature_list), device)
            self.__t_net.eval()
            self.__net.eval()
            self.__results_save_path = os.path.join(self.__results_save_path,
                                                    'timesnet_modified_add_spatial_' + task_desc + str(
                                                        series_len) + '_melt_emb')
        elif 'timesnet_modified_add_spatial_' + task_desc + '_melt_geo' in pretrained_model_path:  # 去地理数据的消融
            self.__t_net = T_Model(len(self.__input_feature_list), len(self.__normal_feature_list),
                                   self.__results_save_path,
                                   series_len, device)
            self.__net = TS_Model_melt_geo(self.__t_net, len(self.__normal_feature_list), device)
            self.__t_net.eval()
            self.__net.eval()
            self.__results_save_path = os.path.join(self.__results_save_path,
                                                    'timesnet_modified_add_spatial_' + task_desc + str(
                                                        series_len) + '_melt_geo')
        elif 'timesnet_modified_add_spatial_' + task_desc + '_melt_res' in pretrained_model_path:  # 去剩余特征的消融（比3更进一步的消融）
            self.__t_net = T_Model(len(self.__input_feature_list), len(self.__normal_feature_list),
                                   self.__results_save_path,
                                   series_len, device)
            self.__net = TS_Model_melt_residual_spatial(self.__t_net, len(self.__normal_feature_list), device)
            self.__t_net.eval()
            self.__net.eval()
            self.__results_save_path = os.path.join(self.__results_save_path,
                                                    'timesnet_modified_add_spatial_' + task_desc + str(
                                                        series_len) + '_melt_res')
        elif 'timesnet_modified_add_spatial_' + task_desc + '_tMLP2Linear' in pretrained_model_path:
            self.__t_net = T_Model_trans_tMLP2Linear(len(self.__input_feature_list), len(self.__normal_feature_list),
                                                     series_len, device)
            self.__net = TS_Model(self.__t_net, len(self.__normal_feature_list), self.__results_save_path, device)
            self.__t_net.eval()
            self.__net.eval()
            self.__results_save_path = os.path.join(self.__results_save_path,
                                                    'timesnet_modified_add_spatial_' + task_desc + str(
                                                        series_len) + '_tMLP2Linear')
        elif 'timesnet_modified_add_spatial_' + task_desc + '_sMLP2Linear' in pretrained_model_path:
            self.__t_net = T_Model(len(self.__input_feature_list), len(self.__normal_feature_list),
                                   self.__results_save_path,
                                   series_len, device)
            self.__net = TS_Model_trans_sMLP2Linear(self.__t_net, len(self.__normal_feature_list), device)
            self.__t_net.eval()
            self.__net.eval()
            self.__results_save_path = os.path.join(self.__results_save_path,
                                                    'timesnet_modified_add_spatial_' + task_desc + str(
                                                        series_len) + '_sMLP2Linear')
        # else:
        #     raise ValueError('[ERROR] if-else Lack this section')
        else:  # 正常模型
            self.__t_net = T_Model(len(self.__input_feature_list), len(self.__normal_feature_list),
                                   self.__results_save_path,
                                   series_len, device)
            self.__net = TS_Model(self.__t_net, len(self.__normal_feature_list), self.__results_save_path, device)
            self.__t_net.eval()
            self.__net.eval()
            # self.__eval_loss_threshold = 3.5  # 缩小阈值
            self.__results_save_path = os.path.join(self.__results_save_path,
                                                    'timesnet_modified_add_spatial_' + task_desc + str(series_len))

        model = copy.deepcopy(self.__net)
        print_net_params_list(model)
        count_flops_params(model,
                           (
                               (1, series_len, len(input_feature_list)),
                               (1, series_len, len(normal_feature_list)),
                           ),
                           device
                           )

        self.__antiNormalize_function = antiNormalize_function.to(device)
        self.__antiNormalize_function.cuda_device(device)
        self.__antiNormalize_function.location(self.__output_feature_list)

        self.__pred_reader = reader

        import time
        time_str = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        self.__results_save_path = os.path.join(self.__results_save_path, time_str)

        del input_feature_list, output_feature_list, normal_feature_list, (
            dataset_input_feature_list), dataset_output_feature_list

        # count_flops(self.__net, (30, 365, ), device)
        print(' all requires_grad -> False (use "with torch.no_grad():" during predicting)')

        self.__results = torch.Tensor([])

        self.__series_len = series_len

    def forward(self, save_file_type: str = 'csv'):
        # 创建目录
        os.makedirs(self.__results_save_path, exist_ok=True)  # 结果保存根目录
        os.makedirs(os.path.join(self.__results_save_path, 'test_true1pred'), exist_ok=True)  # 二级分目录 -> 测试集: 真实值-预测值
        os.makedirs(os.path.join(self.__results_save_path, 'train_true1pred'), exist_ok=True)  # 二级分目录 -> 训练集: 真实值-预测值
        os.makedirs(os.path.join(self.__results_save_path, 'residuals_plot'), exist_ok=True)  # 二级分目录 -> 收敛状况图
        os.makedirs(os.path.join(self.__results_save_path, 'tsne'), exist_ok=True)  # 二级分目录 -> tsne
        # 运行模型/保存预测结果/保存拟合图
        if save_file_type.lower() == 'csv':
            self.save_csv()
        elif save_file_type.lower() == 'xlsx':
            self.save_xlsx()
        with open(os.path.join(self.__results_save_path, 'LOG.json'), 'w') as f:
            json.dump({'test': self.__mode_dict, 'train': self.__train_mode_dict}, f, indent=4)
        print('Results hava all been saved.\n')

    # 通用“逐行写”函数
    def _write_sheet(self, reader_type, reader_phase, sheet_normal, sheet_denorm):
        header_normal_done = False
        header_denorm_done = False
        if reader_type == 'train':
            reader = self.__train_reader
        elif reader_type == 'test':
            reader = self.__test_reader
        else:
            raise NotImplementedError
        for one_sample_normal, one_sample in self.results(reader, reader_phase):
            # 归一化头
            if not header_normal_done:
                sheet_normal.append(list(one_sample_normal.keys()))
                header_normal_done = True
            sheet_normal.append(list(one_sample_normal.values()))

            # 反归一化头
            if not header_denorm_done:
                sheet_denorm.append(list(one_sample.keys()))
                header_denorm_done = True
            sheet_denorm.append(list(one_sample.values()))

    def save_xlsx(self):
        xlsx_path = os.path.join(self.__results_save_path, 'Results.xlsx')
        wb = Workbook()

        # 提前把 4 个 sheet 建好，顺序固定
        sheets = {
            'test_normal': wb.create_sheet('test_normal'),
            'test_deNormal': wb.create_sheet('test_deNormal'),
            'train_normal': wb.create_sheet('train_normal'),
            'train_deNormal': wb.create_sheet('train_deNormal')
        }
        # 删除默认空 sheet
        wb.remove(wb.active)

        # 1. test 部分
        self._write_sheet('test',
                          sheets['test_normal'],
                          sheets['test_deNormal'])
        print('Test has already been saved.\n')

        # 2. train 部分（可选）
        print(f'(is_test_train: {self.is_test_train})')
        if self.is_test_train:
            self._write_sheet('train',
                              sheets['train_normal'],
                              sheets['train_deNormal'])
            print('Train has already been saved.\n')

        # 落盘
        wb.save(xlsx_path)

    def save_csv(self):
        # 保存预测结果csv.DictWriter/运行过程self.results/保存 真实值-预测值 拟合图
        with open(os.path.join(self.__results_save_path, 'test_RESULTS.csv'), 'w', newline='') as f_norm, \
                open(os.path.join(self.__results_save_path, 'test_RESULTS_denormalized.csv'), 'w',
                     newline='') as f_denorm:
            writer_normal = None
            writer = None
            for (one_sample_normal, one_sample) in self.results(self.__test_reader, 'test'):
                if writer_normal is None:
                    writer_normal = csv.DictWriter(f_norm, fieldnames=list(one_sample_normal.keys()))
                    writer_normal.writeheader()
                writer_normal.writerow(one_sample_normal)
                if writer is None:
                    writer = csv.DictWriter(f_denorm, fieldnames=list(one_sample.keys()))
                    writer.writeheader()
                writer.writerow(one_sample)
        print(f'Results has already been saved.\n')

    def results(self, reader) -> Generator[tuple[dict[str, Any], dict[str, Any]], Any, None]:
        for time_steps, temporal_data, station_id in reader:
            temporal_map = temporal_data.clone()
            normal_map = temporal_data.clone()
            if isinstance(temporal_map[0], pd.DataFrame):
                for i in range(len(temporal_map)):
                    temporal_map[i] = temporal_map[i][self.__input_feature_list].values
                    normal_map[i] = normal_map[i][self.__normal_feature_list].values
            else:
                temporal_map = temporal_map[:, :, self.__input_feature_list]
                normal_map = normal_map[:, :, self.__normal_feature_list]
            temporal_map, normal_map = (torch.Tensor(temporal_map).to(self.__device),
                                        torch.Tensor(normal_map).to(self.__device))

            # 预测
            with torch.no_grad():
                # 预测隐向量和结果
                hidden_t_feature, hidden_t_results, \
                    hidden_s_feature, hidden_g_data, hidden_m_feature, \
                    s_w, s_b, results, \
                    B, T = self.__net.hidden_params(temporal_map, normal_map)

                results = results.reshape(B, T, -1)

                # 额外的数据标注：tsne需要，包括时间的标注（季节、月份、闰年等）、地理环境 terrain_labels
                for sid in station_id:  # 一个batch的 s id  （需要挨个根据站点对应其地理环境标签）
                    g_label = torch.cat(
                        [g_label, torch.tensor([int(terrain_labels[str(sid)])]).to(self.__train_g_labels.device)],
                        # 挨个拿地理标签
                        dim=0  # 按第一个维度堆叠
                    )  # (batch_size)
                # 拼接进总记录-隐向量/结果中
                self.__results = torch.cat([self.__results.to("cpu"), results.to("cpu")], dim=0)

            # 处理输出，保证向量的结构格式统一，防止出现最后一个维度为1时会输出省略的问题
            results = results.to(self.__device)
            results = results.reshape(results.shape[0], results.shape[1], -1)

            # 构建需要保存的内容
            merged = torch.cat([time_steps, temporal_map, normal_map, results], dim=-1)  # BTF
            merged = merged.reshape(temporal_map.shape[0] * temporal_map.shape[1], -1)  # SF
            results_dict = [dict(zip(self.__results_head, row.tolist())) for row in merged]

            # 反归一化
            antiNormalize_pred = self.__antiNormalize_function.work(results)

            # 构建另一个需要保存的内容
            merged = torch.cat([time_steps, temporal_map, normal_map, antiNormalize_pred], dim=-1)
            merged = merged.reshape(temporal_map.shape[0] * temporal_map.shape[1], -1)
            results_dict_anti = [dict(zip(self.__results_head, row.tolist())) for row in merged]

            # 一行一行地返回 未归一化 和 已归一化 的数据
            for result, result_anti in zip(results_dict, results_dict_anti):
                yield result, result_anti  # -> 用于保存预测结果

            del time_steps, temporal_data, station_id, temporal_map, normal_map, results, antiNormalize_pred, merged

    def load_model(self, pretrained_model_path: str):
        if not os.path.exists(pretrained_model_path):
            raise FileNotFoundError(f'pretrained model path: ({pretrained_model_path}) does not exist')
        self.__net.load_state_dict(torch.load(pretrained_model_path))

    @property
    def __device__(self):
        return self.__device

    @property
    def eval_loss_threshold(self):
        return self.__eval_loss_threshold

    @property
    def OUTPUT(self):
        return self.__output
