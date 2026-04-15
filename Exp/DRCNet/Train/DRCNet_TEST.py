import copy
import csv
import json
import os
import time
from typing import Union, Any, Generator

import numpy as np
import pandas as pd
import torch
from openpyxl import Workbook
from torch import nn

from ModelRegister.DRCNet import TS_Model, TS_Model_ENHANCE
from ModelRegister.Frequency_Encoder import T_Model_new
from Tools.Normalizer_1D import StdMeanAntiNormalize, MinMaxAntiNormalize
from Tools.Tools import print_net_params_list, count_flops_params
from Tools.draw import draw_tsne_by_ss, draw_residuals, draw_true1pred_img


class Procedure(nn.Module):
    def __init__(self, series_len: int, pretrained_model_path: str, device: str,
                 antiNormalize_function: Union[MinMaxAntiNormalize, StdMeanAntiNormalize],
                 train_reader, test_reader, results_save_path: str, task_desc: str,
                 t_features, s_features, o_features, in_features_name, out_features_name,
                 additional_Info=None, is_test_train: bool = False):
        super().__init__()

        self.__t_features = t_features
        self.__s_features = s_features

        self.__results_head = in_features_name  # ['DATE'] + in_features_name 还是 in_features_name
        for idx, feature in enumerate(list(out_features_name)):
            self.__results_head += [feature, 'predict' + str(idx + 1)]
        self.__results_head += ["error"]

        self.__device = device

        if results_save_path is None or results_save_path == '' or results_save_path == '/':
            results_save_path = 'results'
        self.__results_save_path = results_save_path

        self.__t_net = T_Model_new(len(t_features), len(s_features), None,
                                   series_len, device, nn.LayerNorm)
        if 'base' in pretrained_model_path:  # 去空间优化模块的消融 (纯时序模型)
            self.__net = TS_Model(self.__t_net, len(s_features), None, device, nn.LayerNorm)
            self.__net.eval()
            self.__results_save_path = os.path.join(self.__results_save_path, self.__net.NAME, task_desc,
                                                    f"{self.__t_net.PARAMS}_{self.__net.PARAMS()}")
        elif 'enhance' in pretrained_model_path:
            self.__net = TS_Model_ENHANCE(self.__t_net, len(s_features), None, device, nn.LayerNorm)
            self.__net.eval()
            self.__results_save_path = os.path.join(self.__results_save_path, self.__net.NAME, task_desc,
                                                    f"{self.__t_net.PARAMS}_{self.__net.PARAMS()}")
        else:
            raise Exception('[❌ ERROR] Train model first')

        try:
            model = copy.deepcopy(self.__net)
            print_net_params_list(model)
            count_flops_params(model,
                               ((1, series_len, len(t_features)), (1, series_len, len(s_features)),),
                               device)
        except Exception as e:
            print(f'[❌ ERROR] skip count_flops_params because: {e}')

        self.__antiNormalize_function = antiNormalize_function.to(device)
        self.__antiNormalize_function.cuda_device(device)

        self.__test_reader = test_reader
        self.__train_reader = train_reader

        import time
        time_str = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        self.__results_save_path = os.path.join(self.__results_save_path, time_str)

        # count_flops(self.__net, (30, 365, ), device)
        print(' all requires_grad -> False (use "with torch.no_grad():" during predicting)')

        self.__hidden_t_results = torch.Tensor([])
        self.__hidden_t_feature = torch.Tensor([])
        self.__hidden_s_feature = torch.Tensor([])
        self.__hidden_m_feature = torch.Tensor([])
        self.__hidden_g_data = torch.Tensor([])
        self.__results = torch.Tensor([])
        self.__targets = torch.Tensor([])
        # self.__g_labels = torch.Tensor([])
        self.__s_w = torch.Tensor([])
        self.__s_b = torch.Tensor([])

        self.__mode_dict = {
            'Information': 'This is test the forcast of TEST dataset. ' if not additional_Info else f'This is test the forcast of TEST dataset. Attention: {additional_Info}',
            'mse': 0., 'mae': 0.,
            'mse_normal': 0., 'mae_normal': 0.,
            'r2': 0.,
            'mse_one_sample': 0., 'mae_one_sample': 0.,
            'mse_normal_one_sample': 0.,
            'mae_normal_one_sample': 0.,
            'P50': 0., 'P90': 0., 'P50_normal': 0., 'P90_normal': 0.,
            'pretrained_model_path': pretrained_model_path}
        self.__train_hidden_t_feature = torch.Tensor([])
        self.__train_hidden_t_results = torch.Tensor([])
        self.__train_hidden_s_feature = torch.Tensor([])
        self.__train_hidden_m_feature = torch.Tensor([])
        self.__train_hidden_g_data = torch.Tensor([])
        self.__train_mode_dict = {
            'Information': 'This is test the forcast of TRAIN dataset. ' if not additional_Info else f'This is test the forcast of TRAIN dataset. Attention: {additional_Info}',
            'mse': 0., 'mae': 0.,
            'mse_normal': 0., 'mae_normal': 0.,
            'r2': 0.,
            'mse_one_sample': 0., 'mae_one_sample': 0.,
            'mse_normal_one_sample': 0.,
            'mae_normal_one_sample': 0.,
            'P50': 0., 'P90': 0., 'P50_normal': 0., 'P90_normal': 0., }
        self.__train_results = torch.Tensor([])
        self.__train_targets = torch.Tensor([])
        # self.__train_g_labels = torch.Tensor([])
        self.__train_s_w = torch.Tensor([])
        self.__train_s_b = torch.Tensor([])

        self.__series_len = series_len
        self.is_test_train = is_test_train

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
        # 接下来绘图时要修改哪些格式
        # self.__train_g_labels = self.__train_g_labels.reshape(self.__train_g_labels.shape[0], 1,
        #                                                       1)  # 增加维度 (sample_num, 1, 1)
        # self.__train_g_labels = self.__train_g_labels.repeat(1, self.__train_results.shape[1], 1)  # (sample_num, T, 1)
        # 绘制tsne降维图
        # 第一个：各隐向量在目标结果中的作用
        draw_tsne_by_ss(self.__hidden_t_feature, self.__targets,
                        'test : t feature <-> label', os.path.join(self.__results_save_path, 'tsne'))
        draw_tsne_by_ss(self.__hidden_t_results, self.__targets,
                        'test : t result <-> label', os.path.join(self.__results_save_path, 'tsne'))
        draw_tsne_by_ss(self.__hidden_s_feature, self.__targets,
                        'test : s feature <-> label', os.path.join(self.__results_save_path, 'tsne'))
        draw_tsne_by_ss(self.__hidden_m_feature, self.__targets,
                        'test : recall feature <-> label', os.path.join(self.__results_save_path, 'tsne'))
        draw_tsne_by_ss(self.__hidden_g_data, self.__targets,
                        'test : g data <-> label', os.path.join(self.__results_save_path, 'tsne'))
        # 第二个：各隐向量在预测结果中的作用
        draw_tsne_by_ss(self.__hidden_t_feature, self.__results,
                        'test : t feature <-> pred result', os.path.join(self.__results_save_path, 'tsne'))
        draw_tsne_by_ss(self.__hidden_t_results, self.__results,
                        'test : t result <-> pred result', os.path.join(self.__results_save_path, 'tsne'))
        draw_tsne_by_ss(self.__hidden_s_feature, self.__results,
                        'test : s feature <-> pred result', os.path.join(self.__results_save_path, 'tsne'))
        draw_tsne_by_ss(self.__hidden_m_feature, self.__results,
                        'test : recall feature <-> pred result', os.path.join(self.__results_save_path, 'tsne'))
        draw_tsne_by_ss(self.__hidden_g_data, self.__results,
                        'test : g data <-> pred result', os.path.join(self.__results_save_path, 'tsne'))
        # 第三个：以上改train
        draw_tsne_by_ss(self.__train_hidden_t_feature, self.__train_targets,
                        'train : t feature <-> label', os.path.join(self.__results_save_path, 'tsne'))
        draw_tsne_by_ss(self.__train_hidden_t_results, self.__train_targets,
                        'train : t result <-> label', os.path.join(self.__results_save_path, 'tsne'))
        draw_tsne_by_ss(self.__train_hidden_s_feature, self.__train_targets,
                        'train : s feature <-> label', os.path.join(self.__results_save_path, 'tsne'))
        draw_tsne_by_ss(self.__train_hidden_m_feature, self.__train_targets,
                        'train : recall feature <-> label', os.path.join(self.__results_save_path, 'tsne'))
        draw_tsne_by_ss(self.__train_hidden_g_data, self.__train_targets,
                        'train : g data <-> label', os.path.join(self.__results_save_path, 'tsne'))
        draw_tsne_by_ss(self.__train_hidden_t_feature, self.__train_results,
                        'train : t feature <-> pred result', os.path.join(self.__results_save_path, 'tsne'))
        draw_tsne_by_ss(self.__train_hidden_t_results, self.__train_results,
                        'train : t result <-> pred result', os.path.join(self.__results_save_path, 'tsne'))
        draw_tsne_by_ss(self.__train_hidden_s_feature, self.__train_results,
                        'train : s feature <-> pred result', os.path.join(self.__results_save_path, 'tsne'))
        draw_tsne_by_ss(self.__train_hidden_m_feature, self.__train_results,
                        'train : recall feature <-> pred result', os.path.join(self.__results_save_path, 'tsne'))
        draw_tsne_by_ss(self.__train_hidden_g_data, self.__train_results,
                        'train : g data <-> pred result', os.path.join(self.__results_save_path, 'tsne'))
        # 第四个：两空间隐向量和预测结果在地理上是否有明确区分(因为train是多空间节点的，test只是空间节点的延伸)
        # draw_tsne_by_sss(self.__train_hidden_s_feature, self.__train_g_labels,
        #                  'train : s feature <-> g-env', os.path.join(self.__results_save_path, 'tsne'))
        # draw_tsne_by_sss(self.__train_hidden_m_feature, self.__train_g_labels,
        #                  'train : recall feature <-> g-env', os.path.join(self.__results_save_path, 'tsne'))
        # draw_tsne_by_sss(self.__train_results, self.__train_g_labels,
        #                  'train : pred result <-> g-env', os.path.join(self.__results_save_path, 'tsne'))
        # draw_tsne_by_sss(self.__train_targets, self.__train_g_labels,
        #                  'train : label(compare) <-> g-env', os.path.join(self.__results_save_path, 'tsne'))
        # 第七个：看时序预测结果是否对时间的预测精准(毕竟单纯时序预测有问题，不能单独先训练冻结，后单独训练空间优化模块，所以是不是这里时序预测是真的在做时序预测)

        # 绘制收敛图
        draw_residuals(self.__results, self.__targets, 'test : residuals_plot',
                       os.path.join(self.__results_save_path, 'residuals_plot'))
        draw_residuals(self.__train_results, self.__train_targets, 'train : residuals_plot',
                       os.path.join(self.__results_save_path, 'residuals_plot'))
        print('Matrics hava all been saved.\n')

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
        print(f'Test has already been saved.\n')
        print(f'(is_test_train: {self.is_test_train})')
        if self.is_test_train:
            with open(os.path.join(self.__results_save_path, 'test_train_RESULTS.csv'), 'w', newline='') as f_norm, \
                    open(os.path.join(self.__results_save_path, 'test_train_RESULTS_denormalized.csv'), 'w',
                         newline='') as f_denorm:
                writer_normal = None
                writer = None
                for (one_sample_normal, one_sample) in self.results(self.__train_reader, 'train'):
                    if writer_normal is None:
                        writer_normal = csv.DictWriter(f_norm, fieldnames=list(one_sample_normal.keys()))
                        writer_normal.writeheader()
                    writer_normal.writerow(one_sample_normal)
                    if writer is None:
                        writer = csv.DictWriter(f_denorm, fieldnames=list(one_sample.keys()))
                        writer.writeheader()
                    writer.writerow(one_sample)
            print('Train has already been saved.\n')

    def results(self, reader, mode='test') -> Generator[tuple[dict[str, Any], dict[str, Any]], Any, None]:
        start_time = time.time()
        total_steps = len(reader)
        idxcount = 0
        # t_one_sample = None
        # s_id_one_sample = None
        # target_one_sample = None
        # pred_one_sample = None
        errors = []
        errors_anti = []
        for t, target in reader:
            # 处理数据
            temporal_map = t.clone()
            normal_map = t.clone()
            if type(temporal_map[0]) == pd.DataFrame:
                # 结构是[DataFrame, DataFrame, DataFrame, DataFrame...]
                for i in range(len(temporal_map)):
                    temporal_map[i] = temporal_map[i][self.__t_features].values
                    normal_map[i] = normal_map[i][self.__s_features].values
                # for i in range(len(target)):
                #     target[i] = target[i][self.__output_feature_list].values
            else:
                temporal_map = temporal_map[:, :, self.__t_features]
                normal_map = normal_map[:, :, self.__s_features]
            temporal_map, normal_map, target = (torch.Tensor(temporal_map).to(self.__device),
                                                torch.Tensor(normal_map).to(self.__device),
                                                torch.Tensor(target).to(self.__device))

            # 预测
            with torch.no_grad():
                # 预测隐向量和结果
                hidden_t_feature, hidden_t_results, \
                    hidden_s_feature, hidden_g_data, hidden_m_feature, \
                    s_w, s_b, results, \
                    B, T = self.__net.hidden_params(temporal_map, normal_map)

                # 保证三维结构
                hidden_t_feature = hidden_t_feature.reshape(B, T, -1)
                hidden_t_results = hidden_t_results.reshape(B, T, -1)
                hidden_s_feature = hidden_s_feature.reshape(B, T, -1)
                hidden_g_data = hidden_g_data.reshape(B, T, -1)
                hidden_m_feature = hidden_m_feature.reshape(B, T, -1)
                s_w = s_w.reshape(B, T, -1)
                s_b = s_b.reshape(B, T, -1)
                results = results.reshape(B, T, -1)
                target = target.reshape(B, T, -1)

                # 额外的数据标注：tsne需要，包括时间的标注（季节、月份、闰年等）、地理环境 terrain_labels
                # g_label = torch.tensor([int(terrain_labels[str(s_id[0])])]).to(self.__train_g_labels.device)
                # g_label = torch.Tensor([]).to(self.__train_g_labels.device)
                # for sid in s_id:  # 一个batch的 s id  （需要挨个根据站点对应其地理环境标签）
                #     g_label = torch.cat(
                #         [g_label, torch.tensor([int(terrain_labels[str(sid)])]).to(self.__train_g_labels.device)],
                #         # 挨个拿地理标签
                #         dim=0  # 按第一个维度堆叠
                #     )  # (batch_size)
                # 拼接进总记录-隐向量/结果中
                if mode == 'train':
                    self.__train_hidden_t_feature = torch.cat([self.__train_hidden_t_feature.to("cpu"),
                                                               hidden_t_feature.to("cpu")], dim=0)
                    self.__train_hidden_t_results = torch.cat([self.__train_hidden_t_results.to("cpu"),
                                                               hidden_t_results.to("cpu")], dim=0)
                    self.__train_hidden_s_feature = torch.cat([self.__train_hidden_s_feature.to("cpu"),
                                                               hidden_s_feature.to("cpu")], dim=0)
                    self.__train_hidden_g_data = torch.cat([self.__train_hidden_g_data.to("cpu"),
                                                            hidden_g_data.to("cpu")], dim=0)
                    self.__train_hidden_m_feature = torch.cat([self.__train_hidden_m_feature.to("cpu"),
                                                               hidden_m_feature.to("cpu")], dim=0)
                    self.__train_s_w = torch.cat([self.__train_s_w.to("cpu"), s_w.to("cpu")], dim=0)
                    self.__train_s_b = torch.cat([self.__train_s_b.to("cpu"), s_b.to("cpu")], dim=0)
                    self.__train_results = torch.cat([self.__train_results.to("cpu"), results.to("cpu")], dim=0)
                    self.__train_targets = torch.cat([self.__train_targets.to("cpu"), target.to("cpu")], dim=0)
                    # self.__train_g_labels = torch.cat([self.__train_g_labels.to("cpu"), g_label.to("cpu")], dim=0)
                elif mode == 'test':
                    self.__hidden_t_feature = torch.cat([self.__hidden_t_feature.to("cpu"),
                                                         hidden_t_feature.to("cpu")], dim=0)
                    self.__hidden_t_results = torch.cat([self.__hidden_t_results.to("cpu"),
                                                         hidden_t_results.to("cpu")], dim=0)
                    self.__hidden_s_feature = torch.cat([self.__hidden_s_feature.to("cpu"),
                                                         hidden_s_feature.to("cpu")], dim=0)
                    self.__hidden_g_data = torch.cat([self.__hidden_g_data.to("cpu"),
                                                      hidden_g_data.to("cpu")], dim=0)
                    self.__hidden_m_feature = torch.cat([self.__hidden_m_feature.to("cpu"),
                                                         hidden_m_feature.to("cpu")], dim=0)
                    self.__s_w = torch.cat([self.__s_w.to("cpu"), s_w.to("cpu")], dim=0)
                    self.__s_b = torch.cat([self.__s_b.to("cpu"), s_b.to("cpu")], dim=0)
                    self.__results = torch.cat([self.__results.to("cpu"), results.to("cpu")], dim=0)
                    self.__targets = torch.cat([self.__targets.to("cpu"), target.to("cpu")], dim=0)
                    # self.__g_labels = torch.cat([self.__g_labels.to("cpu"), g_label.to("cpu")], dim=0)

            # 处理输出，保证向量的结构格式统一，防止出现最后一个维度为1时会输出省略的问题
            results = results.to(self.__device)
            results = results.reshape(results.shape[0], results.shape[1], -1)
            target = target.to(self.__device)
            target = target.reshape(target.shape[0], target.shape[1], -1)

            # 构建需要保存的内容
            error = results - target
            interleaved = torch.stack([target, results], dim=-1).view(results.shape[0], results.shape[1], -1)
            merged = torch.cat([temporal_map, normal_map, interleaved, error], dim=-1)  # BTF
            merged = merged.reshape(temporal_map.shape[0] * temporal_map.shape[1], -1)  # SF
            results_dict = [dict(zip(self.__results_head, row.tolist())) for row in merged]

            # 反归一化
            antiNormalize_pred = self.__antiNormalize_function.work(results)
            antiNormalize_label = self.__antiNormalize_function.work(target.to(self.__device))

            # 构建另一个需要保存的内容
            error = antiNormalize_pred - antiNormalize_label
            interleaved = torch.stack([antiNormalize_label, antiNormalize_pred], dim=-1).view(results.shape[0],
                                                                                              results.shape[1], -1)
            merged = torch.cat([temporal_map, normal_map, interleaved, error], dim=-1)
            merged = merged.reshape(temporal_map.shape[0] * temporal_map.shape[1], -1)
            results_dict_anti = [dict(zip(self.__results_head, row.tolist())) for row in merged]

            # 一行一行地返回 未归一化 和 已归一化 的数据
            for result, result_anti in zip(results_dict, results_dict_anti):
                yield result, result_anti  # -> 用于保存预测结果
                errors.append(result["error"])
                errors_anti.append(result_anti["error"])

            idxcount += 1
            # 拼接进总记录-误差中
            if mode == 'test':
                mse = nn.functional.mse_loss(antiNormalize_pred, antiNormalize_label)
                self.__mode_dict['mse'] += mse.item()
                mae = nn.functional.l1_loss(antiNormalize_pred, antiNormalize_label)
                self.__mode_dict['mae'] += mae.item()
                # 计算总平方和（SST）
                ss_total = torch.sum((antiNormalize_label - torch.mean(antiNormalize_label)) ** 2)
                # 计算残差平方和（SSE）
                ss_residual = torch.sum((antiNormalize_label - antiNormalize_pred) ** 2)
                # 计算 R²
                r2 = 1 - (ss_residual / ss_total)
                self.__mode_dict['r2'] += r2.item()
                # no-anti-normalize
                mae_normal = nn.functional.l1_loss(results, target.to(self.__device))
                self.__mode_dict['mae_normal'] += mae_normal.item()
                mse_normal = nn.functional.mse_loss(results, target.to(self.__device))
                self.__mode_dict['mse_normal'] += mse_normal.item()

                current_test_error = self.__mode_dict['mae'] / idxcount
                print(
                    f'TEST: {idxcount} / {total_steps}, current time: {time.time() - start_time}, '
                    f'current test error: {current_test_error}')

                # # 保存最后一个数据的偏差图
                # if idxcount == total_steps:
                #     draw_true1pred_img(os.path.join(self.__results_save_path, f'test_true1pred.png'), results[0],
                #                        target[0], self.__series_len)
                draw_true1pred_img(
                    os.path.join(self.__results_save_path, 'test_true1pred', f'test_true1pred{idxcount}_1.png'),
                    results[0], target[0], self.__series_len)
                draw_idx = len(results) // 2
                draw_true1pred_img(
                    os.path.join(self.__results_save_path, 'test_true1pred', f'test_true1pred{idxcount}_2.png'),
                    results[draw_idx], target[draw_idx], self.__series_len)
                draw_true1pred_img(
                    os.path.join(self.__results_save_path, 'test_true1pred', f'test_true1pred{idxcount}_3.png'),
                    results[-1], target[-1], self.__series_len)
            else:
                mse = nn.functional.mse_loss(antiNormalize_pred, antiNormalize_label)
                self.__train_mode_dict['mse'] += mse.item()
                mae = nn.functional.l1_loss(antiNormalize_pred, antiNormalize_label)
                self.__train_mode_dict['mae'] += mae.item()
                # 计算总平方和（SST）
                ss_total = torch.sum((antiNormalize_label - torch.mean(antiNormalize_label)) ** 2)
                # 计算残差平方和（SSE）
                ss_residual = torch.sum((antiNormalize_label - antiNormalize_pred) ** 2)
                # 计算 R²
                r2 = 1 - (ss_residual / ss_total)
                self.__train_mode_dict['r2'] += r2.item()
                # no-anti-normalize
                mae_normal = nn.functional.l1_loss(results, target.to(self.__device))
                self.__train_mode_dict['mae_normal'] += mae_normal.item()
                mse_normal = nn.functional.mse_loss(results, target.to(self.__device))
                self.__train_mode_dict['mse_normal'] += mse_normal.item()
                current_train_error = self.__train_mode_dict['mae'] / idxcount
                print(
                    f'TRAIN: {idxcount} / {total_steps}, current time: {time.time() - start_time}, '
                    f'current test error: {current_train_error}')

                # # 保存最后一个数据的偏差图
                # if idxcount == total_steps:
                #     draw_true1pred_img(os.path.join(self.__results_save_path, 'train_true1pred.png'), results[0],
                #                        target[0], self.__series_len)
                draw_true1pred_img(
                    os.path.join(self.__results_save_path, 'train_true1pred', f'train_true1pred{idxcount}_1.png'),
                    results[0], target[0], self.__series_len)
                draw_idx = len(results) // 2
                draw_true1pred_img(
                    os.path.join(self.__results_save_path, 'train_true1pred', f'train_true1pred{idxcount}_2.png'),
                    results[draw_idx], target[draw_idx], self.__series_len)
                draw_true1pred_img(
                    os.path.join(self.__results_save_path, 'train_true1pred', f'train_true1pred{idxcount}_3.png'),
                    results[-1], target[-1], self.__series_len)

            if idxcount == total_steps:  # 取最后一个batch的第一个数据做一次单例预测
                t_one_sample = temporal_map[0, :, :]
                normal_one_sample = normal_map[0, :, :]
                target_one_sample = target[0, :, :]
                if len(t_one_sample.shape) == 2:
                    t_one_sample = t_one_sample.unsqueeze(0)
                    normal_one_sample = normal_one_sample.unsqueeze(0)
                    target_one_sample = target_one_sample.unsqueeze(0)
                pred_one_sample = self.__net(t_one_sample, normal_one_sample)
                if len(pred_one_sample.shape) == 2:
                    pred_one_sample = pred_one_sample.unsqueeze(0)
                mae_normal = nn.functional.l1_loss(pred_one_sample, target_one_sample)
                mse_normal = nn.functional.mse_loss(pred_one_sample, target_one_sample)
                pred_one_sample = self.__antiNormalize_function.work(pred_one_sample)
                target_one_sample = self.__antiNormalize_function.work(target_one_sample)
                mae = nn.functional.l1_loss(pred_one_sample, target_one_sample)
                mse = nn.functional.mse_loss(pred_one_sample, target_one_sample)
                if mode == 'test':
                    self.__mode_dict['mse_normal_one_sample'] = mse_normal.item()
                    self.__mode_dict['mae_normal_one_sample'] = mae_normal.item()
                    self.__mode_dict['mse_one_sample'] = mse.item()
                    self.__mode_dict['mae_one_sample'] = mae.item()
                else:
                    self.__train_mode_dict['mse_normal_one_sample'] = mse_normal.item()
                    self.__train_mode_dict['mae_normal_one_sample'] = mae_normal.item()
                    self.__train_mode_dict['mse_one_sample'] = mse.item()
                    self.__train_mode_dict['mae_one_sample'] = mae.item()

            del t, target, temporal_map, normal_map, results, antiNormalize_label, antiNormalize_pred, error, interleaved, merged

        # 将误差总和改为误差均值
        if mode == 'test':
            self.__mode_dict['mse'] = self.__mode_dict['mse'] / total_steps
            self.__mode_dict['mae'] = self.__mode_dict['mae'] / total_steps
            self.__mode_dict['r2'] = self.__mode_dict['r2'] / total_steps
            self.__mode_dict['mse_normal'] = self.__mode_dict['mse_normal'] / total_steps
            self.__mode_dict['mae_normal'] = self.__mode_dict['mae_normal'] / total_steps
            self.__mode_dict['P50_normal'] = np.percentile(errors, 50)
            self.__mode_dict['P90_normal'] = np.percentile(errors, 90)
            self.__mode_dict['P50'] = np.percentile(errors_anti, 50)
            self.__mode_dict['P90'] = np.percentile(errors_anti, 90)
        elif mode == 'train':
            self.__train_mode_dict['mse'] = self.__train_mode_dict['mse'] / total_steps
            self.__train_mode_dict['mae'] = self.__train_mode_dict['mae'] / total_steps
            self.__train_mode_dict['r2'] = self.__train_mode_dict['r2'] / total_steps
            self.__train_mode_dict['mse_normal'] = self.__train_mode_dict['mse_normal'] / total_steps
            self.__train_mode_dict['mae_normal'] = self.__train_mode_dict['mae_normal'] / total_steps
            self.__train_mode_dict['P50_normal'] = np.percentile(errors, 50)
            self.__train_mode_dict['P90_normal'] = np.percentile(errors, 90)
            self.__train_mode_dict['P50'] = np.percentile(errors_anti, 50)
            self.__train_mode_dict['P90'] = np.percentile(errors_anti, 90)

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
