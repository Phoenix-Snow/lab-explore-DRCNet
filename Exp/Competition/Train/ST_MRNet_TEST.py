import csv
import json
from openpyxl import Workbook
import os
import time

import pandas as pd
import torch
from torch import nn
from typing import Union, Any, Generator

from ModelRegister.Frequency_Encoder import T_Model
from ModelRegister.DRCNet import Competition_Model
from Tools.Normalizer_1D import StdMeanAntiNormalize, MinMaxAntiNormalize, NoAnti
from Tools.Tools import print_net_params_list, count_flops
from Tools.draw import draw_tsne_by_ss, draw_residuals, draw_tsne_by_sss, draw_tsne_by_feature_sss

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
    def __init__(self, method: int, device: str,
                 antiNormalize_function: Union[MinMaxAntiNormalize, StdMeanAntiNormalize],
                 test_reader, train_reader, spatial_reader, results_save_path: str, target_mode,
                 is_test_train: bool):
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

        self.__t_net = T_Model(len(self.__input_feature_list), len(self.__normal_feature_list), device)
        self.__t_net.eval()
        self.__net = Competition_Model(self.__t_net, len(self.__normal_feature_list), device)
        self.__net.eval()

        self.__antiNormalize_function = antiNormalize_function.to(device)
        self.__antiNormalize_function.cuda_device(device)
        self.__antiNormalize_function.location(self.__output_feature_list)

        self.__test_reader = test_reader
        self.__train_reader = train_reader
        self.__spatial_reader = spatial_reader

        self.__results_save_path = results_save_path
        import time
        time_str = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        self.__results_save_path = os.path.join(results_save_path, time_str)

        del input_feature_list, output_feature_list, normal_feature_list, (
            dataset_input_feature_list), dataset_output_feature_list

        print_net_params_list(self.__net)
        # count_flops(self.__net, (30, 365, ), device)
        print(' all requires_grad -> False (use "with torch.no_grad():" during predicting)')

        self.__hidden_t_results = torch.Tensor([])
        self.__hidden_t_feature = torch.Tensor([])
        self.__hidden_s_feature = torch.Tensor([])
        self.__hidden_g_data = torch.Tensor([])
        self.__results = torch.Tensor([])
        self.__targets = torch.Tensor([])
        self.__g_labels = torch.Tensor([])
        self.__s_w = torch.Tensor([])
        self.__s_b = torch.Tensor([])

        self.__mode_dict = {'predict_mode': target_mode, 'mse': 0, 'mae': 0, 'mse_noAnti': 0, 'mae_noAnti': 0, 'r2': 0}
        self.__train_hidden_t_feature = torch.Tensor([])
        self.__train_hidden_t_results = torch.Tensor([])
        self.__train_hidden_s_feature = torch.Tensor([])
        self.__train_hidden_g_data = torch.Tensor([])
        self.__train_mode_dict = {'predict_mode': target_mode, 'mse': 0, 'mae': 0, 'mse_noAnti': 0, 'mae_noAnti': 0,
                                  'r2': 0}
        self.__train_results = torch.Tensor([])
        self.__train_targets = torch.Tensor([])
        self.__train_g_labels = torch.Tensor([])
        self.__train_s_w = torch.Tensor([])
        self.__train_s_b = torch.Tensor([])

        self.is_test_train = is_test_train

    # def forward(self):
    #     mae_loss_sum = 0
    #     mse_loss_sum = 0
    #     r2_loss_sum = 0
    #     mae_loss_sum_noAnti = 0
    #     mse_loss_sum_noAnti = 0
    #     self.__results = self.__results.to('cpu')
    #     self.__results_noAnti = self.__results_noAnti.to('cpu')
    #
    #     # 加载数据
    #     reader = self.__test_reader
    #     total_steps = len(reader)
    #
    #     # 进行预测
    #     current_time = time.time()
    #     countidx = 0
    #     for t, s_id, target in reader:
    #         mse, mae, r2, mae_noAnti, mse_noAnti = self.predict(t, s_id, target)
    #
    #         # 计入总和
    #         mae_loss_sum += mae.item()
    #         mse_loss_sum += mse.item()
    #         r2_loss_sum += r2.item()
    #         mae_loss_sum_noAnti += mae_noAnti.item()
    #         mse_loss_sum_noAnti += mse_noAnti.item()
    #
    #         current_cost = time.time() - current_time
    #         countidx += 1
    #         print(countidx, '/', total_steps, ', time ', current_cost, ' seconds. current loss: ', mae_loss_sum / countidx)
    #
    #         del t, s_id, target
    #
    #     # 记载评判标准
    #     self.__mode_dict = {'mse': mse_loss_sum / total_steps,
    #                         'mae': mae_loss_sum / total_steps,
    #                         'r2': r2_loss_sum / total_steps,
    #                         'mae_noAnti': mae_loss_sum_noAnti / total_steps,
    #                         'mse_noAnti': mse_loss_sum_noAnti / total_steps, }
    #
    #     # 保存
    #     print('Test Dataset has already been predicted. Now, clean the results and save.')
    #     self.save_results()
    #
    #     print('Test results have already been saved.')

    # def predict(self, t, s_id, target):
    #     temporal_map = t.clone()
    #     normal_map = t.clone()
    #     # partition = torch.Tensor([[torch.nan for _ in range(len(self.__results_head))])
    #     if type(temporal_map[0]) == pd.DataFrame:
    #         # 结构是[DataFrame, DataFrame, DataFrame, DataFrame...]
    #         for i in range(len(temporal_map)):
    #             temporal_map[i] = temporal_map[i][self.__input_feature_list].values
    #             normal_map[i] = normal_map[i][self.__normal_feature_list].values
    #         for i in range(len(target)):
    #             target[i] = target[i][self.__output_feature_list].values
    #     else:
    #         temporal_map = temporal_map[:, :, self.__input_feature_list]
    #         normal_map = normal_map[:, :, self.__normal_feature_list]
    #         target = target[:, :, self.__output_feature_list]
    #     temporal_map, normal_map, target = (torch.Tensor(temporal_map).to(self.__device),
    #                                         torch.Tensor(normal_map).to(self.__device),
    #                                         torch.Tensor(target).to(self.__device))
    #     # print(temporal_map.shape, normal_map.shape, target.shape)
    #
    #     with torch.no_grad():
    #         results = self.__net.forward(temporal_map, s_id, normal_map).to(self.__device)
    #     results = results.reshape(results.shape[0], results.shape[1], 1)
    #     # print(results.shape, target.shape)
    #     error = results - target
    #     interleaved = torch.stack([target, results], dim=-1).view(results.shape[0], results.shape[1], -1)
    #     merged = torch.cat([temporal_map, normal_map, interleaved, error], dim=-1)
    #     # self.__results_noAnti = torch.cat([self.__results_noAnti, merged.to('cpu')], dim=0)
    #
    #
    #     # anti-normalize
    #     antiNormalize_pred = self.__antiNormalize_function.work(results)
    #     antiNormalize_label = self.__antiNormalize_function.work(target.to(self.__device))
    #     error = antiNormalize_pred - antiNormalize_label
    #     interleaved = torch.stack([antiNormalize_label, antiNormalize_pred], dim=-1).view(results.shape[0],
    #                                                                                       results.shape[1], -1)
    #     merged = torch.cat([temporal_map, normal_map, interleaved, error], dim=-1)
    #     self.__results = torch.cat([self.__results, merged.to('cpu')], dim=0)
    #
    #     # error
    #     mse = nn.functional.mse_loss(antiNormalize_pred, antiNormalize_label)
    #     mae = nn.functional.l1_loss(antiNormalize_pred, antiNormalize_label)
    #     # 计算总平方和（SST）
    #     ss_total = torch.sum((antiNormalize_label - torch.mean(antiNormalize_label)) ** 2)
    #     # 计算残差平方和（SSE）
    #     ss_residual = torch.sum((antiNormalize_label - antiNormalize_pred) ** 2)
    #     # 计算 R²
    #     r2 = 1 - (ss_residual / ss_total)
    #     # no-anti-normalize
    #     mae_noAnti = nn.functional.l1_loss(results, target.to(self.__device))
    #     mse_noAnti = nn.functional.mse_loss(results, target.to(self.__device))
    #
    #     del t, s_id, target, temporal_map, normal_map, results, antiNormalize_label, antiNormalize_pred, error, interleaved, merged
    #
    #     return mse, mae, r2, mae_noAnti, mse_noAnti
    #
    #
    # def save_results(self):
    #     time_count = 0
    #     path = self.__results_save_path
    #     while os.path.exists(path):
    #         path = os.path.join(self.__results_save_path, '%04d' % time_count)
    #         time_count += 1
    #
    #     os.makedirs(path, exist_ok=True)
    #     # json保存指标等 mode + evaluate_loss_type + evaluate_loss + evaluate_min_loss
    #     # + mae_loss + mse_loss + r2_loss
    #     with open(os.path.join(path, 'LOG.json'), 'w') as f:
    #         json.dump(self.__mode_dict, f, indent=4)
    #     # csv保存标签和结果
    #     # 清洗异常值
    #     # print(self.__results)  # [sample_num,time_series,feature_results]
    #     self.__results = self.__results.reshape(-1, self.__results.shape[-1])
    #     results_dict = [dict(zip(self.__results_head, row)) for row in self.__results]
    #     with open(os.path.join(path, 'test_RESULTS.csv'), 'w') as f:
    #         writer = csv.DictWriter(f, fieldnames=self.__results_head)
    #         writer.writeheader()
    #         writer.writerows(results_dict)
    #     self.__results_noAnti = self.__results.reshape(-1, self.__results_noAnti.shape[-1])
    #     results_dict = [dict(zip(self.__results_head, row)) for row in self.__results_noAnti]
    #     with open(os.path.join(path, 'test_RESULTS_noAnti.csv'), 'w') as f:
    #         writer = csv.DictWriter(f, fieldnames=self.__results_head)
    #         writer.writeheader()
    #         writer.writerows(results_dict)

    def forward(self, save_file_type: str = 'csv'):
        if save_file_type.lower() == 'csv':
            self.save_csv()
        elif save_file_type.lower() == 'xlsx':
            self.save_xlsx()
        with open(os.path.join(self.__results_save_path, 'LOG.json'), 'w') as f:
            json.dump({'test': self.__mode_dict, 'train': self.__train_mode_dict}, f, indent=4)
        # 接下来绘图时要修改哪些格式
        self.__train_g_labels = self.__train_g_labels.reshape(self.__train_g_labels.shape[0], 1,
                                                              1)  # 增加维度 (sample_num, 1, 1)
        self.__train_g_labels = self.__train_g_labels.repeat(1, self.__train_results.shape[1], 1)  # (sample_num, T, 1)
        # 第一个：各隐向量在目标结果中的作用
        draw_tsne_by_ss(self.__hidden_t_feature, self.__targets,
                        'test : t feature <-> label', self.__results_save_path)
        draw_tsne_by_ss(self.__hidden_t_results, self.__targets,
                        'test : t result <-> label', self.__results_save_path)
        draw_tsne_by_ss(self.__hidden_s_feature, self.__targets,
                        'test : s feature <-> label', self.__results_save_path)
        draw_tsne_by_ss(self.__hidden_g_data, self.__targets,
                        'test : g data <-> label', self.__results_save_path)
        # 第二个：各隐向量在预测结果中的作用
        draw_tsne_by_ss(self.__hidden_t_feature, self.__results,
                        'test : t feature <-> pred result', self.__results_save_path)
        draw_tsne_by_ss(self.__hidden_t_results, self.__results,
                        'test : t result <-> pred result', self.__results_save_path)
        draw_tsne_by_ss(self.__hidden_s_feature, self.__results,
                        'test : s feature <-> pred result', self.__results_save_path)
        draw_tsne_by_ss(self.__hidden_g_data, self.__results,
                        'test : g data <-> pred result', self.__results_save_path)
        # 第三个：以上改train
        draw_tsne_by_ss(self.__train_hidden_t_feature, self.__train_targets,
                        'train : t feature <-> label', self.__results_save_path)
        draw_tsne_by_ss(self.__train_hidden_t_results, self.__train_targets,
                        'train : t result <-> label', self.__results_save_path)
        draw_tsne_by_ss(self.__train_hidden_s_feature, self.__train_targets,
                        'train : s feature <-> label', self.__results_save_path)
        draw_tsne_by_ss(self.__train_hidden_g_data, self.__train_targets,
                        'train : g data <-> label', self.__results_save_path)
        draw_tsne_by_ss(self.__train_hidden_t_feature, self.__train_results,
                        'train : t feature <-> pred result', self.__results_save_path)
        draw_tsne_by_ss(self.__train_hidden_t_results, self.__train_results,
                        'train : t result <-> pred result', self.__results_save_path)
        draw_tsne_by_ss(self.__train_hidden_s_feature, self.__train_results,
                        'train : s feature <-> pred result', self.__results_save_path)
        draw_tsne_by_ss(self.__train_hidden_g_data, self.__train_results,
                        'train : g data <-> pred result', self.__results_save_path)
        # 第四个：两空间隐向量和预测结果在地理上是否有明确区分(因为train是多空间节点的，test只是空间节点的延伸)
        draw_tsne_by_sss(self.__train_hidden_s_feature, self.__train_g_labels,
                         'train : s feature <-> g-env', self.__results_save_path)
        draw_tsne_by_sss(self.__train_results, self.__train_g_labels,
                         'train : pred result <-> g-env', self.__results_save_path)
        draw_tsne_by_sss(self.__train_targets, self.__train_g_labels,
                         'train : label(compare) <-> g-env', self.__results_save_path)
        # 第七个：看时序预测结果是否对时间的预测精准(毕竟单纯时序预测有问题，不能单独先训练冻结，后单独训练空间优化模块，所以是不是这里时序预测是真的在做时序预测)

        # 画误差图
        draw_residuals(self.__results, self.__targets, 'test : plot_btf_residuals', self.__results_save_path)
        draw_residuals(self.__train_results, self.__train_targets, 'train : plot_btf_residuals',
                       self.__results_save_path)
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
        for one_sample_noAnti, one_sample in self.results(reader, reader_phase):
            # 归一化头
            if not header_normal_done:
                sheet_normal.append(list(one_sample_noAnti.keys()))
                header_normal_done = True
            sheet_normal.append(list(one_sample_noAnti.values()))

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
        os.makedirs(self.__results_save_path, exist_ok=True)
        with open(os.path.join(self.__results_save_path, 'test_RESULTS.csv'), 'w', newline='') as f_norm, \
                open(os.path.join(self.__results_save_path, 'test_RESULTS_denormalized.csv'), 'w',
                     newline='') as f_denorm:
            writer_noAnti = None
            writer = None
            for (one_sample_noAnti, one_sample) in self.results(self.__test_reader, 'test'):
                if writer_noAnti is None:
                    writer_noAnti = csv.DictWriter(f_norm, fieldnames=list(one_sample_noAnti.keys()))
                    writer_noAnti.writeheader()
                writer_noAnti.writerow(one_sample_noAnti)
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
                writer_noAnti = None
                writer = None
                for (one_sample_noAnti, one_sample) in self.results(self.__train_reader, 'train'):
                    if writer_noAnti is None:
                        writer_noAnti = csv.DictWriter(f_norm, fieldnames=list(one_sample_noAnti.keys()))
                        writer_noAnti.writeheader()
                    writer_noAnti.writerow(one_sample_noAnti)
                    if writer is None:
                        writer = csv.DictWriter(f_denorm, fieldnames=list(one_sample.keys()))
                        writer.writeheader()
                    writer.writerow(one_sample)
            print('Train has already been saved.\n')

    def results(self, reader, mode='test') -> Generator[tuple[dict[str, Any], dict[str, Any]], Any, None]:
        start_time = time.time()
        total_steps = len(reader)
        idxcount = 0
        for t, s_id, target in reader:
            # 处理数据
            temporal_map = t.clone()
            normal_map = t.clone()
            # partition = torch.Tensor([[torch.nan for _ in range(len(self.__results_head))])
            if isinstance(temporal_map[0], pd.DataFrame):
                # 结构是[DataFrame, DataFrame, DataFrame, DataFrame...]
                for i in range(len(temporal_map)):
                    temporal_map[i] = temporal_map[i][self.__input_feature_list].values
                    normal_map[i] = normal_map[i][self.__normal_feature_list].values
                for i in range(len(target)):
                    target[i] = target[i][self.__output_feature_list].values
            else:
                temporal_map = temporal_map[:, :, self.__input_feature_list]
                normal_map = normal_map[:, :, self.__normal_feature_list]
                target = target[:, :, self.__output_feature_list]
            temporal_map, normal_map, target = (torch.Tensor(temporal_map).to(self.__device),
                                                torch.Tensor(normal_map).to(self.__device),
                                                torch.Tensor(target).to(self.__device))

            # 预测
            with ((torch.no_grad())):
                # 预测隐向量和结果
                hidden_t_feature, hidden_t_results, \
                    hidden_s_feature, hidden_g_data, \
                    s_w, s_b, results, \
                    B, T = self.__net.hidden_params(temporal_map, s_id, normal_map)

                # 保证三维结构
                hidden_t_feature = hidden_t_feature.reshape(B, T, -1)
                hidden_t_results = hidden_t_results.reshape(B, T, -1)
                hidden_s_feature = hidden_s_feature.reshape(B, T, -1)
                hidden_g_data = hidden_g_data.reshape(B, T, -1)
                s_w = s_w.reshape(B, T, -1)
                s_b = s_b.reshape(B, T, -1)
                results = results.reshape(B, T, -1)
                target = target.reshape(B, T, -1)

                # 额外的数据标注：tsne需要，包括时间的标注（季节、月份、闰年等）、地理环境 terrain_labels
                # g_label = torch.tensor([int(terrain_labels[str(s_id[0])])]).to(self.__train_g_labels.device)
                g_label = torch.Tensor([]).to(self.__train_g_labels.device)
                for sid in s_id:  # 一个batch的 s id  （需要挨个根据站点对应其地理环境标签）
                    g_label = torch.cat(
                        [g_label, torch.tensor([int(terrain_labels[str(sid)])]).to(self.__train_g_labels.device)],
                        # 挨个拿地理标签
                        dim=0  # 按第一个维度堆叠
                    )  # (batch_size)
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
                    self.__train_s_w = torch.cat([self.__train_s_w.to("cpu"), s_w.to("cpu")], dim=0)
                    self.__train_s_b = torch.cat([self.__train_s_b.to("cpu"), s_b.to("cpu")], dim=0)
                    self.__train_results = torch.cat([self.__train_results.to("cpu"), results.to("cpu")], dim=0)
                    self.__train_targets = torch.cat([self.__train_targets.to("cpu"), target.to("cpu")], dim=0)
                    self.__train_g_labels = torch.cat([self.__train_g_labels.to("cpu"), g_label.to("cpu")], dim=0)
                elif mode == 'test':
                    self.__hidden_t_feature = torch.cat([self.__hidden_t_feature.to("cpu"),
                                                         hidden_t_feature.to("cpu")], dim=0)
                    self.__hidden_t_results = torch.cat([self.__hidden_t_results.to("cpu"),
                                                         hidden_t_results.to("cpu")], dim=0)
                    self.__hidden_s_feature = torch.cat([self.__hidden_s_feature.to("cpu"),
                                                         hidden_s_feature.to("cpu")], dim=0)
                    self.__hidden_g_data = torch.cat([self.__hidden_g_data.to("cpu"),
                                                      hidden_g_data.to("cpu")], dim=0)
                    self.__s_w = torch.cat([self.__s_w.to("cpu"), s_w.to("cpu")], dim=0)
                    self.__s_b = torch.cat([self.__s_b.to("cpu"), s_b.to("cpu")], dim=0)
                    self.__results = torch.cat([self.__results.to("cpu"), results.to("cpu")], dim=0)
                    self.__targets = torch.cat([self.__targets.to("cpu"), target.to("cpu")], dim=0)
                    self.__g_labels = torch.cat([self.__g_labels.to("cpu"), g_label.to("cpu")], dim=0)

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
                mae_noAnti = nn.functional.l1_loss(results, target.to(self.__device))
                self.__mode_dict['mae_noAnti'] += mae_noAnti.item()
                mse_noAnti = nn.functional.mse_loss(results, target.to(self.__device))
                self.__mode_dict['mse_noAnti'] += mse_noAnti.item()
                current_test_error = self.__mode_dict['mae'] / idxcount
                print(
                    f'TEST: {idxcount} / {total_steps}, current time: {time.time() - start_time}, '
                    f'current test error: {current_test_error}')
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
                mae_noAnti = nn.functional.l1_loss(results, target.to(self.__device))
                self.__train_mode_dict['mae_noAnti'] += mae_noAnti.item()
                mse_noAnti = nn.functional.mse_loss(results, target.to(self.__device))
                self.__train_mode_dict['mse_noAnti'] += mse_noAnti.item()
                current_train_error = self.__train_mode_dict['mae'] / idxcount
                print(
                    f'TRAIN: {idxcount} / {total_steps}, current time: {time.time() - start_time}, '
                    f'current test error: {current_train_error}')

            del t, s_id, target, temporal_map, normal_map, results, antiNormalize_label, antiNormalize_pred, error, interleaved, merged

        # 将误差总和改为误差均值
        if mode == 'test':
            self.__mode_dict['mse'] = self.__mode_dict['mse'] / total_steps
            self.__mode_dict['mae'] = self.__mode_dict['mae'] / total_steps
            self.__mode_dict['r2'] = self.__mode_dict['r2'] / total_steps
            self.__mode_dict['mse_noAnti'] = self.__mode_dict['mse_noAnti'] / total_steps
            self.__mode_dict['mae_noAnti'] = self.__mode_dict['mae_noAnti'] / total_steps
        elif mode == 'train':
            self.__train_mode_dict['mse'] = self.__train_mode_dict['mse'] / total_steps
            self.__train_mode_dict['mae'] = self.__train_mode_dict['mae'] / total_steps
            self.__train_mode_dict['r2'] = self.__train_mode_dict['r2'] / total_steps
            self.__train_mode_dict['mse_noAnti'] = self.__mode_dict['mse_noAnti'] / total_steps
            self.__train_mode_dict['mae_noAnti'] = self.__mode_dict['mae_noAnti'] / total_steps

    def load_model(self, add_spatial_model_path: str):
        self.__net.load_state_dict(torch.load(add_spatial_model_path))

    @property
    def __device__(self):
        return self.__device

    @property
    def eval_loss_threshold(self):
        return self.__eval_loss_threshold

    @property
    def OUTPUT(self):
        return self.__output
