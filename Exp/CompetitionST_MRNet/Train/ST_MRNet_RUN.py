<<<<<<< Updated upstream:Exp/CompetitionST_MRNet/Train/ST_MRNet_RUN.py
import json
import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from typing import Dict, Union, Any

from ModelRegister.Frequency_Encoder import T_Model
from ModelRegister.DRCNet import Competition_Model
from Integration.ClassCenter import TrainVal_Output
from Tools.Normalizer_1D import StdMeanAntiNormalize, MinMaxAntiNormalize
from Tools.Tools import print_net_params_list


class Procedure(nn.Module):
    def __init__(self, method: int, learning_rate: float, device: str,
                 antiNormalize_function: Union[MinMaxAntiNormalize, StdMeanAntiNormalize], is_output_antiNormal: bool,
                 train_reader, valid_reader, spatial_reader, model_save_path: str):
        super().__init__()
        # 'T_ave', 'T_max', 'T_min', 'AtmospherePressure_avg', 'WindSpeed_avg',
        # 'RelativeHumidity', 'theory_sunDuration', 'SunDuration', 'sunDuration_theoryPlus',
        # 'theory_DailySunRadiation', 'Longitude', 'Latitude', 'Elevation',
        # '3d_x', '3d_y', '3d_z', 'slope', 'aspect',
        input_feature_list = [
            'T_ave', 'T_max', 'T_min', 'AtmospherePressure_avg', 'WindSpeed_avg', 'RelativeHumidity', 'SunDuration',
        ]
        normal_feature_list = [
            'theory_sunDuration', 'theory_DailySunRadiation', 'Longitude', 'Latitude', 'Elevation', 'slope', 'aspect',
        ]
        # 'DailySunRadiation', 'dailySunRadiation_theoryPlus'
        output_feature_list = ['DailySunRadiation']
        self.__eval_loss_threshold = 10

        # dataset
        dataset_input_feature_list = [
            'T_ave', 'T_max', 'T_min', 'AtmospherePressure_avg', 'WindSpeed_avg',
            'RelativeHumidity', 'theory_sunDuration', 'SunDuration', 'sunDuration_theoryPlus',
            'theory_DailySunRadiation', 'Longitude', 'Latitude', 'Elevation',
            '3d_x', '3d_y', '3d_z', 'slope', 'aspect',
        ]
        dataset_output_feature_list = [
            'DailySunRadiation', 'dailySunRadiation_theoryPlus'
        ]
        self.__input_feature_list = [dataset_input_feature_list.index(f) for f in input_feature_list]
        self.__normal_feature_list = [dataset_input_feature_list.index(f) for f in normal_feature_list]
        self.__output_feature_list = [dataset_output_feature_list.index(f) for f in output_feature_list]

        self.__device = device
        self.__t_net = T_Model(len(self.__input_feature_list), len(self.__normal_feature_list), device)
        self.__net = Competition_Model(self.__t_net, len(self.__normal_feature_list), device)
        self.__net_optimizer = torch.optim.Adam(self.__net.parameters(), lr=learning_rate, weight_decay=1e-5)
        # self.__net_optimizer = torch.optim.Adam(self.__net.parameters(), lr=learning_rate)
        self.__net_optimizer.zero_grad()
        self.__trainVal_loss_func = nn.MSELoss().to(device)
        self.__evaluate_loss_func = nn.L1Loss().to(device)
        self.__output = TrainVal_Output()
        self.__antiNormalize_function = antiNormalize_function.to(device)
        self.__antiNormalize_function.cuda_device(device)
        self.__antiNormalize_function.location(self.__output_feature_list)
        # if not is_output_antiNormal:
        #     self.__antiNormalize_function = NoAnti()

        self.__model_save_path = model_save_path
        self.__last_save_path = None

        self.__train_reader = train_reader
        self.__valid_reader = valid_reader
        self.__spatial_reader = spatial_reader

        self.__record = []

        del input_feature_list, output_feature_list, normal_feature_list, (
            dataset_input_feature_list), dataset_output_feature_list

        print('[INFO] model save path:', self.__model_save_path)

        print_net_params_list(self.__net)

    def forward(self, arg: Dict[str, Any]):
        """
        训练模型, 分轮次、批次ReLoad, 需要每轮先运行disorder_dataset_during_train函数
        """
        self.__net.train()

        temporal_map = arg['map'].clone()
        normal_map = arg['map'].clone()
        label = arg['label']
        if type(temporal_map[0]) == pd.DataFrame:
            # 结构是[DataFrame, DataFrame, DataFrame, DataFrame...]
            for i in range(len(temporal_map)):
                temporal_map[i] = temporal_map[i][self.__input_feature_list].values
                normal_map[i] = normal_map[i][self.__normal_feature_list].values
            for i in range(len(label)):
                label[i] = label[i][self.__output_feature_list].values
        else:
            temporal_map = temporal_map[:, :, self.__input_feature_list]
            normal_map = normal_map[:, :, self.__normal_feature_list]
            label = label[:, :, self.__output_feature_list]
        temporal_map, normal_map, label = (torch.Tensor(temporal_map).to(self.__device),
                                           torch.Tensor(normal_map).to(self.__device),
                                           torch.Tensor(label).to(self.__device))
        # spatial_map = self.__spatial_reader[arg['spatial']]

        print('``` Training...')
        self.__output.pred = self.__net.forward(temporal_map, arg['spatial'], normal_map).to(self.__device)
        training_loss = self.__trainVal_loss_func(self.__output.pred.reshape(-1),
                                                  label.reshape(-1))
        self.__net_optimizer.zero_grad()
        training_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.__net.parameters(), max_norm=0.1)
        self.__net_optimizer.step()
        print('Train Already! ```')
        antiNormalize_label = self.__antiNormalize_function.work(label.reshape(-1).to(self.__device))
        self.__output.pred = self.__net.forward(temporal_map, arg['spatial'], normal_map).to('cuda')
        self.__output.pred = self.__antiNormalize_function.work(self.__output.pred.reshape(-1))
        training_loss = self.__evaluate_loss_func(self.__output.pred.reshape(-1), antiNormalize_label.reshape(-1)).item()

        del temporal_map, normal_map, label

        self.__record.append(float(training_loss))

        return training_loss

    def train_predict(self, mode='val'):
        eval_loss_sum = 0
        mae_loss_sum = 0
        mse_loss_sum = 0
        r2_loss_sum = 0
        mae_loss_sum_noAnti = 0
        mse_loss_sum_noAnti = 0

        # 加载数据
        if mode == 'val':
            reader = self.__valid_reader
            total_steps = len(self.__valid_reader)
        elif mode == 'train':
            reader = self.__train_reader
            total_steps = len(self.__train_reader)
        else:
            raise ValueError('[ERROR] Mode Error')

        # 进行预测
        for t, s_id, target in reader:
            temporal_map = t.clone()
            normal_map = t.clone()

            if type(temporal_map[0]) == pd.DataFrame:
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

            self.__output.pred = self.__net.forward(temporal_map, s_id, normal_map).to(self.__device)
            # anti-normalize
            antiNormalize_pred = self.__antiNormalize_function.work(self.__output.pred.reshape(-1))
            antiNormalize_label = self.__antiNormalize_function.work(target.reshape(-1).to(self.__device))
            eval_loss = self.__evaluate_loss_func(antiNormalize_pred, antiNormalize_label)
            mse = nn.functional.mse_loss(antiNormalize_pred, antiNormalize_label)
            mae = nn.functional.l1_loss(antiNormalize_pred, antiNormalize_label)
            # 计算总平方和（SST）
            ss_total = torch.sum((antiNormalize_label - torch.mean(antiNormalize_label)) ** 2)
            # 计算残差平方和（SSE）
            ss_residual = torch.sum((antiNormalize_label - antiNormalize_pred) ** 2)
            # 计算 R²
            r2 = 1 - (ss_residual / ss_total)
            # no-anti-normalize
            mae_noAnti = nn.functional.l1_loss(self.__output.pred.reshape(-1),
                                               target.reshape(-1).to(self.__device))
            mse_noAnti = nn.functional.mse_loss(self.__output.pred.reshape(-1),
                                                target.reshape(-1).to(self.__device))
            # 计入总和
            eval_loss_sum += eval_loss.item()
            mae_loss_sum += mae.item()
            mse_loss_sum += mse.item()
            r2_loss_sum += r2.item()
            mae_loss_sum_noAnti += mae_noAnti.item()
            mse_loss_sum_noAnti += mse_noAnti.item()

        # 返回评判标准
        return {'eval': eval_loss_sum / total_steps,
                'mse': mse_loss_sum / total_steps,
                'mae': mae_loss_sum / total_steps,
                'r2': r2_loss_sum / total_steps,
                'mae_noAnti': mae_loss_sum_noAnti / total_steps,
                'mse_noAnti': mse_loss_sum_noAnti / total_steps, }

    def save_results(self, path: str, mode_dict: dict):
        # 当前时间
        import time
        time_str = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        # print(path, time_str)
        desc = 'timesnet_modified_add_spatial'
        mode = mode_dict['mode'] + '_' + str(mode_dict['eval_loss'])
        path = os.path.join(path, f'{desc}', time_str, mode)
        os.makedirs(path, exist_ok=True)
        # 模型保存 mode + evaluate_loss
        self.save_model(path)
        # json保存 mode + evaluate_loss_type + evaluate_loss + evaluate_min_loss
        # + mae_loss + mse_loss + r2_loss
        with open(os.path.join(path, 'LOG.json'), 'w') as f:
            json.dump(mode_dict, f, indent=4)
        # self.__record.append([mode_dict['epoch'], mode_dict['batch'],
        #                       mode_dict['eval_loss'], mode_dict['eval_min_loss']])
        # if len(self.__record) >= 7 and self.__record[-1][0] - self.__record[1][0] >= 7:
        if len(self.__record) >= 10:
            y_train_loss = self.__record.copy()
            # if len(self.__record) > 2999:
            #     y_train_loss = y_train_loss[-2999:]
            if True:
                y_train_loss.append(float(mode_dict['eval_loss']))
                y_train_loss = np.asarray(y_train_loss, float)  # loss值, 即y轴
                x_train_loss = range(len(y_train_loss))  # loss的数量, 即x轴

                plt.figure()

                # 去除顶部和右边框框
                ax = plt.axes()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                plt.xlabel('iters')  # x轴标签
                plt.ylabel('loss(anti-normalization)')  # y轴标签

                # 以x_train_loss为横坐标, y_train_loss为纵坐标, 曲线宽度为1, 实线, 增加标签, 训练损失, 
                plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
                plt.legend()
                plt.title('Loss curve')
                plt.savefig(os.path.join(path, 'loss_decline_chart.png'))
                # plt.show()
                plt.close('all')
            if True:
                y_train_loss[-1] = float(mode_dict['eval_min_loss'])  # loss值, 即y轴 (替换最后一个值为最小值)
                x_train_loss = range(len(y_train_loss))  # loss的数量, 即x轴

                plt.figure()

                # 去除顶部和右边框框
                ax = plt.axes()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                plt.xlabel('iters')  # x轴标签
                plt.ylabel('loss')  # y轴标签

                # 以x_train_loss为横坐标, y_train_loss为纵坐标, 曲线宽度为1, 实线, 增加标签, 训练损失, 
                plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
                plt.legend()
                plt.title('Loss curve')
                plt.savefig(os.path.join(path, 'min_loss_decline_chart.png'))
                # plt.show()
                plt.close('all')
            del y_train_loss

    def save_model(self, base_path: str, desc: str | None = None):
        if desc is None:
            torch.save(self.__net.state_dict(), os.path.join(base_path, 'params.pth'))
            torch.save(self.__net, os.path.join(base_path, 'model.pth'))
        else:
            torch.save(self.__net.state_dict(), os.path.join(base_path, f'params_{desc}.pth'))
            torch.save(self.__net, os.path.join(base_path, f'model_{desc}.pth'))

    def load_model(self, add_spatial_model_path: str = None):
        if add_spatial_model_path:
            self.__net.load_state_dict(torch.load(add_spatial_model_path))
            self.__output.min_loss = float(add_spatial_model_path.split('/')[-2].split('_')[-1])
            self.__eval_loss_threshold = self.__output.min_loss
            self.__last_save_path = add_spatial_model_path


    def evaluate(self, epoch_num: int, batch_num: int):
        self.__net.eval()
        print('``` Evaluation...')
        # 训练集预测
        train_predict_eval: dict = self.train_predict('train')
        # 验证集预测
        val_predict_eval = self.train_predict('val')
        # 打印信息
        self.__output.train_loss = train_predict_eval['eval']
        self.__output.val_loss = val_predict_eval['eval']
        print(f' Train Loss: {self.__output.train_loss}, '
              f' Val Loss: {self.__output.val_loss}')
        # 确认是否保存模型
        mode = 'Overfitting'
        self.__output.loss = self.__output.train_loss * 0.8 + self.__output.val_loss * 0.2
        print(f'                             Current Loss: {self.__output.loss}')
        if self.__output.val_loss <= self.__output.train_loss:
            mode = 'Underfitting'
        mode_dict = {'epoch': epoch_num,
                     'batch': batch_num,
                     'mode': mode,
                     'eval_loss': self.__output.loss,
                     'eval_min_loss': self.__output.min_loss,
                     'mae': train_predict_eval["mae"] * 0.8 + val_predict_eval["mae"] * 0.2,
                     'mse': train_predict_eval["mse"] * 0.8 + val_predict_eval["mse"] * 0.2,
                     'r2': train_predict_eval["r2"] * 0.8 + val_predict_eval["r2"] * 0.2,
                     'mae_noAnti': train_predict_eval["mae_noAnti"] * 0.8 + val_predict_eval["mae_noAnti"] * 0.2,
                     'mse_noAnti': train_predict_eval["mse_noAnti"] * 0.8 + val_predict_eval["mse_noAnti"] * 0.2, }
        if mode == 'Underfitting' and self.__output.loss < self.__output.min_loss:
            print(self.__output.min_loss)
            self.__output.update_min_loss()
            self.save_results(self.__model_save_path, mode_dict)
            print('                            Underfitting, Saving Model!')
        else:
            if (self.__output.loss < self.__output.min_loss
                    and self.__output.val_loss / self.__output.train_loss < 1.01):
                self.save_results(self.__model_save_path, mode_dict)
                print('                            Overfitting, but not too much, Saving Model!')
        if self.__output.min_loss != 10000000:
            print(f'                             Min Loss: {self.__output.min_loss}')
        print('\n\n')

    def calc_loss(self):
        loss_cache = 0

        # 加载数据
        reader = self.__test_reader
        total_steps = len(self.__train_reader)

        # 进行预测
        for t, s_id, target in reader:
            temporal_map = t.clone()
            normal_map = t.clone()

            if type(temporal_map[0]) == pd.DataFrame:
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

            self.__output.pred = self.__net.forward(temporal_map, s_id, normal_map).to(self.__device)
            self.__output.pred = self.__antiNormalize_function.work(self.__output.pred.reshape(-1))
            antiNormalize_label = self.__antiNormalize_function.work(target.reshape(-1).to(self.__device))
            antiNormal_loss = self.__evaluate_loss_func(self.__output.pred, antiNormalize_label)
            loss_cache += antiNormal_loss.item()

        # 返回评判标准
        return loss_cache / total_steps

    @property
    def __device__(self):
        return self.__device

    @property
    def eval_loss_threshold(self):
        return self.__eval_loss_threshold

    @property
    def OUTPUT(self):
        return self.__output

    @property
    def model_path(self):
        return self.__last_save_path

    @model_path.setter
    def model_path(self, value):
        self.__last_save_path = value
=======
import json
import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from typing import Dict, Union, Any

from ModelRegister.Frequency_Encoder import T_Model
from ModelRegister.DRCNet import Competition_Model
from Tools.ClassCenter import TrainVal_Output
from Tools.Normalizer_1D import StdMeanAntiNormalize, MinMaxAntiNormalize
from Tools.Tools import print_net_params_list


class Procedure(nn.Module):
    def __init__(self, method: int, learning_rate: float, device: str,
                 antiNormalize_function: Union[MinMaxAntiNormalize, StdMeanAntiNormalize], is_output_antiNormal: bool,
                 train_reader, valid_reader, spatial_reader, model_save_path: str):
        super().__init__()
        # 'T_ave', 'T_max', 'T_min', 'AtmospherePressure_avg', 'WindSpeed_avg',
        # 'RelativeHumidity', 'theory_sunDuration', 'SunDuration', 'sunDuration_theoryPlus',
        # 'theory_DailySunRadiation', 'Longitude', 'Latitude', 'Elevation',
        # '3d_x', '3d_y', '3d_z', 'slope', 'aspect',
        input_feature_list = [
            'T_ave', 'T_max', 'T_min', 'AtmospherePressure_avg', 'WindSpeed_avg', 'RelativeHumidity', 'SunDuration',
        ]
        normal_feature_list = [
            'theory_sunDuration', 'theory_DailySunRadiation', 'Longitude', 'Latitude', 'Elevation', 'slope', 'aspect',
        ]
        # 'DailySunRadiation', 'dailySunRadiation_theoryPlus'
        output_feature_list = ['DailySunRadiation']
        self.__eval_loss_threshold = 10

        # dataset
        dataset_input_feature_list = [
            'T_ave', 'T_max', 'T_min', 'AtmospherePressure_avg', 'WindSpeed_avg',
            'RelativeHumidity', 'theory_sunDuration', 'SunDuration', 'sunDuration_theoryPlus',
            'theory_DailySunRadiation', 'Longitude', 'Latitude', 'Elevation',
            '3d_x', '3d_y', '3d_z', 'slope', 'aspect',
        ]
        dataset_output_feature_list = [
            'DailySunRadiation', 'dailySunRadiation_theoryPlus'
        ]
        self.__input_feature_list = [dataset_input_feature_list.index(f) for f in input_feature_list]
        self.__normal_feature_list = [dataset_input_feature_list.index(f) for f in normal_feature_list]
        self.__output_feature_list = [dataset_output_feature_list.index(f) for f in output_feature_list]

        self.__device = device
        self.__t_net = T_Model(len(self.__input_feature_list), len(self.__normal_feature_list), device)
        self.__net = Competition_Model(self.__t_net, len(self.__normal_feature_list), device)
        self.__net_optimizer = torch.optim.Adam(self.__net.parameters(), lr=learning_rate, weight_decay=1e-5)
        # self.__net_optimizer = torch.optim.Adam(self.__net.parameters(), lr=learning_rate)
        self.__net_optimizer.zero_grad()
        self.__trainVal_loss_func = nn.MSELoss().to(device)
        self.__evaluate_loss_func = nn.L1Loss().to(device)
        self.__output = TrainVal_Output()
        self.__antiNormalize_function = antiNormalize_function.to(device)
        self.__antiNormalize_function.cuda_device(device)
        self.__antiNormalize_function.location(self.__output_feature_list)
        # if not is_output_antiNormal:
        #     self.__antiNormalize_function = NoAnti()

        self.__model_save_path = model_save_path
        self.__last_save_path = None

        self.__train_reader = train_reader
        self.__valid_reader = valid_reader
        self.__spatial_reader = spatial_reader

        self.__record = []

        del input_feature_list, output_feature_list, normal_feature_list, (
            dataset_input_feature_list), dataset_output_feature_list

        print('[INFO] model save path:', self.__model_save_path)

        print_net_params_list(self.__net)

    def forward(self, arg: Dict[str, Any]):
        """
        训练模型, 分轮次、批次ReLoad, 需要每轮先运行disorder_dataset_during_train函数
        """
        self.__net.train()

        temporal_map = arg['map'].clone()
        normal_map = arg['map'].clone()
        label = arg['label']
        if type(temporal_map[0]) == pd.DataFrame:
            # 结构是[DataFrame, DataFrame, DataFrame, DataFrame...]
            for i in range(len(temporal_map)):
                temporal_map[i] = temporal_map[i][self.__input_feature_list].values
                normal_map[i] = normal_map[i][self.__normal_feature_list].values
            for i in range(len(label)):
                label[i] = label[i][self.__output_feature_list].values
        else:
            temporal_map = temporal_map[:, :, self.__input_feature_list]
            normal_map = normal_map[:, :, self.__normal_feature_list]
            label = label[:, :, self.__output_feature_list]
        temporal_map, normal_map, label = (torch.Tensor(temporal_map).to(self.__device),
                                           torch.Tensor(normal_map).to(self.__device),
                                           torch.Tensor(label).to(self.__device))
        # spatial_map = self.__spatial_reader[arg['spatial']]

        print('``` Training...')
        self.__output.pred = self.__net.forward(temporal_map, arg['spatial'], normal_map).to(self.__device)
        training_loss = self.__trainVal_loss_func(self.__output.pred.reshape(-1),
                                                  label.reshape(-1))
        self.__net_optimizer.zero_grad()
        training_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.__net.parameters(), max_norm=0.1)
        self.__net_optimizer.step()
        print('Train Already! ```')
        antiNormalize_label = self.__antiNormalize_function.work(label.reshape(-1).to(self.__device))
        self.__output.pred = self.__net.forward(temporal_map, arg['spatial'], normal_map).to('cuda')
        self.__output.pred = self.__antiNormalize_function.work(self.__output.pred.reshape(-1))
        training_loss = self.__evaluate_loss_func(self.__output.pred.reshape(-1), antiNormalize_label.reshape(-1)).item()

        del temporal_map, normal_map, label

        self.__record.append(float(training_loss))

        return training_loss

    def train_predict(self, mode='val'):
        eval_loss_sum = 0
        mae_loss_sum = 0
        mse_loss_sum = 0
        r2_loss_sum = 0
        mae_loss_sum_noAnti = 0
        mse_loss_sum_noAnti = 0

        # 加载数据
        if mode == 'val':
            reader = self.__valid_reader
            total_steps = len(self.__valid_reader)
        elif mode == 'train':
            reader = self.__train_reader
            total_steps = len(self.__train_reader)
        else:
            raise ValueError('[ERROR] Mode Error')

        # 进行预测
        for t, s_id, target in reader:
            temporal_map = t.clone()
            normal_map = t.clone()

            if type(temporal_map[0]) == pd.DataFrame:
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

            self.__output.pred = self.__net.forward(temporal_map, s_id, normal_map).to(self.__device)
            # anti-normalize
            antiNormalize_pred = self.__antiNormalize_function.work(self.__output.pred.reshape(-1))
            antiNormalize_label = self.__antiNormalize_function.work(target.reshape(-1).to(self.__device))
            eval_loss = self.__evaluate_loss_func(antiNormalize_pred, antiNormalize_label)
            mse = nn.functional.mse_loss(antiNormalize_pred, antiNormalize_label)
            mae = nn.functional.l1_loss(antiNormalize_pred, antiNormalize_label)
            # 计算总平方和（SST）
            ss_total = torch.sum((antiNormalize_label - torch.mean(antiNormalize_label)) ** 2)
            # 计算残差平方和（SSE）
            ss_residual = torch.sum((antiNormalize_label - antiNormalize_pred) ** 2)
            # 计算 R²
            r2 = 1 - (ss_residual / ss_total)
            # no-anti-normalize
            mae_noAnti = nn.functional.l1_loss(self.__output.pred.reshape(-1),
                                               target.reshape(-1).to(self.__device))
            mse_noAnti = nn.functional.mse_loss(self.__output.pred.reshape(-1),
                                                target.reshape(-1).to(self.__device))
            # 计入总和
            eval_loss_sum += eval_loss.item()
            mae_loss_sum += mae.item()
            mse_loss_sum += mse.item()
            r2_loss_sum += r2.item()
            mae_loss_sum_noAnti += mae_noAnti.item()
            mse_loss_sum_noAnti += mse_noAnti.item()

        # 返回评判标准
        return {'eval': eval_loss_sum / total_steps,
                'mse': mse_loss_sum / total_steps,
                'mae': mae_loss_sum / total_steps,
                'r2': r2_loss_sum / total_steps,
                'mae_noAnti': mae_loss_sum_noAnti / total_steps,
                'mse_noAnti': mse_loss_sum_noAnti / total_steps, }

    def save_results(self, path: str, mode_dict: dict):
        # 当前时间
        import time
        time_str = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        # print(path, time_str)
        desc = 'timesnet_modified_add_spatial'
        mode = mode_dict['mode'] + '_' + str(mode_dict['eval_loss'])
        path = os.path.join(path, f'{desc}', time_str, mode)
        os.makedirs(path, exist_ok=True)
        # 模型保存 mode + evaluate_loss
        self.save_model(path)
        # json保存 mode + evaluate_loss_type + evaluate_loss + evaluate_min_loss
        # + mae_loss + mse_loss + r2_loss
        with open(os.path.join(path, 'LOG.json'), 'w') as f:
            json.dump(mode_dict, f, indent=4)
        # self.__record.append([mode_dict['epoch'], mode_dict['batch'],
        #                       mode_dict['eval_loss'], mode_dict['eval_min_loss']])
        # if len(self.__record) >= 7 and self.__record[-1][0] - self.__record[1][0] >= 7:
        if len(self.__record) >= 10:
            y_train_loss = self.__record.copy()
            # if len(self.__record) > 2999:
            #     y_train_loss = y_train_loss[-2999:]
            if True:
                y_train_loss.append(float(mode_dict['eval_loss']))
                y_train_loss = np.asarray(y_train_loss, float)  # loss值, 即y轴
                x_train_loss = range(len(y_train_loss))  # loss的数量, 即x轴

                plt.figure()

                # 去除顶部和右边框框
                ax = plt.axes()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                plt.xlabel('iters')  # x轴标签
                plt.ylabel('loss(anti-normalization)')  # y轴标签

                # 以x_train_loss为横坐标, y_train_loss为纵坐标, 曲线宽度为1, 实线, 增加标签, 训练损失, 
                plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
                plt.legend()
                plt.title('Loss curve')
                plt.savefig(os.path.join(path, 'loss_decline_chart.png'))
                # plt.show()
                plt.close('all')
            if True:
                y_train_loss[-1] = float(mode_dict['eval_min_loss'])  # loss值, 即y轴 (替换最后一个值为最小值)
                x_train_loss = range(len(y_train_loss))  # loss的数量, 即x轴

                plt.figure()

                # 去除顶部和右边框框
                ax = plt.axes()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                plt.xlabel('iters')  # x轴标签
                plt.ylabel('loss')  # y轴标签

                # 以x_train_loss为横坐标, y_train_loss为纵坐标, 曲线宽度为1, 实线, 增加标签, 训练损失, 
                plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
                plt.legend()
                plt.title('Loss curve')
                plt.savefig(os.path.join(path, 'min_loss_decline_chart.png'))
                # plt.show()
                plt.close('all')
            del y_train_loss

    def save_model(self, base_path: str, desc: str | None = None):
        if desc is None:
            torch.save(self.__net.state_dict(), os.path.join(base_path, 'params.pth'))
            torch.save(self.__net, os.path.join(base_path, 'model.pth'))
        else:
            torch.save(self.__net.state_dict(), os.path.join(base_path, f'params_{desc}.pth'))
            torch.save(self.__net, os.path.join(base_path, f'model_{desc}.pth'))

    def load_model(self, add_spatial_model_path: str = None):
        if add_spatial_model_path:
            self.__net.load_state_dict(torch.load(add_spatial_model_path))
            self.__output.min_loss = float(add_spatial_model_path.split('/')[-2].split('_')[-1])
            self.__eval_loss_threshold = self.__output.min_loss
            self.__last_save_path = add_spatial_model_path


    def evaluate(self, epoch_num: int, batch_num: int):
        self.__net.eval()
        print('``` Evaluation...')
        # 训练集预测
        train_predict_eval: dict = self.train_predict('train')
        # 验证集预测
        val_predict_eval = self.train_predict('val')
        # 打印信息
        self.__output.train_loss = train_predict_eval['eval']
        self.__output.val_loss = val_predict_eval['eval']
        print(f' Train Loss: {self.__output.train_loss}, '
              f' Val Loss: {self.__output.val_loss}')
        # 确认是否保存模型
        mode = 'Overfitting'
        self.__output.loss = self.__output.train_loss * 0.8 + self.__output.val_loss * 0.2
        print(f'                             Current Loss: {self.__output.loss}')
        if self.__output.val_loss <= self.__output.train_loss:
            mode = 'Underfitting'
        mode_dict = {'epoch': epoch_num,
                     'batch': batch_num,
                     'mode': mode,
                     'eval_loss': self.__output.loss,
                     'eval_min_loss': self.__output.min_loss,
                     'mae': train_predict_eval["mae"] * 0.8 + val_predict_eval["mae"] * 0.2,
                     'mse': train_predict_eval["mse"] * 0.8 + val_predict_eval["mse"] * 0.2,
                     'r2': train_predict_eval["r2"] * 0.8 + val_predict_eval["r2"] * 0.2,
                     'mae_noAnti': train_predict_eval["mae_noAnti"] * 0.8 + val_predict_eval["mae_noAnti"] * 0.2,
                     'mse_noAnti': train_predict_eval["mse_noAnti"] * 0.8 + val_predict_eval["mse_noAnti"] * 0.2, }
        if mode == 'Underfitting' and self.__output.loss < self.__output.min_loss:
            print(self.__output.min_loss)
            self.__output.update_min_loss()
            self.save_results(self.__model_save_path, mode_dict)
            print('                            Underfitting, Saving Model!')
        else:
            if (self.__output.loss < self.__output.min_loss
                    and self.__output.val_loss / self.__output.train_loss < 1.01):
                self.save_results(self.__model_save_path, mode_dict)
                print('                            Overfitting, but not too much, Saving Model!')
        if self.__output.min_loss != 10000000:
            print(f'                             Min Loss: {self.__output.min_loss}')
        print('\n\n')

    def calc_loss(self):
        loss_cache = 0

        # 加载数据
        reader = self.__test_reader
        total_steps = len(self.__train_reader)

        # 进行预测
        for t, s_id, target in reader:
            temporal_map = t.clone()
            normal_map = t.clone()

            if type(temporal_map[0]) == pd.DataFrame:
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

            self.__output.pred = self.__net.forward(temporal_map, s_id, normal_map).to(self.__device)
            self.__output.pred = self.__antiNormalize_function.work(self.__output.pred.reshape(-1))
            antiNormalize_label = self.__antiNormalize_function.work(target.reshape(-1).to(self.__device))
            antiNormal_loss = self.__evaluate_loss_func(self.__output.pred, antiNormalize_label)
            loss_cache += antiNormal_loss.item()

        # 返回评判标准
        return loss_cache / total_steps

    @property
    def __device__(self):
        return self.__device

    @property
    def eval_loss_threshold(self):
        return self.__eval_loss_threshold

    @property
    def OUTPUT(self):
        return self.__output

    @property
    def model_path(self):
        return self.__last_save_path

    @model_path.setter
    def model_path(self, value):
        self.__last_save_path = value
>>>>>>> Stashed changes:Exp/Competition/Train/ST_MRNet_RUN.py
