import torch
from torch import nn
from typing import Union

from ModelRegister.Frequency_Encoder import T_Model
from ModelRegister.DRCNet import TS_Model
from Integration.ClassCenter import TrainVal_Output
from Tools.Normalizer_1D import StdMeanAntiNormalize, MinMaxAntiNormalize, NoAnti


class Procedure(nn.Module):
    def __init__(self, device4predict: str, model_path: str,
                 antiNormalize_function: Union[MinMaxAntiNormalize, StdMeanAntiNormalize],
                 is_output_antiNormal: bool):
        super().__init__()
        input_feature_list = [
            'T_ave', 'T_max', 'T_min', 'AtmospherePressure_avg', 'WindSpeed_avg', 'RelativeHumidity', 'SunDuration',
        ]
        normal_feature_list = [
            'theory_sunDuration', 'theory_DailySunRadiation', 'Longitude', 'Latitude', 'Elevation', 'slope', 'aspect',
        ]
        # 'DailySunRadiation', 'dailySunRadiation_theoryPlus'
        output_feature_list = ['DailySunRadiation']
        self.current_step = 2

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

        self.__device = device4predict

        if self.current_step == 1:
            self.__net = T_Model(len(self.__input_feature_list), len(self.__normal_feature_list), device4predict)
            self.__net.eval()
        else:
            self.__t_net = T_Model(len(self.__input_feature_list), len(self.__normal_feature_list), device4predict)
            self.__t_net.eval()
            self.__net = TS_Model(self.__t_net, len(self.__normal_feature_list), device4predict)
            self.__net.eval()
        self.__evaluate_loss_func = nn.L1Loss().to(device4predict)
        self.__output = TrainVal_Output()
        self.__antiNormalize_function = antiNormalize_function.to(device4predict)
        self.__antiNormalize_function.cuda_device4predict(device4predict)
        self.__antiNormalize_function.location(self.__output_feature_list)
        if not is_output_antiNormal:
            self.__antiNormalize_function = NoAnti()

        self.__model_path = model_path

    def forward(self, input_data, normal_map, station_id, batch_size):
        total_steps = len(input_data) // batch_size
        if len(input_data) % batch_size != 0:
            total_steps += 1
        for idx in len(input_data) // batch_size:
            temporal_map, normal_map = (
                torch.Tensor(input_data[idx:idx + batch_size]).to(self.__device),
                torch.Tensor(normal_map[idx:idx + batch_size]).to(self.__device))
            pred = self.__net.forward(temporal_map, station_id, normal_map).to(self.__device)
        return pred

    def load_model(self, step1_model_path: str, step2_model_path: str = None):
        if self.current_step == 1:
            self.__net.load_state_dict(torch.load(step1_model_path))
        elif self.current_step == 2:
            self.__net.load_state_dict(torch.load(step2_model_path))

    @property
    def __device4predict__(self):
        return self.__device

    @property
    def eval_loss_threshold(self):
        return self.__eval_loss_threshold

    @property
    def OUTPUT(self):
        return self.__output
