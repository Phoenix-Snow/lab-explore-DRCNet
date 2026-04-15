import yaml

from Integration.Competition.Merge import MergeByYaml


class Params:
    def __init__(self):
        self.__resource = {}
        with open('Integration/Competition/config.yaml', 'r', encoding='utf-8') as file:
            self.__config = yaml.safe_load(file)
            file.close()
        if self.__config is None:
            raise ValueError('Config file is empty or not found.')
        self.__task = self.__config['task'].lower()
        print(f'Task: {self.__task} - ')
        self.__background = self.__config['background']
        self.__device = 'cuda' if self.__config['device'].lower() == 'gpu' else 'cpu'
        print('Device: {}'.format(self.__config['device']))
        if self.__task == 'merge':
            a, b, c, d, e, f = MergeByYaml().work(self.__background['time_series_length'],
                                                  self.__background['spatial_target_size'])
            self.__config['is_merged'] = True
            self.__config['dataset_number'] = a
            self.__resource['station_path_dict'] = b
            self.__resource['station_map_dict'] = c
            self.__resource['station_id_list'] = list(b.keys())
            self.__resource['train_index_list'] = d
            self.__resource['valid_index_list'] = e
            self.__resource['normal_params'] = f
        elif self.__task == 'train':
            self.__config = self.__config['train']
            predict_target = self.__config['predict_target'].lower()
            print(f'target: {predict_target}')
            if not self.__config['merge']['is_merged']:
                a, b, c, d, e, f = MergeByYaml().work(self.__background['time_series_length'],
                                                      self.__background['spatial_target_size'])
                self.__config['merge']['is_merged'] = True
                self.__config['dataset_number'] = a
                self.__resource['station_path_dict'] = b
                self.__resource['station_map_dict'] = c
                self.__resource['station_id_list'] = list(b.keys())
                self.__resource['train_index_list'] = d
                self.__resource['valid_index_list'] = e
                self.__resource['normal_params'] = f
            else:
                a, b, c, d, e, f = MergeByYaml().not_work(self.__config['merge']['dataset_number'])
                self.__config['dataset_number'] = a
                self.__resource['station_path_dict'] = b
                self.__resource['station_map_dict'] = c
                self.__resource['station_id_list'] = list(b.keys())
                self.__resource['train_index_list'] = d
                self.__resource['valid_index_list'] = e
                self.__resource['normal_params'] = f
        elif self.__task == 'test':
            self.__config = self.__config['test']
            predict_target = self.__config['predict_target'].lower()
            print(f'target: {predict_target}')
            print('LOAD TEST')
            if not self.__config['merge']['is_merged']:
                a, b, c, d, e, f = MergeByYaml().work(self.__background['time_series_length'],
                                                      self.__background['spatial_target_size'])
                self.__config['merge']['is_merged'] = True
                self.__config['dataset_number'] = a
                self.__resource['station_path_dict'] = b
                self.__resource['station_map_dict'] = c
                self.__resource['station_id_list'] = {}
                self.__resource['station_id_list']['test'] = list(b.keys())
            else:
                a, b, c, d, e, f = MergeByYaml().not_work(self.__config['merge']['dataset_number'])
                self.__config['dataset_number'] = a
                self.__resource['station_path_dict'] = {}
                self.__resource['station_path_dict']['test'] = b
                self.__resource['station_map_dict'] = c
                self.__resource['station_id_list'] = {}
                self.__resource['station_id_list']['test'] = list(b.keys())
            print('LOAD TRAIN')
            self.__config['merge']['is_merged'] = True
            a, b, c, d, e, f = MergeByYaml().not_work(self.__config['merge']['train_dataset_number'])  # 这一次给的是train的编号
            self.__config['train_dataset_number'] = a
            self.__resource['station_path_dict']['train'] = b  # 合并
            self.__resource['station_map_dict'] = {**self.__resource['station_map_dict'], **c}  # 合并
            # 按照训练、测试字典罗列station_id_list
            self.__resource['station_id_list']['train'] = b.keys()
            self.__resource['normal_params'] = f
        elif self.__task == 'predict':
            self.__config = self.__config['predict']
            if not self.__config['merge']['is_merged']:
                a, b, c, d, e, f = MergeByYaml().work(self.__background['time_series_length'],
                                                      self.__background['spatial_target_size'])
                self.__config['merge']['is_merged'] = True
                self.__config['dataset_number'] = a
                self.__resource['station_path_dict'] = b
                self.__resource['station_map_dict'] = c
                self.__resource['station_id_list'] = list(b.keys())
            else:
                b, c, d, e, f = MergeByYaml().not_work(self.__config['merge']['dataset_number'])
                self.__resource['station_path_dict'] = b
                self.__resource['station_map_dict'] = c
                self.__resource['station_id_list'] = list(b.keys())
        else:
            raise ValueError('Task is not supported.')

    @property
    def TASK(self):
        return self.__task

    @property
    def DEVICE(self):
        return self.__device

    @property
    def CONFIG(self):
        return self.__config

    @property
    def RESOURCE(self):
        return self.__resource

    @property
    def DATASET_NUMBER(self):
        return self.__config['dataset_number']

    @property
    def TIME_SERIES_LENGTH(self):
        return self.__background['time_series_length']

    @property
    def SPATIAL_TARGET_SIZE(self):
        return self.__background['spatial_target_size']

    @property
    def STATION(self):
        return self.__background['stat']
