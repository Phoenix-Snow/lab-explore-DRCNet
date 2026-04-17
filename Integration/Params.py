import yaml

<<<<<<<< Updated upstream:Integration/privateDataset/Params.py
from Integration.privateDataset.Merge import MergeByYaml
========
from Integration.DRCNet.Merge import MergeByYaml
>>>>>>>> Stashed changes:Integration/Params.py


class Params:
    def __init__(self):
        self.__resource = {}
<<<<<<<< Updated upstream:Integration/privateDataset/Params.py
        with open('Integration/privateDataset/config.yaml', 'r', encoding='utf-8') as file:
========
        with open('Integration/DRCNet/config.yaml', 'r', encoding='utf-8') as file:
>>>>>>>> Stashed changes:Integration/Params.py
            self.__config = yaml.safe_load(file)
            file.close()
        if self.__config is None:
            raise ValueError('Config file is empty or not found.')
        self.__task = self.__config['task'].lower()
        print(f'Task: {self.__task}')
<<<<<<<< Updated upstream:Integration/privateDataset/Params.py
        self.__background = self.__config['background']
        self.__device = 'cuda' if self.__config['device'].lower() == 'gpu' else 'cpu'
        print('Device: {}'.format(self.__config['device']))
        if self.__task == 'merge':
            self.__config = self.__config['merge']
            a, b, c, d, e, f = MergeByYaml().work(self.__background['time_series_length'],
                                                  self.__background['spatial_target_size'],
                                                  self.__config['temporal_series_path'])
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
========
        self.__background = {}
        self.__split_columns = self.__config['feature_split']
        self.__device = 'cuda' if self.__config['device'].lower() == 'gpu' else 'cpu'
        print('Device: {}'.format(self.__config['device']))
        if self.__task == 'merge':
            self.__config = {}
            a, b, d, e, f, g = MergeByYaml().load_config('work')  # 按照merge的配置来
            self.__config['is_merged'] = True
            self.__config['dataset_name'] = a
            self.__resource['station_path_dict'] = b
            self.__resource['station_id_list'] = list(b.keys())
            self.__resource['train_list'] = d
            self.__resource['valid_list'] = e
            self.__resource['normal_params'] = f
            self.__background = g
        elif self.__task == 'train':  # 按照主程序的config配置来
            self.__config = self.__config['train']
            if not self.__config['merge']['is_merged']:
                a, b, d, e, f, g = MergeByYaml().load_config('work', self.__config['merge']['dataset_name'])
                self.__config['merge']['is_merged'] = True
                self.__config['dataset_name'] = a
                self.__resource['station_path_dict'] = b
                self.__resource['station_id_list'] = list(b.keys())
                self.__resource['train_list'] = d
                self.__resource['valid_list'] = e
                self.__resource['normal_params'] = f
                self.__background = g
            else:
                a, b, d, e, f, g = MergeByYaml().load_config('not_work', self.__config['merge']['dataset_name'])
                self.__config['dataset_name'] = a
                self.__resource['station_path_dict'] = b
                self.__resource['station_id_list'] = list(b.keys())
                self.__resource['train_list'] = d
                self.__resource['valid_list'] = e
                self.__resource['normal_params'] = f
                self.__background = g
>>>>>>>> Stashed changes:Integration/Params.py
        elif self.__task == 'test':
            self.__config = self.__config['test']
            print('LOAD TEST')
            if not self.__config['merge']['is_merged']:
<<<<<<<< Updated upstream:Integration/privateDataset/Params.py
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
========
                a, b, d, e, f, g = MergeByYaml().load_config('work', self.__config['merge']['dataset_name'])
                self.__config['merge']['is_merged'] = True
                self.__config['dataset_name'] = a
                self.__resource['station_path_dict'] = b
                self.__resource['station_id_list'] = {}
                self.__resource['station_id_list']['test'] = list(b.keys())
                self.__background = g
            else:
                a, b, d, e, f, g = MergeByYaml().load_config('not_work', self.__config['merge']['dataset_name'])
                self.__config['dataset_name'] = a
                self.__resource['station_path_dict'] = {}
                self.__resource['station_path_dict']['test'] = b
                self.__resource['station_id_list'] = {}
                self.__resource['station_id_list']['test'] = list(b.keys())
                self.__background = g
            print('LOAD TRAIN')
            self.__config['merge']['is_merged'] = True
            a, b, d, e, f, g = MergeByYaml().load_config('not_work', self.__config['merge']['train_dataset_name'])  # 这一次给的是train的编号
            self.__config['train_dataset_name'] = a
            self.__resource['station_path_dict']['train'] = b  # 合并
            # 按照训练、测试字典罗列station_id_list
            self.__resource['station_id_list']['train'] = b.keys()
            self.__resource['normal_params'] = f
            self.__background = g
        elif self.__task == 'predict':
            self.__config = self.__config['predict']
            if not self.__config['merge']['is_merged']:
                a, b, d, e, f, g = MergeByYaml().load_config('work', self.__config['merge']['dataset_name'])
                self.__config['merge']['is_merged'] = True
                self.__config['dataset_name'] = a
                self.__resource['station_path_dict'] = b
                self.__resource['station_id_list'] = list(b.keys())
                self.__background = g
            else:
                a, b, d, e, f, g = MergeByYaml().load_config('not_work', self.__config['merge']['dataset_name'])
                self.__resource['station_path_dict'] = b
                self.__resource['station_id_list'] = list(b.keys())
                self.__background = g
>>>>>>>> Stashed changes:Integration/Params.py
        else:
            raise ValueError('Task is not supported.')

    @property
<<<<<<<< Updated upstream:Integration/privateDataset/Params.py
========
    def NAME(self):
        return self.__config['dataset_name']

    @property
>>>>>>>> Stashed changes:Integration/Params.py
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
<<<<<<<< Updated upstream:Integration/privateDataset/Params.py
    def DATASET_NUMBER(self):
        return self.__config['dataset_number']

    @property
========
>>>>>>>> Stashed changes:Integration/Params.py
    def TIME_SERIES_LENGTH(self):
        return self.__background['time_series_length']

    @property
<<<<<<<< Updated upstream:Integration/privateDataset/Params.py
    def SPATIAL_TARGET_SIZE(self):
        return self.__background['spatial_target_size']

    @property
    def STATION(self):
        return self.__background['stat']
========
    def STATION(self):
        return self.__background['station_3d']

    @property
    def BACKGROUND(self):
        return self.__background

    @property
    def loc_column_date(self):
        return self.__split_columns['date_column']
    
    @property
    def in_columns_1st(self):
        return self.__split_columns['first_level']
    
    @property
    def in_columns_2nd(self):
        return self.__split_columns['second_level']
    
    @property
    def in_columns_3rd(self):
        return self.__split_columns['third_level']

    @property
    def out_column(self):
        return self.__split_columns['output_columns']
>>>>>>>> Stashed changes:Integration/Params.py
