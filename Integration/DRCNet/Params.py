<<<<<<< Updated upstream
import yaml

from Integration.DRCNet.Merge import MergeByYaml


class Params:
    def __init__(self):
        self.__resource = {}
        with open('Integration/DRCNet/config.yaml', 'r', encoding='utf-8') as file:
            self.__config = yaml.safe_load(file)
            file.close()
        if self.__config is None:
            raise ValueError('Config file is empty or not found.')
        self.__task = self.__config['task'].lower()
        print(f'Task: {self.__task}')
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
        elif self.__task == 'test':
            self.__config = self.__config['test']
            print('LOAD TEST')
            if not self.__config['merge']['is_merged']:
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
        else:
            raise ValueError('Task is not supported.')

    @property
    def NAME(self):
        return self.__config['dataset_name']

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
    def TIME_SERIES_LENGTH(self):
        return self.__background['time_series_length']

    @property
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
=======
import yaml

from Integration.DRCNet.Merge import MergeByYaml


class Params:
    def __init__(self):
        self.__resource = {}
        with open('Integration/DRCNet/config.yaml', 'r', encoding='utf-8') as file:
            self.__config = yaml.safe_load(file)
            file.close()
        if self.__config is None:
            raise ValueError('Config file is empty or not found.')
        self.__task = self.__config['task'].lower()
        print(f'Task: {self.__task}')
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
        elif self.__task == 'test':
            self.__config = self.__config['test']
            print('LOAD TEST')
            if not self.__config['merge']['is_merged']:
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
        else:
            raise ValueError('Task is not supported.')

    @property
    def NAME(self):
        return self.__config['dataset_name']

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
    def TIME_SERIES_LENGTH(self):
        return self.__background['time_series_length']

    @property
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
>>>>>>> Stashed changes
