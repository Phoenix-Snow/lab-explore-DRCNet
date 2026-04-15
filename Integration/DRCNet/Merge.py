import copy
import os
import shutil
import pandas as pd
import json
import yaml

from datetime import timedelta
from PIL import Image

from Tools.Calculate import calculate_solar_radiation, trans_LonLat2CCDEM
from Tools.AnyFile2CSV import excel2csv


# 数据集格式：每个站点的数据给予序号, 并按照序号保存对应的地图数据, 同时保存每个站点的数据, 
# 然后根据80%/20%的比例划分每个站点的训练集、验证集, 并给出相应的乱序数组
# 合并相应的训练集、验证集, 得到完整的训练集、 验证集, 保存train_random_index.json、val_random_index.json。

class MergeByYaml:
    def __init__(self):
        self.__train_ratio = None
        self.__station_3dim = None
        self.__map_width = None
        self.__map_height = None
        self.__longitude_range = None
        self.__latitude_range = None
        self.__geographical_target_size = None
        self.__time_series_length = None
        self.__random_index_save_path = None
        self.__normal_params_save_path = None
        self.__all_station_data_save_path = None
        self.__temporal_dataset_save_base = None
        self.__spatial_dataset_save_base = None
        self.__count = None
        self.__spatial_map_path = None
        self.__temporal_series_path = None
        self.__dem_path = None
        self.__is_imputation = False
        self.__config = {}

    def load_config(self, state, count=0):
        with open('Integration/DRCNet/merge.yaml', 'r', encoding='utf-8') as file:
            self.__config = yaml.safe_load(file)
            file.close()
        # 数据源
        self.__temporal_series_path = self.__config['temporal_series_path']
        if not os.path.exists(self.__temporal_series_path):
            raise FileNotFoundError('[self.__configuration error: merge.yaml] temporal_series_path false')
        self.__spatial_map_path = self.__config['spatial_map_path']
        if not os.path.exists(self.__spatial_map_path):
            raise FileNotFoundError('[self.__configuration error: merge.yaml] spatial_map_path false')
        # 保存路径
        if state == 'work':
            self.__count = 0
            dataset = 'dataset/' + str(self.__count).zfill(2)
            while os.path.exists(dataset):
                self.__count += 1
                if self.__count > 99:
                    self.__count = 0
                    dataset = 'dataset/' + str(self.__count).zfill(2)
                    print('[dataset_save_path initialization warning] dataset count is over 99, overlay 00')
                    break
                dataset = 'dataset/' + str(self.__count).zfill(2)
            print('[INFO] dataset root path:', dataset)
            self.__spatial_dataset_save_base = os.path.join(dataset, self.__config['spatial_dataset_save_base'])
            self.__temporal_dataset_save_base = os.path.join(dataset, self.__config['temporal_dataset_save_base'])
            self.__all_station_data_save_path = os.path.join(dataset, self.__config['all_station_data_save_path'])
            self.__normal_params_save_path = os.path.join(dataset, 'temporal_NormalProcess.json')
            self.__random_index_save_path = os.path.join(dataset, 'train_random_index.json')
            # 训练集占比
            self.__train_ratio = self.__config['train_ratio']
            # 规格参数
            self.__time_series_length = self.__config['time_series_length']
            print('[INFO] time series length:', self.__time_series_length)
            self.__geographical_target_size = self.__config['spatial_target_size']
            # 地图源参数
            self.__latitude_range = self.__config['latitude_range']
            self.__longitude_range = self.__config['longitude_range']
            self.__map_height = self.__config['map_height']
            self.__map_width = self.__config['map_width']
            self.__dem_path = self.__config['dem_file_path']
            self.__is_imputation = self.__config['is_imputation']
            # 创建文件夹
            os.makedirs(self.__temporal_dataset_save_base, exist_ok=True)
            os.makedirs(self.__spatial_dataset_save_base, exist_ok=True)
        elif state == 'not_work':
            self.__count = count
            dataset = 'dataset/' + str(self.__count).zfill(2)
            self.__temporal_dataset_save_base = os.path.join(dataset, self.__config['temporal_dataset_save_base'])
            self.__spatial_dataset_save_base = os.path.join(dataset, self.__config['spatial_dataset_save_base'])
            self.__random_index_save_path = os.path.join(dataset, 'train_random_index.json')
            self.__normal_params_save_path = os.path.join(dataset, 'temporal_NormalProcess.json')
        # 站点三维空间信息
        self.__station_3dim = self.__config['stat']

    def not_work(self, count):
        print('>>>>>>>>>>>>>> load resource and config >>>>>>>>>>>>>>>>')
        self.load_config('not_work', count)
        # 时序
        file_list = os.listdir(self.__temporal_dataset_save_base)
        station_path_dict = {}
        for file_name in file_list:
            file_path = os.path.join(self.__temporal_dataset_save_base, file_name)
            station_data = pd.read_csv(file_path)
            station_path_dict[file_name.split('.')[0]] = station_data
        # 地理图
        file_list = os.listdir(self.__spatial_dataset_save_base)
        station_geographical_dict = {}
        for file_name in file_list:
            file_path = os.path.join(self.__spatial_dataset_save_base, file_name)
            station_map = Image.open(file_path)
            station_geographical_dict[file_name.split('.')[0]] = station_map
        # 判断数据集是否全面
        if station_path_dict.keys() != station_geographical_dict.keys():
            print('[ERROR] station data and map data are not match!')
            print('[REMINDER] Please check merge.yaml and the data source before answering the question below '
                  'if you want to merge new dataset.')
            is_merge_new_dataset = input('[Question] IS MERGE NEW DATASET?(y/n)  ')
            if is_merge_new_dataset.lower() == 'y':
                return self.work()
            else:
                raise SystemExit("[User interrupt: not merge new dataset] -1")
        # 序列索引, 训练集、验证集区分
        with open(self.__random_index_save_path, 'r') as f:
            train_random_index = json.load(f)
        # 计算训练集的数量
        train_num = int(len(train_random_index['train_index_list'][file_list[0].split('.')[0]]))
        print('[INFO] dataset resource number:', self.__count,
              ';\n       station id list: ', list(station_path_dict.keys()),
              ';\n       train dataset num: ', train_num)
        with open(self.__normal_params_save_path, 'r') as f:
            normal_params = json.load(f)
            # **在函数返回前强制深拷贝**
            normal_params = copy.deepcopy(normal_params)  # 防止json关闭时导致缓存损坏问题，这里在内存里另开辟一处空间复制一下
        print('[INFO] normalization params: \n', normal_params)
        print('>>>>>>>>>>>>>> load end >>>>>>>>>>>>>>>>')
        return (self.__count, station_path_dict, station_geographical_dict, train_random_index['train_index_list'],
                train_random_index['val_index_list'], normal_params)

    def work(self, time_series_len=None, spatial_target_size=None, temporal_series_path=None, ):
        # 加载配置
        self.load_config('work')
        # 使用传参指定，否则使用默认的merge.yaml文件里的
        if time_series_len is not None:
            self.__time_series_length = time_series_len
        if spatial_target_size is not None:
            self.__geographical_target_size = spatial_target_size
        if temporal_series_path is not None:
            self.__temporal_series_path = temporal_series_path
        # print('1', self.__time_series_length, self.__geographical_target_size)
        print('>>>>>>>>>>>>>> start merge >>>>>>>>>>>>>>>>\n'
              'The merge operation\'s purpose is to avoid the effects of discontinuities in the original time series.')

        # 格式判断与处理
        if self.__temporal_series_path.endswith('.xlsx') or self.__temporal_series_path.endswith('.xls'):
            excel2csv(self.__temporal_series_path, 'dataset/' + str(self.__count).zfill(2) + '/time_series_xlsx.csv')
            # 更新temporal_series_path
            self.__temporal_series_path = 'dataset/' + str(self.__count).zfill(2) + '/time_series_xlsx.csv'
        elif self.__temporal_series_path.endswith('.csv'):
            pass
        else:
            raise ValueError('[ERROR] temporal_series_path is not a valid file type')

        # 读取时间序列数据
        time_series_data = pd.read_csv(self.__temporal_series_path)
        # 处理异常问题, 标题行名称里yeares拼写错误, 改为year等（为方便后续处理, years等虽拼写正确, 也改为year等单数形式）
        time_series_data.rename(columns={'yeares': 'year', 'monthes': 'month', 'dayes': 'day', 'datees': 'date',
                                         'years': 'year', 'months': 'month', 'days': 'day', 'dates': 'date',
                                         'houres': 'hour', 'minutees': 'minute', 'secondes': 'second',
                                         'hours': 'hour', 'minutes': 'minute', 'seconds': 'second'}, inplace=True)
        # 列仅保留station、date、year、month、day、Tave、Tmax、Tmin、平均气压、平均风、相对湿度、日照、日总辐射
        is_exist_date = False
        # 如果station列不存在, 则报错
        if 'station' not in time_series_data.columns:
            raise ValueError('[ERROR] time_series_data lack a column named "station"')
        # 如果year、month、day列不存在, 则
        if 'year' not in time_series_data.columns or 'month' not in time_series_data.columns or 'day' not in time_series_data.columns:
            # 判断date列存不存在, 如果存在, 则跳过, 如果不存在, 则报错
            if 'date' in time_series_data.columns:
                time_series_data['date'] = pd.to_datetime(time_series_data['date'])
                time_series_data['year'] = time_series_data['date'].dt.year
                time_series_data['month'] = time_series_data['date'].dt.month
                time_series_data['day'] = time_series_data['date'].dt.day
                is_exist_date = True
            else:
                raise ValueError('[ERROR] time_series_data lack a column named "year" or "month" or "day" or "date"')

        # 划分有用的数据, 划分时'station', 'year', 'month', 'day'四列排在最前, 方便iloc切片切除
        if not is_exist_date:
            # 补全date列
            time_series_data['date'] = pd.to_datetime(time_series_data[['year', 'month', 'day']])
        # 划分重排
        time_series_data = time_series_data[
            ['station', 'year', 'month', 'day', 'date', 'Tave', 'Tmax', 'Tmin', '平均气压', '平均风', '相对湿度',
             '日照', '日总辐射']
        ]

        # (如果需要进行缺失值补全) 检查是否有缺失值, 如果有就进行补全
        if self.__is_imputation and time_series_data.isnull().values.any():
            # time_series_data[['Tave', 'Tmax', 'Tmin', '平均气压', '平均风', '相对湿度', '日照', '日总辐射']] = Imputation(
            #     time_series_data[['Tave', 'Tmax', 'Tmin', '平均气压', '平均风', '相对湿度', '日照', '日总辐射']],
            #     'median')
            time_series_data[['Tave', 'Tmax', 'Tmin', '平均气压', '平均风', '相对湿度', '日照', '日总辐射']] = \
                time_series_data[
                    ['Tave', 'Tmax', 'Tmin', '平均气压', '平均风', '相对湿度', '日照', '日总辐射']].interpolate(
                    method='linear', limit_direction='both')

        # 按站点划分数据（为时序切割做准备） & 按站点保存地理截图
        station_path_dict = {}
        # 加载地图图片
        map_image = Image.open(self.__spatial_map_path)
        station_geograph_dict = {}
        # for index, row in tqdm(time_series_data.iterrows(), desc='loading station data '):
        for index, row in time_series_data.iterrows():
            # print('3', self.__time_series_length, self.__geographical_target_size)
            station_id = int(row['station'])
            # 截取地理图并保存, 但是每个站点只用保存一次
            if station_id not in station_path_dict:
                # print('2', time_series_len, spatial_target_size)
                # 新建时序数据字典
                station_path_dict[station_id] = []
                # 同时, 根据站点坐标, 获取周围100*100的站点数据
                # 中心点坐标
                center_longitude = self.__station_3dim[station_id][0]
                center_latitude = self.__station_3dim[station_id][1]
                # 计算经纬度到像素的转换比例
                lat_scale = self.__map_height / (self.__latitude_range[1] - self.__latitude_range[0])
                lon_scale = self.__map_width / (self.__longitude_range[1] - self.__longitude_range[0])
                # 计算中心点在地图上的像素坐标
                center_pixel_x = (center_longitude - self.__longitude_range[0]) * lon_scale
                center_pixel_y = (center_latitude - self.__latitude_range[0]) * lat_scale
                # 计算切出区域的左上角像素坐标
                start_pixel_x = int(center_pixel_x - self.__geographical_target_size / 2)
                start_pixel_y = int(center_pixel_y - self.__geographical_target_size / 2)
                # 切出 spatial_target_size ** 2 的区域
                cropped_area = map_image.crop(
                    (start_pixel_x, start_pixel_y, start_pixel_x + self.__geographical_target_size,
                     start_pixel_y + self.__geographical_target_size))
                # 保存切出的区域
                # cropped_area = cropped_area.convert('L')
                station_geograph_dict[station_id] = cropped_area
                cropped_area.save(os.path.join(self.__spatial_dataset_save_base, str(station_id) + ".png"))
            # # 如果年份不是2023年, 则跳过
            # if row['year'] not in [2023, 2024, '2023', '2024']:
            #     continue
            station_path_dict[station_id].append(row)
        print('[INFO] station geographic map has already been saved.')

        # # 按时间划分数据（为空序切割做准备）
        # time_data_dict = {}
        # 加载cc和dem数据
        CCDEM = trans_LonLat2CCDEM(self.__station_3dim, self.__dem_path)
        # 保存配置信息, temporal_dataset_save_base、spatial_dataset_save_base、time_series_length、spatial_target_size、stat
        with open(os.path.join('dataset', str(self.__count).zfill(2), 'config.json'), 'w') as f:
            json.dump({'temporal_dataset_save_base': self.__temporal_dataset_save_base,
                       'geographic_images_save_base': self.__spatial_dataset_save_base,
                       'station_list': list(station_path_dict.keys()),
                       'time_series_length': self.__time_series_length,
                       'geographic_img_size': self.__geographical_target_size,
                       'station_3d': self.__station_3dim}, f)
        # station_series_data = time_series_data.copy()[
        #     ['year', 'month', 'day', 'date', 'station', 'Tave', 'Tmax', 'Tmin', '平均气压', '平均风', '相对湿度',
        #      '日照', '日总辐射']
        # ]

        # 时空序列全保存（站点接站点-后续全连接准备） & 按站点保存时序数据
        all_data = []
        for station_id in station_path_dict:
            # print('3', time_series_len, spatial_target_size)
            station_data = pd.DataFrame(station_path_dict[station_id])
            # 保存站点数据
            station_data['theory_sunDuration'], station_data['theory_DailySunRadiation'] = calculate_solar_radiation(
                station_data['year'], station_data['month'], station_data['day'], self.__station_3dim[station_id][1]
            )
            station_data['sunDuration_theoryPlus'] = station_data['日照'] - station_data['theory_sunDuration']
            station_data['DailySunRadiation_theoryPlus'] = station_data['日总辐射'] - station_data[
                'theory_DailySunRadiation']
            # 排序确保日期是按顺序的, 方便后续的顺序遍历
            station_data.sort_values('date', inplace=True)
            # 重置索引
            station_data.reset_index(drop=True, inplace=True)
            # 更新格式
            station_data = station_data.iloc[:, 4:].reindex(
                columns=['date', 'Tave', 'Tmax', 'Tmin', '平均气压', '平均风', '相对湿度', 'theory_sunDuration', '日照',
                         'sunDuration_theoryPlus', 'theory_DailySunRadiation', '日总辐射',
                         'DailySunRadiation_theoryPlus']
            )
            # station_data数据列表
            station_data['station_id'] = station_id
            station_data['经度'] = self.__station_3dim[station_id][0]
            station_data['纬度'] = self.__station_3dim[station_id][1]
            station_data['海拔'] = self.__station_3dim[station_id][2]
            station_data['cc_x'] = CCDEM[station_id][0]
            station_data['cc_y'] = CCDEM[station_id][1]
            station_data['cc_z'] = CCDEM[station_id][2]
            station_data['slope'] = CCDEM[station_id][3]
            station_data['aspect'] = CCDEM[station_id][4]
            station_data = station_data.reindex(
                columns=['station_id', 'date', 'Tave', 'Tmax', 'Tmin', '平均气压', '平均风', '相对湿度',
                         'theory_sunDuration', '日照', 'sunDuration_theoryPlus', 'theory_DailySunRadiation', '经度',
                         '纬度', '海拔', 'cc_x', 'cc_y', 'cc_z', 'slope', 'aspect', '日总辐射',
                         'DailySunRadiation_theoryPlus']
            )
            # 缓存station_path_dict字典
            station_path_dict_cache = station_data.copy()
            station_path_dict[station_id] = station_path_dict_cache[
                ['date', 'Tave', 'Tmax', 'Tmin', '平均气压', '平均风', '相对湿度', 'theory_sunDuration', '日照',
                 'sunDuration_theoryPlus', 'theory_DailySunRadiation', '经度', '纬度', '海拔', 'cc_x', 'cc_y', 'cc_z',
                 'slope', 'aspect', '日总辐射', 'DailySunRadiation_theoryPlus']
            ]
            # 缓存all_data列表
            all_data.extend(station_data.values.tolist())

        # 重构DataFrame
        all_data = pd.DataFrame(all_data,
                                columns=['Station_ID', 'Date', 'T_ave', 'T_max', 'T_min', 'AtmospherePressure_avg',
                                         'WindSpeed_avg', 'RelativeHumidity', 'theory_sunDuration', 'SunDuration',
                                         'sunDuration_theoryPlus', 'theory_DailySunRadiation', 'Longitude', 'Latitude',
                                         'Elevation', '3d_x', '3d_y', '3d_z', 'slope', 'aspect', 'DailySunRadiation',
                                         'dailySunRadiation_theoryPlus']
                                )
        # 全数据本地保存
        all_data.to_csv(self.__all_station_data_save_path, index=False)
        print('[INFO] all station text map data has already been saved.')

        # 通过all_data保存normalize所需的数值(去除Station_ID和Date)
        max_values = all_data.iloc[:, 2:].max().values.tolist()  # 去日期等信息, 取最大值, 转换格式
        min_values = all_data.iloc[:, 2:].min().values.tolist()
        mean_values = all_data.iloc[:, 2:].mean().values.tolist()
        std_values = all_data.iloc[:, 2:].std().values.tolist()
        # 保存json
        normal_params = {
            'max_values_input': max_values[:-2], 'min_values_input': min_values[:-2],
            'mean_values_input': mean_values[:-2], 'std_values_input': std_values[:-2],
            'max_values_output': max_values[-2:], 'min_values_output': min_values[-2:],
            'mean_values_output': mean_values[-2:], 'std_values_output': std_values[-2:]
        }
        with open(self.__normal_params_save_path, 'w') as f:
            json.dump(normal_params, f)

        # 切分时序片段, 并按站点保存数据
        # for station_id in tqdm(station_path_dict, desc='saving station data '):
        for station_id in station_path_dict:
            # print('5', time_series_len, spatial_target_size)
            station_data = pd.DataFrame(station_path_dict[station_id])

            # 初始化一个空的DataFrame来存储结果
            result_station_data = pd.DataFrame()

            # 初始化起始日期和结束日期
            start_date = station_data['date'].iloc[0]
            end_date = start_date + timedelta(days=self.__time_series_length - 1)
            # print('6', time_series_len, spatial_target_size)

            # 遍历数据集, 每28天分为一组
            while start_date in station_data['date'].values:
                # print('7', self.__time_series_length, self.__geographical_target_size)
                # 选取当前28天的数据
                current_group = station_data[(station_data['date'] >= start_date) & (station_data['date'] <= end_date)]

                # 如果当前组的数据不足28天, 即存在不连续的情况, 这个断点必定是最后一个数据, 所以只要数量不对, 则从最后一个的下一个开始重新分组
                if len(current_group) < self.__time_series_length:
                    start_date = end_date + timedelta(days=1)
                    end_date = start_date + timedelta(days=self.__time_series_length - 1)
                    continue

                # 将当前组的数据添加到结果DataFrame中
                result_station_data = pd.concat([result_station_data, current_group])

                # 更新起始日期和结束日期
                start_date = start_date + timedelta(days=1)
                end_date = start_date + timedelta(days=self.__time_series_length - 1)

            # 判断是否为空, 如果为空, 报错
            if result_station_data.empty:
                # 删除之前创建的文件夹以及其中的内容
                path = os.path.join('dataset', str(self.__count).zfill(2))
                if os.path.exists(path):
                    shutil.rmtree(path)
                raise ValueError(f"The dataset is misshapen too much. "
                                 f"station {station_id}: The time series length of one batch({self.__time_series_length}) can not be divided.")

            # 重置索引
            result_station_data.reset_index(drop=True, inplace=True)

            # 保存结果到新的CSV文件
            result_station_data.columns = ['Date', 'T_ave', 'T_max', 'T_min', 'AtmospherePressure_avg', 'WindSpeed_avg',
                                           'RelativeHumidity', 'theory_sunDuration', 'SunDuration',
                                           'sunDuration_theoryPlus', 'theory_DailySunRadiation', 'Longitude',
                                           'Latitude', 'Elevation', '3d_x', '3d_y', '3d_z', 'slope', 'aspect',
                                           'DailySunRadiation', 'dailySunRadiation_theoryPlus']
            result_station_data.to_csv(os.path.join(self.__temporal_dataset_save_base, str(station_id) + '.csv'),
                                       index=False)
            station_path_dict[station_id] = result_station_data
        print('[INFO] each station text map data has already been saved.')

        # 挨个加载每个站点的CSV文件, 获取总长度/28, 即每个站点的训练集数量, 然后生成不重复的整型随机数, 作为训练集的索引
        import random

        # 读取每个站点的CSV文件, 获取总长度 // time_series_length, 即每个站点的数据集数量
        station_sort_index_list = {}
        station_data_len_list = []
        for station_id in station_path_dict:
            station_sample_num = int(len(station_path_dict[station_id]) // self.__time_series_length)
            station_data_len_list.append(station_sample_num)
            # 生成 [0, station_sample_num) 的打乱整型数, 作为训练集的索引
            # random_index = [0 + i * self.__time_series_length for i in range(station_sample_num)]
            # random.shuffle(random_index)
            random_index = random.sample(range(station_sample_num), station_sample_num)
            # 缓存训练集索引
            station_sort_index_list[station_id] = random_index

        min_len = min(station_data_len_list)
        cache_index = 0
        while station_data_len_list[cache_index] != min_len:
            cache_index += 1
        print("最小数据集长度为：", min_len, ", 训练集占比为：", self.__train_ratio)
        train_len = int(min_len * self.__train_ratio)
        print("训练集长度为：", train_len)
        # 保存训练集索引
        train_index_list = {}
        val_index_list = {}
        for station_id in station_sort_index_list:
            random_index = station_sort_index_list[station_id]
            train_index_list[station_id] = random_index[:train_len]
            val_index_list[station_id] = random_index[train_len:int(train_len / 0.8)]

        # 保存训练集索引
        train_random_index = {'train_index_list': train_index_list, 'val_index_list': val_index_list}
        with open(self.__random_index_save_path, 'w') as f:
            json.dump(train_random_index, f)

        print('>>>>>>>>>>>>>> merge end >>>>>>>>>>>>>>>>\n')

        return self.__count, station_path_dict, station_geograph_dict, train_index_list, val_index_list, normal_params
