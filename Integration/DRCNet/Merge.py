import copy
import json
import math
import os
import queue
import random
import shutil
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from Tools.AnyFile2CSV import excel2csv
from Tools.Calculate_pandas import calc_slp_residual, calc_fog_risk, calc_petEst_daylightDuration_dailyRadiation, \
    calc_geo_zoning
from Tools.Guardian import ProcessGuardian
from Tools.Imputation import Imputation
from Tools.Save import save_structured_npz
from Tools.Tools import move_cursor_up, normal_city, normal_province, extract_coordinates, extract_coordinates_plus, \
    jsonDumps
from Tools.identify_location.by_GeoJson import China_GeoJson


# 数据集格式：每个站点的数据按照ID保存,
# 然后根据80%/20%的比例划分每个站点的训练集、验证集, 并给出相应的乱序数组。
# 合并相应的训练集、验证集, 得到完整的训练集、 验证集, 保存。

class MergeByYaml:
    def __init__(self):
        # resource
        self.__temporal_series_path = ""
        self.__dem_path = ""
        # datasets
        self.__name: int | str = 0
        self.__temporal_dataset_save_base = ""
        self.__config_save_path = ""
        self.__domain_split_save_path = ""
        self.__normalization_params_save_path = ""
        self.__cleaned_data_save_path = ""
        self.__normalized_data_save_path = ""
        self.__readme_path = ""
        # yaml
        self.__domain_type = 0  # 0: province; 1: city
        self.__station_domain = []
        self.__station_3d: Dict[str, List[str | float]] = {}  # 当前数据来源用不到
        self.__columns = {}
        self.__time_series_length = 365
        self.__train_ratio = 0.8
        self.__target_domain_ratio = 0.1
        self.__is_imputation = False
        self.__NAN_placeholder = {}
        # details
        self.__station_ID_list = []
        self.__domain_split = {"train_list": {}, "val_list": {}}
        self.__target_domain_ids = []
        self.__normal_params = {
            "max_values_input": [], "min_values_input": [], "max_values_output": [], "min_values_output": [],
            "mean_values_input": [], "std_values_input": [], "mean_values_output": [], "std_values_output": [],
        }
        self.__config_datasets = {}
        # Tools
        self.geojson = China_GeoJson()

        # thread
        # if multi files
        # queue
        # self.ID_queue = queue.Queue(maxsize=2)
        # self.input_queue = queue.Queue(maxsize=2)
        # self.output_queue = queue.Queue(maxsize=2)
        # self.reader_queue = queue.Queue(maxsize=2)
        self.cleaner_queue = queue.Queue(maxsize=2)
        self.saver_queue_split = queue.Queue(maxsize=2)  # temporal_resource
        self.saver_queue_cleaned = queue.Queue(maxsize=2)  # temporalSpacial_series_cleaned.csv
        self.saver_queue_normal = queue.Queue(maxsize=2)  # temporalSpacial_series_normal.csv
        # signal
        self.all_data_scanned = False
        self.valid_station_ids = []
        # thread stop event
        self.stop_event = threading.Event()
        #
        self._merged = [False, False, False]

        # Guardian
        self.guardian = ProcessGuardian(os.getpid())

    def load_config(self, state, name: int | str | None = None):
        while True:
            with open("Integration/DRCNet/merge.yaml", "r", encoding="utf-8") as f:
                config_yaml = yaml.safe_load(f)
                f.close()
            if state == "work":
                # resource
                self.__temporal_series_path = config_yaml["temporal_series_path"]
                if not os.path.exists(self.__temporal_series_path):
                    self.stop_event.set()
                    raise FileNotFoundError("[❌ configuration error: merge.yaml] temporal_series_path false")
                self.__dem_path = Path(config_yaml["dem_path"])
                if not os.path.exists(self.__dem_path):
                    self.stop_event.set()
                    raise FileNotFoundError("[❌ configuration error: merge.yaml] dem_path false")
                # datasets
                if name is None:  # 使用merge配置时，name使用默认的None
                    if "dataset_name" in config_yaml.keys() and \
                            (config_yaml["dataset_name"] is not None and
                             os.path.join("datasets", config_yaml["dataset_name"]) != Path("datasets")):  # merge配置里有这个名称
                        if os.path.exists(os.path.join("datasets", config_yaml["dataset_name"])):
                            print('🚧🚧🚧 [INFO] 检测到已存在该文件夹，未避免后续保存的文件有问题，请选择:')
                            options = ["1. 我配置文件忘改了，请终止程序; ", "2. 之前的文件不需要了，请删除重新生成; ",
                                       "3. 就使用这个数据集，跳过生成阶段; ", "4. 配置文件已经重新修改，请重新读取并生成; ",
                                       "5. 不想修改配置，备份一下重新开始生成."]
                            # selected_index = 0
                            # # 1. 初始打印菜单
                            # draw_menu(options, selected_index)
                            # while True:
                            #     # 2. 获取按键 (非阻塞或单字符读取)
                            #     key = readchar.readkey()
                            #
                            #     # 逻辑处理
                            #     if key == 'UP':
                            #         selected_index = (selected_index - 1) % len(options)
                            #     elif key == 'DOWN':
                            #         selected_index = (selected_index + 1) % len(options)
                            #     elif key == 'ENTER':
                            #         print(f"\n>> 确认选择: {options[selected_index]}")  # 这里会换行，保留在历史记录里
                            #         break
                            #     # 3. 核心刷新步骤
                            #     # 第一步：光标回到菜单起始位置 (上移 len(options) 行)
                            #     move_cursor_up(len(options))
                            #
                            #     # 第二步：重新绘制菜单 (覆盖旧内容)
                            #     draw_menu(options, selected_index)

                            for option in options:
                                print(option)
                            selected_index = int(input()) - 1
                            move_cursor_up(len(options) + 1)
                            print(f">> 确认选择: {options[selected_index]}")  # 这里会换行，保留在历史记录里

                            if selected_index == 0:
                                self.stop_event.set()
                                exit()
                            elif selected_index == 1:
                                try:
                                    shutil.rmtree(os.path.join("datasets", config_yaml["dataset_name"]))
                                    print(
                                        f"✅ 文件夹 {os.path.join("datasets", config_yaml["dataset_name"])} 及其内容已成功删除。")
                                except PermissionError:
                                    self.stop_event.set()
                                    raise Exception(
                                        f"❌ 权限不足，无法删除 {os.path.join("datasets", config_yaml["dataset_name"])}。请检查文件是否被占用或尝试以管理员身份运行。")
                                except Exception as e:
                                    self.stop_event.set()
                                    raise Exception(
                                        f"❌ 删除{os.path.join("datasets", config_yaml["dataset_name"])}时发生错误: {e}")
                            elif selected_index == 2:
                                state = "not_work"
                                os.system('cls' if os.name == 'nt' else 'clear')  # 清屏
                                continue
                            elif selected_index == 3:
                                os.system('cls' if os.name == 'nt' else 'clear')  # 清屏
                                continue
                            elif selected_index == 4:
                                try:
                                    self.__name = 0
                                    dataset = os.path.join("datasets", "Backup" + str(self.__name).zfill(2))
                                    while os.path.exists(dataset):
                                        self.__name += 1
                                        if self.__name > 99:
                                            self.__name = 0
                                            dataset = os.path.join("datasets", "Backup" + str(self.__name).zfill(2))
                                            print(
                                                "[⚠️ dataset_save_path initialization warning] dataset count is over 99, overlay 00")
                                            shutil.rmtree(dataset)
                                            break
                                        else:
                                            dataset = os.path.join("datasets", "Backup" + str(self.__name).zfill(2))
                                    os.rename(os.path.join("datasets", config_yaml["dataset_name"]),
                                              dataset)
                                    print(f"成功：已将 '{os.path.join("datasets", config_yaml["dataset_name"])}' "
                                          f"备份为 '{dataset}'")
                                except Exception as e:
                                    self.stop_event.set()
                                    raise Exception(
                                        f"备份(更名)'{os.path.join("datasets", config_yaml["dataset_name"])}'时发生错误：{e}")

                        self.__name = config_yaml["dataset_name"]
                        dataset = os.path.join("datasets", str(self.__name))
                    else:  # merge配置里没有这个名称
                        self.__name = 0
                        dataset = os.path.join("datasets/", str(self.__name).zfill(2))
                        while os.path.exists(dataset):
                            self.__name += 1
                            if self.__name > 99:
                                self.__name = 0
                                dataset = os.path.join("datasets/", str(self.__name).zfill(2))
                                print("[⚠️ dataset_save_path initialization warning] dataset count is over 99, overlay 00")
                                shutil.rmtree(dataset)
                                break
                            else:
                                dataset = os.path.join("datasets/", str(self.__name).zfill(2))
                else:  # 使用主程序的config配置
                    self.__name = name
                    dataset = os.path.join("datasets/", str(self.__name).zfill(2))
                print("[🔔 INFO] dataset root path:", dataset)
                self.__temporal_dataset_save_base = os.path.join(dataset, "temporal_resource")
                os.makedirs(self.__temporal_dataset_save_base, exist_ok=True)
                self.__config_save_path = os.path.join(dataset, "config.json")
                self.__domain_split_save_path = os.path.join(dataset, "domain_split.json")
                self.__normalization_params_save_path = os.path.join(dataset, "temporal_normalization_params.json")
                self.__cleaned_data_save_path = os.path.join(dataset, "temporalSpacial_series_cleaned.csv")
                self.__normalized_data_save_path = os.path.join(dataset, "temporalSpacial_series_normal.csv")
                self.__readme_path = os.path.join(dataset, "README.md")
                # details
                self.__domain_type = config_yaml["stat"]["type"]
                if self.__domain_type == "city" or "市" in self.__domain_type:
                    self.__station_domain = [normal_city(address) for address in config_yaml["stat"]["domain"]]
                elif self.__domain_type == "province" or "省" in self.__domain_type:
                    self.__station_domain = [normal_province(address) for address in config_yaml["stat"]["domain"]]
                else:
                    self.stop_event.set()
                    raise Exception('[❌ configuration error: merge.yaml] stat - type should be "city" or "province"')
                self.__columns = config_yaml["columns"]
                self.__time_series_length = config_yaml["time_series_length"]
                self.__station_3d = config_yaml["station_3d"]
                self.__train_ratio = config_yaml["train_ratio"]
                if "target_domain_ratio" in config_yaml.keys():
                    self.__target_domain_ratio = config_yaml["target_domain_ratio"]
                self.__is_imputation = config_yaml["is_imputation"]
                self.__NAN_placeholder = config_yaml["NAN_placeholder"]
                print(f"[😀 INFO] domain: {self.__station_domain}, train_ratio: {self.__train_ratio}, "
                      f"target domain ratio: {self.__target_domain_ratio}, time series length: {self.__time_series_length}, "
                      f"imputation or not: {self.__is_imputation}")
                return self.work()
            elif state == "not_work":
                # datasets
                self.__name = name
                if type(self.__name) == int:
                    dataset = os.path.join("datasets/", str(self.__name).zfill(2))
                else:
                    dataset = os.path.join("datasets/", str(self.__name))
                self.__temporal_dataset_save_base = os.path.join(dataset, "temporal_resource")
                self.__config_save_path = os.path.join(dataset, "config.json")
                self.__domain_split_save_path = os.path.join(dataset, "domain_split.json")
                self.__normalization_params_save_path = os.path.join(dataset, "temporal_normalization_params.json")
                self.__cleaned_data_save_path = os.path.join(dataset, "temporalSpacial_series_cleaned.csv")
                self.__normalized_data_save_path = os.path.join(dataset, "temporalSpacial_series_normal.csv")
                self.__readme_path = os.path.join(dataset, "README.md")
                print("[😀 INFO] dataset load base path:", dataset)
                return self.not_work()
            break  # 加载完毕就跳出来，否则就中间被拦截了

    def not_work(self):
        # self.load_config("not_work", name)
        print(">>>>>>>>>>>>>> 📄 load resource and config >>>>>>>>>>>>>>>>")
        # dataset config
        with open(self.__config_save_path, "r", encoding='utf-8') as f:
            config_dataset = json.load(f)
            config_dataset = copy.deepcopy(config_dataset)
            f.close()
        if "temporal_dataset_save_base" in config_dataset.keys() and \
                config_dataset["temporal_dataset_save_base"] is not None:
            self.__temporal_dataset_save_base = config_dataset["temporal_dataset_save_base"]
        if "domain_split_path" in config_dataset.keys() and config_dataset["domain_split_path"] is not None:
            self.__domain_split_save_path = config_dataset["domain_split_path"]
        if "temporal_normalization_params_path" in config_dataset.keys() and \
                config_dataset["temporal_normalization_params_path"] is not None:
            self.__normalization_params_save_path = config_dataset["temporal_normalization_params_path"]
        self.__domain_type = config_dataset["domain_type"]
        self.__station_domain = config_dataset["domain"]
        self.__time_series_length = config_dataset["time_series_length"]
        self.__station_ID_list = config_dataset["station_list"]
        self.__target_domain_ids = config_dataset["target_domain_ids"]
        # datasets list
        file_list = os.listdir(self.__temporal_dataset_save_base)
        station_path_dict = {}
        for file_name in file_list:
            if not file_name.endswith(".npz"):
                continue
            file_path = os.path.join(self.__temporal_dataset_save_base, file_name)
            station_path_dict[file_name.split(".")[0]] = file_path
        # domain split: train val
        with open(self.__domain_split_save_path, "r") as f:
            self.__domain_split = json.load(f)
        # calc train num
        train_num = int(len(self.__domain_split["train_list"][file_list[0].split(".")[0]]))
        val_num = int(len(self.__domain_split["val_list"][file_list[0].split(".")[0]]))
        print("[😀 INFO] dataset resource name:", self.__name,
              ";\n       station id list: ", list(station_path_dict.keys()),
              ";\n       train num: ", train_num, "val num: ", val_num)
        # normalization
        with open(self.__normalization_params_save_path, "r") as f:
            self.__normal_params = json.load(f)
            # **在函数返回前强制深拷贝**
            self.__normal_params = copy.deepcopy(self.__normal_params)  # 防止json关闭时导致缓存损坏问题，这里在内存里另开辟一处空间复制一下
            f.close()
        print("[😀 INFO] normalization params: \n", self.__normal_params)
        print(">>>>>>>>>>>>>> ✅ load end >>>>>>>>>>>>>>>>\n")
        return (self.__name, station_path_dict, self.__domain_split["train_list"], self.__domain_split["val_list"],
                self.__normal_params, config_dataset)

    def work(self, time_series_len=None, temporal_series_path=None, dem_path=None):
        # 加载配置
        # self.load_config("work")

        # 使用传参指定，否则使用默认的merge.yaml文件里的(因为是二级函数，所以基本上不使用)
        if time_series_len is not None:
            self.__time_series_length = time_series_len
        if temporal_series_path is not None:
            self.__temporal_series_path = temporal_series_path
        if dem_path is not None:
            self.__dem_path = dem_path

        print(">>>>>>>>>>>>>> 🚀 start merge >>>>>>>>>>>>>>>>\n"
              "😉 The merge operation\"s purpose is to avoid the effects of discontinuities in the original time series.")

        # 员工 的线程
        t_provider = threading.Thread(target=self.provider_thread)
        t_worker = threading.Thread(target=self.worker_thread)
        t_writer1 = threading.Thread(target=self.consumer_thread_split)
        t_writer2 = threading.Thread(target=self.consumer_thread_normal)
        t_writer3 = threading.Thread(target=self.consumer_thread_cleaned)

        # 1. 🛡️ [监护者] 进程的初始化
        self.guardian.init()

        # 2. 🚀 [主进程] 的心跳汇报线程
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
        # 其他的退了，这个肯定也退了（常规）；子进程没有退（异常），那么这个进程就开始杀子进程

        try:
            # 3. 启动 员工 的进程
            t_provider.start()
            t_worker.start()
            t_writer1.start()
            t_writer2.start()
            t_writer3.start()
            heartbeat_thread.start()

            # 等待所有线程结束
            t_provider.join()
            t_worker.join()
            t_writer1.join()
            t_writer2.join()
            t_writer3.join()

            self.stop_event.set()
            heartbeat_thread.join()

            print("🎉 [👮‍👷✍️] 任务完成！")
        except KeyboardInterrupt:
            print("\n⚠️ 捕获到 Ctrl+C，准备退出...")
        except Exception as e:
            print(f"\n❌ 发生严重错误: {e}")
        finally:
            # 4. 无论成功还是失败，都要停止心跳和监护进程（常规退出）
            self.stop_event.set()
            heartbeat_thread.join()
            self.guardian.stop()

        if all(self._merged) and self.__station_ID_list is not None:
            print(">>>>>>>>>>>>>> 🎉 merge end >>>>>>>>>>>>>>>>\n")
        else:
            print(">>>>>>>>>>>>>> 😭 Unfortunately ! merge error >>>>>>>>>>>>>>>>\n")
            self.stop_event.set()
            exit()

        # datasets list
        file_list = os.listdir(self.__temporal_dataset_save_base)
        station_path_dict = {}
        for file_name in file_list:
            if not file_name.endswith(".npz"):
                continue
            file_path = os.path.join(self.__temporal_dataset_save_base, file_name)
            station_path_dict[file_name.split(".")[0]] = file_path

        return (self.__name, station_path_dict, self.__domain_split["train_list"], self.__domain_split["val_list"],
                self.__normal_params, self.__config_datasets)

    def provider_thread(self, ):
        print("👮‍♂️ [provider] 启动...")
        # 向守护进程汇报TID
        self.guardian.record_TID(threading.get_ident(), '👮‍♂️ provider')

        # 非需要的站点，需要的站点保存在self.__station_ID_list里面了
        drop_station = []
        # 判断输入是文件夹还是单文件
        if os.path.isdir(self.__temporal_series_path):
            files = os.listdir(self.__temporal_series_path)
            for f_name in files:
                if not f_name.endswith((".csv", ".xlsx", ".xls")):
                    continue
                f_path = os.path.join(self.__temporal_series_path, f_name)
                # 判断是否需要写，不需要continue
                station_id = f_name.split(".")[0]
                if station_id not in self.__station_ID_list and station_id not in drop_station:
                    if len(self.__columns['spatial_column']) == 2:
                        coordinates = extract_coordinates(f_path, self.__columns['spatial_column'][0],
                                                          self.__columns['spatial_column'][1])
                        loc = self.geojson.IDENTIFY_PROVINCE(coordinates[0], coordinates[1])
                        if self.__domain_type == "city" or "市" in self.__domain_type:
                            loc = normal_city(loc)
                        else:
                            loc = normal_province(loc)
                        # ↓ 核心判断
                        if loc not in self.__station_domain:
                            drop_station.append(station_id)  # fail
                            continue
                        self.__station_ID_list.append(station_id)  # success
                        # ↑ 核心判断
                        self.__station_3d[str(station_id)] = [coordinates[0], coordinates[1], loc]
                    elif len(self.__columns['spatial_column']) >= 3:
                        coordinates = extract_coordinates_plus(f_path, self.__columns['spatial_column'][0],
                                                               self.__columns['spatial_column'][1],
                                                               self.__columns['spatial_column'][2])
                        loc = self.geojson.IDENTIFY_PROVINCE(coordinates[0], coordinates[1])
                        if self.__domain_type == "city" or "市" in self.__domain_type:
                            loc = normal_city(loc)
                        else:
                            loc = normal_province(loc)
                        if loc not in self.__station_domain:
                            drop_station.append(station_id)
                            continue
                        self.__station_ID_list.append(station_id)
                        self.__station_3d[str(station_id)] = [coordinates[0], coordinates[1], coordinates[2], loc]
                    elif len(self.__columns['spatial_column']) == 0:
                        if str(station_id) in self.__station_3d:
                            coordinates = self.__station_3d[str(station_id)]
                        else:
                            self.stop_event.set()
                            raise Exception('[❌ configuration error: merge.yaml] '
                                            'if coordinates in tabel: columns - spatial_column error | at least 2 elements are required. [\'LATITUDE\', \'LONGITUDE\'; \n'
                                            '                                     '
                                            'if coordinates not in tabel: station_3d error | need [LATITUDE, LONGITUDE]')
                        loc = self.geojson.IDENTIFY_PROVINCE(coordinates[0], coordinates[1])
                        if self.__domain_type == "city" or "市" in self.__domain_type:
                            loc = normal_city(loc)
                        else:
                            loc = normal_province(loc)
                        self.__station_3d[str(station_id)].append(loc)
                        if loc not in self.__station_domain:
                            drop_station.append(station_id)
                            continue
                        self.__station_ID_list.append(station_id)
                    else:
                        self.stop_event.set()
                        raise Exception(
                            '[❌ configuration error: merge.yaml] columns - spatial_column error, at least 2 elements are required. [\'LATITUDE\', \'LONGITUDE\'')
                # 写入队列(分配任务)
                input_columns = ([self.__columns["station_id"]] + [self.__columns["date_column"]] +
                                 self.__columns["temporal_column"] + self.__columns["spatial_column"])
                output_columns = self.__columns["output_column"]
                if f_path.endswith(".csv"):
                    input_data = pd.read_csv(f_path, usecols=input_columns, iterator=False)
                    output_data = pd.read_csv(f_path, usecols=output_columns, iterator=False)
                elif f_path.endswith((".xlsx", "xls")):
                    input_data = pd.read_excel(f_path, usecols=input_columns)
                    output_data = pd.read_excel(f_path, usecols=output_columns)
                else:
                    self.stop_event.set()
                    raise Exception(
                        '[❌ configuration error: merge.yaml] temporal_series_path error. Sorry, current code only support .csv or .xlsx file.')
                # self.ID_queue.put(station_id)
                # self.input_queue.put(input_data)
                # self.output_queue.put(output_data)
                self.cleaner_queue.put({str(station_id): {'input': input_data, 'output': output_data}})
            self.all_data_scanned = True
        else:
            # ===============================
            # === 🔍 筛选, 分站点保存临时文件 ===
            # ===============================
            if self.__columns["station_id"] is None:
                self.stop_event.set()
                raise Exception(
                    '[❌ configuration error: merge.yaml] columns - station_id error, as for single file with muti station data, station_id is required.')
            input_columns = ([self.__columns["station_id"]] + [self.__columns["date_column"]] +
                             self.__columns["temporal_column"] + self.__columns["spatial_column"])
            output_columns = self.__columns["output_column"]
            # 先流式转换文件格式-csv，再流式读取
            if self.__temporal_series_path.endswith(".xlsx"):
                original_path = deepcopy(self.__temporal_series_path)
                self.__temporal_series_path = self.__temporal_series_path[:-5] + ".csv"
                excel2csv(original_path, self.__temporal_series_path)
            elif self.__temporal_series_path.endswith(".xls"):
                original_path = deepcopy(self.__temporal_series_path)
                self.__temporal_series_path = self.__temporal_series_path[:-4] + ".csv"
                excel2csv(original_path, self.__temporal_series_path)
            elif self.__temporal_series_path.endswith(".csv"):
                pass
            else:
                self.stop_event.set()
                raise Exception(
                    '[❌ configuration error: merge.yaml] temporal_series_path error. Sorry, current code only support .csv or .xlsx file.')
            input_reader = pd.read_csv(self.__temporal_series_path, usecols=input_columns,
                                       iterator=True, chunksize=10000)
            output_reader = pd.read_csv(self.__temporal_series_path, usecols=output_columns,
                                        iterator=True, chunksize=10000)
            id_col = self.__columns["station_id"]  # 必定有
            for input_chunk, output_chunk in zip(input_reader, output_reader):
                # 1. 索引重置
                input_chunk.reset_index(drop=True, inplace=True)
                output_chunk.reset_index(drop=True, inplace=True)

                # 2. 获取当前 chunk 中所有站点ID

                # === 🔍 核心逻辑：动态发现并校验新站点 ===
                # 我们只在第一次遇到某个站点时进行地理围栏判断
                chunk_station_ids = input_chunk[id_col].unique()
                for station_id in chunk_station_ids:
                    # 如果该站点还没被判定过（既不在白名单，也不在黑名单）
                    if station_id not in self.__station_ID_list and station_id not in drop_station:

                        try:
                            coordinates = None

                            # --- A. 尝试从当前 Chunk 的数据中直接获取坐标 (最高效) ---
                            station_mask = input_chunk[id_col] == station_id
                            if station_mask.any():
                                # 获取该站点在当前 chunk 的第一行数据
                                first_row = input_chunk[station_mask].iloc[0]

                                # 根据配置提取经纬度
                                if len(self.__columns['spatial_column']) >= 2:
                                    lat_col = self.__columns['spatial_column'][0]
                                    lon_col = self.__columns['spatial_column'][1]
                                    # 假设列存在
                                    if lat_col in first_row and lon_col in first_row:
                                        coordinates = [first_row[lat_col], first_row[lon_col]]
                                        # 如果有高程列
                                        if len(self.__columns['spatial_column']) >= 3:
                                            ele_col = self.__columns['spatial_column'][2]
                                            if ele_col in first_row:
                                                coordinates.append(first_row[ele_col])

                            # --- B. 如果 Chunk 里没坐标，尝试从预定义的 station_3d 获取 ---
                            if coordinates is None and len(self.__columns['spatial_column']) == 0:
                                if str(station_id) in self.__station_3d:
                                    coordinates = self.__station_3d[str(station_id)]

                            # --- C. 执行地理围栏/行政区划判断 ---
                            if coordinates is not None:
                                lat, lon = coordinates[0], coordinates[1]

                                # 调用你的地理围栏逻辑
                                loc = self.geojson.IDENTIFY_PROVINCE(lon, lat, )

                                # 标准化
                                if self.__domain_type == "city" or "市" in str(self.__domain_type):
                                    loc = normal_city(loc)
                                else:
                                    loc = normal_province(loc)

                                # 判定是否在目标区域
                                if loc in self.__station_domain:
                                    self.__station_ID_list.append(station_id)
                                    # 记录坐标信息供后续使用
                                    self.__station_3d[str(station_id)] = coordinates + [loc]
                                else:
                                    drop_station.append(station_id)  # 加入黑名单
                                    continue
                            else:
                                # 无法获取坐标，视情况处理（这里暂时加入黑名单防止报错）
                                drop_station.append(station_id)
                                continue
                        except Exception as e:
                            # 发生错误，加入黑名单跳过
                            drop_station.append(station_id)
                            continue

                        # 更新白名单(黑名单都continue了)
                        if station_id not in self.__station_ID_list:
                            self.__station_ID_list.append(station_id)

                # === ✅ 数据筛选与入队 ===
                # 只有属于“白名单”的站点数据才会被处理 (注: 上述 🔍 核心逻辑 已经将新出现的未知站点写入黑名单或白名单中)
                chunk_station_buffer = {}  # chunk 缓冲区
                for station_id in chunk_station_ids:
                    # 双重过滤：必须在白名单中，且不在黑名单中
                    if station_id in self.__station_ID_list:
                        # 筛选数据
                        mask = input_chunk[id_col] == station_id
                        current_input_part = input_chunk[mask].copy()
                        current_output_part = output_chunk[mask].copy()

                        # 放入缓冲区 (处理跨 Chunk 拼接)
                        chunk_station_buffer[str(station_id)] = {
                            'input': current_input_part,
                            'output': current_output_part
                        }
                # === 🚀 写入队列 ===
                if chunk_station_buffer:
                    for sid in chunk_station_buffer.keys():
                        # 构造任务包：包含数据和操作指令
                        task_package = {
                            'data': pd.concat([chunk_station_buffer[sid]["input"],
                                               chunk_station_buffer[sid]["output"]], axis=1),
                            'mode': 'a',  # 指令：追加保存模式
                            'id': sid
                        }
                        self.saver_queue_split.put(task_package)
            self.all_data_scanned = True

            # # ===============================
            # # === 📩 通知 consumer 进程等待 ===
            # # ===============================
            # self.saver_queue.put({'WAIT': None})  # 显性等待符, consumer不离开
            self.saver_queue_split.put({
                'data': None,
                'mode': "WAIT"
            })

            # =======================================
            # === 📨 分发任务, 让 worker 开始清洗数据 ===
            # =======================================
            for f_name in os.listdir(self.__temporal_dataset_save_base):
                if not f_name.endswith(".csv"):
                    continue
                f_path = os.path.join(self.__temporal_dataset_save_base, f_name)
                station_id = f_name.split(".")[0]
                # 写入队列(分配任务)
                csv_data = pd.read_csv(f_path)
                input_data = csv_data.loc[:, input_columns]
                output_data = csv_data.loc[:, output_columns]
                self.cleaner_queue.put({str(station_id): {'input': input_data, 'output': output_data}})
                # 处理训练集索引和验证集索引
                total_len = len(input_data) - self.__time_series_length
                train_len = int(total_len * self.__train_ratio)
                # val_len = total_len - train_len
                # {"train_list": {}, "val_list": {}}
                shuffle_index_list = random.sample(range(total_len), total_len)
                self.__domain_split["train_list"][str(station_id)] = shuffle_index_list[:train_len]
                self.__domain_split["val_list"][str(station_id)] = shuffle_index_list[train_len:]

        self.cleaner_queue.put({'STOP': None})  # 显性终止符, 独立,单次
        print(f"👮‍♂️ [provider] 数据提供完毕, 共 {len(self.__station_ID_list)} 个站点。")

        # write
        count = math.ceil(len(self.__station_ID_list) * 0.1)
        self.__target_domain_ids = random.sample(self.__station_ID_list, count)
        self.__config_datasets = {"temporal_dataset_save_base": self.__temporal_dataset_save_base,
                                  "temporal_normalization_params_path": self.__normalization_params_save_path,
                                  "domain_split_path": self.__domain_split_save_path,
                                  "domain_type": self.__domain_type,
                                  "domain": self.__station_domain,
                                  "station_list": list(self.__station_ID_list),
                                  "station_3d": self.__station_3d,
                                  "time_series_length": self.__time_series_length,
                                  "target_domain_ids": self.__target_domain_ids}
        with open(self.__config_save_path, "w", encoding='utf-8') as f:
            f.write(jsonDumps(self.__config_datasets))
            f.close()
        with open(self.__domain_split_save_path, "w", encoding='utf-8') as f:
            f.write(jsonDumps(self.__domain_split))
            f.close()

        print(f"👮‍♂️ [provider] 配置文件已保存, 任务完毕, 下线。")

    def worker_thread(self):
        print("👷 [worker] 启动...")
        self.guardian.record_TID(threading.get_ident(), '👷 worker')

        # 两种保存方式需要发送的json指令格式
        # 一种是分站点保存:
        """
        task_package = {
            'data': station_buffer.loc[:, station_save_columns],
            'mode': 'w',  # 指令：覆盖保存模式
            'stage': 'simple-station-save'
        }
        """
        # 一种是整体保存:
        """
        task_package = {
            'data': station_buffer.loc[:, total_save_columns],
            'mode': 'a',  # 指令：追加保存模式
            'stage': 'cleaned-total-series-save'
        }
        task_package = {
            'data': station_buffer.loc[:, total_save_columns],
            'mode': 'a',  # 指令：追加保存模式
            'stage': 'normal-total-series-save'
        }
        """

        # 归一化参数缓存
        norparam_cache = {
            # --- 极值 (用于 Max/Min) ---
            "max_values_input": None,
            "min_values_input": None,
            "max_values_output": None,
            "min_values_output": None,

            # --- 累加值 (用于 Mean/Std) ---
            "count": 0,
            "sum_input": None,  # 每列的总和
            "sum_output": None,  # 每列的总和
            "sum_sq_input": None,  # 每列的平方和 (x^2)
            "sum_sq_output": None  # 每列的平方和 (x^2)
        }
        while True:
            try:
                chunks = self.cleaner_queue.get()
                sids = chunks.keys()
                if 'STOP' in sids and chunks['STOP'] is None:
                    if len(sids) != 1:
                        self.stop_event.set()
                        raise Exception(
                            '[❌ code error] worker_thread called, provider_thread error, "self.cleaner_queue.put" stop signal')
                    if not self.all_data_scanned:
                        self.stop_event.set()
                        raise Exception(
                            '[❌ code error] worker_thread called, provider_thread error, "self.all_data_scanned" not changed')
                    print("👷 [worker] 文件读取结束。")
                    self.saver_queue_split.put({
                        'data': None,
                        'mode': "STOP"
                    })  # 显性终止符
                    self.saver_queue_cleaned.put({
                        'data': None,
                        'mode': "STOP"
                    })
                    self.saver_queue_normal.put({
                        'data': None,
                        'mode': "STOP"
                    })
                    break
                # try:
                #     # file_path = self.file_queue.get(timeout=1)
                #     # station_id = self.Id_queue.get()
                #     # input_chunk = self.input_queue.get()
                #     # output_chunk = self.output_queue.get()
                #     chunks = self.reader_queue.get()
                #     sids = chunks.keys()
                # except queue.Empty:  # 如果空了
                #     if self.all_data_scanned:
                #         print("👷 [cleaner] 文件读取结束。")
                #         self.cleaner_queue.put(None)  # 发送结束信号
                #         break
                #     continue

                for sid in sids:
                    input_chunk = chunks[sid]['input']
                    output_chunk = chunks[sid]['output']

                    # ===========
                    # === 插值 ===
                    # ===========
                    # 清洗成NAN，NAN_placeholder
                    input_chunk = input_chunk.replace(self.__NAN_placeholder, np.nan)
                    output_chunk = output_chunk.replace(self.__NAN_placeholder, np.nan)

                    def check_empty_columns(df, df_name):
                        # isna() 判断是否为空 -> all() 判断该列是否全为 True -> 返回全空列的列名列表
                        all_nan_cols = df.columns[df.isna().all()].tolist()

                        if all_nan_cols:
                            print(f"❌ [ERROR] 在 {df_name} 中检测到 {len(all_nan_cols)} 个全空列！")
                            print(f"   列名列表: {all_nan_cols}")
                            # 可以选择打印具体是哪些占位符导致的，如果需要的话

                            # 3. 结束进程
                            # sys.exit(1) 会以非零状态码退出，表示发生错误
                            self.stop_event.set()
                            sys.exit(1)

                    # 4. 执行检测
                    check_empty_columns(input_chunk, "input_chunk")
                    check_empty_columns(output_chunk, "output_chunk")
                    # 插值
                    input_chunk.iloc[:, 2:] = Imputation(input_chunk.iloc[:, 2:])
                    output_chunk.iloc[:, :] = Imputation(output_chunk.iloc[:, :])

                    # =============
                    # === 计算列 ===
                    # =============
                    # 这里的逻辑是先生成, 再筛选 --- 拼接的有
                    #  calc_column: ['Slope', 'Aspect',  # 坡度坡向
                    #                'SLP_Residual', 'PET_Est', 'Solar', 'Fog_Risk',  # 剔除海拔干扰, 水循环驱动力, 太阳辐射能量, 相对湿度趋势
                    #                'Geo_Zoning_Season', 'Geo_Zoning_Continentality', 'Geo_Zoning_Lat_Abs', 'Geo_Zoning_Lat_2',
                    #                'Geo_Zoning_Atmos_Thickness', 'Geo_Zoning_Atmos_Log_Elevation']
                    #                # 南北半球季节区分, 沿海内陆区分, 赤道极地区分(线性, 非线性), 海拔高度区分(e函数, log函数)
                    Date = pd.to_datetime(input_chunk['DATE'])
                    LATITUDE = input_chunk['LATITUDE']
                    LONGITUDE = input_chunk['LONGITUDE']
                    ELEVATION = input_chunk['ELEVATION']
                    if 'TEMP' in input_chunk.columns:
                        TEMP = input_chunk['TEMP']
                    else:
                        TEMP = None
                    temp_ = 'C'
                    # STP = input_chunk['STP']
                    # SLP = input_chunk['SLP']
                    # DEWP = input_chunk['DEWP']
                    # dewp_ = 'C'
                    # WDSP = input_chunk['WDSP']
                    if 'STP' in input_chunk.columns:
                        STP = input_chunk['STP']
                    else:
                        STP = None
                    if 'SLP' in input_chunk.columns:
                        SLP = input_chunk['SLP']
                    else:
                        SLP = None
                    if 'DEWP' in input_chunk.columns:
                        DEWP = input_chunk['DEWP']
                    else:
                        DEWP = None
                    dewp_ = 'C'
                    if 'WDSP' in input_chunk.columns:
                        WDSP = input_chunk['WDSP']
                    else:
                        WDSP = None
                    loc = self.__station_3d[sid][-1]
                    if type(loc) != str:
                        self.stop_event.set()
                        raise Exception(
                            '[❌ code error] worker_thread called, provider_thread error, loc in self.__station_3d[sid] error')
                    dem_path = None
                    for filename in os.listdir(self.__dem_path):
                        if filename.lower().endswith(('.tif', '.tiff')) and \
                                all(keyword in filename for keyword in [loc, '30m', 'NASA']):
                            dem_path = os.path.join(self.__dem_path, filename)
                            break
                    if TEMP is not None and STP is not None and SLP is not None:
                        SLP_Residual_DataFrame = calc_slp_residual(ELEVATION, TEMP, STP, SLP, temp_)
                    else:
                        SLP_Residual_DataFrame = None
                    if TEMP is not None:
                        PETEst_Sun_DataFrame = calc_petEst_daylightDuration_dailyRadiation(Date, LATITUDE, TEMP, temp_)
                    else:
                        PETEst_Sun_DataFrame = None
                    if TEMP is not None and DEWP is not None and WDSP is not None:
                        Fog_Risk_DataFrame = calc_fog_risk(TEMP, DEWP, WDSP, temp_, dewp_)
                    else:
                        Fog_Risk_DataFrame = None
                    Geo_Zoning_DataFrame = calc_geo_zoning(Date, LATITUDE, LONGITUDE, ELEVATION, dem_path)
                    Calc_DataFrame = pd.concat(
                        [df for df in [SLP_Residual_DataFrame, PETEst_Sun_DataFrame, Fog_Risk_DataFrame,
                                       Geo_Zoning_DataFrame] if df is not None], axis=1)
                    # input_chunk = pd.concat([input_chunk, Calc_DataFrame.loc[:, self.__columns['calc_column']]], axis=1)
                    input_chunk = pd.concat([input_chunk, Calc_DataFrame], axis=1)  # 更改为根据时序特征决定物理驱动模型输出哪些特征，先不切
                    input_chunk.columns = input_chunk.columns.str.strip()
                    input_Normal_chunk = input_chunk.drop(self.__columns['station_id'], axis=1).drop(
                        self.__columns['date_column'], axis=1)

                    # ================
                    # === 归一化参数 ===
                    # ================
                    # === 最大最小值 ===
                    # INPUT
                    cur_max = input_Normal_chunk.max()
                    cur_min = input_Normal_chunk.min()
                    if norparam_cache["max_values_input"] is None:
                        norparam_cache["max_values_input"] = cur_max.values
                        norparam_cache["min_values_input"] = cur_min.values
                    else:
                        norparam_cache["max_values_input"] = np.maximum(norparam_cache["max_values_input"],
                                                                        cur_max.values)
                        norparam_cache["min_values_input"] = np.minimum(norparam_cache["min_values_input"],
                                                                        cur_min.values)
                    # OUTPUT
                    cur_max_out = output_chunk.max()
                    cur_min_out = output_chunk.min()
                    if norparam_cache["max_values_output"] is None:
                        norparam_cache["max_values_output"] = cur_max_out.values
                        norparam_cache["min_values_output"] = cur_min_out.values
                    else:
                        norparam_cache["max_values_output"] = np.maximum(norparam_cache["max_values_output"],
                                                                         cur_max_out.values)
                        norparam_cache["min_values_output"] = np.minimum(norparam_cache["min_values_output"],
                                                                         cur_min_out.values)
                    # === 均值方差 ===
                    # 更新样本总数
                    norparam_cache["count"] += len(input_Normal_chunk)
                    # INPUT
                    if norparam_cache["sum_input"] is None:
                        norparam_cache["sum_input"] = input_Normal_chunk.sum()
                        norparam_cache["sum_sq_input"] = (input_Normal_chunk ** 2).sum()
                    else:
                        norparam_cache["sum_input"] += input_Normal_chunk.sum()
                        norparam_cache["sum_sq_input"] += (input_Normal_chunk ** 2).sum()
                    # OUTPUT
                    if norparam_cache["sum_output"] is None:
                        norparam_cache["sum_output"] = output_chunk.sum()
                        norparam_cache["sum_sq_output"] = (output_chunk ** 2).sum()
                    else:
                        norparam_cache["sum_output"] += output_chunk.sum()
                        norparam_cache["sum_sq_output"] += (output_chunk ** 2).sum()

                    del input_Normal_chunk

                    # ==================
                    # === 拼接完整数据 ===
                    # ==================
                    station_buffer = pd.concat([input_chunk, output_chunk], axis=1)

                    # 写入队列
                    task_package = {
                        'data': station_buffer.drop(self.__columns['station_id'], axis=1),
                        'mode': 'w',
                        'id': sid
                    }
                    self.saver_queue_split.put(task_package)
                    task_package = {
                        'data': station_buffer,
                        'mode': 'a',
                    }
                    self.saver_queue_cleaned.put(task_package)
                    station_buffer = (station_buffer - cur_min) / (cur_max - cur_min)
                    task_package = {
                        'data': station_buffer,
                        'mode': 'a',
                    }
                    self.saver_queue_normal.put(task_package)
                    # 清除一次task的计数
                    self.cleaner_queue.task_done()
            except Exception as e:
                self.stop_event.set()
                raise Exception(f"👷 [worker] 发生错误: {e}")

        print(f"👷 [worker] 数据清洗完毕。")

        if norparam_cache["count"] != 0:
            # write
            # --- 计算 Input 的最终统计量 ---
            # 1. 均值 = 总和 / 数量
            mean_input = norparam_cache["sum_input"] / norparam_cache["count"]
            # 2. 方差 = (平方和 / 数量) - (均值 ^ 2)
            # 公式推导：Var(X) = E[X^2] - (E[X])^2
            variance_input = (norparam_cache["sum_sq_input"] / norparam_cache["count"]) - (mean_input ** 2)
            # 3. 标准差 = sqrt(方差)
            # 使用 np.maximum(..., 0) 防止因浮点数精度问题导致方差出现微小的负数
            std_input = np.sqrt(np.maximum(variance_input, 0))

            # --- 计算 Output 的最终统计量 ---
            mean_output = norparam_cache["sum_output"] / norparam_cache["count"]
            variance_output = (norparam_cache["sum_sq_output"] / norparam_cache["count"]) - (mean_output ** 2)
            std_output = np.sqrt(np.maximum(variance_output, 0))

            self.__normal_params = {
                "max_values_input": norparam_cache['max_values_input'],
                "min_values_input": norparam_cache['min_values_input'],
                "max_values_output": norparam_cache['max_values_output'],
                "min_values_output": norparam_cache['min_values_output'],
                "mean_values_input": mean_input, "std_values_input": std_input,
                "mean_values_output": mean_output, "std_values_output": std_output,
            }

            for key, value in self.__normal_params.items():
                if hasattr(value, "tolist"):  # 判断是否为 numpy array 或 pandas series
                    self.__normal_params[key] = value.tolist()
                else:
                    # 如果是单个 numpy 数值 (如 np.float32)，转为原生 python 类型
                    self.__normal_params[key] = value.item() if hasattr(value, "item") else value

            with open(self.__normalization_params_save_path, "w", encoding="utf-8") as f:
                f.write(jsonDumps(self.__normal_params))
                f.close()

        print(f"👷 [worker] 配置文件已保存, 任务完毕, 下线。")

    def consumer_thread_split(self):
        print("✏️ [writer 1] 启动 (支持 CSV + NPY 双存)...")
        self.guardian.record_TID(threading.get_ident(), '✏️ writer 1')
        while True:
            try:
                task_package = self.saver_queue_split.get()
                mode = task_package['mode']
                if mode == "STOP":
                    break
                elif mode == "WAIT":
                    print("✏️ [writer 1] 收到，等待下一组任务...")
                    continue
                station_buffer = task_package["data"]
                sid = task_package['id']
                save_path = os.path.join(self.__temporal_dataset_save_base, str(sid) + '.csv')
                if mode == "a" and os.path.exists(save_path):  # 只有追加保存时，需要判断是否存在，如果已经存在就不要表头
                    header = False
                else:
                    header = True
                # 保存csv
                with open(os.path.join(self.__temporal_dataset_save_base, str(sid) + '.csv'), mode,
                          encoding="utf-8", newline='') as f:
                    station_buffer.to_csv(f, header=header, index=False)
                # 保存npy
                if mode == "w" and not os.path.exists(
                        os.path.join(self.__temporal_dataset_save_base, str(sid) + '.npz')):
                    # 写入模式 (w) 且 文件不存在：直接保存
                    np.save(os.path.join(self.__temporal_dataset_save_base, str(sid) + '.npy'), station_buffer.values)
                    # np.savez_compressed(os.path.join(self.__temporal_dataset_save_base, str(sid) + '.npz'), data=station_buffer.values, columns=station_buffer.columns)
                    # **必须将列名转换为Python字符串列表**
                    columns_list = station_buffer.columns.tolist()
                    # 确保列名是普通字符串（非pandas Index对象）
                    columns_list = [str(col) for col in columns_list if col != self.__columns['date_column']]
                    # 保存为NPZ（包含data和columns两个数组）
                    np.savez_compressed(
                        os.path.join(self.__temporal_dataset_save_base, str(sid) + '.npz'),
                        data=station_buffer.drop(self.__columns['date_column'], axis=1).values.astype(np.float32),  # 强制类型
                        date=station_buffer.loc[:, self.__columns['date_column']].values,
                        columns=columns_list,
                        date_column = self.__columns['date_column']
                    )

            except Exception as e:
                self.stop_event.set()
                raise Exception(f"✏️ [writer 1] 发生错误: {e}")
        self._merged[0] = True

    def consumer_thread_cleaned(self):
        print("🖍️ [writer 2] 启动...")
        self.guardian.record_TID(threading.get_ident(), '🖍️ writer 2')
        while True:
            try:
                task_package = self.saver_queue_cleaned.get()
                mode = task_package['mode']
                if mode == "STOP":
                    break
                station_buffer = task_package["data"]
                if mode == "a" and os.path.exists(self.__cleaned_data_save_path):
                    header = False
                else:
                    header = True
                with open(self.__cleaned_data_save_path, mode, encoding="utf-8", newline='') as f:
                    station_buffer.to_csv(f, header=header, index=False)
            except Exception as e:
                self.stop_event.set()
                raise Exception(f"🖍️ [writer 2] 发生错误: {e}")
        self._merged[1] = True

    def consumer_thread_normal(self):
        print("🖋️ [writer 3] 启动...")
        self.guardian.record_TID(threading.get_ident(), '🖋️ writer 3')
        while True:
            try:
                task_package = self.saver_queue_normal.get()
                mode = task_package['mode']
                if mode == "STOP":
                    break
                station_buffer = task_package["data"]
                if mode == "a" and os.path.exists(self.__normalized_data_save_path):
                    header = False
                else:
                    header = True
                with open(self.__normalized_data_save_path, mode, encoding="utf-8", newline='') as f:
                    station_buffer.to_csv(f, header=header, index=False)
            except Exception as e:
                self.stop_event.set()
                raise Exception(f"🖋️️ [writer 3] 发生错误: {e}")
        self._merged[2] = True

    def _heartbeat_loop(self):
        """后台线程：不断发送心跳"""
        while not self.stop_event.is_set():  # 需要一个 stop_event 来控制这个线程
            self.guardian.record_heartbeat()
            time.sleep(2)  # 每2秒发送一次心跳信号
