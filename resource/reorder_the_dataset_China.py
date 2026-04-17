import os
import pandas as pd
import json
import time
from pathlib import Path
from Function_identify_location.by_GeoJson import China_GeoJson


# --- 配置区域 ---
# 请将此处修改为你电脑上的实际路径
# 例如: r'D:\Data\WeatherData'
input_folder = 'world/'  # 原始数据文件夹
output_dir = 'china/'


def merge_stations_streaming(source_dir, output_dir):
    """
    流式处理版本：逐个站点合并，避免内存溢出。
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 第一次遍历：只记录文件路径，不读取内容
    # 字典结构: {'站点A': ['path/to/2020/A.csv', 'path/to/2022/A.csv'], ...}
    station_files = {}

    print("🔍 正在扫描并建立文件索引...")
    for year_folder in sorted(source_path.iterdir()):
        if year_folder.is_dir():
            for csv_file in year_folder.glob("*.csv"):
                station_id = csv_file.stem
                if station_id not in station_files:
                    station_files[station_id] = []
                station_files[station_id].append(csv_file)

    print(f"📊 扫描完成。共发现 {len(station_files)} 个唯一站点，开始合并...\n")

    # 2. 第二次遍历：逐个站点处理，处理完即释放内存
    for i, (station_id, file_paths) in enumerate(station_files.items(), 1):
        output_file_path = output_path / f"{station_id}.csv"

        # 为当前站点创建一个列表，用于暂存该站点的DataFrame
        # 这个列表只包含一个站点的数据，内存占用可控
        df_list = []

        print(f"[{i}/{len(station_files)}] 正在合并站点: {station_id}")

        for file_path in file_paths:
            # 读取单个文件
            df = pd.read_csv(file_path)
            # 可选：添加年份信息
            df['Source_Year'] = file_path.parent.name
            df_list.append(df)
            # 读取完并加入列表后，原始的 df 对象就可以被垃圾回收了

        # 合并当前站点的所有数据
        merged_df = pd.concat(df_list, ignore_index=True)

        # 立即写入硬盘
        merged_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')

        # 关键一步：处理完一个站点后，清空列表，显式释放内存
        del df_list
        del merged_df

    print(f"\n✅ 所有站点处理完毕！")


from concurrent.futures import ThreadPoolExecutor, as_completed


def process_single_station(station_id, file_paths, output_dir):
    """
    处理单个站点的所有文件，并将其合并保存。
    这个函数将被多线程并发执行。
    """
    output_file_path = output_dir / f"{station_id}.csv"
    df_list = []

    try:
        # 读取当前站点的所有年份文件
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            df['Source_Year'] = file_path.parent.name
            df_list.append(df)

        # 合并并保存
        merged_df = pd.concat(df_list, ignore_index=True)
        merged_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')

        return f"✅ 成功: {station_id}"
    except Exception as e:
        return f"❌ 失败: {station_id}, 错误: {e}"
    finally:
        # 无论成功失败，都清理内存
        del df_list


def merge_stations_multithreaded(source_dir, output_dir, max_workers=8):
    """
    使用多线程并行处理所有站点。

    参数:
    source_dir: 源文件夹路径
    output_dir: 输出文件夹路径
    max_workers: 最大线程数，默认为8，可根据电脑性能调整
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 扫描并建立文件索引 (与之前相同)
    station_files = {}
    print("🔍 正在扫描并建立文件索引...")

    def is_station_in_china(file_path, lat_col='LATITUDE', lon_col='LONGITUDE'):
        """
        辅助函数：读取文件的第一行来获取经纬度，并调用你的类进行判断
        """
        try:
            # 关键点：使用 nrows=1 只读取第一行，速度极快且不占内存
            df_head = pd.read_csv(file_path, nrows=1)

            # 检查列是否存在
            if lat_col not in df_head.columns or lon_col not in df_head.columns:
                print(f"   ⚠️ 警告: 文件 {file_path} 缺少经纬度列 ({lat_col}, {lon_col})，默认视为中国站点。")
                return True

            lat = df_head[lat_col].iloc[0]
            lon = df_head[lon_col].iloc[0]

            # 调用你定义的类方法
            # 注意：请根据你实际的类定义调整调用方式
            # 如果你的类需要实例化，请确保在多线程环境下它是线程安全的，或者在这里实例化
            return China_GeoJson.IDENTIFY_China(lon, lat)

        except Exception as e:
            print(f"   ❌ 读取经纬度失败 {file_path}: {e}")
            return False  # 如果读取失败，为了安全起见，可以选择丢弃或保留，这里设为丢弃

    for year_folder in sorted(source_path.iterdir()):
        if year_folder.is_dir():
            for csv_file in year_folder.glob("*.csv"):
                station_id = csv_file.stem
                if station_id not in station_files:
                    station_files[station_id] = []
                station_files[station_id].append(csv_file)

    print(f"📊 扫描完成。共发现 {len(station_files)} 个唯一站点。\n")
    print(f"c启动多线程处理，线程池大小: {max_workers}...")

    # 2. 使用线程池并行处理
    # ThreadPoolExecutor 会自动管理线程的创建和销毁
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务到线程池
        # future 代表一个将来会完成的任务
        futures = [
            executor.submit(process_single_station, station_id, files, output_path)
            for station_id, files in station_files.items()
        ]

        # 等待所有任务完成，并打印结果
        for future in as_completed(futures):
            # future.result() 会获取任务函数的返回值
            print(future.result())

    print(f"\n🎉 所有任务处理完毕！")


def extract_coordinates(file_path, lat_col='LATITUDE', lon_col='LONGITUDE'):
    """
    辅助函数：读取文件的第一行来获取经纬度，并调用你的类进行判断
    """
    try:
        # 关键点：使用 nrows=1 只读取第一行，速度极快且不占内存
        df_head = pd.read_csv(file_path, nrows=1)

        # 检查列是否存在
        if lat_col not in df_head.columns or lon_col not in df_head.columns:
            print(f"   ⚠️ 警告: 文件 {file_path} 缺少经纬度列 ({lat_col}, {lon_col})，默认视为中国站点。")
            return True

        lat = df_head[lat_col].iloc[0]
        lon = df_head[lon_col].iloc[0]

        # 调用你定义的类方法
        # 注意：请根据你实际的类定义调整调用方式
        # 如果你的类需要实例化，请确保在多线程环境下它是线程安全的，或者在这里实例化
        return lon, lat

    except Exception as e:
        print(f"   ❌ 读取经纬度失败 {file_path}: {e}")
        return False  # 如果读取失败，为了安全起见，可以选择丢弃或保留，这里设为丢弃


def merge_china_stations_with_geo_filter(source_dir, output_dir, max_workers=8):
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    geojson = China_GeoJson()

    # 1. 建立索引并进行地理围栏判断
    station_files = {}  # 存储中国站点
    drop_stations = {}  # 存储非中国站点，方便后续排查

    print("🔍 正在扫描目录并进行地理位置过滤...")

    # 遍历所有年份文件夹
    year_folders = sorted([f for f in source_path.iterdir() if f.is_dir()])

    for year_folder in year_folders:
        print(f"📅 扫描年份: {year_folder.name}")

        for csv_file in year_folder.glob("*.csv"):
            station_id = csv_file.stem

            # --- 核心修改逻辑开始 ---

            # 只有当这个站点ID第一次出现时，才进行地理位置判断
            if station_id not in station_files and station_id not in drop_stations:

                # 中国内地编号以5开头，凡是不是5开头的保险起见都去掉
                if not station_id.startswith('5'):
                    drop_stations[station_id] = []
                    continue

                # 调用辅助函数判断是否在中国
                lon, lat = extract_coordinates(csv_file)
                if geojson.IDENTIFY_China(lon, lat):
                    # 是中国站点，初始化列表
                    station_files[station_id] = []
                    print(f"   ✅ 确认站点 {station_id} 位于中国") # 调试用
                else:
                    # 不是中国站点，加入丢弃列表
                    drop_stations[station_id] = []
                    print(f"   🚫 排除站点 {station_id} (不在中国境内)")

            # 根据判断结果，将文件路径加入对应的列表
            if station_id in station_files:
                station_files[station_id].append(csv_file)
            elif station_id in drop_stations:
                drop_stations[station_id].append(csv_file)

            # --- 核心修改逻辑结束 ---

    print(f"\n📊 扫描完成。")
    print(f"   中国站点数: {len(station_files)}")
    print(f"   排除站点数: {len(drop_stations)}")

    # # 可选：保存被排除的站点列表到文件，方便后续查看
    # if drop_stations:
    #     with open(output_path / "dropped_stations_list.txt", 'w', encoding='utf-8') as f:
    #         f.write("以下站点因不在中国境内被排除:\n")
    #         f.write("\n".join(drop_stations.keys()))
    #     print(f"   ℹ️ 被排除的站点列表已保存至: {output_path / 'dropped_stations_list.txt'}")

    # 2. 保存 YAML 配置文件
    # 保存一个包含所有有效站点ID的列表
    valid_station_ids = list(station_files.keys())
    json_file_path = f"{output_path}/stationIDs.json"

    try:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            # default_flow_style=False 让 YAML 以列表形式展示，更易读
            json.dump({'station_ids': valid_station_ids}, f)
        print(f"\n📄 中国大陆站点已保存至: {json_file_path}")
    except Exception as e:
        print(f"   read_excelYAML 保存失败: {e}")

    # 3. 多线程合并数据
    print(f"\n🚀 开始多线程合并中国站点数据...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_station, station_id, files, output_path)
            for station_id, files in station_files.items()
        ]

        for future in as_completed(futures):
            print(future.result())

    print(f"\n🎉 所有任务处理完毕！")


#写入同一个文件组成单文件时空序列数据集
from pathlib import Path
import threading
import queue
from Guardian import ProcessGuardian


class OrderedStreamMerger:
    def __init__(self, source_dir, output_dir, output_filename="all_china_stations.csv"):
        self.source_path = Path(source_dir)
        self.output_path = Path(output_dir)
        os.makedirs(self.output_path, exist_ok=True)
        self.output_file = os.path.join(output_dir, output_filename)
        self.geojson = China_GeoJson()

        # 核心队列
        self.file_queue = queue.Queue(maxsize=2)
        self.data_queue = queue.Queue()

        # 状态标记
        self.all_files_scanned = False
        self.valid_station_ids = []

        # 监护者
        self.guardian = ProcessGuardian(os.getpid())

        # 用于控制线程停止的事件
        self.stop_event = threading.Event()

    def extract_coordinates(self, file_path, lat_col='LATITUDE', lon_col='LONGITUDE'):
        try:
            df_head = pd.read_csv(file_path, nrows=1)
            if lat_col not in df_head.columns or lon_col not in df_head.columns:
                return None, None
            return df_head[lon_col].iloc[0], df_head[lat_col].iloc[0]
        except:
            return None, None

    # ==========================================
    # 线程 1：调度员
    # 职责：扫描、过滤、【关键排序】、入队
    # ==========================================
    def dispatcher_thread(self):
        print("👮‍♂️ [调度员] 启动，正在扫描并排序...")
        # 向主进程汇报TID
        self.guardian.record_TID(threading.get_ident(), '👮‍♂️ dispatcher')

        station_files_map = {}  # 用于暂存 {站点ID: [文件列表]}

        # 1. 扫描所有文件
        year_folders = sorted([f for f in self.source_path.iterdir() if f.is_dir()])

        for year_folder in year_folders:
            for csv_file in year_folder.glob("*.csv"):
                station_id = csv_file.stem

                # 第一次遇到该站点，进行过滤判断
                if station_id not in station_files_map:
                    if not station_id.startswith('5'): continue

                    lon, lat = self.extract_coordinates(csv_file)
                    if lon is None or not self.geojson.IDENTIFY_China(lon, lat):
                        continue

                    station_files_map[station_id] = []

                # 将文件加入该站点的列表
                station_files_map[station_id].append(csv_file)

        # 2. 【关键步骤】对站点ID进行排序
        # 这样可以保证先处理完站点A的所有年份，再处理站点B
        sorted_station_ids = sorted(station_files_map.keys())
        self.valid_station_ids = sorted_station_ids

        # 3. 将文件按顺序放入队列
        # 这里的顺序是：A_2020, A_2021, B_2020, B_2021...
        for station_id in sorted_station_ids:
            # 对同一个站点的年份文件也排个序（可选）
            files = sorted(station_files_map[station_id], key=lambda x: x.parent.name)
            for f in files:
                self.file_queue.put(f)

        self.all_files_scanned = True
        print(f"👮‍♂️ [调度员] 任务生成完毕，共 {len(sorted_station_ids)} 个站点。")

        # 保存站点ID列表
        self.output_path.mkdir(parents=True, exist_ok=True)
        json_path = self.output_path / "stationIDs.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({'station_ids': self.valid_station_ids}, f, ensure_ascii=False, indent=4)
        print(f"👮‍♂️ [调度员] 相关信息保存完毕，地址：\'{json_path}\' 。")

    # ==========================================
    # 线程 2：处理工
    # 职责：读一个文件，放一个数据
    # ==========================================
    def worker_thread(self):
        print("👷 [处理工] 启动...")
        # 向主进程汇报TID
        self.guardian.record_TID(threading.get_ident(), '👷 worker')
        while True:
            try:
                try:
                    file_path = self.file_queue.get(timeout=1)
                except queue.Empty:
                    if self.all_files_scanned:
                        print("👷 [处理工] 文件读取结束。")
                        self.data_queue.put(None)  # 发送结束信号
                        break
                    continue

                # 读取单个文件
                try:
                    df = pd.read_csv(file_path)

                    # 放入结果队列
                    self.data_queue.put(df)
                except Exception as e:
                    print(f"⚠️ 读取失败 {file_path}: {e}")

                self.file_queue.task_done()
            except Exception as e:
                print(f"👷 [处理工] 发生错误: {e}")

    # ==========================================
    # 线程 3：写入者
    # 职责：简单追加
    # ==========================================
    def writer_thread(self):
        print("✍️ [写入者] 启动，准备追加数据...")
        # 向主进程汇报TID
        self.guardian.record_TID(threading.get_ident(), '✍️ writer')

        first_chunk = True

        with open(self.output_file, 'a', encoding='utf-8-sig', newline='') as f:
            while True:
                df = self.data_queue.get()

                if df is None:
                    print("✍️ [写入者] 文件保存完成。")
                    break

                # 写入数据
                # 如果是第一块数据，写表头；否则只写数据（追加模式）
                df.to_csv(f, index=False, header=first_chunk)

                if first_chunk:
                    first_chunk = False
                    # 获取第一个站点的ID用于提示（可选）
                    if 'Station_ID' in df.columns:
                        print(f"🆕 开始写入站点: {df['Station_ID'].iloc[0]}")
                else:
                    # 简单的进度提示，避免刷屏
                    pass

                self.data_queue.task_done()

    def _heartbeat_loop(self):
        """后台线程：不断发送心跳"""
<<<<<<< Updated upstream
        while not self.stop_event.is_set():  # 需要一个 stop_event 来控制这个线程
=======
        while not self.stop_event.is_set():  # 需要一个 __stop_event 来控制这个线程
>>>>>>> Stashed changes
            self.guardian.record_heartbeat()
            time.sleep(2)  # 每2秒发送一次心跳信号

    def RUN(self):
        # 初始化事件对象用于停止心跳线程
        self.stop_event = threading.Event()

        # 员工 的线程
        t_dispatcher = threading.Thread(target=self.dispatcher_thread)
        t_worker = threading.Thread(target=self.worker_thread)
        t_writer = threading.Thread(target=self.writer_thread)

        # 1. 🛡️ [监护者] 进程的初始化
        self.guardian.init()

        # 2. 🚀 [主进程] 的心跳汇报线程
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
        # 其他的退了，这个肯定也退了（常规）；子进程没有退（异常），那么这个进程就开始杀子进程

        try:
            # 3. 启动 员工 的进程
            t_dispatcher.start()
            t_worker.start()
            t_writer.start()
            heartbeat_thread.start()

            # 等待所有线程结束
            t_dispatcher.join()
            t_worker.join()
            t_writer.join()

            print("🎉 [👮‍👷✍️] 任务完成！")

        except KeyboardInterrupt:
            print("\n⚠️ 捕获到 Ctrl+C，准备退出...")
        except Exception as e:
            print(f"\n❌ 发生严重错误: {e}")
        finally:
            # 4. 无论成功还是失败，都要停止心跳和监护进程（常规退出）
            self.stop_event.set()
            heartbeat_thread.join()
            self.guardian._stop()


if __name__ == "__main__":
    if os.path.exists(input_folder):
        try:
            print("🚀 主程序开始运行...")
            merger = OrderedStreamMerger(input_folder, output_dir)
            merger.RUN()
        except KeyboardInterrupt:
            print("⌨️ 用户中断，正在退出...")
    else:
        print(f"错误：找不到源文件夹 '{input_folder}'，请检查路径。")