__all__ = ['list_parameters', 'read_grib']

import os

import numpy as np

try:
    import pygrib
except ImportError:
    print(f'\033[96m'
          f'              请按以下命令安装 (环境：python3.10)：\n',
          f'                pip3 install proj \n',
          f'                pip3 install pyproj \n',
          f'                pip3 install eccodes  （建议使用pip3工具, 用conda安装后运行遇到过DLL冲突）\n',
          f'                conda install -c conda-forge pygrib  （只能在conda环境中安装, pip安装时报错）\n'
          f'\033[0m')
    raise ImportError(f''
                      f'\033[96m import pygrib \033[0m 未完成环境安装')


def list_parameters(grib_file_path, test=False):
    """
    列出GRIB文件中的所有参数及其详细信息

    Args:
        grib_file_path (str): GRIB文件路径
        test (bool): 测试模式标志, 为True时显示调试信息

    Returns:
        list: 包含每个参数详细信息的字典列表
    """
    if test:
        print(f"Reading grib file: {grib_file_path}   list\n")
    grbs = pygrib.open(grib_file_path)
    parameter_names = set()
    details = []

    for grb in grbs:
        param_name = grb.parameterName
        valid_date = getattr(grb, 'validDate', 'N/A')  # 部分字段可能无时间属性

        parameter_names.add(param_name)
        details.append({
            "Name": grb.name,
            "parameterName": param_name,
            "shortName": grb.shortName,
            "typeOfLevel": grb.typeOfLevel,
            "Level": grb.level,
            "Date": valid_date,
            "Structure": grb.values.shape  # GRIB2的数组是np.ndarray格式
        })

    grbs.close()

    print(" List（parameterName）：")
    for param_name in sorted(parameter_names):
        print(f" - {param_name}")

    print()

    print(" Details（Name, parameterName, shortName, typeOfLevel, Level, Date）：")
    for entry in details:
        print(' - ', entry)

    print()

    return details


def read_grib(grib_file_path, target_variables, test=False):
    """
    读取GRIB文件并提取指定变量的数据

    Args:
        grib_file_path (str): GRIB文件路径
        target_variables (list): 包含变量信息的字典列表, 每个字典应包含:
            'Name' - 变量名称
            'parameterName' - 参数名称
            'typeOfLevel' - 层级类型
            'Level' - 层级值
        test (bool): 测试模式开关, 为True时打印GRIB文件路径

    Returns:
        tuple: 包含三个元素的元组:
            1. grbArrays (list): 包含所有目标变量数据的3D列表（每个变量一个2D数组）
            2. grbArrays_time (str): 数据有效时间的格式化字符串（YYYY-MM-DD HH:MM）

        """
    if not os.path.exists(grib_file_path):
        raise FileNotFoundError(f"GRIB文件不存在: {grib_file_path}")
    if test:
        print(f"Reading grib file: {grib_file_path}   print: {target_variables}\n")
    grbObjects = pygrib.open(grib_file_path)
    grbArrays = []
    grbArrays_time = ''
    # min_value, max_value, mean_value, std_value = [], [], [], []
    for target_variable in target_variables:
        selected_object = grbObjects.select(name=target_variable['Name'],
                                            parameterName=target_variable['parameterName'],
                                            typeOfLevel=target_variable['typeOfLevel'],
                                            level=target_variable['Level'])[0]
        # select(Name, parameterName, shortName, typeOfLevel, level)
        grbArray = selected_object.values  # GRIB2的数组是np.ndarray格式
        # min_value.append(np.min(grbArray))
        # max_value.append(np.max(grbArray))
        # mean_value.append(np.mean(grbArray))
        # std_value.append(np.std(grbArray))
        grbArrays.append(grbArray.tolist())  # grb.value 二维矩阵 W,H -> grbArrays 三维矩阵（通道为target_variables）C,W,H
        grbArrays_time = selected_object.validDate.strftime("%Y-%m-%d %H:%M")
    grbObjects.close()
    return grbArrays, grbArrays_time
