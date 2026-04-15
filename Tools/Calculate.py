import math
from typing import Union

import numpy as np
import pandas as pd
from osgeo import gdal, gdalconst


def calculate_solar_radiation(year: pd.Series, month: pd.Series, day: pd.Series,
                              latitude: Union[int, pd.Series]):
    """
    根据日序列号、纬度, 理论计算日地距离, 太阳赤纬弧度, 太阳赤纬, 日没时角, 日照时长, 日辐射量
    :param year: 年, pandas Series
    :param month: 月, pandas Series
    :param day: 日, pandas Series
    :param latitude: 纬度, pandas Series 或 int（广播）
    :return: 日照时长, 日辐射量
    """
    # 转弧度
    latitude_rad = np.radians(latitude)

    # 定义当前日期
    current_date = pd.to_datetime((year, month, day))
    # 获取今年的第一天
    first_day_of_year = pd.to_datetime(year, format='%Y')
    # 计算日期序列号
    day_count = (current_date - first_day_of_year).dt.days + 1
    day_count = np.where((year % 4 == 0) & (month > 2), day_count - 1, day_count)

    # 计算日地距离
    sun_earth_distance = 1 + 0.033 * np.cos(day_count * 2 * np.pi / 365)

    # 计算太阳赤纬弧度
    sun_declination = 0.409 * np.sin(2 * np.pi * day_count / 365 - 1.39)

    # 计算日没时角
    sun_noon_angle = np.arccos(-1 * np.tan(latitude_rad) * np.tan(sun_declination))

    # 计算日照时长
    sun_duration = 24 * sun_noon_angle / np.pi

    # 计算日辐射量
    sun_radiation = (24 * 60 / np.pi) * 0.082 * sun_earth_distance * (
            sun_noon_angle * np.sin(latitude_rad) * np.sin(sun_declination) +
            np.cos(latitude_rad) * np.cos(sun_declination) * np.sin(sun_noon_angle))

    return sun_duration, sun_radiation


def transform_slope_aspect(dem_file, x, y):
    """
    获取指定坐标的坡度和坡向
    :param dem_file: DEM文件路径
    :param x: 指定的X坐标（地理坐标）
    :param y: 定的Y坐标（地理坐标）
    :return: 坡度（度）和坡向（度）
    """
    # 打开DEM文件
    dataset = gdal.Open(dem_file, gdalconst.GA_ReadOnly)
    if dataset is None:
        raise ValueError("无法打开DEM文件, 请检查文件路径是否正确！")

    # 获取地理坐标到像素坐标的转换参数
    geotransform = dataset.GetGeoTransform()
    if geotransform is None:
        raise ValueError("DEM文件中缺少地理坐标转换参数！")

    # 将地理坐标转换为像素坐标
    col = int((x - geotransform[0]) / geotransform[1])
    row = int((y - geotransform[3]) / geotransform[5])

    # 获取DEM数据
    band = dataset.GetRasterBand(1)
    dem_data = band.ReadAsArray()

    # 获取像素大小
    cell_size_x = geotransform[1]
    cell_size_y = -geotransform[5]

    # 获取当前像素及其周围像素的高程值
    try:
        z = dem_data[row, col]
        z_n = dem_data[row - 1, col]  # 北
        z_s = dem_data[row + 1, col]  # 南
        z_e = dem_data[row, col + 1]  # 东
        z_w = dem_data[row, col - 1]  # 西
    except IndexError:
        raise ValueError("指定的坐标超出了DEM数据范围！")

    # 计算坡度和坡向
    dz_dx = (z_e - z_w) / (2 * cell_size_x)
    dz_dy = (z_n - z_s) / (2 * cell_size_y)
    slope = math.degrees(math.atan(math.sqrt(dz_dx ** 2 + dz_dy ** 2)))
    aspect = math.degrees(math.atan2(dz_dy, -dz_dx))
    if aspect < 0:
        aspect += 360

    return slope, aspect


def batch_transform_slope_aspect(dem_file, coords):
    """批量计算坡度坡向"""
    dataset = gdal.Open(dem_file, gdalconst.GA_ReadOnly)
    geotransform = dataset.GetGeoTransform()
    band = dataset.GetRasterBand(1)
    dem_data = band.ReadAsArray()

    slopes, aspects = [], []
    for x, y in coords:
        col = int((x - geotransform[0]) / geotransform[1])
        row = int((y - geotransform[3]) / geotransform[5])
        try:
            # 计算周围高程值（中心差分法）
            z = dem_data[row, col]
            z_n = dem_data[row - 1, col]  # 北
            z_s = dem_data[row + 1, col]  # 南
            z_e = dem_data[row, col + 1]  # 东
            z_w = dem_data[row, col - 1]  # 西

            # 计算梯度
            dz_dx = (z_e - z_w) / (2 * geotransform[1])
            dz_dy = (z_n - z_s) / (2 * abs(geotransform[5]))
            slope = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))
            aspect = np.degrees(np.arctan2(dz_dy, -dz_dx)) % 360

            slopes.append(slope)
            aspects.append(aspect)
        except IndexError:
            slopes.append(np.nan)
            aspects.append(np.nan)
    return slopes, aspects


def latlon_to_CC3d(longitude, latitude, radius=6371):
    """
    将经纬度转换为三维笛卡尔坐标。

    参数：
        latitude (float): 纬度（度数）
        longitude (float): 经度（度数）
        radius (float): 平均地球半径（默认为 6371 km, 新疆地区）

    返回：
        tuple: (input_map, y, z)
    """
    # 将经纬度从度数转换为弧度
    lat_rad = np.radians(latitude)
    lon_rad = np.radians(longitude)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)

    return x, y, z


def trans_LonLat2CCDEM(coords, dem_path):
    df = pd.DataFrame.from_dict(
        coords,
        orient='index',
        columns=['longitude', 'latitude', 'altitude']
    )
    # 计算三维坐标（向量化加速）
    df['input_map'], df['y'], df['z'] = latlon_to_CC3d(
        df['longitude'].values,
        df['latitude'].values
    )
    # 批量计算坡度坡向
    coords = df[['longitude', 'latitude']].values
    slopes, aspects = batch_transform_slope_aspect(dem_path, coords)
    df['slope'] = slopes
    df['aspect'] = aspects
    # 转回字典格式（列表值）
    return df[['input_map', 'y', 'z', 'slope', 'aspect', 'altitude']] \
        .apply(lambda row: row.tolist(), axis=1) \
        .to_dict()


# ==============================
# === SLP_Residual (气压残差) ===
# ==============================
def calc_slp_residual(elevation, temp, stp, slp_obs, temp_='C'):
    """
    计算海平面气压残差。
    通过物理压高公式，根据测站气压和温度推算理论海平面气压，
    并计算其与观测值的差值，用于修正气压数据。

    参数:
        stp (float): 测站气压 (hPa/mb)
        slp_obs (float): 观测海平面气压 (hPa/mb)
        elevation (float): 海拔高度 (米)
        temp (float): 气温数值
        temp_ (str): 气温单位 ('F', 'C', 或 'K')

    返回:
        float: 气压残差 (观测值 - 理论值)
    """
    if temp_ == 'F':
        temp = (temp - 32) * (5 / 9) + 273.15
    elif temp_ == 'K':
        temp = temp
    elif temp_ == 'C':
        temp = temp + 273.15
    else:
        raise ValueError("Invalid temp_. Use 'F', 'C', or 'K'.")

    g = 9.80665  # 重力加速度
    M = 0.0289644  # 干空气摩尔质量 (kg/mol)
    R = 8.31446  # 通用气体常数 (J/(mol·K))

    temp = np.clip(temp, 1, None)

    exponent = (g * M * elevation) / (R * temp)
    slp_calc = stp * np.exp(exponent)

    return slp_obs - slp_calc


# ==========================
# === PET_Est (潜在蒸散发) ===
# ==========================
def calc_pet_est(date_obj, latitude, temp, temp_):
    """
    计算潜在蒸散发估算值。
    基于温度和日照时长（由经纬度和日期计算得出）估算水分蒸发潜力。

    参数:
        temp (float): 气温数值
        temp_ (str): 气温单位 ('F', 'C', 或 'K')
        latitude (float): 纬度 (-90 ~ 90)
        date_obj (datetime): 日期对象，用于计算太阳赤纬

    返回:
        float: 潜在蒸散发估算值 (基于摄氏度比例)
    """
    # --- 第一步：统一转换为摄氏度 ---
    if temp_ == 'F':
        temp = (temp - 32) * (5 / 9)
    elif temp_ == 'K':
        temp = temp - 273.15
    elif temp_ == 'C':
        temp = temp
    else:
        raise ValueError("Invalid temp_.")

    # --- 第二步：计算日照时长 ---
    doy = date_obj.timetuple().tm_yday
    delta = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)  # 太阳赤纬

    lat_rad = np.radians(latitude)
    cos_ws = -np.tan(lat_rad) * np.tan(delta)
    cos_ws = np.clip(cos_ws, -1, 1)  # 防止极昼极夜数学错误

    omega_s = np.arccos(cos_ws)
    daylight_hours = (24 / np.pi) * omega_s

    # --- 第三步：计算 PET ---
    pet_est = temp * (daylight_hours / 12.0)

    return pet_est


# ========================
# === Fog_Risk (雾风险) ===
# ========================
def calc_fog_risk(temp, dewp_val, wdsp, temp_, dewp_unit):
    """
    计算雾风险指数。
    基于相对湿度（由气温和露点计算）和风速，评估形成雾的可能性。

    参数:
        temp (float): 气温数值
        temp_ (str): 气温单位 ('F', 'C', 或 'K')
        dewp_val (float): 露点数值
        dewp_unit (str): 露点单位 ('F', 'C', 或 'K')
        wdsp (float): 风速

    返回:
        float: 雾风险指数 (0 ~ 1)
    """
    # --- 第一步：气温转摄氏度 ---
    if temp_ == 'F':
        temp = (temp - 32) * (5 / 9)
    elif temp_ == 'K':
        temp = temp - 273.15
    else:
        temp = temp

    # --- 第二步：露点转摄氏度 ---
    if dewp_unit == 'F':
        td_c = (dewp_val - 32) * (5 / 9)
    elif dewp_unit == 'K':
        td_c = dewp_val - 273.15
    else:
        td_c = dewp_val

    # --- 第三步：计算相对湿度 (Magnus 公式) ---
    numerator = 17.625 * td_c
    denominator = 243.04 + td_c
    denominator = np.clip(denominator, 1e-6, None)

    rh = 100 * np.exp((numerator / denominator) - (17.625 * temp / (243.04 + temp)))
    rh = np.clip(rh, 0, 100)

    # --- 第四步：结合风速计算风险 ---
    # 湿度越高、风速越小，风险越大
    fog_risk = (rh / 100.0) ** 3 * np.exp(-wdsp / 5.0)

    return fog_risk


# ============================
# === Geo_Zoning (地理特征) ===
# ============================
def calc_geo_zoning(date_obj, latitude, longitude, elevation):
    """
    计算综合地理特征。
    基于经纬度、海拔和日期，生成用于气象预测的物理特征，
    包括季节相位、纬度热力、大陆度、大气厚度和对数海拔。

    参数:
        latitude (float): 纬度 (-90 ~ 90)
        longitude (float): 经度 (-180 ~ 180)
        elevation (float): 海拔 (米)
        date_obj (datetime): 日期对象

    返回:
        dict: 包含5个特征的字典
            - Season_Phase: 季节信号 (-1 ~ 1)
            - Lat_Abs: 纬度距离信号 (0 ~ 90)
            - Continentality: 海陆信号 (0 ~ 1)
            - Atmos_Thickness: 海拔物理信号 (0 ~ 1)
            - Log_Elevation: 海拔统计信号 (0 ~ ~9.2)
    """
    # --- 1. 季节相位 ---
    hemisphere = 1 if latitude >= 0 else -1
    doy = date_obj.timetuple().tm_yday
    seasonal_phase = hemisphere * np.sin(2 * np.pi * doy / 365)

    # --- 2. 纬度热力特征 ---
    lat_abs = abs(latitude)

    # --- 3. 大陆度指数 ---
    dist_atlantic = abs(longitude)
    dist_pacific = 180 - abs(longitude)
    min_dist_to_ocean = min(dist_atlantic, dist_pacific)
    continentality = np.tanh(min_dist_to_ocean * 0.05)

    # --- 4. 大气厚度因子 ---
    atmospheric_thickness = np.exp(-elevation / 8400.0)

    # --- 5. 对数海拔 ---
    log_elevation = np.log(elevation + 1)

    return {
        "Season_Phase": seasonal_phase,
        "Lat_Abs": lat_abs,
        "Continentality": continentality,
        "Atmos_Thickness": atmospheric_thickness,
        "Log_Elevation": log_elevation
    }



