from datetime import date
from datetime import datetime
from typing import Union, List, Tuple

import numpy as np
import pandas as pd
from osgeo import gdal, gdalconst


# ============================
# === 辅助函数：数据格式统一化 ===
# ============================
def to_series(data: Union[List, Tuple, np.ndarray, datetime, date, float, int, str, pd.Series]) -> pd.Series:
    """
    将输入数据转换为 pandas Series，并确保数据类型正确（特别是日期类型）。

    参数:
        data: 输入数据，可以是列表、元组、numpy数组、时间对象、数值、字符串或已有的Series

    返回:
        pd.Series: 转换后的pandas Series对象
    """
    # 1. 如果已经是 Series，直接返回
    if isinstance(data, pd.Series):
        return data

    # 2. 如果是列表或数组，转为 Series
    if isinstance(data, (list, tuple, np.ndarray)):
        return pd.Series(data)

    # 3. 如果是单个时间对象 (datetime 或 date)
    # 必须显式转换为 pd.Timestamp，否则 pd.Series([datetime_obj]) 会生成 dtype=object
    # 导致后续无法使用 .dt 访问器
    if isinstance(data, (datetime, date)):
        return pd.Series([pd.Timestamp(data)])

    # 4. 其他单个值 (数字、字符串等)
    return pd.Series([data])


# ==============================
# === SLP_Residual (气压残差) ===
# ==============================
def calc_slp_residual(elevation: Union[float, List, pd.Series], temp: Union[float, List, pd.Series],
                      stp: Union[float, List, pd.Series], slp_obs: Union[float, List, pd.Series],
                      temp_: str = 'C') -> pd.DataFrame:
    """
    计算海平面气压残差。

    参数:
        elevation: 海拔高度 (米)，支持单个值或批量数据
        temp: 温度，支持单个值或批量数据
        stp: 站点气压，支持单个值或批量数据
        slp_obs: 观测海平面气压，支持单个值或批量数据
        temp_: 温度单位，默认为 'C'，可选 'F', 'C', 'K'

    返回:
        pd.DataFrame: 包含 'SLP_Residual' 列的DataFrame
    """
    # 1. 统一转换为 Series
    elevation = to_series(elevation)
    temp = to_series(temp)
    stp = to_series(stp)
    slp_obs = to_series(slp_obs)

    # 2. 温度单位转换
    if temp_ == 'F':
        temp_k = (temp - 32) * (5 / 9) + 273.15
    elif temp_ == 'K':
        temp_k = temp
    elif temp_ == 'C':
        temp_k = temp + 273.15
    else:
        raise ValueError("Invalid temp_. Use 'F', 'C', or 'K'.")

    # 3. 物理计算
    g = 9.80665
    M = 0.0289644
    R = 8.31446

    temp_k = temp_k.clip(lower=1.0)  # 防止温度过低
    exponent = (g * M * elevation) / (R * temp_k)
    slp_calc = stp * np.exp(exponent)

    residual = slp_obs - slp_calc

    # 4. 返回指定格式的 DataFrame
    return pd.DataFrame({'SLP_Residual': residual})


# ========================================================================
# === PET_Est & DaylightDuration & DailyRadiation (蒸散发、日照与日辐射量) ===
# ========================================================================
def calc_petEst_daylightDuration_dailyRadiation(date_obj: Union[pd.Series, List, datetime, date],
                                                latitude: Union[float, List, pd.Series],
                                                temp: Union[float, List, pd.Series],
                                                temp_: str = 'C') -> pd.DataFrame:
    """
    计算潜在蒸散发、日照时长及日天文辐射量。
    支持批量处理：date_obj 必须是 pandas Series (日期列)。

    参数:
        date_obj: 日期列，支持单个日期或pandas Series (datetime64类型)
        latitude: 纬度，支持单个值或批量数据
        temp: 气温，支持单个值或批量数据
        temp_: 气温单位 ('C', 'F', 'K')

    返回:
        pd.DataFrame: 包含 ['PET_Est', 'DaylightDuration', 'DailyRadiation'] 列的DataFrame
    """
    # 1. 统一转换为 Series
    latitude = to_series(latitude)
    temp = to_series(temp)
    date_obj = to_series(date_obj)

    # 2. 温度单位转换 (向量化)
    if temp_ == 'F':
        temp_c = (temp - 32) * (5 / 9)
    elif temp_ == 'K':
        temp_c = temp - 273.15
    elif temp_ == 'C':
        temp_c = temp
    else:
        raise ValueError("Invalid temp_. Use 'F', 'C', or 'K'.")

    # 3. 天文参数计算 (基于 date_obj Series)
    # 3.1 计算积日 (Day of Year) - 使用 Pandas 向量化属性
    # 注意：这里假设 date_obj 已经是 datetime64 类型
    doy = date_obj.dt.dayofyear

    # 3.2 纬度转弧度
    lat_rad = np.radians(latitude)

    # 3.3 太阳赤纬 (Solar Declination)
    delta = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)

    # 3.4 日落时角 (Sunset Hour Angle)
    cos_ws = -np.tan(lat_rad) * np.tan(delta)
    # 防止极昼极夜导致的数学域错误
    cos_ws = cos_ws.clip(-1, 1)
    omega_s = np.arccos(cos_ws)

    # 4. 计算日照时长 (Daylight Duration)
    daylight_hours = (24 / np.pi) * omega_s

    # 5. 计算日天文辐射量 (Daily Radiation)
    # 5.1 日地距离修正系数 (Inverse Relative Distance Earth-Sun)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)

    # 5.2 辐射计算公式 (FAO-56 标准)
    Gsc = 0.0820  # 太阳常数 MJ/(m^2·min)
    daily_radiation = (24 * 60 / np.pi) * Gsc * dr * (
            omega_s * np.sin(lat_rad) * np.sin(delta) +
            np.cos(lat_rad) * np.cos(delta) * np.sin(omega_s)
    )

    # 6. 计算潜在蒸散发 (PET Estimation)
    pet_est = temp_c * (daylight_hours / 12.0)

    # 7. 返回 DataFrame
    return pd.DataFrame({
        'PET_Est': pet_est,
        'DaylightDuration': daylight_hours,
        'DailyRadiation': daily_radiation
    })


# ========================
# === Fog_Risk (雾风险) ===
# ========================
def calc_fog_risk(temp: Union[float, List, pd.Series], dewp: Union[float, List, pd.Series],
                  wdsp: Union[float, List, pd.Series], temp_: str = 'C', dewp_: str = 'C') -> pd.DataFrame:
    """
    计算雾风险指数。

    参数:
        temp: 温度，支持单个值或批量数据
        dewp: 露点温度值，支持单个值或批量数据
        wdsp: 风速，支持单个值或批量数据
        temp_: 温度单位 ('C', 'F', 'K')
        dewp_: 露点温度单位 ('C', 'F', 'K')

    返回:
        pd.DataFrame: 包含 'Fog_Risk' 列的DataFrame
    """
    # 1. 统一转换为 Series
    temp = to_series(temp)
    dewp = to_series(dewp)
    wdsp = to_series(wdsp)

    # 2. 温度转换
    if temp_ == 'F':
        temp_c = (temp - 32) * (5 / 9)
    elif temp_ == 'K':
        temp_c = temp - 273.15
    else:
        temp_c = temp

    if dewp_ == 'F':
        td_c = (dewp - 32) * (5 / 9)
    elif dewp_ == 'K':
        td_c = dewp - 273.15
    else:
        td_c = dewp

    # 3. 相对湿度计算
    numerator = 17.625 * td_c
    denominator = 243.04 + td_c
    denominator = denominator.clip(lower=1e-6)

    rh = 100 * np.exp((numerator / denominator) - (17.625 * temp_c / (243.04 + temp_c)))
    rh = rh.clip(0, 100)

    # 4. 风险计算
    fog_risk = (rh / 100.0) ** 3 * np.exp(-wdsp / 5.0)

    return pd.DataFrame({'Fog_Risk': fog_risk})


# ============================
# === Geo_Zoning (地理特征) ===
# ============================
def calc_geo_zoning(date_obj: Union[datetime, pd.Series],
                    latitude: Union[float, pd.Series],
                    longitude: Union[float, pd.Series],
                    elevation: Union[float, pd.Series],
                    dem_path: str = None) -> pd.DataFrame:
    """
    计算全量综合地理特征。
    融合了基础地理统计特征（季节、大陆度等）与高阶地形特征（坡度、坡向、3D坐标）。

    参数:
        date_obj: 日期对象 (datetime 或 pandas Series)
        latitude: 纬度
        longitude: 经度
        elevation: 海拔 (米)
        dem_path: DEM文件路径 (可选，若提供则计算坡度和坡向)

    返回:
        pandas.DataFrame: 包含以下列
            - Geo_Zoning_Season: 季节相位 (-1 ~ 1)
            - Geo_Zoning_Continentality: 大陆度 (0 ~ 1)
            - Geo_Zoning_Lat_Abs: 纬度绝对值
            - Geo_Zoning_Lat_2: 纬度平方
            - Geo_Zoning_Atmos_Thickness: 大气厚度因子
            - Geo_Zoning_Atmos_Log_Elevation: 对数海拔
            - Geo_Spatial_X/Y/Z: 地心三维坐标 (km)
            - Terrain_Slope: 坡度 (度) - 仅当提供 dem_path 时计算
            - Terrain_Aspect: 坡向 (度) - 仅当提供 dem_path 时计算
    """
    # 1. 数据标准化
    lat_s = to_series(latitude)
    lon_s = to_series(longitude)
    elev_s = to_series(elevation)

    # 2. 基础地理特征计算 (原有逻辑)

    # 2.1 季节相位 (Seasonal Phase)
    # 处理日期输入，兼容 Series 和 单个 datetime 对象
    if isinstance(date_obj, pd.Series):
        if np.issubdtype(date_obj.dtype, np.datetime64):
            doy = date_obj.dt.dayofyear
        else:
            doy = date_obj.apply(lambda x: x.timetuple().tm_yday)
    else:
        doy = to_series(date_obj.timetuple().tm_yday)

    hemisphere = np.where(lat_s >= 0, 1, -1)
    seasonal_phase = hemisphere * np.sin(2 * np.pi * doy / 365)

    # 2.2 大陆度 (Continentality)
    dist_atlantic = np.abs(lon_s)
    dist_pacific = 180 - np.abs(lon_s)
    min_dist = np.minimum(dist_atlantic, dist_pacific)
    continentality = np.tanh(min_dist * 0.05)

    # 2.3 纬度特征
    lat_abs = np.abs(lat_s)
    lat_2 = lat_s ** 2

    # 2.4 大气与海拔特征
    atmos_thickness = np.exp(-elev_s / 8400.0)
    log_elevation = np.log(elev_s + 1)

    # 3. 高阶地形与空间特征计算 (融合 calc_terrain_features 逻辑)

    # 3.1 计算三维笛卡尔坐标 (Earth-Centered Coordinates)
    radius = 6371  # 地球平均半径 km
    lat_rad = np.radians(lat_s)
    lon_rad = np.radians(lon_s)

    x_3d = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y_3d = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z_3d = radius * np.sin(lat_rad)

    # 初始化结果字典
    result_data = {
        'Geo_Zoning_Season': seasonal_phase,
        'Geo_Zoning_Continentality': continentality,
        'Geo_Zoning_Lat_Abs': lat_abs,
        'Geo_Zoning_Lat_2': lat_2,
        'Geo_Zoning_Atmos_Thickness': atmos_thickness,
        'Geo_Zoning_Atmos_Log_Elevation': log_elevation,
        'Geo_Spatial_X': x_3d,
        'Geo_Spatial_Y': y_3d,
        'Geo_Spatial_Z': z_3d
    }

    # 3.2 批量计算坡度坡向 (如果提供了 DEM 路径)
    if dem_path:
        try:
            dataset = gdal.Open(dem_path, gdalconst.GA_ReadOnly)
            if dataset:
                geotransform = dataset.GetGeoTransform()
                band = dataset.GetRasterBand(1)
                dem_data = band.ReadAsArray()

                slopes, aspects = [], []

                # 遍历坐标点计算地形参数
                for x, y in zip(lon_s.values, lat_s.values):
                    col = int((x - geotransform[0]) / geotransform[1])
                    row = int((y - geotransform[3]) / geotransform[5])

                    try:
                        # 3x3 窗口中心差分法
                        z_n = dem_data[row - 1, col]
                        z_s = dem_data[row + 1, col]
                        z_e = dem_data[row, col + 1]
                        z_w = dem_data[row, col - 1]

                        cell_size_x = geotransform[1]
                        cell_size_y = abs(geotransform[5])

                        dz_dx = (z_e - z_w) / (2 * cell_size_x)
                        dz_dy = (z_n - z_s) / (2 * cell_size_y)

                        slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))
                        aspect_rad = np.arctan2(dz_dy, -dz_dx)
                        aspect_deg = np.degrees(aspect_rad)
                        if aspect_deg < 0: aspect_deg += 360

                        slopes.append(slope_deg)
                        aspects.append(aspect_deg)
                    except IndexError:
                        slopes.append(np.nan)
                        aspects.append(np.nan)

                result_data['Terrain_Slope'] = slopes
                result_data['Terrain_Aspect'] = aspects
        except Exception:
            # 如果 DEM 读取失败，忽略地形特征
            pass

    # 4. 返回 DataFrame
    return pd.DataFrame(result_data)
