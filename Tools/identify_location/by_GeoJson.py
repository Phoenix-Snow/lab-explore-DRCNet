import json
import os

from shapely.geometry import Point, Polygon

GeoJson_China_path = "Tools/identify_location/china.geojson"
GeoJson_province_path = "Tools/identify_location/china_province.geojson"
GeoJson_city_path = "Tools/identify_location/china_city.geojson"


def IDENTIFY_PROVINCE(longitude, latitude, border_data):
    """
        # 加载省份边界数据
        # 假设 data 是从文件中读取的 GeoJSON 数据
        with open(>>PATH<<, encoding='utf-8') as f:
            border_data = json.load(f)
        # 使用示例
        print(IDENTIFY_PROVINCE(119.25, 31.54, mock_geojson_data))
    """
    point = Point(latitude, longitude, )

    # 遍历所有省份
    for feature in border_data['features']:
        # 获取省份名称
        province_name = feature['properties']['name']

        # 处理坐标数组 (GeoJSON 格式差异，需根据实际情况调整)
        coords = feature['geometry']['coordinates']
        # 注意：这里需要根据 GeoJSON 的具体结构解析出多边形坐标点列表
        # 这里仅作逻辑演示，实际需处理 MultiPolygon 嵌套
        if feature['geometry']['type'] == 'Polygon':
            polygon = Polygon(coords[0])
        else:
            # MultiPolygon 取第一个最大的面，或者遍历所有面
            polygon = Polygon(coords[0][0])

        # 核心判断：点是否在多边形内
        if polygon.contains(point):
            return province_name

    return "Unknown province"


class China_GeoJson:
    def __init__(self):
        if os.path.exists(GeoJson_China_path):
            with open(GeoJson_China_path, encoding='utf-8') as f:
                self.__china_border_data = json.load(f)
        else:
            print('[warning] GeoJson - no china data')
            self.__china_border_data = None
        if os.path.exists(GeoJson_province_path):
            with open(GeoJson_province_path, encoding='utf-8') as f:
                self.__province_border_data = json.load(f)
        else:
            print('[warning] GeoJson - no province data')
            self.__province_border_data = None
        if os.path.exists(GeoJson_city_path):
            with open(GeoJson_city_path, encoding='utf-8') as f:
                self.__city_border_data = json.load(f)
        else:
            print('[warning] GeoJson - no city data')
            self.__city_border_data = None

    def IDENTIFY_PROVINCE(self, longitude, latitude):
        if self.__province_border_data is None:
            raise Exception('[error] GeoJson - no province data - IDENTIFY_PROVINCE cannot continue')

        point = Point(longitude, latitude)

        # 遍历所有省份
        for feature in self.__province_border_data['features']:
            # 获取省份名称
            province_name = feature['properties']['name']

            # 处理坐标数组 (GeoJSON 格式差异，需根据实际情况调整)
            coords = feature['geometry']['coordinates']
            # 注意：这里需要根据 GeoJSON 的具体结构解析出多边形坐标点列表
            # 这里仅作逻辑演示，实际需处理 MultiPolygon 嵌套
            if feature['geometry']['type'] == 'Polygon':
                polygon = Polygon(coords[0])
            else:
                # MultiPolygon 取第一个最大的面，或者遍历所有面
                polygon = Polygon(coords[0][0])

                # 核心判断：点是否在多边形内
            if polygon.contains(point):
                return province_name

        return "Unknown province"

    def IDENTIFY_CITY(self, longitude, latitude):
        if self.__city_border_data is None:
            raise Exception('[error] GeoJson - no city data - IDENTIFY_CITY cannot continue')

        point = Point(longitude, latitude)

        # 遍历所有城市
        for feature in self.__city_border_data['features']:
            # 获取城市名称
            city_name = feature['properties']['name']

            # 处理坐标数组 (GeoJSON 格式差异，需根据实际情况调整)
            coords = feature['geometry']['coordinates']
            # 注意：这里需要根据 GeoJSON 的具体结构解析出多边形坐标点列表
            # 这里仅作逻辑演示，实际需处理 MultiPolygon 嵌套
            if feature['geometry']['type'] == 'Polygon':
                polygon = Polygon(coords[0])
            else:
                # MultiPolygon 取第一个最大的面，或者遍历所有面
                polygon = Polygon(coords[0][0])

                # 核心判断：点是否在多边形内
            if polygon.contains(point):
                return city_name

        return "Unknown city"

    def IDENTIFY_China(self, longitude, latitude):
        if self.__china_border_data is None:
            raise Exception('[error] GeoJson - no china data - IDENTIFY_China cannot continue')

        point = Point(longitude, latitude)

        feature = self.__china_border_data['features'][0]

        # 处理坐标数组 (GeoJSON 格式差异，需根据实际情况调整)
        coords = feature['geometry']['coordinates']
        # 注意：这里需要根据 GeoJSON 的具体结构解析出多边形坐标点列表
        # 这里仅作逻辑演示，实际需处理 MultiPolygon 嵌套
        if feature['geometry']['type'] == 'Polygon':
            polygon = Polygon(coords[0])
        else:
            # MultiPolygon 取第一个最大的面，或者遍历所有面
            polygon = Polygon(coords[0][0])

        # 核心判断：点是否在多边形内
        return polygon.contains(point)
