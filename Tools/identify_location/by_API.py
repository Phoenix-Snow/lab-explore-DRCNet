import requests


def IDENTIFY_DETAIL(longitude, latitude, api_key):
    """
    高德地图
    """
    url = "https://restapi.amap.com/v3/geocode/regeo"
    params = {
        "key": api_key,
        "location": f"{longitude},{latitude}",  # 高德要求 经度,纬度
        "extensions": "base"
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data['status'] == '1':
        # 获取省份
        province = data['regeocode']['addressComponent']['province']
        return province
    else:
        return "Unknown province"


class GeoAPI:
    def __init__(self):
        self.url = {
            '高德': 'https://restapi.amap.com/v3/geocode/regeo',
        }

    def IDENTIFY_PROVINCE(longitude, latitude, api, api_key):
        """
        高德地图
        """
        url = self.url[api]
        params = {
            "key": api_key,
            "location": f"{longitude},{latitude}",  # 高德要求 经度,纬度
            "extensions": "base"
        }

        response = requests.get(url, params=params)
        data = response.json()

        if data['status'] == '1':
            # 获取省份
            province = data['regeocode']['addressComponent']['province']
            return province
        else:
            return "Unknown province"
