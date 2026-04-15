import json
from by_GeoJson import IDENTIFY_PROVINCE
from by_GeoCoder import IDENTIFY_CITY

if __name__ == '__main__':
    with open("china_province.geojson", encoding='utf-8') as f:
        border_data = json.load(f)

    print('JSON: ', IDENTIFY_PROVINCE(29.7172627, 118.3324811, border_data))
    print('Coder: ', IDENTIFY_CITY(29.7172627, 118.3324811, ))
    # 答案：安徽省黄山市

