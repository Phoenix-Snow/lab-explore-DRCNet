import reverse_geocoder as rg

# # 坐标点 (经度, 纬度)
# coordinates = (119.257591, 31.542629)
#
# # 查询
# result = rg.search(coordinates)
#
# # 输出结果
# print(f"省份/地区: {result[0]['name']}") # 通常返回城市名，可结合行政区划表推导省份
# print(f"行政代码: {result[0]['admin1']}") # 部分版本包含行政区域代码

def IDENTIFY_CITY(longitude, latitude):
    coordinates = (longitude, latitude)
    return rg.search(coordinates)[0]['name']
