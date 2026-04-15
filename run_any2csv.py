from Tools.AnyFile2CSV import excel2csv, npy2csv, npy2csv_with_autoHeader

# original_path = 'resource/time_series_noDate.xlsx'
# csv_path = 'resource/time_series_noDate.csv'

# original_path = 'resource/time_series.xlsx'
# csv_path = 'resource/time_series.csv'

# original_path = 'resource/features_aggregated_perfile.xlsx'
# csv_path = 'resource/dataset.csv'

# original_path = 'resource/XinJiangTS/源域数据集.xlsx'
# csv_path = 'resource/XinJiangTS/源域数据集.csv'

original_path = 'resource/XinJiangTS/目标域数据集.xlsx'
csv_path = 'resource/XinJiangTS/目标域数据集.csv'
header = []

if original_path.endswith('.xlsx'):
    excel2csv(original_path, csv_path)
elif original_path.endswith('.npy') and header:
    npy2csv(original_path, csv_path, header)
elif original_path.endswith('.npy') and not header:
    npy2csv_with_autoHeader(original_path, csv_path)

