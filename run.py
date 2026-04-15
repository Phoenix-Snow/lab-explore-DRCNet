import csv
import numpy as np


def calculate_percentiles_from_csv(csv_file_path, has_header=True):
    """
    从CSV文件的最后一列读取数据，计算绝对误差的P50和P90

    参数:
        csv_file_path: CSV文件路径
        has_header: 是否有标题行，默认为True

    返回:
        dict: 包含P50和P90的字典
    """
    # 读取CSV文件最后一列的数据
    errors = []
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)

        # 如果有标题行，跳过它
        if has_header:
            next(reader)

        # 提取最后一列并转换为绝对误差
        for row in reader:
            if row:  # 跳过空行
                try:
                    # 获取最后一列的值，转换为浮点数并取绝对值
                    value = float(row[-1])
                    abs_error = abs(value)
                    errors.append(abs_error)
                except (ValueError, IndexError):
                    print(f"警告: 跳过无效行: {row}")
                    continue

    if not errors:
        raise ValueError("未读取到有效数据！")

    # 将列表转换为numpy数组以便计算
    errors_array = np.array(errors)

    # 计算P50和P90
    p50 = np.percentile(errors_array, 50)  # 中位数
    p90 = np.percentile(errors_array, 90)  # 第90百分位数

    return {
        'P50': p50,
        'P90': p90,
        'count': len(errors)
    }


# 使用示例
if __name__ == "__main__":
    # CSV文件路径
    csv_file = "results/XingJiangTS/DRCNet/timesnet_modified_add_spatial_future365/20251109003536/test_train_RESULTS.csv"  # 请替换为你的CSV文件路径

    try:
        # 计算百分位数
        result = calculate_percentiles_from_csv(csv_file, has_header=True)

        # 输出结果
        print("=" * 40)
        print(f"数据总数: {result['count']}")
        print(f"P50 (前50%中的最大值): {result['P50']:.6f}")
        print(f"P90 (前90%中的最大值): {result['P90']:.6f}")
        print("=" * 40)

    except FileNotFoundError:
        print(f"错误: 文件 '{csv_file}' 不存在")
    except Exception as e:
        print(f"发生错误: {e}")