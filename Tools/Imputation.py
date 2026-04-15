import pandas as pd
# 先启用实验性功能
from sklearn.experimental import enable_iterative_imputer
# 再导入实际模块
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer


def Imputation(data, method='Iterative'):
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    data_imputed = None
    imputer = None
    if method == 'Iterative' or method == 'iterative':
        # 组合线性回归插值
        imputer = IterativeImputer(
            max_iter=10,  # 最大迭代次数
            random_state=42,  # 随机种子
            sample_posterior=True  # 启用贝叶斯采样
        )
    elif method.lower() == '__mean':
        # 均值填充缺失值
        imputer = SimpleImputer(strategy='__mean')
    elif method.lower() == 'median':
        # 平均值插值
        imputer = SimpleImputer(strategy='median')
    elif method.lower() == 'linear':
        data_imputed = data.interpolate(method='linear', limit_direction='both')
    else:
        raise Exception("Invalid imputation method.")

    if data_imputed is None:
        data_imputed = pd.DataFrame(imputer.fit_transform(data.values), columns=data.columns)

    return data_imputed
