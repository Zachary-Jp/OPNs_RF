import numpy as np
import pandas as pd  # 新增：用于读取网络 CSV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_breast_cancer, load_wine, load_digits


def load_and_preprocess_data(dataset_name):
    """
    加载对应的数据集，并严格执行原论文要求的“零均值归一化”和标签编码。
    """
    if dataset_name == 'BCM':
        data = load_breast_cancer()
        X, y = data.data, data.target
    elif dataset_name == 'WNM':
        data = load_wine()
        X, y = data.data, data.target
    elif dataset_name == 'LED':
        data = load_digits()
        X, y = data.data, data.target
    elif dataset_name == 'WNQ':
        # 新增：从 UCI 官方仓库拉取 Wine Quality (Red) 数据集
        print("正在从 UCI 仓库下载 WNQ 数据集，请稍候...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        df = pd.read_csv(url, sep=';')
        X = df.drop('quality', axis=1).values
        y = df['quality'].values
    else:
        raise ValueError(f"暂时未内置数据集: {dataset_name}，请自行添加读取逻辑。")

    # 1. 标签编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 2. 零均值归一化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(
        f"[{dataset_name}] 数据集加载成功! 样本数: {X_scaled.shape[0]}, 特征数: {X_scaled.shape[1]}, 类别数: {len(np.unique(y_encoded))}")
    return X_scaled, y_encoded