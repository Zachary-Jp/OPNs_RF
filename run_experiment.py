import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 导入我们自己写的数据加载器和 OPNs 森林
from utils.data_loader import load_and_preprocess_data
from models.opns_forest import OPNsRandomForestClassifier

if __name__ == "__main__":
    print("====== OPNs 随机森林真实数据集测试 (BCM) ======")

    # 1. 加载并预处理数据 (sklearn 自动下载并加载)
    dataset_name = 'BCM'
    X, y = load_and_preprocess_data(dataset_name)

    # 2. 划分训练集和测试集 (80% 训练，20% 测试)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\n开始训练 OPNs 随机森林 (数据集: {dataset_name})...")

    # 3. 实例化 OPNs 森林
    # 乳腺癌数据有 30 个特征，我们按照随机森林的经验法则，每次抽取 sqrt(30) ≈ 5 个特征
    forest = OPNsRandomForestClassifier(
        n_estimators=10,  # 先用 10 棵树试试水
        max_depth=5,  # 限制深度防止过拟合
        max_features=5,  # 每次节点分裂随机抽取 5 个特征去组装 OPNs
        n_jobs=-1,  # CPU 满载并行运算
        random_state=2026
    )

    # 4. 训练模型并计时
    start_time = time.time()
    forest.fit(X_train, y_train)
    print(f"训练完成！耗时: {time.time() - start_time:.4f} 秒")

    # 5. 测试与评估
    y_pred = forest.predict(X_test)

    # 原论文 Table 2 中使用的四个核心评估指标
    print("\n====== 最终测试结果 ======")
    print(f"Accuracy  (准确率): {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"Precision (精确率): {precision_score(y_test, y_pred, average='weighted') * 100:.2f}%")
    print(f"Recall    (召回率): {recall_score(y_test, y_pred, average='weighted') * 100:.2f}%")
    print(f"F1-Score  (F1分数): {f1_score(y_test, y_pred, average='weighted') * 100:.2f}%")