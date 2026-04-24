import numpy as np
from collections import Counter
from joblib import Parallel, delayed
import time

# 导入我们刚刚写好的单棵 OPNs 树
from models.opns_tree import OPNsDecisionTreeClassifier


# 内部辅助函数：用于并行化训练单棵树
def _train_single_tree(X, y, max_depth, min_samples_split, max_features, random_state):
    # 设置随机种子，保证每棵树抽样不同
    np.random.seed(random_state)
    n_samples = X.shape[0]

    # 1. Bagging: 样本的有放回抽样 (Bootstrap)
    bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
    X_bootstrap = X[bootstrap_indices]
    y_bootstrap = y[bootstrap_indices]

    # 2. 实例化并训练单棵树
    tree = OPNsDecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=max_features
    )
    tree.fit(X_bootstrap, y_bootstrap)
    return tree


class OPNsRandomForestClassifier:
    """
    基于 OPNs 的随机森林分类器
    结合了 Bagging (样本随机) 和 Random Subspace (OPNs特征对随机)
    """

    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2,
                 max_features=None, n_jobs=-1, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_jobs = n_jobs  # -1 表示使用所有可用的 CPU 核心
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        """
        并行训练森林中的所有树
        """
        # 生成一组随机种子用于各棵树的抽样
        base_seed = self.random_state if self.random_state is not None else np.random.randint(1e5)
        seeds = [base_seed + i for i in range(self.n_estimators)]

        # 使用 joblib 开启多进程并行训练
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(_train_single_tree)(
                X, y, self.max_depth, self.min_samples_split, self.max_features, seeds[i]
            ) for i in range(self.n_estimators)
        )
        return self

    def predict(self, X):
        """
        所有树进行预测，并进行多数投票
        """
        # 收集所有树的预测结果, 形状为 (n_estimators, n_samples)
        # 预测过程通常很快，也可以不并行，这里先用串行提取
        all_tree_preds = np.array([tree.predict(X) for tree in self.trees])

        # 沿着树的维度(axis=0)进行多数投票
        majority_votes = []
        for i in range(X.shape[0]):
            sample_preds = all_tree_preds[:, i]
            # 找到票数最多的类别
            most_common = Counter(sample_preds).most_common(1)[0][0]
            majority_votes.append(most_common)

        return np.array(majority_votes)


# ==========================================
# 测试 OPNs 随机森林多进程加速效果
# ==========================================
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    print("====== OPNs 随机森林并行测试 ======")

    X, y = make_classification(n_samples=300, n_features=6, n_informative=4,
                               n_redundant=0, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"数据准备完毕: 训练集 {X_train_scaled.shape}, 测试集 {X_test_scaled.shape}")

    # 实例化包含 10 棵树的随机森林，n_jobs=-1 拉满 CPU
    forest = OPNsRandomForestClassifier(n_estimators=10, max_depth=5, max_features=4, n_jobs=-1, random_state=2026)

    print("开始并行训练 OPNs 随机森林...")
    start_time = time.time()
    forest.fit(X_train_scaled, y_train)
    print(f"训练完成！包含 10 棵树的森林，总耗时: {time.time() - start_time:.4f} 秒")

    y_pred_test = forest.predict(X_test_scaled)
    acc_test = accuracy_score(y_test, y_pred_test)

    print(f"测试集准确率 : {acc_test * 100:.2f}%")