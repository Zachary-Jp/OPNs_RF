import numpy as np
from models.opns_core import opns_less_equal
from collections import Counter
from models.opns_core import opns_less_equal, opns_less_equal_vectorized


# ==========================================
# 1. 纯度评估与增益计算
# ==========================================
def calculate_entropy(y):
    """
    根据论文公式 (8) 计算数据集 S 的信息熵 H(S)
    """
    if len(y) == 0:
        return 0.0
    # 统计各个类别的数量
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    # 计算香农熵，加上 1e-9 防止 log2(0) 报错
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))


def calculate_gain_ratio(y_parent, y_left, y_right):
    """
    根据论文公式 (9)-(11) 计算广义增益率 GR_OPNs。
    它能有效惩罚那些把大部分样本分到一边、只剥离出极少数样本的“偏科”切分。
    """
    h_parent = calculate_entropy(y_parent)
    n_total = len(y_parent)

    if n_total == 0:
        return 0.0

    weight_left = len(y_left) / n_total
    weight_right = len(y_right) / n_total

    # 1. 计算传统的广义信息增益 G_OPNs
    h_children = (weight_left * calculate_entropy(y_left)) + (weight_right * calculate_entropy(y_right))
    gain = h_parent - h_children

    # 2. 计算本征值 / 惩罚项 (Intrinsic Value, IV_OPNs) [论文公式 10]
    iv = 0.0
    if weight_left > 0:
        iv -= weight_left * np.log2(weight_left)
    if weight_right > 0:
        iv -= weight_right * np.log2(weight_right)

    # 如果完全没有切分开（都分到了同一边），IV 为 0，增益率直接给 0
    if iv < 1e-9:
        return 0.0

    # 3. 计算广义增益率 GR_OPNs [论文公式 11]
    return gain / iv


# ==========================================
# 2. 核心探测：寻找最佳 OPNs 分割阈值
# ==========================================
def find_best_split_for_feature(X_feature, y):
    """
    遍历单个 OPNs 特征对，使用广义增益率寻找最佳分割点。
    """
    best_gain_ratio = -1.0
    best_theta = None
    best_left_mask = None
    best_right_mask = None

    unique_thetas = np.unique(X_feature, axis=0)

    for theta in unique_thetas:
        # left_mask = np.array([opns_less_equal(x, theta) for x in X_feature])
        left_mask = opns_less_equal_vectorized(X_feature, theta)
        right_mask = ~left_mask

        y_left = y[left_mask]
        y_right = y[right_mask]

        if len(y_left) == 0 or len(y_right) == 0:
            continue

        # 核心改动：这里换成了调用 calculate_gain_ratio
        current_gain_ratio = calculate_gain_ratio(y, y_left, y_right)

        if current_gain_ratio > best_gain_ratio:
            best_gain_ratio = current_gain_ratio
            best_theta = theta
            best_left_mask = left_mask
            best_right_mask = right_mask

    return best_gain_ratio, best_theta, best_left_mask, best_right_mask


# ==========================================
# 3. 树节点数据结构
# ==========================================
class OPNsNode:
    """
    存储决策树节点的结构信息
    """

    def __init__(self, is_leaf=False, predicted_class=None,
                 split_feature_indices=None, split_theta=None,
                 left=None, right=None):
        self.is_leaf = is_leaf
        self.predicted_class = predicted_class
        # 记录分裂用的原始特征组合 (比如第 0 和第 2 个特征)
        self.split_feature_indices = split_feature_indices
        # 记录最佳探索阈值 OPNs 坐标
        self.split_theta = split_theta
        self.left = left
        self.right = right


# ==========================================
# 4. 单棵 OPNs 决策树分类器
# ==========================================
class OPNsDecisionTreeClassifier:
    """
    基于 OPNs 的分类决策树，实现了递归建树与预测逻辑
    """

    def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features  # 专为随机森林设计的参数
        self.root = None

    def fit(self, X, y):
        """
        训练模型。注意：传入的 X 必须是已经做过零均值归一化的实数矩阵！
        """
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)

        # 1. 检查停止条件 (Stopping Criteria)
        if (n_samples < self.min_samples_split or
                len(unique_classes) == 1 or
                (self.max_depth is not None and depth >= self.max_depth)):
            return OPNsNode(is_leaf=True, predicted_class=self._majority_class(y))

        # 2. 特征子空间采样 (为 OPNs-RF 埋下伏笔)
        if self.max_features is None:
            feat_idxs = range(n_features)
        else:
            # 随机抽取 max_features 个特征
            feat_idxs = np.random.choice(n_features, self.max_features, replace=False)

        # 3. 全连接配对与最优分裂点寻找
        best_gain = -1.0
        best_split = None

        # 仅对抽中的特征进行两两配对生成 OPNs
        for i in range(len(feat_idxs)):
            for j in range(i + 1, len(feat_idxs)):
                f1, f2 = feat_idxs[i], feat_idxs[j]

                # 提取特征对作为 OPNs 坐标 (虚拟映射)
                X_pair = X[:, [f1, f2]]

                gain, theta, left_mask, right_mask = find_best_split_for_feature(X_pair, y)

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature_indices': (f1, f2),
                        'theta': theta,
                        'left_mask': left_mask,
                        'right_mask': right_mask
                    }

        # 4. 如果找不到有效的增益（例如切分后一边为空），直接变叶子节点
        if best_gain < 1e-6 or best_split is None:
            return OPNsNode(is_leaf=True, predicted_class=self._majority_class(y))

        # 5. 递归构建左右子树
        left_child = self._grow_tree(X[best_split['left_mask']], y[best_split['left_mask']], depth + 1)
        right_child = self._grow_tree(X[best_split['right_mask']], y[best_split['right_mask']], depth + 1)

        return OPNsNode(
            is_leaf=False,
            split_feature_indices=best_split['feature_indices'],
            split_theta=best_split['theta'],
            left=left_child,
            right=right_child
        )

    def _majority_class(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf:
            return node.predicted_class

        # 提取分裂节点记录的两个特征构成 OPNs 对
        f1, f2 = node.split_feature_indices
        x_pair = np.array([x[f1], x[f2]])

        # 使用我们写的核心 OPNs 比较算子
        if opns_less_equal(x_pair, node.split_theta):
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    import time

    print("====== OPNs 单棵决策树全链路测试 ======")

    # 1. 生成虚拟数据集 (4个特征，2个有信息量的特征，非线性分布)
    X, y = make_classification(n_samples=200, n_features=4, n_informative=3,
                               n_redundant=0, n_classes=2, random_state=42)

    # 2. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3. 严格执行论文中的“零均值归一化” (Zero-mean normalization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"数据准备完毕: 训练集 {X_train_scaled.shape}, 测试集 {X_test_scaled.shape}")

    # 4. 实例化我们的 OPNs 决策树
    # 设置最大深度防止过拟合，并测试 max_features 接口
    tree = OPNsDecisionTreeClassifier(max_depth=5, min_samples_split=2, max_features=3)

    # 5. 训练模型并计时
    print("开始训练 OPNs 决策树...")
    start_time = time.time()
    tree.fit(X_train_scaled, y_train)
    print(f"训练完成！耗时: {time.time() - start_time:.4f} 秒")

    # 6. 预测与评估
    y_pred_train = tree.predict(X_train_scaled)
    y_pred_test = tree.predict(X_test_scaled)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)

    print(f"\n模型表现:")
    print(f"训练集准确率 : {acc_train * 100:.2f}%")
    print(f"测试集准确率 : {acc_test * 100:.2f}%")