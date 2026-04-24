import time
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 导入我们自己写的数据加载器和 OPNs 森林
from utils.data_loader import load_and_preprocess_data
from models.opns_forest import OPNsRandomForestClassifier

if __name__ == "__main__":
    print("====== OPNs-RF 5x5 折交叉验证 (学术标准评估) ======")

    # 1. 选择数据集
    # 强烈建议：第一次测试先用 'WNM' (葡萄酒，178个样本，13个特征)，跑得快！
    # 如果用 'BCM' (569样本)，25 次交叉验证可能需要 1 个多小时。
    dataset_name = 'WNQ'
    X, y = load_and_preprocess_data(dataset_name)

    # 2. 设置交叉验证参数：5 折，重复 5 次 = 总共 25 次训练
    n_splits = 5
    n_repeats = 5
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=2026)

    # 记录 25 次验证的指标
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    print(f"\n开始执行 {n_splits}x{n_repeats} CV (共 {n_splits * n_repeats} 轮训练)...")
    total_start_time = time.time()

    # 3. 循环每一折
    for fold, (train_index, test_index) in enumerate(rskf.split(X, y)):
        fold_start_time = time.time()

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 实例化森林 (为了速度，特征抽样比例可以适当调小)
        # WNM 有 13 个特征，抽取 sqrt(13) ≈ 4
        forest = OPNsRandomForestClassifier(
            # n_estimators=10,
            # max_depth=5,
            # max_features=4,
            # n_jobs=-1,
            # random_state=fold
            n_estimators=30,  # 增加到 50 棵树（标准随机森林的起步量）
            max_depth=None,  # 解除深度限制，让 C4.5 增益率自然停止树的生长
            max_features=5,  # 稍微增加一点特征透视率
            n_jobs=-1,
            random_state=fold  # 每折使用不同的随机种子增加多样性

        )

        # 训练与预测
        forest.fit(X_train, y_train)
        y_pred = forest.predict(X_test)

        # 计算当前折的指标并记录
        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100

        metrics['accuracy'].append(acc)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['f1'].append(f1)

        print(f"  - 第 {fold + 1:02d}/25 轮完成 | 耗时: {time.time() - fold_start_time:.2f} 秒 | F1: {f1:.2f}%")

    # 4. 汇总报告 Mean ± Std
    print("\n====== 最终学术报告 (可直接填入论文表格) ======")
    print(f"数据集: {dataset_name}")
    print(f"总耗时: {time.time() - total_start_time:.2f} 秒\n")

    print(f"Accuracy  : {np.mean(metrics['accuracy']):.2f} ± {np.std(metrics['accuracy']):.2f}")
    print(f"Precision : {np.mean(metrics['precision']):.2f} ± {np.std(metrics['precision']):.2f}")
    print(f"Recall    : {np.mean(metrics['recall']):.2f} ± {np.std(metrics['recall']):.2f}")
    print(f"F1-Score  : {np.mean(metrics['f1']):.2f} ± {np.std(metrics['f1']):.2f}")