import time
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

from utils.data_loader import load_and_preprocess_data
from models.opns_forest import OPNsRandomForestClassifier


def run_single_dataset(dataset_name, n_estimators=30, max_features=None):
    print(f"\n{'=' * 40}")
    print(f"🚀 开始评估数据集: {dataset_name}")
    print(f"{'=' * 40}")

    try:
        X, y = load_and_preprocess_data(dataset_name)
    except Exception as e:
        print(f"加载 {dataset_name} 失败: {e}")
        return None

    n_splits, n_repeats = 5, 5
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=2026)

    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    # # 动态调整 max_features：如果是 WNQ 这种特征较多的，抽样；小特征全放开
    # current_max_features = 5 if dataset_name == 'WNQ' else max_features

    total_start_time = time.time()

    for fold, (train_index, test_index) in enumerate(rskf.split(X, y)):
        fold_start = time.time()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        forest = OPNsRandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            max_features=max_features,
            n_jobs=-1,
            random_state=fold
        )

        forest.fit(X_train, y_train)
        y_pred = forest.predict(X_test)

        metrics['accuracy'].append(accuracy_score(y_test, y_pred) * 100)
        metrics['precision'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100)
        metrics['recall'].append(recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100)
        metrics['f1'].append(f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100)

        print(
            f"  [{dataset_name}] Fold {fold + 1:02d}/25 | 耗时: {time.time() - fold_start:.1f}s | F1: {metrics['f1'][-1]:.2f}%")

    total_time = time.time() - total_start_time

    # 整理结果字符串
    result_dict = {
        'Dataset': dataset_name,
        'Time(s)': round(total_time, 2),
        'Accuracy': f"{np.mean(metrics['accuracy']):.2f} ± {np.std(metrics['accuracy']):.2f}",
        'Precision': f"{np.mean(metrics['precision']):.2f} ± {np.std(metrics['precision']):.2f}",
        'Recall': f"{np.mean(metrics['recall']):.2f} ± {np.std(metrics['recall']):.2f}",
        'F1-Score': f"{np.mean(metrics['f1']):.2f} ± {np.std(metrics['f1']):.2f}"
    }

    print(f"\n✅ {dataset_name} 评估完成！总耗时: {total_time:.2f} 秒")
    return result_dict


if __name__ == "__main__":
    # 为每个数据集量身定制参数，防止高维数据引发算力灾难
    # 规则：特征数较少的可以全放开(None)或选多点，特征数大的必须严格限制在 5~8 之间
    dataset_configs = {
        # 'WNM': {'n_estimators': 30, 'max_features': None},  # 13个特征 -> 78 对 (很快)
        # 'BCM': {'n_estimators': 30, 'max_features': 6},  # 30个特征 -> 选6个产生 15 对
        'WNQ': {'n_estimators': 30, 'max_features': 5},  # 11个特征 -> 选5个产生 10 对
        # 'LED': {'n_estimators': 30, 'max_features': 8}  # 64个特征 -> 选8个产生 28 对 (救命参数！)
    }

    all_results = []

    for ds, config in dataset_configs.items():
        # 这里把 config 解包传进去
        res = run_single_dataset(ds, n_estimators=config['n_estimators'], max_features=config['max_features'])
        if res:
            all_results.append(res)

            # 每跑完一个就立刻覆盖保存
            df = pd.DataFrame(all_results)
            os.makedirs('results', exist_ok=True)
            df.to_csv('results/final_metrics.csv', index=False, encoding='utf-8-sig')

    print("\n" + "=" * 50)
    print("🎉 所有实验批处理完毕！结果已保存至 results/final_metrics.csv")
    print("=" * 50)