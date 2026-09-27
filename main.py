import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_curve, auc
from typing import Optional
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer

class AdvancedOutlierDetector:
    def __init__(self, categorical_cols_indices=None, numerical_cols_indices=None):
        self.numerical_cols_indices = numerical_cols_indices if numerical_cols_indices is not None else []
        self.categorical_cols_indices = categorical_cols_indices if categorical_cols_indices is not None else []
        self.numerical_cols = []
        self.categorical_cols = []
        self.min_vals = {}  # 存储每个数值属性的归一化参数
        self.max_vals = {}
        self.attribute_weights = None  # 存储属性权重
        self.weight1 = []
        self.weight2 = []
        self.weight = []
        self.adj = []
        self.fuzzy_similarity = []
        self.factor_outlier = []

    def _normalize(self, X, feature_idx):
        """对单个数值属性进行归一化"""
        feature = X[:, feature_idx]
        min_val = np.min(feature)
        max_val = np.max(feature)
        self.min_vals[feature_idx] = min_val
        self.max_vals[feature_idx] = max_val
        return (feature - min_val) / (max_val - min_val + 1e-8)

    def _discretize_single_attribute(self, X, feature_idx):
        """对单个数值属性进行离散化"""
        X_normalized = self._normalize(X, feature_idx)
        distance_matrix = np.abs(X_normalized.reshape(-1, 1) - X_normalized)

        np.fill_diagonal(distance_matrix, np.nan)  # 将对角线设为 NaN
        avg_dis = np.nanmean(distance_matrix, axis=1)

        # 计算分子和分母的广播矩阵
        i_matrix = avg_dis.reshape(-1, 1)  # 列向量
        j_matrix = avg_dis.reshape(1, -1)  # 行向量
        numerator = np.abs(i_matrix - j_matrix)
        denominator = i_matrix + j_matrix + 1e-8

        # 计算相似度矩阵
        similarity = 1 - numerator / denominator
        # 生成邻接矩阵
        adj_matrix = similarity > 0.95
        csr_adj = csr_matrix(adj_matrix)
        n_components, labels = connected_components(csgraph=csr_adj, directed=False)
        return labels, similarity

    def _calc_categorical_similarity(self, X_categorical, col):
        """计算类别属性的相似度矩阵"""
        col_vals = np.asarray(X_categorical)[:, col]
        # 广播比较 + 直接转int（用np.int_替代astype，等价但更稳妥）
        similarity_matrix = np.int_(col_vals[:, None] == col_vals)
        return similarity_matrix

    def _discretize_numerical_attributes(self, X_numerical, X_categorical):
        """对所有数值属性进行离散化"""
        n_samples = X_numerical.shape[0] if X_numerical.size > 0 else X_categorical.shape[0]
        n_features = len(self.numerical_cols_indices) + len(self.categorical_cols_indices)
        self.fuzzy_similarity = np.zeros((n_features, n_samples, n_samples), dtype=float)
        fs = np.zeros((n_features, n_samples, n_samples), dtype=float)
        discrete_labels = np.zeros((n_samples, len(self.numerical_cols_indices)), dtype=int)

        # 处理数值列
        for i, col in enumerate(self.numerical_cols_indices):
            idx = i  # 数值列从0开始索引
            discrete_labels[:, i], fs[idx] = self._discretize_single_attribute(X_numerical, col)

        # 处理类别列
        if X_categorical is not None and len(self.categorical_cols_indices) > 0:
            for i, col in enumerate(self.categorical_cols_indices):
                idx = len(self.numerical_cols_indices) + i  # 类别列接在数值列后面
                fs[idx] = self._calc_categorical_similarity(X_categorical, col)

        self.fuzzy_similarity = fs
        return discrete_labels

    def _calculate_mutual_information(self, X, y):
        """计算互信息"""
        n_features = X.shape[1]
        mi_values = np.zeros(n_features)
        labeled_indices = ~np.isnan(y)

        # 处理无标签的情况
        if np.sum(labeled_indices) < 2:
            return mi_values

        X_labeled = X[labeled_indices]
        y_labeled = y[labeled_indices]

        # 计算互信息
        summ = 0
        for h in range(n_features):
            feature_values = X_labeled[:, h]
            contingency = pd.crosstab(feature_values, y_labeled, normalize=True)
            mi = 0
            for i in range(contingency.shape[0]):
                for j in range(contingency.shape[1]):
                    p_xy = contingency.iloc[i, j]
                    p_x = contingency.iloc[i, :].sum()
                    p_y = contingency.iloc[:, j].sum()
                    if p_xy > 0 and p_x > 0 and p_y > 0:
                        mi += p_xy * np.log2(p_xy / (p_x * p_y))

            mi_values[h] = max(mi, 0)  # 互信息非负
            summ += mi_values[h]

        # 归一化权重
        self.weight1 = [mi / (summ + 1e-8) for mi in mi_values]
        return mi_values

    def _calculate_fuzzy_entropy(self, X):
        """计算模糊熵"""
        n_features = X.shape[1]
        n_samples = X.shape[0]
        fe_values = np.zeros(n_features)
        summ = 0
        fs = np.array(self.fuzzy_similarity)

        for h in range(n_features):
            fs_matrix = fs[h]
            # 计算每个样本的模糊邻域大小
            fuzzy_neighborhood_sizes = np.sum(fs_matrix, axis=1)
            prob = fuzzy_neighborhood_sizes / n_samples
            entropy_sum = np.sum(np.log2(prob + 1e-8))  # 向量化对数计算
            fe_values[h] = -entropy_sum / n_samples

            # 计算熵差
            entropy_max = abs(np.log2(1 / n_samples + 1e-8))
            entropy_diff = entropy_max - fe_values[h]
            summ += entropy_diff

        # 归一化权重
        self.weight2 = [diff / (summ + 1e-8) for diff in (abs(np.log2(1 / n_samples + 1e-8)) - fe_values)]
        return fe_values

    def _calculate_outlier_factor(self, X):
        """计算离群因子"""
        n_samples = X.shape[0]
        n_features = X.shape[1]
        fd_matrix = np.zeros((n_samples, n_features))

        for h in range(n_features):
            fs_matrix = self.fuzzy_similarity[h]
            for i in range(n_samples):
                neighborhood_size = np.sum(fs_matrix[i])
                # 计算离群因子
                fd_matrix[i, h] = 1.0 - neighborhood_size / n_samples

        self.factor_outlier.append(fd_matrix)
        return fd_matrix

    def _calculate_weights(self, X, y):
        """计算属性权重"""
        n_features = X.shape[1]
        attr_weight = np.zeros(n_features)

        # 1. 计算标签完备度
        labeled_count = np.sum(~np.isnan(y))
        total_count = len(y)
        r = labeled_count / total_count if total_count > 0 else 0.5

        w1 = np.array(self.weight1)
        w2 = np.array(self.weight2)

        # 2. 计算权重差异
        G = np.abs(w1 - w2)

        # 3. 计算动态系数
        alpha = np.zeros(n_features)
        beta = np.zeros(n_features)

        for i in range(n_features):
            if w1[i] + w2[i] > 0:
                # 计算权重比值
                ratio = w1[i] / (w1[i] + w2[i])
                # 计算动态系数
                alpha[i] = r + (1 - r) * (G[i] * ratio)
                alpha[i] = np.clip(alpha[i], 0, 1)  # 限制在[0,1]范围内
            else:
                alpha[i] = 0.5
            beta[i] = 1 - alpha[i]

        attr_weight = alpha * w1 + beta * w2
        self.weight.append(attr_weight / (np.sum(attr_weight) + 1e-8))

        return attr_weight

    def fit_transform(self, X, y):
        """
        完整流程：
        1. 分离数值和类别属性
        2. 离散化数值属性
        3. 结合所有属性计算离群值
        :return: (离散化后的数值属性标签, 离群值分数)
        """
        # 处理空输入
        if X.size == 0:
            return np.array([]), np.array([])

        # 分离数据
        X_numerical = X[:, self.numerical_cols_indices].astype(float) if self.numerical_cols_indices else np.array([])
        X_categorical = X[:, self.categorical_cols_indices] if self.categorical_cols_indices and len(
            self.categorical_cols_indices) > 0 else None

        self.categorical_cols =self.categorical_cols_indices
        self.numerical_cols =self.numerical_cols_indices

        self.numerical_cols_indices = list(range(len(self.numerical_cols_indices)))
        self.categorical_cols_indices = list(range(len(self.categorical_cols_indices)))

        # 第一部分：离散化数值属性
        discrete_numerical = self._discretize_numerical_attributes(X_numerical, X_categorical)

        # 准备所有属性数据（编码后的类别属性 + 离散化后的数值属性）
        X_combined_parts = []
        if X_categorical is not None and len(X_categorical) > 0:
            X_combined_parts.append(X_categorical)
        if discrete_numerical.size > 0:
            X_combined_parts.append(discrete_numerical)

        X_combined = np.hstack(X_combined_parts) if X_combined_parts else np.array([])
        if X_combined.size == 0:
            X_combined = X_numerical

        # 第二部分：计算离群值
        n_samples = X.shape[0]
        mi_values = self._calculate_mutual_information(X_combined, y)
        fe_values = self._calculate_fuzzy_entropy(X_combined)
        weights = self._calculate_weights(X_combined, y)
        fd = self._calculate_outlier_factor(X_combined)

        # 计算最终离群分数
        outlier_scores = np.zeros(n_samples)
        if len(self.weight) > 0 and fd.size > 0:
            outlier_scores = fd @ self.weight[-1]

        return discrete_numerical, outlier_scores


def generate_partial_labels(y_full: np.ndarray, label_percent: float = 0.3, random_seed: int = 42) -> np.ndarray:
    """
    二分类标签(0/1)按百分比生成部分标签
    """
    # 参数与标签校验
    if not (0 <= label_percent <= 1):
        raise ValueError("label_percent需在0-1之间")
    if not np.array_equal(np.unique(y_full), [0, 1]):
        raise ValueError("标签仅支持0和1")

    # 固定随机种子
    np.random.seed(random_seed)
    y_partial = np.full_like(y_full, np.nan, dtype=float)
    if label_percent == 1.0:
        return y_full.astype(float)

    # 分离0/1索引
    idx_0, idx_1 = np.where(y_full == 0)[0], np.where(y_full == 1)[0]

    coef_1 = 0.015  # 1类抽 1.5%
    coef_0 = 0.035  # 0类抽 3.5%

    n_1 = int(round(len(y_full) * coef_1))
    n_0 = int(round(len(y_full) * coef_0))
    n_1 = max(1, min(n_1, len(idx_1)))
    n_0 = max(1, min(n_0, len(idx_0)))

    labeled_idx = np.hstack([
        np.random.choice(idx_1, n_1, replace=False),
        np.random.choice(idx_0, n_0, replace=False),
    ])

    # 赋值并返回
    y_partial[np.unique(labeled_idx)] = y_full[np.unique(labeled_idx)]
    return y_partial


if __name__ == "__main__":
    DATA_PATH = ''  # 数据文件路径
    RANDOM_SEED = 42 # 随机种子
    numerical_cols_indices = []
    categorical_cols_indices = []

    # 定义要测试的标签百分比列表
    LABEL_PERCENTS = [0.05]

    # 循环遍历每个标签百分比
    for label_percent in LABEL_PERCENTS:
        # 读取完整数据集
        data = pd.read_csv(DATA_PATH)
        X = data.iloc[:, :-1].values  # 特征数据
        y_full = data.iloc[:, -1].values  # 完整标签

        # 生成当前百分比的部分标签
        y_partial = generate_partial_labels(y_full, label_percent=label_percent, random_seed=RANDOM_SEED)
        # 初始化并执行离群检测
        detector = AdvancedOutlierDetector(categorical_cols_indices=categorical_cols_indices,
                                           numerical_cols_indices=numerical_cols_indices)
        discrete_labels, outlier_scores = detector.fit_transform(X, y_partial)
