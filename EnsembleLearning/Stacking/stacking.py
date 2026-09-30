"""手写 Stacking 核心。

分两层，且刻意解耦：
- generate_oof():   第一层，只关心 base learner，输出 OOF 矩阵（base 与 meta 之间的"契约"）
- StackingEnsemble: 把两层串起来；meta-learner 只看得到 OOF 矩阵，看不到 base 模型本身
"""
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold


def _predict_one(model, X, task):
    """分类用 P(y=1) 作为元特征（比硬标签信息更多），回归直接用预测值。"""
    if task == "classification":
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def make_folds(y, n_splits=5, task="classification", seed=42):
    """统一的 fold 划分。工程中会把它持久化，保证所有 base 模型用同一套切分。"""
    splitter = (StratifiedKFold if task == "classification" else KFold)(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    return list(splitter.split(np.zeros(len(y)), y))


def generate_oof(base_models, X, y, folds, task="classification"):
    """第一层：对每个 base model 做 K 折，返回 OOF 矩阵 Z，形状 (n_samples, M)。

    Z[i, m] = 第 m 个模型在"没见过样本 i"的那一折上训练后，对样本 i 的预测。
    """
    Z = np.zeros((len(X), len(base_models)))
    for m, (name, model) in enumerate(base_models):
        for train_idx, valid_idx in folds:
            h = clone(model).fit(X[train_idx], y[train_idx])  # 在 K-1 折上训练
            Z[valid_idx, m] = _predict_one(h, X[valid_idx], task)  # 只预测留出折
    return Z


class StackingEnsemble:
    def __init__(self, base_models, meta_model, task="classification", n_splits=5, seed=42):
        self.base_models = base_models  # [(name, estimator), ...]
        self.meta_model = meta_model
        self.task = task
        self.n_splits = n_splits
        self.seed = seed

    def fit(self, X, y):
        # 步骤 1-3：第一层产出 OOF 元特征
        folds = make_folds(y, self.n_splits, self.task, self.seed)
        self.Z_train_ = generate_oof(self.base_models, X, y, folds, self.task)

        # 步骤 4：meta-learner 单独训练，输入只是 Z（一张普通的 n×M 特征表）
        self.meta_ = clone(self.meta_model).fit(self.Z_train_, y)

        # 步骤 5：base 模型在全量训练集上重训，供推理使用
        self.fitted_bases_ = [(n, clone(m).fit(X, y)) for n, m in self.base_models]
        return self

    def meta_features(self, X):
        """推理时的第一层：全量 base 模型的预测拼成 Z。"""
        return np.column_stack([_predict_one(m, X, self.task) for _, m in self.fitted_bases_])

    def predict(self, X):
        # 步骤 6：y_hat = g(h_1(x), ..., h_M(x))
        return self.meta_.predict(self.meta_features(X))

    def predict_proba(self, X):
        return self.meta_.predict_proba(self.meta_features(X))

    def meta_weights(self):
        """线性 meta-learner 学到的权重 {模型名: beta_m}，外加截距 beta_0。"""
        coef = np.ravel(self.meta_.coef_)
        weights = {name: float(c) for (name, _), c in zip(self.base_models, coef)}
        weights["intercept"] = float(np.ravel(self.meta_.intercept_)[0])
        return weights
