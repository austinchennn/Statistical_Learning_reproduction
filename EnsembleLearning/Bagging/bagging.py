"""手写 Bagging 核心。

- bootstrap_sample(): 有放回抽 N 个样本，同时返回袋外（OOB）索引
- BaggingEnsemble:    在 M 个 bootstrap 集上训练同一种模型，回归取平均、分类软/硬投票，并给出 OOB 估计
"""
import numpy as np
from sklearn.base import clone


def bootstrap_sample(n, rng):
    """返回 (in_bag_idx, oob_idx)。in_bag 可重复；oob 为一次都没被抽中的样本。"""
    in_bag = rng.integers(0, n, size=n)
    oob_mask = np.ones(n, dtype=bool)
    oob_mask[in_bag] = False
    return in_bag, np.flatnonzero(oob_mask)


class BaggingEnsemble:
    def __init__(self, base_model, n_estimators=100, task="classification", voting="soft", seed=42):
        self.base_model = base_model
        self.n_estimators = n_estimators  # M
        self.task = task
        self.voting = voting  # 分类时："soft" 平均概率，"hard" 多数投票
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n = len(X)
        self.estimators_, self.oob_indices_ = [], []
        if self.task == "classification":
            self.classes_ = np.unique(y)
        for _ in range(self.n_estimators):
            in_bag, oob = bootstrap_sample(n, rng)
            self.estimators_.append(clone(self.base_model).fit(X[in_bag], y[in_bag]))  # 每个 h_m 独立训练，可并行
            self.oob_indices_.append(oob)
        return self

    def _proba_aligned(self, h, X):
        """某个 bootstrap 集可能缺少某些类别，把 h 的概率列对齐到全部 classes_。"""
        P = np.zeros((len(X), len(self.classes_)))
        P[:, np.searchsorted(self.classes_, h.classes_)] = h.predict_proba(X)
        return P

    def _combine(self, estimators, X):
        if self.task == "regression":
            return np.mean([h.predict(X) for h in estimators], axis=0)  # (1/M) Σ h_m(x)
        if self.voting == "soft":
            P = np.mean([self._proba_aligned(h, X) for h in estimators], axis=0)
            return self.classes_[P.argmax(axis=1)]
        votes = np.array([h.predict(X) for h in estimators])  # (M, n)
        counts = np.stack([(votes == c).sum(axis=0) for c in self.classes_], axis=1)
        return self.classes_[counts.argmax(axis=1)]  # majority vote

    def predict(self, X):
        return self._combine(self.estimators_, X)

    def predict_proba(self, X):
        return np.mean([self._proba_aligned(h, X) for h in self.estimators_], axis=0)

    def oob_predict(self, X):
        """对每个训练样本 i，只用"没见过 i"的那些 h_m 来预测它；从未 OOB 的样本返回 NaN/None。"""
        n = len(X)
        members = [[] for _ in range(n)]
        for m, oob in enumerate(self.oob_indices_):
            for i in oob:
                members[i].append(m)
        preds = np.full(n, np.nan, dtype=object if self.task == "classification" else float)
        for i, ms in enumerate(members):
            if ms:
                preds[i] = self._combine([self.estimators_[m] for m in ms], X[i : i + 1])[0]
        return preds

    def oob_score(self, X, y):
        """OOB 估计：分类返回准确率，回归返回 MSE。无需额外验证集。"""
        preds = self.oob_predict(X)
        mask = np.array([p is not None and not (isinstance(p, float) and np.isnan(p)) for p in preds])
        if self.task == "classification":
            return float(np.mean(preds[mask] == y[mask]))
        return float(np.mean((preds[mask].astype(float) - y[mask]) ** 2))
