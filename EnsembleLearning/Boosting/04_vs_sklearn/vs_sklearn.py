"""例 4：手写 GBDT 与 sklearn GradientBoosting 对比（回归 + 分类），验证实现正确。"""
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gbdt import GBDT
from sklearn.datasets import make_regression, make_classification
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=400, n_features=5, noise=10, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
ours = GBDT("squared", n_estimators=100, learning_rate=0.1, max_depth=3, lam=0.0).fit(Xtr, ytr)
sk = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0).fit(Xtr, ytr)
mse = lambda a, b: np.mean((a - b) ** 2)
print(f"回归 测试 MSE   手写: {mse(ours.predict(Xte), yte):8.1f}   sklearn: {mse(sk.predict(Xte), yte):8.1f}")

X, y = make_classification(n_samples=400, n_features=6, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
ours = GBDT("log", n_estimators=100, learning_rate=0.1, max_depth=3).fit(Xtr, ytr)
sk = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0).fit(Xtr, ytr)
print(f"分类 测试准确率 手写: {(ours.predict(Xte) == yte).mean():.3f}      sklearn: {sk.score(Xte, yte):.3f}")
