"""例 4：Bagging 只降方差不降偏差。
在同一真实函数上反复抽训练集，分别估计单模型与 Bagging 的 bias² 和 variance：
- 深决策树（高方差）：variance 大幅下降
- 线性回归（高偏差、低方差）：几乎没变化
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from bagging import BaggingEnsemble

rng = np.random.default_rng(0)
f = lambda x: np.sin(2 * x)  # 真实函数（非线性，线性回归必有偏差）
X_grid = np.linspace(-3, 3, 200).reshape(-1, 1)


def bias_variance(make_model, n_trials=50, n=100, noise=0.3):
    preds = []
    for _ in range(n_trials):
        X = rng.uniform(-3, 3, size=(n, 1))
        y = f(X[:, 0]) + rng.normal(0, noise, n)
        preds.append(make_model().fit(X, y).predict(X_grid))
    preds = np.array(preds)
    bias2 = np.mean((preds.mean(axis=0) - f(X_grid[:, 0])) ** 2)
    var = np.mean(preds.var(axis=0))
    return bias2, var


cases = {
    "深决策树": lambda: DecisionTreeRegressor(),
    "Bagging 深决策树": lambda: BaggingEnsemble(DecisionTreeRegressor(), 50, task="regression",
                                           seed=int(rng.integers(1e9))),
    "线性回归": lambda: LinearRegression(),
    "Bagging 线性回归": lambda: BaggingEnsemble(LinearRegression(), 50, task="regression",
                                           seed=int(rng.integers(1e9))),
}
print(f"{'模型':16s} {'bias²':>8s} {'variance':>9s}")
for name, make in cases.items():
    b, v = bias_variance(make)
    print(f"{name:16s} {b:8.4f} {v:9.4f}")
