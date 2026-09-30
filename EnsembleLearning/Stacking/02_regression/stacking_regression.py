"""例 2：回归。线性回归 + 随机森林 + 神经网络 → Ridge 元模型。

y_hat = beta_lin * h_lin + beta_rf * h_rf + beta_nn * h_nn + beta_0
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sklearn.datasets import make_friedman1
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stacking import StackingEnsemble

# Friedman #1：同时含线性项和非线性项，不同模型各有擅长的部分
X, y = make_friedman1(n_samples=1500, noise=1.0, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

base_models = [
    ("lin", LinearRegression()),
    ("rf", RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)),
    ("nn", make_pipeline(StandardScaler(),
                         MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=0))),
]
meta_model = Ridge(alpha=1.0)

stack = StackingEnsemble(base_models, meta_model, task="regression").fit(X_train, y_train)

print("单个基模型测试 MSE：")
for name, model in stack.fitted_bases_:
    print(f"  {name:4s} {mean_squared_error(y_test, model.predict(X_test)):.4f}")
print(f"Stacking 测试 MSE： {mean_squared_error(y_test, stack.predict(X_test)):.4f}")

print("\nRidge meta-learner 学到的组合权重：")
for name, w in stack.meta_weights().items():
    print(f"  {name:9s} {w:+.3f}")
