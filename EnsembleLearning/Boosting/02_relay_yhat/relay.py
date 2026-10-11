"""例 2：建树之前 ŷ 从哪来？—— 接力跑。打印每一轮建树前用到的 ŷ^(t-1)、g、以及树补上的量。"""
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gbdt import GBDT

# 4 套房子，一个特征：面积
X = np.array([[50], [60], [100], [120]], dtype=float)
y = np.array([60, 70, 100, 130], dtype=float)

model = GBDT(loss="squared", n_estimators=5, learning_rate=0.5, max_depth=1, lam=0.0).fit(X, y)

y_hat = np.full(len(y), model.base_score_)
print(f"第 0 步：没有树，ŷ^(0) = mean(y) = {model.base_score_:.1f}\n")
for t, (tree, rec) in enumerate(zip(model.trees_, model.history_), 1):
    step = model.learning_rate * tree.predict(X)
    print(f"--- 第 {t} 棵树 ---")
    print(f"  建树前 ŷ^({t-1}) = {np.round(y_hat, 1)}")
    print(f"  g = ŷ - y      = {np.round(rec['g'], 1)}   ← 只用 y 和 ŷ，没用 x")
    print(f"  树补上 η·f_t   = {np.round(step, 1)}       ← x 只用来决定谁和谁在同一片叶子")
    y_hat = rec["y_hat"]
    print(f"  新 ŷ^({t})      = {np.round(y_hat, 1)}   MSE={np.mean((y-y_hat)**2):.1f}")
print(f"\n真实 y = {y}")
