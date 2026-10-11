"""例 1：伪残差 = 先公式求导，再代入数字。
MSE 下 -g 恰好等于残差；LogLoss 下 -g = y - p，不再是 y - ŷ，所以叫“伪”残差。"""
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gbdt import SquaredLoss, LogLoss, sigmoid

print("== MSE：房价 y=100 万，前 t-1 棵树累加预测 ŷ=70 万 ==")
y, y_hat = np.array([100.0]), np.array([70.0])
g, h = SquaredLoss().grad_hess(y, y_hat)
print(f"公式 g = ŷ - y → {g[0]:+.1f}     h = 1 → {h[0]:.1f}")
print(f"真正残差 y - ŷ = {y[0]-y_hat[0]:+.1f}   负梯度 -g = {-g[0]:+.1f}   （完全相等）")

# 数值验证：用有限差分去“数值求导”，和公式结果对比
L = lambda yh: 0.5 * (100 - yh) ** 2
eps = 1e-4
print(f"有限差分 (L(70+ε)-L(70-ε))/2ε = {(L(70+eps)-L(70-eps))/(2*eps):+.4f}  ← 和公式一致\n")

print("== LogLoss：二分类，ŷ 是 logit，p = sigmoid(ŷ) ==")
y = np.array([1.0, 1.0, 0.0, 0.0])
y_hat = np.array([2.0, -1.0, 0.5, -3.0])
g, h = LogLoss().grad_hess(y, y_hat)
p = sigmoid(y_hat)
print(" y   ŷ(logit)   p      g=p-y    h=p(1-p)   y-ŷ(不是伪残差)")
for i in range(len(y)):
    print(f"{y[i]:.0f}   {y_hat[i]:+5.1f}    {p[i]:.3f}   {g[i]:+.3f}    {h[i]:.3f}      {y[i]-y_hat[i]:+.1f}")
print("→ 第 2 个样本 y=1 但 p=0.269，g 很负：下一棵树会把它往上推；h 小 = 曲率小，步子可以迈大。")
