"""例 3：离散特征（晴/阴/雨）不需要可导。它只是把 g、h 分流到左右子树，然后比较 Gain。"""
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gbdt import GBDT, GradientTree, LogLoss

rng = np.random.default_rng(0)
n = 300
weather = rng.choice(["晴", "阴", "雨"], size=n)
temp = rng.uniform(5, 35, size=n)
# 真实规律：雨天基本不去打球；其他天气温度适中才去
logit = np.where(weather == "雨", -3.0, 1.5) - 0.02 * (temp - 22) ** 2
y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(float)
X = np.column_stack([weather, temp]).astype(object)
X[:, 1] = temp
names = ["天气", "温度"]

# 手算第一棵树根节点：所有可能的天气子集切分 Gain
loss = LogLoss()
y_hat0 = np.full(n, loss.base_score(y))
g, h = loss.grad_hess(y, y_hat0)
G, H, lam = g.sum(), h.sum(), 1.0
print("第 1 棵树之前，每个类别的 G、H（只是把该类样本的 g、h 加起来）：")
for c in ["晴", "阴", "雨"]:
    m = weather == c
    print(f"  {c}: n={m.sum():3d}  G={g[m].sum():+7.2f}  H={h[m].sum():6.2f}")
print("\n枚举天气的二分切分：")
for left in [{"雨"}, {"晴"}, {"阴"}]:
    m = np.isin(weather, list(left))
    GL, HL = g[m].sum(), h[m].sum()
    GR, HR = G - GL, H - HL
    gain = 0.5 * (GL**2/(HL+lam) + GR**2/(HR+lam) - G**2/(H+lam))
    print(f"  {sorted(left)} vs 其余   Gain = {gain:.3f}")
print("→ {雨} vs {晴,阴} 的 Gain 最大，和真实规律一致。整个过程没有对“天气”求过导。\n")

model = GBDT(loss="log", n_estimators=30, learning_rate=0.3, max_depth=2, categorical=[0]).fit(X, y)
print("第 1 棵树结构：")
print(model.trees_[0].describe(names))
print(f"训练准确率: {(model.predict(X) == y).mean():.3f}")
test = np.array([["雨", 22.0], ["晴", 22.0], ["晴", 35.0]], dtype=object)
for row, p in zip(test, model.predict_proba(test)):
    print(f"  {row[0]}, {row[1]:.0f}°C → P(打球)={p:.3f}")
