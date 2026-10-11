# Boosting：梯度提升树（GBDT）

## 目录结构

| 路径 | 内容 |
|---|---|
| `gbdt.py` | 手写 XGBoost 风格二阶 GBDT：`SquaredLoss`/`LogLoss`（$g,h$）、`GradientTree`（梯度 Gain 分裂，支持连续阈值 + 离散子集切分）、`GBDT`（加法模型） |
| `01_pseudo_residual/` | 例 1：伪残差 = 公式求导 + 代数字；有限差分验证 |
| `02_relay_yhat/` | 例 2：建树前的 $\hat y$ 来自前面树的累加，逐轮打印 |
| `03_categorical_split/` | 例 3：晴/阴/雨离散特征，手算各子集切分的 Gain |
| `04_vs_sklearn/` | 例 4：与 sklearn `GradientBoosting*` 对比，验证实现 |
| `concepts/` | [伪残差：对谁求导](concepts/01_伪残差：对谁求导.md)、[建树前 ŷ 从哪来](concepts/02_建树前ŷ从哪来.md)、[梯度 Gain 与离散特征](concepts/03_梯度Gain与离散特征.md) |

运行：`cd EnsembleLearning/Boosting && python 03_categorical_split/categorical.py`

## 1. 基本定义

**串行**训练一串弱学习器（浅回归树），每棵新树去修正前面所有树的错误，最后**相加**。核心作用：**降低偏差**（对比 Bagging 降方差）。

## 2. 算法与公式

1. 初始化 $\hat y_i^{(0)} = $ 常数（MSE：均值；LogLoss：对数几率）。
2. 对 $t = 1,\dots,T$：
   - 用当前预测算每个样本的 $g_i = \dfrac{\partial L(y_i,\hat y_i)}{\partial \hat y_i}\Big|_{\hat y_i^{(t-1)}}$，$h_i = \dfrac{\partial^2 L}{\partial \hat y_i^2}\Big|_{\hat y_i^{(t-1)}}$
   - 用 $\{(x_i, g_i, h_i)\}$ 建一棵回归树：分裂选 Gain 最大，叶子值 $w^* = -\dfrac{G}{H+\lambda}$
   $$\text{Gain}=\frac12\left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{G^2}{H+\lambda}\right]-\gamma$$
   - 更新 $\hat y^{(t)} = \hat y^{(t-1)} + \eta f_t(x)$
3. 预测：$\hat y = \hat y^{(0)} + \eta\sum_t f_t(x)$（分类再过 sigmoid）

## 3. 我曾经卡住的三个问题

| 问题 | 一句话答案 | 详见 |
|---|---|---|
| 离散特征（阴天/雨天）怎么对 $L$ 求导？ | **不对 $x$ 求导**。只对 $\hat y$ 求导；$x$ 只负责把 $g,h$ 分到左右子树 | [03](concepts/03_梯度Gain与离散特征.md) |
| 还没建树，$\hat y$ 哪来？ | 用**前面已建好的树**的累加（第 0 步用常数） | [02](concepts/02_建树前ŷ从哪来.md) |
| 伪残差是公式求导还是数值计算？ | **先**把 $\hat y$ 当自变量公式求导（MSE: $g=\hat y-y$），**再**代入 $y$ 和 $\hat y^{(t-1)}$ 两个数 | [01](concepts/01_伪残差：对谁求导.md) |

附带结论：GBDT 分裂用**梯度 Gain**，不用 Gini/IG；里面的树**全是回归树**。

## 4. 例子（实测结果）

- **例 1**：$y=100,\hat y=70$ → $g=-30,h=1$，有限差分也得 $-30.0000$；LogLoss 下 $-g=y-p\neq y-\hat y$。
- **例 2**：4 套房子，5 棵树桩，训练 MSE 281.2 → 92.2 → 44.9 → 16.7 → 5.6。
- **例 3**：根节点 {雨} vs {晴,阴} Gain=41.64，远大于其它切分；之后在晴/阴里再按温度切。
- **例 4**：回归测试 MSE 手写 365.3 vs sklearn 364.7；分类准确率 0.920 vs 0.910。

## 5. 优缺点与注意事项

**优点**：表格数据上通常最强；任意可导损失；天然处理混合类型特征、缺失值（XGBoost/LightGBM）。
**缺点**：串行训练（树内可并行，树间不能）；对噪声/离群点较敏感；超参数多。

- $\eta$ 小 + $T$ 大一般更好，但慢；**$T$ 太大会过拟合**（与 Bagging 不同），用早停。
- 树要浅（深度 3–8），弱学习器 → 低方差，偏差交给 boosting 降。
- $\lambda,\gamma$、`min_child_weight` 控制复杂度。
