# 梯度 Gain 取代 Gini/IG，离散特征怎么用

> 我当时的卡点：(1) 特征是"阴天/雨天"这种离散值，怎么对 $L$ 求导？(2) 所以 GBDT 分裂不用 Gini/IG 了？

## 1. 离散特征不需要可导——它只是"分流器"

GBDT **永远不对 $x$ 求导**（见 [01](01_伪残差：对谁求导.md)）。$g_i,h_i$ 算好之后，特征的唯一作用是决定**哪些样本的 $g,h$ 归到左边、哪些归到右边**。

试一个切分 {雨} vs {晴, 阴}：

$$G_L=\sum_{i\in\text{雨}} g_i,\ H_L=\sum_{i\in\text{雨}} h_i,\qquad G_R,H_R \text{ 同理}$$

$$\text{Gain}=\frac12\left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right]-\gamma$$

选 Gain 最大的切分。叶子值 $w^*=-G/(H+\lambda)$。

**这些公式从哪来**：对每片叶子，把 $L$ 在 $\hat y^{(t-1)}$ 处二阶泰勒展开，$\sum_i [g_i w + \tfrac12 h_i w^2] + \tfrac12\lambda w^2$，对 $w$ 求最小 → $w^*=-G/(H+\lambda)$，最小值 $=-\tfrac12 G^2/(H+\lambda)$。Gain = 分裂后 loss 下降量。

## 2. 带数据的例子（`03_categorical_split/categorical.py`）

规律：雨天基本不打球。第 1 棵树之前各类别的 $G,H$：

| 天气 | $G$ | $H$ |
|---|---|---|
| 晴 | −14.50 | 20.47 |
| 阴 | −22.75 | 21.61 |
| 雨 | +37.25 | 26.16 |

| 切分 | Gain |
|---|---|
| {雨} vs {晴,阴} | **41.64** |
| {阴} vs 其余 | 16.88 |
| {晴} vs 其余 | 7.05 |

雨天 $G>0$（预测偏高 → 要往下压），晴/阴 $G<0$ → 把符号相反的分开 Gain 最大。全程没对"天气"求导。

**K 个类别要试 $2^{K-1}$ 种子集吗？** 不用。按 $G_c/H_c$ 给类别排序，最优二分一定是"排序后的前缀 vs 剩余"，只需试 $K-1$ 种（LightGBM 的做法，`_categorical_candidates` 实现了它）。

## 3. Gini/IG vs 梯度 Gain

| | 传统决策树 / 随机森林 | GBDT |
|---|---|---|
| 分裂准则 | Gini、信息增益 | 基于 $g,h$ 的 Gain |
| 盯着什么 | 标签 $y$ 的纯度 | 全局 loss 能降多少 |
| 树的类型 | 分类树或回归树 | **永远是回归树**（叶子输出连续值，哪怕做分类） |
| 损失函数 | 绑定 | 任意二阶可导的 $L$ 都行 |

注意：MSE 下 $h=1$，Gain 退化为"方差下降"，所以 sklearn 的 GBDT（一阶，CART 回归树拟合负梯度）和 XGBoost（二阶）在 MSE 下结构一样。

## 4. 工程中的三种处理方式

- **One-Hot**：拆成 `is_rainy` 等 0/1 列，按阈值 0.5 切分。类别多时树会很深。
- **Target Encoding**（CatBoost）：用该类别 $y$ 的均值替换，变成连续特征。要防目标泄漏（CatBoost 用 ordered TS）。
- **原生支持**（LightGBM、新版 XGBoost `enable_categorical`）：内部直接做子集切分 —— 也就是本目录手写的方式。
