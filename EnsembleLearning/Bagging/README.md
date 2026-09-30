# Bagging（Bootstrap Aggregating）

## 目录结构

| 路径 | 内容 |
|---|---|
| `bagging.py` | 手写的 Bagging 核心：`bootstrap_sample`（有放回抽样 + OOB 索引）+ `BaggingEnsemble`（平均/软投票/硬投票 + OOB 估计） |
| `01_bootstrap/` | 例 1：Bootstrap 抽样，验证 63.2% / 36.8% |
| `02_bagged_trees/` | 例 2：Bagged 决策树，$M$ 增大时测试准确率上升后趋平；OOB 估计 vs 测试集 |
| `03_random_forest/` | 例 3：随机森林（每次分裂只看 $\sqrt{D}$ 个特征），对比树间相关性 $\rho$ |
| `04_bias_vs_variance/` | 例 4：Bagging 只降方差不降偏差（深树 vs 线性回归） |
| `concepts/` | 概念笔记：[Bootstrap 与 63.2%](concepts/01_Bootstrap抽样与63.2%.md)、[OOB 误差](concepts/02_OOB袋外误差.md)、[方差公式 ρσ²](concepts/03_集成方差公式.md) |

运行：`cd EnsembleLearning/Bagging && python 02_bagged_trees/bagged_trees.py`

## 1. 基本定义

对训练集做**有放回抽样（bootstrap）**得到多个不同的数据集，在每个数据集上训练**同一种**模型，最后对预测取平均（回归）或投票（分类）。核心作用：**降低方差**。

## 2. 基本流程与公式

给定训练集 $\mathcal{D} = \{(\mathbf{x}^{(i)}, t^{(i)})\}_{i=1}^N$：

1. 对 $m = 1, \dots, M$：
   - 从 $\mathcal{D}$ 中**有放回**抽 $N$ 个样本，得到 $\mathcal{D}_m$。
   - 在 $\mathcal{D}_m$ 上训练 $h_m$。
2. 合并：
$$\text{回归：}\ \hat{y} = \frac{1}{M}\sum_{m=1}^M h_m(\mathbf{x}) \qquad \text{分类：}\ \hat{y} = \text{majority vote}\{h_m(\mathbf{x})\}$$

**方差分析**：设每个 $h_m$ 方差为 $\sigma^2$，两两相关系数为 $\rho$，则
$$\mathrm{Var}\Big(\frac{1}{M}\sum_m h_m\Big) = \rho\,\sigma^2 + \frac{1-\rho}{M}\,\sigma^2$$
- $M \to \infty$ 时第二项消失，方差降至 $\rho\sigma^2$。
- 偏差基本不变（与单个模型相同）。
- 所以要让 $\rho$ 尽量小 → 随机森林的动机。

推导见 [concepts/03_集成方差公式.md](concepts/03_集成方差公式.md)。

**Bootstrap 性质**：每个样本未被抽中的概率 $(1 - 1/N)^N \to e^{-1} \approx 0.368$，即每个 $\mathcal{D}_m$ 约含 **63.2%** 的不同样本，剩下 ~36.8% 为**袋外（out-of-bag, OOB）**样本，可用来免费估计泛化误差。

## 3. 例子

**例 1：Bootstrap 抽样**（`01_bootstrap/`）
$\mathcal{D} = \{1,2,3,4,5\}$，可能抽到 $\mathcal{D}_1 = \{1,1,3,4,5\}$（2 为 OOB），$\mathcal{D}_2 = \{2,2,2,3,5\}$（1、4 为 OOB）。脚本还验证了 $N$ 增大时不同样本占比收敛到 $1 - 1/e \approx 0.632$。

**例 2：Bagged 决策树**（`02_bagged_trees/`）
单棵深决策树方差很大（数据稍变树就大变）。训练 100 棵在不同 bootstrap 集上的深树，投票后测试误差明显下降。实测：单棵深树测试 0.830 → $M=100$ 时 0.893，且 OOB 估计 0.889 与测试准确率很接近。

**例 3：随机森林（Random Forest）**（`03_random_forest/`）
Bagging + 每次分裂时只从随机选的 $k$ 个特征中挑最佳分裂（分类常用 $k \approx \sqrt{D}$）。这进一步降低树之间的相关性 $\rho$，从而比普通 bagging 方差更低。实测树间 $\rho$：0.468 → 0.421。

**例 4：Bagging 不降偏差**（`04_bias_vs_variance/`）
在同一真实函数 $\sin 2x$ 上重复抽训练集估计 bias² 与 variance：深树的 variance 约减半，而线性回归的 bias²（≈0.436）与 variance 几乎不变。

## 4. 什么时候用

- 基模型是**低偏差、高方差**的：深决策树、1-NN（注：KNN 对 bagging 收益不大，因其对数据扰动较稳定）、未剪枝的模型。
- 数据有噪声、单模型不稳定时。
- 需要并行训练、或想用 OOB 估计误差时。

## 5. 优缺点

**优点**
- 显著降低方差，减少过拟合。
- 完全可并行。
- OOB 误差免去额外验证集。
- 对噪声较鲁棒。

**缺点**
- **几乎不降低偏差**：对线性回归这类高偏差、低方差模型帮助很小。
- 失去单棵树的可解释性。
- 内存/推理开销为 $M$ 倍。

## 6. 注意事项

- 基模型要**不稳定**才有意义（数据小变 → 模型大变）。
- 相关性 $\rho$ 限制了收益上限；想更好就增加随机性（随机森林）。
- $M$ 增大一般**不会导致过拟合**，只是收益递减。
- 分类中可对概率取平均（软投票）而非硬投票。`BaggingEnsemble(voting="soft"|"hard")` 两种都实现了。

## 7. FAQ

**Q1：Bagging 和 Stacking 有什么区别？**

| | Bagging | Stacking |
|---|---|---|
| 基模型 | **同一种**模型，不同 bootstrap 数据 | **不同种**模型，同一份数据 |
| 组合方式 | 固定：平均 / 投票 | **学出来**：meta-learner |
| 主要作用 | 降方差 | 利用模型间互补性 |
| 数据复用 | OOB 样本估计误差 | OOF 预测训练 meta |

OOB 与 OOF 思路相同："只用没见过这个样本的模型来预测它"。区别是 OOF 来自 K 折（每个样本恰好一次留出），OOB 来自 bootstrap（每个样本平均被约 $0.368M$ 个模型留出）。

**Q2：为什么 `_proba_aligned` 要对齐概率列？**

小数据或类别不平衡时，某个 bootstrap 集可能恰好没有某一类，那棵树的 `predict_proba` 就少一列。直接平均会错位，所以按 `classes_` 把每个模型的概率放回正确的列。
