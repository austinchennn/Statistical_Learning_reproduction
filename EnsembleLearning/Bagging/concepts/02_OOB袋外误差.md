# OOB 袋外误差（Out-of-Bag Error）

## 1. 定义与公式

记 $S_i = \{m : i \notin \mathcal{D}_m\}$ 为"没见过样本 $i$"的模型集合。样本 $i$ 的 OOB 预测只用这些模型：

$$\hat{y}_{\text{oob}}^{(i)} = \frac{1}{|S_i|}\sum_{m \in S_i} h_m(\mathbf{x}^{(i)}) \quad(\text{分类则在 } S_i \text{ 内投票})$$

OOB 误差：

$$\text{Err}_{\text{oob}} = \frac{1}{N'}\sum_{i:\,S_i \neq \varnothing} L\big(t^{(i)},\, \hat{y}_{\text{oob}}^{(i)}\big)$$

期望上 $|S_i| \approx 0.368M$。$M$ 较大时几乎所有样本都有 OOB 预测。

## 2. 形象的例子

100 个老师各自用不同的练习题集备课。考学生 $i$ 的某道题时，只请**没用这道题备过课**的老师来答——这相当于每道题都是"新题"，所以得分是诚实的。

## 3. 带数据的例子

$N = 4$，$M = 3$：

| 模型 | $\mathcal{D}_m$（索引） | OOB |
|---|---|---|
| $h_1$ | {1,1,2,4} | {3} |
| $h_2$ | {2,3,3,4} | {1} |
| $h_3$ | {1,2,2,3} | {4} |

- 样本 1：$S_1 = \{h_2\}$ → 用 $h_2$ 预测
- 样本 2：$S_2 = \varnothing$ → 没有 OOB 预测，不计入
- 样本 3：$S_3 = \{h_1\}$；样本 4：$S_4 = \{h_3\}$

$M$ 只有 3 时覆盖不全；$M = 100$ 时某样本从未 OOB 的概率约 $0.632^{100} \approx 10^{-20}$。

## 4. 与 Stacking 中 OOF 的对比

| | OOB（Bagging） | OOF（Stacking） |
|---|---|---|
| 来源 | bootstrap 随机留出 | K 折系统留出 |
| 每个样本被留出次数 | 随机，期望 $0.368M$ | 恰好 1 次 |
| 用途 | 估计集成的泛化误差 | 构造 meta 的训练特征 |

代码：`BaggingEnsemble.oob_predict / oob_score`；`02_bagged_trees` 中 OOB 0.889 vs 测试 0.893。
