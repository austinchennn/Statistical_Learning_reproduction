# OOF 预测文件（OOF Prediction Artifacts）

前置：[02_OOF预测矩阵Z.md](02_OOF预测矩阵Z.md)

## 1. 定义与公式

OOF 预测文件就是把 **Z 的某一列（或几列）持久化到磁盘**，连同足够让下游对齐和追溯的元信息，作为"基模型训练任务"与"meta 训练任务"之间的**数据契约**。

对第 $m$ 个基模型，通常产出两份文件：

$$
\text{oof}_m = \big\{(\text{id}_i,\ k(i),\ z_m^{(i)})\big\}_{i=1}^{n}
\qquad
\text{test}_m = \big\{(\text{id}_j,\ \hat{z}_m^{(j)})\big\}_{j \in \text{test}}
$$

其中 $\hat{z}_m^{(j)}$ 是测试样本的预测，按"全量重训"或"$K$ 个折模型平均"得到：

$$\hat{z}_m^{(j)} = \frac{1}{K}\sum_{k=1}^{K} h_m^{(-k)}\big(\mathbf{x}^{(j)}\big)$$

meta 任务要做的只是：按 `id` 把 $M$ 份 `oof_m` 拼接（join）起来得到 Z，再拼上标签。

## 2. 形象的例子

仍然是招聘：每位面试官不必和总监坐在同一个会议室。他们各自把打分表**写成表格交给 HR 系统**，表里有候选人编号、面试批次、分数和面试官工号。总监只看这些表格做决定，根本不需要见到面试官本人。
- 换了一位面试官，只要他交的表格格式一样，总监的流程不用改。
- 但总监的"信任权重"需要重新学。

## 3. 带数据的例子

接着 Z 那篇的数据，两个基模型各自落盘一个文件（parquet/csv 均可）：

`oof/mean_model__v1.parquet`

| sample_id | fold_id | pred |
|---|---|---|
| 1 | 1 | 7 |
| 2 | 1 | 7 |
| 3 | 2 | 6 |
| 4 | 2 | 6 |
| 5 | 3 | 5 |
| 6 | 3 | 5 |

`oof/knn1__v1.parquet`

| sample_id | fold_id | pred |
|---|---|---|
| 1 | 1 | 4 |
| 2 | 1 | 4 |
| 3 | 2 | 5 |
| 4 | 2 | 6 |
| 5 | 3 | 8 |
| 6 | 3 | 8 |

每个文件旁边还有一份元信息，例如 `knn1__v1.json`：

```json
{
  "model_name": "knn1", "model_version": "v1",
  "fold_scheme": "folds_2026-09-30_seed42", "n_splits": 3,
  "pred_type": "regression_value",
  "oof_score": {"mse": 2.5},
  "train_data_snapshot": "train_2026-09-30",
  "code_commit": "67d1598"
}
```

meta 任务的核心代码只有几行：

```python
Z = (labels                                   # sample_id, t
     .merge(read("oof/mean_model__v1.parquet").rename(columns={"pred": "mean_model"}), on="sample_id")
     .merge(read("oof/knn1__v1.parquet").rename(columns={"pred": "knn1"}),             on="sample_id"))
assert Z.fold_id_x.equals(Z.fold_id_y)       # 两个文件必须来自同一套 fold 划分
meta.fit(Z[["mean_model", "knn1"]], Z["t"])
```


## 4. 为什么用它、来源、解决的问题

- **来源**：Kaggle 社区的实践。多人组队时，每人训练自己的模型，交换 OOF 文件和 test 预测文件来做融合，互不需要对方代码。后来被工业界 ML 平台采纳，变成"预测结果即特征"的标准做法。
- **解决的问题**：
  1. **解耦**：基模型可以用不同语言、框架、机器、团队，meta 只依赖文件格式。
  2. **省算力**：基模型是 $M \times K$ 次训练中最贵的部分。落盘后调 meta、增删基模型都不必重跑已有的基模型。
  3. **可追溯、可复现**：出问题时能查到"是哪个模型哪个版本的哪份 OOF 导致的"。

## 5. 优缺点

**优点**
- 训练流程可并行、可增量：新增一个基模型只要多交一个文件。
- meta 的实验成本极低（秒级）。
- 天然支持版本管理和审计。

**缺点**
- 多了一层"数据管理"：文件版本、fold 方案、训练数据快照必须一一对应，对错了会**静默出错**（不报错，只是结果变差或泄漏）。
- 训练路径（读文件）和上线路径（实时调用模型）是两套代码，存在不一致的风险。

## 6. 工程师必须会 / 必须注意

1. **必须带主键 `sample_id` 并按它 join**，不要依赖行顺序。行顺序在 shuffle、过滤、分布式写入后很容易乱。
2. **必须记录 `fold_id` 和 fold 方案名**，meta 任务启动时校验所有文件用的是同一套划分（见 [04_共用fold划分.md](04_共用fold划分.md)）。
3. **必须记录模型版本和训练数据快照**：基模型重训后 OOF 分布会变，meta 必须跟着重训，否则上线时输入分布和训练时不一致。
4. **写清预测类型**：概率还是 logit、回归值是否做过 log 变换。两个文件语义不同却被当作同一类特征，是常见 bug。
5. **train/serve 一致性**：离线 meta 用的是 OOF，线上 meta 拿的是全量模型的实时预测。上线前要比对两者的分布（均值、分位数），差异大说明有问题。
6. **测试集预测文件与 OOF 文件成对发布**，并标明测试预测是"全量重训"还是"折模型平均"。
