"""例 3：为什么 meta-learner 的训练输入必须是 OOF 预测。

对比两种构造元特征的方式：
- 泄漏版：base 在全量训练集上训练，再对训练集自身预测 → 1-NN 训练集准确率 100%
- OOF 版：用 generate_oof，每个样本的预测来自没见过它的模型
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from stacking import generate_oof, make_folds

# 带 15% 标签噪声的数据：1-NN 会把噪声也背下来
X, y = make_classification(n_samples=1000, n_features=10, n_informative=4,
                           flip_y=0.15, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

base_models = [
    ("1nn", KNeighborsClassifier(n_neighbors=1)),
    ("logreg", LogisticRegression(max_iter=1000)),
]
fitted = [clone(m).fit(X_train, y_train) for _, m in base_models]
Z_test = np.column_stack([m.predict_proba(X_test)[:, 1] for m in fitted])

# 泄漏版元特征：在训练数据上"自己预测自己"
Z_leak = np.column_stack([m.predict_proba(X_train)[:, 1] for m in fitted])
# OOF 版元特征
Z_oof = generate_oof(base_models, X_train, y_train, make_folds(y_train), "classification")

print("1-NN 在训练集上的准确率：")
print(f"  自己预测自己: {accuracy_score(y_train, Z_leak[:, 0] > 0.5):.4f}   <- 假象")
print(f"  OOF:          {accuracy_score(y_train, Z_oof[:, 0] > 0.5):.4f}   <- 真实水平")

for label, Z_train in [("泄漏版", Z_leak), ("OOF 版", Z_oof)]:
    meta = LogisticRegression().fit(Z_train, y_train)
    w = meta.coef_.ravel()
    print(f"\n[{label}] meta 权重: 1nn={w[0]:+.2f}, logreg={w[1]:+.2f}")
    print(f"[{label}] Stacking 测试准确率: {accuracy_score(y_test, meta.predict(Z_test)):.4f}")
