"""例 3：随机森林 = Bagging + 每次分裂只从随机 k≈√D 个特征中选。树间相关性 ρ 更低 → 集成方差更低。"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from bagging import BaggingEnsemble

X, y = make_classification(n_samples=1000, n_features=25, n_informative=6, flip_y=0.1, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)


def mean_pairwise_corr(ens, X):
    """树之间在测试集上 P(y=1) 的平均两两相关系数，即公式里的 ρ。"""
    P = np.array([ens._proba_aligned(h, X)[:, 1] for h in ens.estimators_])
    C = np.corrcoef(P)
    return C[np.triu_indices_from(C, k=1)].mean()


models = {
    "Bagged trees（k=D）": DecisionTreeClassifier(random_state=0),
    "随机森林（k=√D）": DecisionTreeClassifier(max_features="sqrt", random_state=0),  # 每次分裂随机选特征
}
for name, tree in models.items():
    ens = BaggingEnsemble(tree, n_estimators=200).fit(X_train, y_train)
    print(f"{name:18s} 测试 {accuracy_score(y_test, ens.predict(X_test)):.4f}  "
          f"OOB {ens.oob_score(X_train, y_train):.4f}  树间 ρ ≈ {mean_pairwise_corr(ens, X_test):.3f}")
