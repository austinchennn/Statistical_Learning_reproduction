"""例 1：分类。KNN + 决策树 + 逻辑回归 → 逻辑回归元模型（输入是 3 个 P(y=1)）。"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from stacking import StackingEnsemble

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

base_models = [
    ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))),
    ("tree", DecisionTreeClassifier(max_depth=4, random_state=0)),
    ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
]
meta_model = LogisticRegression()  # 简单模型，防止第二层过拟合

stack = StackingEnsemble(base_models, meta_model, task="classification").fit(X_train, y_train)

print("单个基模型测试准确率：")
for name, model in stack.fitted_bases_:
    print(f"  {name:7s} {accuracy_score(y_test, model.predict(X_test)):.4f}")
print(f"Stacking 测试准确率： {accuracy_score(y_test, stack.predict(X_test)):.4f}")

print("\nmeta-learner 学到的权重（作用在各模型的 P(y=1) 上，越大越被信任）：")
for name, w in stack.meta_weights().items():
    print(f"  {name:9s} {w:+.3f}")
