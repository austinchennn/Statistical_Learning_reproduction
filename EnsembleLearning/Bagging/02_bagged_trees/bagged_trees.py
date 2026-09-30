"""例 2：Bagged 决策树。单棵深树方差大；100 棵 bootstrap 深树投票后测试误差下降，OOB 估计接近测试误差。"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from bagging import BaggingEnsemble

X, y = make_classification(n_samples=1000, n_features=20, n_informative=8, flip_y=0.1, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

deep_tree = DecisionTreeClassifier(random_state=0)  # 不剪枝：低偏差、高方差
single = deep_tree.fit(X_train, y_train)
print(f"单棵深树     训练 {accuracy_score(y_train, single.predict(X_train)):.4f}  "
      f"测试 {accuracy_score(y_test, single.predict(X_test)):.4f}")

print("\nM 增大：测试准确率上升后趋平（收益递减，但不会过拟合）")
for M in [1, 5, 10, 25, 50, 100, 200]:
    bag = BaggingEnsemble(deep_tree, n_estimators=M, voting="hard").fit(X_train, y_train)
    print(f"  M={M:3d}  测试 {accuracy_score(y_test, bag.predict(X_test)):.4f}")

bag = BaggingEnsemble(deep_tree, n_estimators=100, voting="soft").fit(X_train, y_train)
print(f"\nM=100 软投票  测试 {accuracy_score(y_test, bag.predict(X_test)):.4f}  "
      f"OOB 估计 {bag.oob_score(X_train, y_train):.4f}（没用测试集）")
