"""例 1：Bootstrap 抽样。验证每个 D_m 约含 63.2% 的不同样本，~36.8% 为 OOB。"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from bagging import bootstrap_sample

rng = np.random.default_rng(0)
D = np.array([1, 2, 3, 4, 5])
print("小例子 D = {1,2,3,4,5}：")
for m in range(1, 4):
    in_bag, oob = bootstrap_sample(len(D), rng)
    print(f"  D_{m} = {sorted(D[in_bag].tolist())}   OOB = {D[oob].tolist()}")

print("\n不同样本占比 vs 理论值 1-(1-1/N)^N：")
for N in [5, 10, 100, 1000, 10000]:
    ratios = [len(np.unique(bootstrap_sample(N, rng)[0])) / N for _ in range(200)]
    print(f"  N={N:6d}  实测 {np.mean(ratios):.4f}  理论 {1 - (1 - 1 / N) ** N:.4f}")
print(f"  N→∞     极限 1-1/e = {1 - np.exp(-1):.4f}")
