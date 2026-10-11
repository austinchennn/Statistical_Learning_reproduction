"""手写梯度提升树（XGBoost 风格的二阶 GBDT）。

- Loss:  SquaredLoss / LogLoss，只负责两件事：g = dL/dŷ, h = d²L/dŷ²（对预测值 ŷ 求导，不对 x 求导）
- GradientTree: 一棵回归树，分裂准则是梯度 Gain（不是 Gini / IG）
    * 连续特征：按阈值 x <= v 切分
    * 离散特征（阴天/雨天…）：按类别子集 {A} vs {B, C} 切分 —— x 只是“分流器”
- GBDT: 加法模型 ŷ^(t) = ŷ^(t-1) + η·f_t(x)，每轮先用“历史总成绩” ŷ^(t-1) 算 g/h，再建树
"""
import numpy as np


# ---------------- 损失函数：公式求导写死在这里，训练时代入数字 ----------------
class SquaredLoss:
    """L = 1/2 (y - ŷ)^2  →  g = ŷ - y,  h = 1"""
    def grad_hess(self, y, y_hat):
        return y_hat - y, np.ones_like(y_hat)

    def base_score(self, y):
        return float(np.mean(y))  # 初始预测 ŷ^(0) = 均值

    def transform(self, raw):
        return raw


class LogLoss:
    """y ∈ {0,1}，ŷ 是 logit（原始分数），p = sigmoid(ŷ)
    L = -[y log p + (1-y) log(1-p)]  →  g = p - y,  h = p(1-p)"""
    def grad_hess(self, y, y_hat):
        p = sigmoid(y_hat)
        return p - y, p * (1 - p)

    def base_score(self, y):
        p = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        return float(np.log(p / (1 - p)))  # 初始 logit = 正类比例的对数几率

    def transform(self, raw):
        return sigmoid(raw)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------------- 用梯度建树 ----------------
def leaf_weight(G, H, lam):
    """最优叶子值 w* = -G / (H + λ)"""
    return -G / (H + lam)


def structure_score(G, H, lam):
    """该节点能让 loss 下降的量 ∝ G² / (H + λ)"""
    return G * G / (H + lam)


class GradientTree:
    def __init__(self, max_depth=3, lam=1.0, gamma=0.0, min_child_weight=1e-3, categorical=()):
        self.max_depth = max_depth
        self.lam = lam          # λ：L2 正则
        self.gamma = gamma      # γ：每多一个叶子的惩罚
        self.min_child_weight = min_child_weight  # 子节点 H 的下限
        self.categorical = set(categorical)       # 哪些列是离散特征

    def fit(self, X, g, h):
        self.root_ = self._build(X, g, h, np.arange(len(g)), depth=0)
        return self

    def _build(self, X, g, h, idx, depth):
        G, H = g[idx].sum(), h[idx].sum()
        node = {"leaf": True, "value": leaf_weight(G, H, self.lam)}
        if depth >= self.max_depth or len(idx) < 2:
            return node

        best = self._best_split(X, g, h, idx, G, H)
        if best is None:
            return node
        gain, j, rule, left_mask = best
        node = {"leaf": False, "feature": j, "rule": rule, "gain": gain,
                "left": self._build(X, g, h, idx[left_mask], depth + 1),
                "right": self._build(X, g, h, idx[~left_mask], depth + 1)}
        return node

    def _gain(self, GL, HL, GR, HR):
        lam = self.lam
        return 0.5 * (structure_score(GL, HL, lam) + structure_score(GR, HR, lam)
                      - structure_score(GL + GR, HL + HR, lam)) - self.gamma

    def _best_split(self, X, g, h, idx, G, H):
        best = None
        for j in range(X.shape[1]):
            col = X[idx, j]
            if j in self.categorical:
                candidates = self._categorical_candidates(col, g[idx], h[idx])
            else:
                candidates = self._numeric_candidates(col)
            for rule, left_mask in candidates:
                GL, HL = g[idx][left_mask].sum(), h[idx][left_mask].sum()
                GR, HR = G - GL, H - HL
                if HL < self.min_child_weight or HR < self.min_child_weight:
                    continue
                gain = self._gain(GL, HL, GR, HR)
                if gain > 0 and (best is None or gain > best[0]):
                    best = (gain, j, rule, left_mask)
        return best

    @staticmethod
    def _numeric_candidates(col):
        vals = np.unique(col.astype(float))
        for a, b in zip(vals[:-1], vals[1:]):
            thr = (a + b) / 2
            yield ("<=", thr), col.astype(float) <= thr

    @staticmethod
    def _categorical_candidates(col, g, h):
        """离散特征的子集切分。把每个类别按 G_c/H_c（≈ 该类别的最优叶子方向）排序，
        最优二分一定是排序后的某个“前缀 vs 剩余”（LightGBM 同款技巧），
        所以 K 个类别只需试 K-1 种，而不是 2^(K-1) 种。"""
        cats = np.unique(col)
        order = sorted(cats, key=lambda c: g[col == c].sum() / (h[col == c].sum() + 1e-12))
        for k in range(1, len(order)):
            left_set = frozenset(order[:k])
            yield ("in", left_set), np.isin(col, list(left_set))

    def predict(self, X):
        return np.array([self._predict_one(self.root_, x) for x in X])

    def _predict_one(self, node, x):
        while not node["leaf"]:
            op, v = node["rule"]
            go_left = (float(x[node["feature"]]) <= v) if op == "<=" else (x[node["feature"]] in v)
            node = node["left"] if go_left else node["right"]
        return node["value"]

    def describe(self, feature_names=None, node=None, indent="", tag=""):
        node = self.root_ if node is None else node
        if node["leaf"]:
            return f"{indent}{tag}leaf: w = {node['value']:+.3f}\n"
        name = feature_names[node["feature"]] if feature_names else f"x{node['feature']}"
        op, v = node["rule"]
        cond = f"{name} <= {v:.3g}" if op == "<=" else f"{name} in {{{', '.join(sorted(map(str, v)))}}}"
        s = f"{indent}{tag}[{cond}]  (Gain={node['gain']:.3f})\n"
        s += self.describe(feature_names, node["left"], indent + "    ", "yes→ ")
        s += self.describe(feature_names, node["right"], indent + "    ", "no → ")
        return s


# ---------------- Boosting：接力跑 ----------------
class GBDT:
    def __init__(self, loss="squared", n_estimators=50, learning_rate=0.3,
                 max_depth=3, lam=1.0, gamma=0.0, categorical=(), verbose=False):
        self.loss = SquaredLoss() if loss == "squared" else LogLoss()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate  # η：每棵树只走一小步
        self.tree_kw = dict(max_depth=max_depth, lam=lam, gamma=gamma, categorical=categorical)
        self.verbose = verbose

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.base_score_ = self.loss.base_score(y)
        y_hat = np.full(len(y), self.base_score_)      # 第 0 步：还没有树，ŷ^(0) = 初始值
        self.trees_, self.history_ = [], []
        for t in range(self.n_estimators):
            g, h = self.loss.grad_hess(y, y_hat)       # 用“历史总成绩” ŷ^(t-1) 算导数 → 纯数值
            tree = GradientTree(**self.tree_kw).fit(X, g, h)  # 新树拟合的是 g/h，不是 y
            y_hat = y_hat + self.learning_rate * tree.predict(X)  # 加到总成绩上
            self.trees_.append(tree)
            self.history_.append({"g": g, "h": h, "y_hat": y_hat.copy()})
        return self

    def decision_function(self, X):
        raw = np.full(len(X), self.base_score_)
        for tree in self.trees_:
            raw += self.learning_rate * tree.predict(X)
        return raw

    def predict(self, X):
        out = self.loss.transform(self.decision_function(X))
        return (out >= 0.5).astype(int) if isinstance(self.loss, LogLoss) else out

    def predict_proba(self, X):
        return self.loss.transform(self.decision_function(X))
