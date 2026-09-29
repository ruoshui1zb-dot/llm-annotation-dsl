"""
Step 2: 用真实的人工标签 + 真实的 DeepSeek 标签，比较几种下游估计方法。
设计是 design-based 的：语料 (N=4838) 固定，随机性只来自「哪些句子被抽去人工标注」。
目标参数 = 用全部人工标签得到的 OLS 系数 (oracle)。
"""
import numpy as np, pandas as pd
from ppi_py import ppi_ols_ci

rng0 = np.random.default_rng(2026)
df = pd.read_csv("corpus.csv")
for p in ["P1", "P2", "P3"]:
    l = pd.read_csv(f"llm_{p}.csv")[["id", p]]
    df = df.merge(l, on="id")
    df[p] = df[p].where(df[p].isin(["positive", "negative", "neutral"]), "neutral")
MODEL_TAG = pd.read_csv("llm_P1.csv").model.dropna().iloc[0]

df["pos"] = (df.human == "positive").astype(float)
for p in ["P1", "P2", "P3"]:
    df[f"pos_{p}"] = (df[p] == "positive").astype(float)
df["lnw"] = np.log(df.nwords)
# 情形 B 的结果变量：已知参数的模拟（例如新闻发布后的超额收益）
df["ret"] = 0.2 + 1.0 * df.pos + 0.5 * df.ambig + 0.3 * df.has_num + rng0.normal(0, 1, len(df))
N = len(df)

# ---------------- 通用工具 ----------------
def design(d, pos_col, case):
    """返回 (y, X)。case A：标签在左边；case B：标签在右边。"""
    if case == "A":
        X = np.column_stack([np.ones(len(d)), d.ambig, d.has_num, d.lnw]); y = d[pos_col].values
    else:
        X = np.column_stack([np.ones(len(d)), d[pos_col], d.ambig, d.has_num]); y = d.ret.values
    return y, X
COEF = {"A": 1, "B": 1}     # 关注的系数：A 中 ambig，B 中 pos
NAMES = {"A": ["const", "ambig", "has_num", "lnw"], "B": ["const", "pos", "ambig", "has_num"]}

def ols(y, X, w=None):
    w = np.ones(len(y)) if w is None else w
    A = (X * w[:, None]).T @ X
    b = np.linalg.solve(A, (X * w[:, None]).T @ y)
    g = X * (w * (y - X @ b))[:, None]
    Ai = np.linalg.inv(A)
    V = Ai @ (g.T @ g) @ Ai                      # HC0 sandwich
    return b, np.sqrt(np.diag(V))

def dsl(d, R, pi, pcol, case):
    """DSL (Egami et al. 2023) 的线性回归版本，预测值直接来自 LLM（不再额外训练）。
    对每个文档构造 design-adjusted 矩条件：
      g~ = g(W_hat; b) + R/pi * [ g(W; b) - g(W_hat; b) ],   g(W; b) = X (y - X'b)
    其中 W 为人工标签版本 (仅 R=1 可见)，W_hat 为 LLM 标签版本。"""
    yh, Xh = design(d, pcol, case)
    yt, Xt = design(d.assign(pos=np.where(R == 1, d.pos, 0.0)), "pos", case)  # 未标注处取值无关
    c = R / pi
    A = Xh.T @ Xh + (Xt * c[:, None]).T @ Xt - (Xh * c[:, None]).T @ Xh
    bvec = Xh.T @ yh + (Xt * c[:, None]).T @ yt - (Xh * c[:, None]).T @ yh
    b = np.linalg.solve(A, bvec)
    g = Xh * (yh - Xh @ b)[:, None] + c[:, None] * (Xt * (yt - Xt @ b)[:, None] - Xh * (yh - Xh @ b)[:, None])
    Ai = np.linalg.inv(A)
    V = Ai @ (g.T @ g) @ Ai
    return b, np.sqrt(np.diag(V))

def ppi(d, R, pcol):
    lab, unl = d[R == 1], d[R == 0]
    y, X = design(lab, "pos", "A"); yh, _ = design(lab, pcol, "A"); yu, Xu = design(unl, pcol, "A")
    lo, hi = ppi_ols_ci(X, y, yh, Xu, yu, alpha=0.05)
    return (lo + hi) / 2, (hi - lo) / (2 * 1.959964)

# ---------------- 1. 标注质量 ----------------
def kappa(a, b):
    cats = ["positive", "negative", "neutral"]
    po = (a == b).mean(); pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return (po - pe) / (1 - pe)
qual = []
for p in ["P1", "P2", "P3"]:
    for nm, s in [("全部", df), ("易判 (一致性≥75%)", df[df.ambig == 0]), ("难判 (一致性<75%)", df[df.ambig == 1])]:
        tp = ((s[p] == "positive") & (s.human == "positive")).sum()
        prec = tp / max((s[p] == "positive").sum(), 1); rec = tp / max((s.human == "positive").sum(), 1)
        qual.append([p, nm, len(s), (s[p] == s.human).mean(), kappa(s[p], s.human), 2 * prec * rec / (prec + rec),
                     s[f"pos_{p}"].mean() - s.pos.mean()])
qual = pd.DataFrame(qual, columns=["prompt", "样本", "N", "accuracy", "kappa", "F1_pos", "LLM正面比例-人工"])

# ---------------- 2. oracle 与 LLM-only ----------------
oracle = {c: ols(*design(df, "pos", c))[0] for c in "AB"}
naive = {(c, p): ols(*design(df, f"pos_{p}", c)) for c in "AB" for p in ["P1", "P2", "P3"]}

# ---------------- 3. 重复抽样 ----------------
def one_draw(rng, n, pcol, strat=False):
    if strat:   # 难判句子按 3 倍概率抽样，期望样本量仍为 n
        w = np.where(df.ambig == 1, 3.0, 1.0); pi = n * w / w.sum()
    else:
        pi = np.full(N, n / N)
    R = (rng.random(N) < pi).astype(int)
    out = {}
    lab = df[R == 1]
    for c in "AB":
        k = COEF[c]
        if not strat:
            b, s = ols(*design(lab, "pos", c)); out[("人工小样本", c)] = (b[k], s[k])
        b, s = dsl(df, R, pi, pcol, c); out[("DSL-分层抽样" if strat else "DSL", c)] = (b[k], s[k])
    if not strat:
        b, s = ppi(df, R, pcol); out[("PPI", "A")] = (b[1], s[1])
    return out

def monte_carlo(n, pcol, B, seed=1):
    rng = np.random.default_rng(seed); rows = []
    for r in range(B):
        for strat in [False, True]:
            for (m, c), (b, s) in one_draw(rng, n, pcol, strat).items():
                t = oracle[c][COEF[c]]
                rows.append([n, pcol, m, c, b, s, abs(b - t) <= 1.959964 * s])
    return pd.DataFrame(rows, columns=["n", "prompt", "method", "case", "b", "se", "cover"])

if __name__ == "__main__":
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print("model:", MODEL_TAG, " N =", N, " ambig share =", df.ambig.mean().round(3), " pos share =", df.pos.mean().round(3))
    print(qual.round(3).to_string(index=False))
    print("\nOracle A:", dict(zip(NAMES["A"], oracle["A"].round(4))))
    print("Oracle B:", dict(zip(NAMES["B"], oracle["B"].round(4))))
    for (c, p), (b, s) in naive.items():
        k = COEF[c]; print(f"LLM-only {c} {p}: {b[k]:.4f} (se {s[k]:.4f})  CI covers oracle: {abs(b[k]-oracle[c][k])<=1.96*s[k]}")
    mc = pd.concat([monte_carlo(300, f"pos_{p}", 1000, seed=i) for i, p in enumerate(["P1", "P2", "P3"])])
    mc.to_csv("mc_n300.csv", index=False)
    summ = mc.groupby(["case", "prompt", "method"]).agg(mean_b=("b", "mean"), sd_b=("b", "std"), mean_se=("se", "mean"),
                                                        coverage=("cover", "mean")).reset_index()
    summ["ci_width"] = 2 * 1.959964 * summ.mean_se
    print(summ.round(4).to_string(index=False))
    curve = pd.concat([monte_carlo(n, "pos_P1", 400, seed=100 + n) for n in [100, 200, 300, 500, 800, 1200]])
    curve.to_csv("mc_curve.csv", index=False)
    cs = curve.groupby(["case", "method", "n"]).agg(ci_width=("se", lambda s: 2 * 1.959964 * s.mean()),
                                                    coverage=("cover", "mean")).reset_index()
    print(cs.round(4).to_string(index=False))
    qual.to_csv("quality.csv", index=False)
