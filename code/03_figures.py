import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont("/Library/Fonts/Arial Unicode.ttf")
plt.rcParams.update({"font.family": "Arial Unicode MS", "axes.unicode_minus": False, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
import importlib.util
spec = importlib.util.spec_from_file_location("a", "02_analysis.py"); a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
df, N, oracle, naive, COEF = a.df, a.N, a.oracle, a.naive, a.COEF
C_ORA, C_LLM, C_HUM, C_COR = "#222222", "#C0392B", "#7F8C8D", "#2E86C1"
import time
TS = time.strftime("%Y%m%d%H%M")
PFX = "Lizhongbin-xxx"       # 图片命名：作者-编号-FigNN-时间戳

# ---------- 图 1：LLM 错误集中在难判句子 ----------
q = pd.read_csv("quality.csv")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
groups = ["易判 (一致性≥75%)", "难判 (一致性<75%)"]; x = np.arange(3); w = 0.36
for j, (g, col) in enumerate(zip(groups, ["#85C1E9", "#C0392B"])):
    s = q[q["样本"] == g]
    ax[0].bar(x + (j - .5) * w, s.accuracy, w, color=col, label=g)
    ax[1].bar(x + (j - .5) * w, s["LLM正面比例-人工"], w, color=col, label=g)
for k in range(2):
    ax[k].set_xticks(x); ax[k].set_xticklabels(["Prompt 1", "Prompt 2", "Prompt 3"])
ax[0].set_ylim(0, 1.18); ax[0].set_title("与人工标签的一致率 (accuracy)")
ax[1].axhline(0, color="k", lw=.8); ax[1].set_title("正面比例：LLM − 人工")
ax[0].legend(frameon=False, loc="upper center", ncol=2, fontsize=9)
fig.tight_layout(); fig.savefig(f"{PFX}-Fig01-{TS}.png", dpi=200)

# ---------- 图 2：同一批数据，三套 prompt ----------
rng = np.random.default_rng(7); n = 300
pi = np.full(N, n / N); R = (rng.random(N) < pi).astype(int)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for pan, c in enumerate("AB"):
    k = COEF[c]; rows = [("人工全样本 (真值)", *[v[k] for v in (oracle[c], [0]*4)], C_ORA)]
    for p in ["P1", "P2", "P3"]:
        b, s = naive[(c, p)]; rows.append((f"直接用 LLM 标签 · {p}", b[k], s[k], C_LLM))
    b, s = a.ols(*a.design(df[R == 1], "pos", c)); rows.append(("只用 300 条人工标签", b[k], s[k], C_HUM))
    for p in ["P1", "P2", "P3"]:
        b, s = a.dsl(df, R, pi, f"pos_{p}", c); rows.append((f"DSL 校正 · {p}", b[k], s[k], C_COR))
    rows = rows[::-1]
    for i, (lab, b, s, col) in enumerate(rows):
        ax[pan].errorbar(b, i, xerr=1.96 * s, fmt="o", color=col, capsize=3, ms=6)
    ax[pan].axvline(oracle[c][k], color=C_ORA, ls="--", lw=1)
    if c == "A": ax[pan].axvline(0, color="#999", lw=.8)
    ax[pan].set_yticks(range(len(rows))); ax[pan].set_yticklabels([r[0] for r in rows] if pan == 0 else [])
    ax[pan].set_title("情形 A：标签作被解释变量\n系数：难判句子 → 正面情绪" if c == "A"
                      else "情形 B：标签作解释变量\n系数：正面情绪 → 模拟收益")
fig.tight_layout(); fig.savefig(f"{PFX}-Fig02-{TS}.png", dpi=200)

# ---------- 图 3：人工标注量 vs 置信区间宽度 ----------
cv = pd.read_csv("mc_curve.csv")
cs = cv.groupby(["case", "method", "n"]).agg(w=("se", lambda s: 2 * 1.959964 * s.mean()), cov=("cover", "mean")).reset_index()
sty = {"人工小样本": (C_HUM, "o", "只用人工标签"), "PPI": ("#8E44AD", "s", "PPI++ (ppi_py)"),
       "DSL": (C_COR, "^", "DSL（随机抽样）"), "DSL-分层抽样": ("#117A65", "D", "DSL（难判句子 3 倍抽样）")}
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
for pan, c in enumerate("AB"):
    for m, (col, mk, lab) in sty.items():
        s = cs[(cs.case == c) & (cs.method == m)]
        if len(s): ax[pan].plot(s.n, s.w, marker=mk, color=col, label=lab)
    ax[pan].set_xlabel("人工标注条数 n"); ax[pan].set_ylabel("95% 置信区间平均宽度")
    ax[pan].set_title("情形 A：标签作被解释变量" if c == "A" else "情形 B：标签作解释变量")
ax[0].legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(f"{PFX}-Fig03-{TS}.png", dpi=200)

# 打印图 2 的数字，供正文引用
for c in "AB":
    k = COEF[c]
    print(c, "human300", np.round(a.ols(*a.design(df[R == 1], "pos", c))[0][k], 4), np.round(a.ols(*a.design(df[R == 1], "pos", c))[1][k], 4), "n_lab", R.sum())
    for p in ["P1", "P2", "P3"]:
        b, s = a.dsl(df, R, pi, f"pos_{p}", c); print(c, p, "DSL", round(b[k], 4), round(s[k], 4))
