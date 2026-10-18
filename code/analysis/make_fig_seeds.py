# -*- coding: utf-8 -*-
"""
多种子结果图
================================================================================
左：四个关键配置的 testA 指标（均值 ± 种子间标准差），叠加阶段0 基线与经典方法
右：BN vs GN 的逐轮验证曲线（展示振荡被消除）
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = r"E:/workbuddy/store/2026-10-12-22-50-08"
FIG = os.path.join(HERE, "figs_pamnet")
OUT = r"E:/project/data/output_pamnet"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10

S = json.load(open(os.path.join(FIG, "seed_agg.json"), encoding="utf-8"))
BL = json.load(open(os.path.join(FIG, "baseline_pam.json"), encoding="utf-8"))
C = S["testA"]["configs"]


def stat(name):
    rows = C[name]
    v = {m: np.array([rows[s][m] for s in rows]) for m in
         ("frmgain_clean", "frmgain_imp", "frmgain_imp_min")}
    return {m: (v[m].mean(), v[m].std(ddof=1) if len(v[m]) > 1 else 0.0, len(v[m]))
            for m in v}


NAMES = ["纯时域 +脉冲分支（GN）", "纯时域 −脉冲分支（GN）",
         "频域精修 +p注入（GN）", "频域精修 −脉冲分支（GN）"]
SHORT = ["纯时域\n+p分支", "纯时域\n−p分支", "频域精修\n+p注入", "频域精修\n−p分支"]

fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), dpi=150)
METRICS = [("frmgain_clean", "干净帧降噪增益 (dB)"),
           ("frmgain_imp", "干扰帧降噪增益 (dB)"),
           ("frmgain_imp_min", "干扰帧最差增益 (dB)")]
xs = np.arange(len(NAMES))
for ax, (m, title) in zip(axs, METRICS):
    st = [stat(n) for n in NAMES]
    vals = [s[m][0] for s in st]
    errs = [s[m][1] for s in st]
    cols = ["#4c78a8", "#9aa0a6", "#e45756", "#f58518"]
    b = ax.bar(xs, vals, 0.6, color=cols, yerr=errs,
               error_kw=dict(ecolor="#333", lw=1.4, capsize=4))
    for r, v, e in zip(b, vals, errs):
        ax.text(r.get_x() + r.get_width() / 2, v + max(errs) * 0.15 + 0.35,
                "%.1f" % v, ha="center", fontsize=9, fontweight="bold")
    # 经典基线
    for nm, c, ls in [("理想带通 798-1500Hz", "#7f7f7f", ":"),
                      ("谱减法（盲）", "#54a24b", "--")]:
        if nm in BL["testA"]:
            ax.axhline(BL["testA"][nm][m], color=c, ls=ls, lw=1.5,
                       label="%s %.2f" % (nm, BL["testA"][nm][m]))
    ax.set_xticks(xs)
    ax.set_xticklabels(SHORT, fontsize=9)
    ax.set_ylabel(title)
    ax.set_ylim(0, max(vals + [max(errs)]) * 1.28)
    ax.grid(alpha=0.25, ls="--", axis="y")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8.5, loc="upper left")
axs[1].set_title("误差棒 = 种子间标准差（n=3）", fontsize=10)
fig.suptitle("PAM-Net 关键配置的多种子结果（testA：新波形 + 见过的信道）", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "figM_seeds.png"))
plt.close(fig)
print("图 -> figs_pamnet/figM_seeds.png")

# ---------------- 右图：BN vs GN 曲线 ----------------
fig2, ax = plt.subplots(figsize=(8.2, 5.0), dpi=150)
PAIRS = [("pamnet", "纯时域 · BatchNorm", "#9aa0a6", "--"),
         ("pamnet_gn", "纯时域 · GroupNorm", "#2563eb", "-"),
         ("pamnet_freq_pf", "频域精修+p · BatchNorm", "#f0a07a", "--"),
         ("pamnet_freq_pf_gn", "频域精修+p · GroupNorm", "#b91c1c", "-")]
for tag, lab, c, ls in PAIRS:
    p = os.path.join(OUT, "train_log_%s.txt" % tag)
    if not os.path.exists(p):
        continue
    try:
        d = np.genfromtxt(p, delimiter=",", names=True)
    except Exception:
        continue
    if d.size == 0 or "val_fg_med" not in d.dtype.names:
        continue
    y = np.asarray(d["val_fg_med"], dtype=float)
    ax.plot(np.arange(1, len(y) + 1), y, marker="o", ms=3.2, lw=1.8, color=c, ls=ls,
            label="%s（逐轮 σ=%.2f）" % (lab, y.std(ddof=1)))
ax.set_xlabel("epoch")
ax.set_ylabel("验证集 帧降噪增益中位数 (dB)")
ax.set_title("归一化方式对训练稳定性的影响", fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.25, ls="--")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig2.tight_layout()
fig2.savefig(os.path.join(FIG, "figN_norm.png"))
plt.close(fig2)
print("图 -> figs_pamnet/figN_norm.png")
