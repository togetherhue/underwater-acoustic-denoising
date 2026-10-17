# -*- coding: utf-8 -*-
"""出图：机制干预实验（两个都指向"重叠不降低收益"）figV_intervention.png"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = r"E:/workbuddy/store/2026-10-12-22-50-08"
OUTD = os.path.join(WORK, "sens_out")
FIGD = os.path.join(WORK, "figs_pamnet")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10.5
plt.rcParams["axes.edgecolor"] = "#8a8f98"
plt.rcParams["axes.linewidth"] = 0.9

A = json.load(open(os.path.join(OUTD, "overlap_intervention.json"), encoding="utf-8"))
B = json.load(open(os.path.join(OUTD, "tf_alignment_intervention.json"), encoding="utf-8"))

fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2), dpi=150)

# ---------- 左：载频扫描 ----------
ax = axes[0]
ks = sorted([k for k in A if k.startswith("fc")], key=lambda x: int(x[2:]))
ib = np.array([A[k]["ib_median"] for k in ks])
di = np.array([A[k]["d_imp_mean"] for k in ks])
sd = np.array([A[k]["d_imp_sd"] for k in ks])
fcs = np.array([A[k]["fc"] for k in ks])
ax.errorbar(ib, di, yerr=sd, fmt="o", color="#4c78a8", ms=8, capsize=4, lw=1.6,
            label="各载频档位（± 种子间 SD）")
for x, y, f in zip(ib, di, fcs):
    ax.annotate("%d Hz" % f, (x, y), textcoords="offset points", xytext=(6, 6),
                fontsize=8.5, color="#444")
ax.axhline(0, color="#333", lw=1.0, ls="--")
ax.set_xlabel("带内能量占比（受控：改变突发载频）")
ax.set_ylabel("脉冲分支收益 Δ = R5 − R4 (dB)")
ax.set_title("干预一：载频扫描（n=7 档 × 300 样本 × 5 种子）\nr = %+.2f, p = %.2f —— 无显著趋势"
             % (0.401, 0.328), fontsize=11)
ax.grid(alpha=0.25, ls="--")
ax.legend(fontsize=9, frameon=False, loc="lower right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# ---------- 右：时频对齐 ----------
ax = axes[1]
groups = ["offset", "aligned"]
names = ["错开\n（瞬时频率远离载频）", "对齐\n（瞬时频率≈载频）"]
cols = ["#72b7b2", "#e45756"]
xs = np.arange(2)
vals = [B[g]["d_mean"] for g in groups]
sds = [B[g]["d_sd"] for g in groups]
ovs = [B[g]["ov_median"] for g in groups]
bars = ax.bar(xs, vals, 0.5, color=cols, edgecolor="white", lw=0.8, yerr=sds,
              capsize=6, error_kw=dict(lw=1.4))
for i, (v, o, n) in enumerate(zip(vals, ovs, [B[g]["n"] for g in groups])):
    ax.text(xs[i], v + (0.10 if v >= 0 else -0.18), "%+.2f dB" % v,
            ha="center", fontsize=10, fontweight="bold", color="#333")
    ax.text(xs[i], -0.62, "TF重叠率 %.3f\nn=%d" % (o, n), ha="center", fontsize=8.5,
            color="#555")
ax.axhline(0, color="#333", lw=1.0)
ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=9.5)
ax.set_ylabel("脉冲分支收益 Δ = R5 − R4 (dB)")
ax.set_ylim(-0.85, 0.75)
ax.set_title("干预二：时频对齐（同一突发波形，只改放置时刻）\n"
             "对齐−错开 = %+.2f dB，跨种子 p = %.2f —— 方向与机制预测相反"
             % (B["aligned"]["d_mean"] - B["offset"]["d_mean"],
                B.get("paired_diff", {}).get("p", float("nan"))), fontsize=11)
ax.grid(alpha=0.25, ls="--", axis="y")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.suptitle("图 V  机制的干预检验：两个独立操纵都【没有】复现"
             "「重叠越高、脉冲分支收益越小」", fontsize=12.5, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(FIGD, "figV_intervention.png"))
plt.close(fig)
print("图 -> figs_pamnet/figV_intervention.png")
