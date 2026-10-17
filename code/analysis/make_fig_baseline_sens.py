# -*- coding: utf-8 -*-
"""
出图：经典基线 vs 神经模型（跨条件）
figU_baseline_sens.png
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = r"E:/workbuddy/store/2026-10-12-22-50-08"
OUTD = os.path.join(WORK, "sens_out")
FIGD = os.path.join(WORK, "figs_pamnet")
os.makedirs(FIGD, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10.5
plt.rcParams["axes.edgecolor"] = "#8a8f98"
plt.rcParams["axes.linewidth"] = 0.9

ORDER = ["base", "a12_r5", "a18_r5", "a20_r5", "a15_r25", "a15_r10", "gauss",
         "m_a12", "m_a18"]
SHORT = {"base": "α=1.5\nr=5\n(base)", "a12_r5": "α=1.2\nr=5", "a18_r5": "α=1.8\nr=5",
         "a20_r5": "α=2.0\nr=5", "a15_r25": "α=1.5\nr=2.5", "a15_r10": "α=1.5\nr=10",
         "gauss": "高斯\nr=5", "m_a12": "α=1.2\n匹配", "m_a18": "α=1.8\n匹配"}

BL = json.load(open(os.path.join(OUTD, "baseline_sens_testA.json"), encoding="utf-8"))
VN = json.load(open(os.path.join(OUTD, "baseline_vs_neural_testA.json"), encoding="utf-8"))
ROWS = {r["cond"]: r for r in VN["rows"]}
conds = [c for c in ORDER if c in BL and c in ROWS]
x = np.arange(len(conds))

fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.6), dpi=150)

# ---------------- 左：干扰帧 ----------------
ax = axes[0]
series = [
    ("理想带通 798–1500 Hz（已知频带·作弊上界）", [BL[c]["理想带通 798-1500Hz"]["frmgain_imp"] for c in conds], "#b279a2", "D"),
    ("谱减法（盲）", [BL[c]["谱减法（盲）"]["frmgain_imp"] for c in conds], "#4c78a8", "o"),
    ("维纳滤波（盲）", [BL[c]["维纳滤波（盲）"]["frmgain_imp"] for c in conds], "#72b7b2", "s"),
    ("中值滤波 k=5（盲）", [BL[c]["中值滤波 k=5"]["frmgain_imp"] for c in conds], "#9d755d", "v"),
    ("R5 神经模型（14,858 参数）", [ROWS[c]["r5"] for c in conds], "#e45756", "^"),
]
for name, ys, col, mk in series:
    lw = 2.6 if "R5" in name else 1.8
    ms = 8 if "R5" in name else 6
    ax.plot(x, ys, marker=mk, color=col, lw=lw, ms=ms, label=name,
            zorder=5 if "R5" in name else 3)
ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in conds], fontsize=8.5)
ax.set_ylabel("干扰帧逐帧降噪增益 (dB)")
ax.set_title("干扰帧：经典基线 vs 神经模型", fontsize=11.5)
ax.grid(alpha=0.25, ls="--")
ax.legend(fontsize=8.8, frameon=False, loc="upper left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
# 标注 R5 与最好盲方法的差距
for i, c in enumerate(conds):
    ax.annotate("%+.0f" % ROWS[c]["gap"], (x[i], ROWS[c]["r5"]),
                textcoords="offset points", xytext=(0, 9), ha="center",
                fontsize=8, color="#9c2b1f")

# ---------------- 右：R5 的优势分解 ----------------
ax = axes[1]
g_blind = [ROWS[c]["gap"] for c in conds]
g_bp = [ROWS[c]["r5"] - ROWS[c]["bp"] for c in conds]
w = 0.38
ax.bar(x - w / 2, g_blind, w, color="#e45756", edgecolor="white", lw=0.6,
       label="R5 − 最好的盲方法")
ax.bar(x + w / 2, g_bp, w, color="#b279a2", edgecolor="white", lw=0.6,
       label="R5 − 理想带通（作弊上界）")
for i, (a, b) in enumerate(zip(g_blind, g_bp)):
    ax.text(x[i] - w / 2, a + 0.4, "%.1f" % a, ha="center", fontsize=7.6, color="#9c2b1f")
    ax.text(x[i] + w / 2, b + 0.4, "%.1f" % b, ha="center", fontsize=7.6, color="#6b4a75")
ax.axhline(0, color="#333", lw=1.0)
ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in conds], fontsize=8.5)
ax.set_ylabel("优势 (dB)")
ax.set_title("R5 的优势幅度（全部条件上为正）", fontsize=11.5)
ax.legend(fontsize=9, frameon=False, loc="lower right")
ax.grid(alpha=0.25, ls="--", axis="y")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.suptitle("图 U  头条结论的跨条件检验：R5 在全部 9 个条件上都远超经典方法"
             "（testA，干扰帧；左图标注为 R5 与最好盲方法的差距）", fontsize=12, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(FIGD, "figU_baseline_sens.png"))
plt.close(fig)
print("图 -> figs_pamnet/figU_baseline_sens.png")
