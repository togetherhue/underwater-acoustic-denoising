# -*- coding: utf-8 -*-
"""图 AB：rate 轴的干预分解 + 与独立 5 种子零样本评估的交叉验证

(a) rate=2.5/5/10/s 上，位置价值与无信息惩罚反向运动，净 Δ 被压缩
(b) 3 种子干预 vs 5 种子零样本（CPU）：排序一致；r10 的 +0.179 dB 差是纯采样
    （右：r10 逐种子 Δ 散点，seed4 为低离群点）
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "sens_out")
FIGD = os.path.join(HERE, "figs_pamnet")
os.makedirs(FIGD, exist_ok=True)

for cand in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"):
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [cand]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.dpi"] = 170

C_POS = "#0f766e"
C_NEG = "#b91c1c"
C_NET = "#1f4e79"
C_CHK = "#b45309"

RCONDS = ["a15_r25", "base", "a15_r10"]
LBL = {"a15_r25": "2.5/s", "base": "5/s\n(base)", "a15_r10": "10/s"}


def main():
    X = json.load(open(os.path.join(OUTD, "pulse_mask_crosscond.json"), encoding="utf-8"))
    S = json.load(open(os.path.join(OUTD, "sens_summary_testA.json"), encoding="utf-8"))
    C = X["conds"]
    cs = [c for c in RCONDS if c in C]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3),
                             gridspec_kw={"width_ratios": [1, 1.15]})

    # ---------------- (a) rate 轴堆叠分解 ----------------
    ax = axes[0]
    xs = np.arange(len(cs))
    base_ = np.array([C[c]["d_zero"] for c in cs])
    pos = np.array([C[c]["position"] for c in cs])
    net = np.array([C[c]["d_identity"] for c in cs])
    ax.bar(xs, base_, width=0.56, color=C_NEG, alpha=0.8,
           edgecolor="white", linewidth=1.2, label="无信息基线（p≡0，R5−R4）", zorder=2)
    ax.bar(xs, pos, width=0.56, bottom=base_, color=C_POS, alpha=0.8,
           edgecolor="white", linewidth=1.2, label="位置信息的价值", zorder=2)
    ax.plot(xs, net, "D", ms=8, color=C_NET, mec="white", mew=1.4, zorder=4,
            label="净 Δ（堆叠顶端）")
    for i, v in enumerate(net):
        ax.annotate("%+.2f" % v, xy=(xs[i], v), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8.6, color=C_NET, fontweight="bold")
    for i, v in enumerate(base_):
        ax.annotate("%+.1f" % v, xy=(xs[i], v), xytext=(0, -13), textcoords="offset points",
                    ha="center", fontsize=7.8, color=C_NEG)
    for i, v in enumerate(pos):
        ax.annotate("+%.1f" % v, xy=(xs[i], base_[i] + v / 2), ha="center", va="center",
                    fontsize=7.8, color="white", fontweight="bold")
    ax.axhline(0, color="#374151", lw=1.1, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([LBL.get(c, c) for c in cs], fontsize=9)
    ax.set_ylabel("dB（相对 R4）")
    ax.set_title("(a) rate 轴：两项反向运动，净 Δ 只变 0.23 dB", fontsize=10.5, pad=9)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(loc="lower left", fontsize=7.6, framealpha=0.93, handlelength=1.8)

    # ---------------- (b) 交叉验证 ----------------
    ax = axes[1]
    w = 0.36
    ours = [C[c]["d_identity"] for c in cs]
    zero = []
    for c in cs:
        row = S["table"][c]
        zero.append(row["R5_freqrefine_pulse"]["imp"] - row["R4_freqrefine_nopulse"]["imp"])
    ax.bar(xs - w / 2, ours, w, color=C_NET, alpha=0.85, edgecolor="white",
           linewidth=1.1, label="干预实验（3 种子）", zorder=2)
    ax.bar(xs + w / 2, zero, w, color=C_CHK, alpha=0.85, edgecolor="white",
           linewidth=1.1, label="零样本评估（5 种子，CPU）", zorder=2)
    for i, v in enumerate(ours):
        ax.annotate("%+.2f" % v, xy=(xs[i] - w / 2, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7.8, color=C_NET)
    for i, v in enumerate(zero):
        ax.annotate("%+.2f" % v, xy=(xs[i] + w / 2, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7.8, color=C_CHK)
    for i, c in enumerate(cs):
        if c == "a15_r10":
            row = S["table"][c]
            d = np.array(row["R5_freqrefine_pulse"]["per_seed_imp"]) - \
                np.array(row["R4_freqrefine_nopulse"]["per_seed_imp"])
            ax.scatter(np.full(len(d), xs[i] + w / 2) + (np.arange(len(d)) - 2) * 0.055,
                       d, s=26, color="#7c2d12", zorder=5, marker="_", lw=1.6)
            ax.annotate("5 个种子逐一画出\n（+0.08 ~ +1.08）", xy=(xs[i] + w / 2, d.min()),
                        xytext=(6, -34), textcoords="offset points", fontsize=7.4,
                        color="#7c2d12")
    ax.set_xticks(xs)
    ax.set_xticklabels([LBL.get(c, c) for c in cs], fontsize=9)
    ax.set_ylabel("净 Δ = R5 − R4（dB）")
    ax.set_title("(b) 排序在两套独立数据中一致；只读方向", fontsize=10.5, pad=9)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(loc="upper right", fontsize=7.6, framealpha=0.93)
    ax.set_ylim(0, max(ours + zero) * 1.42)

    fig.suptitle("rate 轴分解与交叉验证：突发越密，分支净收益越小（方向在两套数据中一致）",
                 fontsize=11.5, y=1.02)
    fig.tight_layout()
    op = os.path.join(FIGD, "figAB_rateaxis.png")
    fig.savefig(op, bbox_inches="tight")
    print("->", op)
    print("  3种子:", ["%+.3f" % v for v in ours])
    print("  5种子:", ["%+.3f" % v for v in zero])
    print("  差  :", ["%+.3f" % (a - b) for a, b in zip(ours, zero)])


if __name__ == "__main__":
    main()
