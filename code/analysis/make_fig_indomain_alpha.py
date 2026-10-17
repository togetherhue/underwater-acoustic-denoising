# -*- coding: utf-8 -*-
"""α 轴域内重训图：零样本 vs 域内重训，看 Δ 的差异是分布偏移还是尾形本身"""
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

C_ZERO = "#a8b0ba"
C_IN = "#dc2626"
C_BASE = "#1f4e79"
C_R4 = "#1f4e79"
C_R5 = "#b91c1c"


def load(sp):
    p = os.path.join(OUTD, "indomain_alpha_%s.json" % sp)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    A = load("testA")
    if not A:
        print("缺 indomain_alpha_testA.json"); return

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7))
    keys = ["a12", "a18"]
    lbls = ["α=1.2（等功率）", "α=1.8（等功率）"]
    x = np.arange(len(keys))

    # ---------- 左：Δ 的三态对比 ----------
    ax = axes[0]
    w = 0.3
    d_base = A["d_base"]
    zeroA = [A["conds"][k]["d"] for k in keys]
    inA = [A["conds"][k]["corrected"] for k in keys]
    sdA = [A["conds"][k]["sd"] for k in keys]

    ax.bar(x - w / 2, zeroA, w, label="零样本（只在 α=1.5 上训练）",
           color=C_ZERO, edgecolor="#4b5563", linewidth=0.6)
    ax.bar(x + w / 2, inA, w, label="域内重训（设备校正后）",
           color=C_IN, edgecolor="#7f1d1d", linewidth=0.6,
           yerr=sdA, error_kw=dict(ecolor="#374151", lw=1.1, capsize=4))
    ax.axhline(d_base, color=C_BASE, ls="--", lw=1.7,
               label="base α=1.5 = %+.3f dB" % d_base)
    for i, (z, v) in enumerate(zip(zeroA, inA)):
        ax.annotate("%+.2f" % z, (i - w / 2, z), ha="center", va="bottom",
                    fontsize=9, color="#374151")
        ax.annotate("%+.2f" % v, (i + w / 2, v + sdA[i]), ha="center", va="bottom",
                    fontsize=9.5, color="#7f1d1d", fontweight="bold")
        # 变化箭头
        ax.annotate("", xy=(i + w / 2 - 0.02, v), xytext=(i - w / 2 + 0.02, z),
                    arrowprops=dict(arrowstyle="->", color="#b45309", lw=1.3,
                                    connectionstyle="arc3,rad=-0.18"))
    ax.set_xticks(x); ax.set_xticklabels(lbls, fontsize=9.6)
    ax.set_ylabel("脉冲分支效应 Δ = R5 − R4（干扰帧, dB）", fontsize=10)
    ax.set_title("(a) 域内重训后：α=1.8 的“尾形增强”几乎完全消失", fontsize=11.4, pad=10)
    ax.legend(fontsize=8.6, loc="upper left", framealpha=0.96)
    ax.grid(axis="y", alpha=0.28, ls=":")
    ax.set_ylim(0, 2.25)
    ax.axhline(0, color="#111827", lw=0.8)
    # 右侧加判定标注
    ax.annotate("落在 base 线上\n→ 差异是分布偏移", xy=(1 + w / 2, inA[1]),
                xytext=(1.30, 1.52), fontsize=8.3, color="#0f766e",
                arrowprops=dict(arrowstyle="->", color="#0f766e", lw=1.0),
                ha="left")
    ax.annotate("仍低于 base\n→ 残余尾形效应", xy=(0 + w / 2 + 0.14, inA[0]),
                xytext=(0.02, 1.30), fontsize=8.3, color="#b45309",
                arrowprops=dict(arrowstyle="->", color="#b45309", lw=1.0),
                ha="left")

    # ---------- 右：域内重训带来的绝对提升（解释 Δ 为何缩小） ----------
    ax = axes[1]
    w2 = 0.32
    g4 = [A["conds"][k]["r4"] - A["conds"][k]["zero_r4"] for k in keys]
    g5 = [A["conds"][k]["r5"] - A["conds"][k]["zero_r5"] for k in keys]
    b1 = ax.bar(x - w2 / 2, g4, w2, label="R4（无脉冲分支）", color=C_R4,
                edgecolor="#0f2a44", linewidth=0.6)
    b2 = ax.bar(x + w2 / 2, g5, w2, label="R5（含脉冲分支）", color=C_R5,
                edgecolor="#7f1d1d", linewidth=0.6)
    for i in range(len(keys)):
        ax.annotate("%+.2f" % g4[i], (x[i] - w2 / 2, g4[i]), ha="center",
                    va="bottom", fontsize=9.5, fontweight="bold", color=C_R4)
        ax.annotate("%+.2f" % g5[i], (x[i] + w2 / 2, g5[i]), ha="center",
                    va="bottom", fontsize=9.5, fontweight="bold", color=C_R5)
        # 差值标注
        gap = g4[i] - g5[i]
        if gap > 0.3:
            ax.annotate("R4 多提升 %.2f dB\n→ Δ 缩小 %.2f dB" % (gap, gap),
                        (x[i], max(g4[i], g5[i]) + 0.30), ha="center", va="bottom",
                        fontsize=8.6, color="#374151")
        else:
            ax.annotate("两模型都几乎无提升\n（零样本已接近该条件最优）",
                        (x[i], max(g4[i], g5[i]) + 0.30), ha="center", va="bottom",
                        fontsize=8.2, color="#6b7280")
    ax.set_xticks(x); ax.set_xticklabels(lbls, fontsize=9.6)
    ax.set_ylabel("域内重训相对零样本的提升（dB）", fontsize=10)
    ax.set_title("(b) 为何 Δ 缩小：α=1.8 上 R4 提升远大于 R5", fontsize=11.4, pad=10)
    ax.legend(fontsize=8.8, loc="upper left", framealpha=0.96)
    ax.grid(axis="y", alpha=0.28, ls=":")
    ax.set_ylim(0, 3.35)
    ax.axhline(0, color="#111827", lw=0.8)

    fig.suptitle("α 轴域内重训：跨 α 的收益差异主要来自分布偏移，而非尾形本身",
                 fontsize=12.8, y=1.015)
    fig.tight_layout()
    out = os.path.join(FIGD, "figW_indomain_alpha.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("已生成", out)


if __name__ == "__main__":
    main()
