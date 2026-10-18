# -*- coding: utf-8 -*-
"""图 AA：跨条件的脉冲掩码干预分解

(a) Δ_id 的跨条件变化，分解成「无信息基线」与「位置信息价值」两个分量（堆叠）
(b) p 的 AUC 几乎跨条件不变（0.95+），但位置价值在变
    ⇒ 变化不是"p 变差了"，而是"位置信息本身值多少钱"变了
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

C_POS = "#0f766e"     # 位置信息价值
C_NEG = "#b91c1c"     # 无信息基线（负）
C_NET = "#1f4e79"     # 净 Δ
C_AUC = "#7c3aed"

ORDER = ["a12_r5", "base", "a18_r5", "a20_r5", "a15_r25", "a15_r10"]
LBL = {"a12_r5": "α=1.2", "base": "base\n(α=1.5,5/s)", "a18_r5": "α=1.8",
       "a20_r5": "α=2.0", "a15_r25": "2.5/s", "a15_r10": "10/s"}


def main():
    p = os.path.join(OUTD, "pulse_mask_crosscond.json")
    if not os.path.exists(p):
        print("缺 pulse_mask_crosscond.json"); return
    R = json.load(open(p, encoding="utf-8"))
    cs = [c for c in ORDER if c in R["conds"]]
    if not cs:
        print("无可用条件"); return

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    # ---------------- (a) 堆叠分解 ----------------
    ax = axes[0]
    xs = np.arange(len(cs))
    base_ = np.array([R["conds"][c]["d_zero"] for c in cs])       # 无信息基线（通常为负）
    pos = np.array([R["conds"][c]["position"] for c in cs])        # 位置价值
    net = np.array([R["conds"][c]["d_identity"] for c in cs])      # 净

    ax.bar(xs, base_, width=0.58, color=C_NEG, alpha=0.8,
           edgecolor="white", linewidth=1.2, label="无信息基线（p≡0，R5−R4）", zorder=2)
    ax.bar(xs, pos, width=0.58, bottom=base_, color=C_POS, alpha=0.8,
           edgecolor="white", linewidth=1.2, label="位置信息的价值（identity−p≡0）", zorder=2)
    # 净 Δ 用菱形标在堆叠顶端
    ax.plot(xs, net, "D", ms=8, color=C_NET, mec="white", mew=1.4,
            zorder=4, label="净 Δ（堆叠顶端）")
    for i, v in enumerate(net):
        ax.annotate("%+.2f" % v, xy=(xs[i], v), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8.2,
                    color=C_NET, fontweight="bold")
    for i, v in enumerate(base_):
        ax.annotate("%+.1f" % v, xy=(xs[i], v), xytext=(0, -13),
                    textcoords="offset points", ha="center", fontsize=7.6, color=C_NEG)
    ax.axhline(0, color="#374151", lw=1.1, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([LBL.get(c, c) for c in cs], fontsize=8.4)
    ax.set_ylabel("dB（相对 R4）")
    ax.set_title("(a) 净 Δ 拆成两项：信息价值远大于净收益", fontsize=10.5, pad=9)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(loc="upper left", fontsize=7.8, framealpha=0.93, handlelength=1.8)

    # ---------------- (b) AUC vs 位置价值 ----------------
    ax = axes[1]
    aucs = np.array([R.get("p_diag", {}).get(c, {}).get("auc_mean", np.nan) for c in cs])
    ok = np.isfinite(aucs)
    if ok.sum() >= 2:
        ax.scatter(aucs[ok], pos[ok], s=64, c=[C_POS if c == "base" else C_NET
                                               for c, o in zip(cs, ok) if o],
                   alpha=0.85, edgecolor="white", linewidth=1.3, zorder=3)
        for c, a, v in zip(cs, aucs, pos):
            if np.isfinite(a):
                ax.annotate(LBL.get(c, c).replace("\n", " "), xy=(a, v),
                            xytext=(5, 5), textcoords="offset points",
                            fontsize=8.0, color="#374151")
        # 相关
        if ok.sum() >= 3:
            r_ = float(np.corrcoef(aucs[ok], pos[ok])[0, 1])
            ax.text(0.03, 0.96, "AUC 极差仅 %.3f  |  位置价值极差 %.2f dB  r = %+.2f"
                    % (aucs[ok].max() - aucs[ok].min(), pos[ok].max() - pos[ok].min(), r_),
                    transform=ax.transAxes, fontsize=8.2, va="top", color="#374151")
        ax.set_xlabel("p 与真实干扰掩码的 AUC（p 的质量）")
        ax.set_ylabel("位置信息的价值 (dB)")
        ax.set_title("(b) p 的质量几乎不变，变的是它值多少钱",
                     fontsize=10.5, pad=9)
        ax.grid(alpha=0.25, zorder=0)
    else:
        ax.text(0.5, 0.5, "缺 p_diag", ha="center", va="center",
                transform=ax.transAxes)

    fig.suptitle("跨条件干预分解：分支的净收益 = 位置信息价值 − 无信息时的惩罚",
                 fontsize=11.5, y=1.005)
    fig.tight_layout()
    op = os.path.join(FIGD, "figAA_crosscond.png")
    fig.savefig(op, bbox_inches="tight")
    print("->", op)


if __name__ == "__main__":
    main()
