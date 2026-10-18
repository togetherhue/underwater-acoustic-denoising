# -*- coding: utf-8 -*-
"""脉冲掩码干预 + 错位剂量响应 的图（图 Z）

(a) 7 种干预下的 Δ = R5−R4：bar + 逐种子散点
(b) 错位剂量响应：Δ 随 p 平移量 k 的衰减，横轴换算成「突发时长」倍数
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

C_KEEP = "#0f766e"      # 保留信息
C_LOSE = "#b91c1c"      # 丢失/错位信息
C_NEUT = "#9ca3af"      # 中性（随机/常数）
C_BASE = "#1f4e79"

MODE_CN = {
    "identity": "原样\n(基线)",
    "shuffle": "换他人 p",
    "shift": "循环平移",
    "flip": "时间反转",
    "zero": "p≡0",
    "one": "p≡1",
    "noise": "随机噪声",
}
MODE_CLR = {"identity": C_KEEP, "shuffle": C_LOSE, "shift": C_LOSE,
            "flip": C_LOSE, "zero": C_LOSE, "one": C_NEUT, "noise": C_NEUT}


def load(n):
    p = os.path.join(OUTD, n)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    I = load("pulse_mask_intervention.json")
    D = load("pulse_mask_dose.json")
    O = load("pulse_mask_oracle.json")
    if not I:
        print("缺 pulse_mask_intervention.json"); return

    modes = I["modes"]
    base_d = I["summary"]["identity"]["mean"]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))

    # ---------------- (a) 7 种干预 ----------------
    ax = axes[0]
    xs = np.arange(len(modes))
    means = [I["summary"][m]["mean"] for m in modes]
    sds = [I["summary"][m]["sd"] for m in modes]
    ax.bar(xs, means, yerr=sds, capsize=4, width=0.62,
           color=[MODE_CLR[m] for m in modes], alpha=0.85,
           edgecolor="white", linewidth=1.2, zorder=2)
    # 逐种子散点
    rng = np.random.default_rng(0)
    for i, m in enumerate(modes):
        v = np.array(list(I["delta"][m].values()), float)
        ax.scatter(np.full(len(v), xs[i]) + rng.uniform(-0.13, 0.13, len(v)),
                   v, s=17, color="#1f2937", alpha=0.75, zorder=3,
                   edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="#374151", lw=1.1, zorder=1,
               label="R4 水平（Δ = 0）")
    ax.axhline(base_d, color=C_BASE, lw=1.0, ls="--", zorder=1,
               label="基线 Δ = %+.2f（正确 p）" % base_d)
    # oracle p 上界（把 p 换成真实干扰掩码）
    if O and O.get("summary"):
        dor = O["summary"]["d_oracle_mean"]
        ax.axhline(dor, color="#7c3aed", lw=1.2, ls="-.", zorder=1,
                   label="oracle p 上界 %+.2f（余量 %+.3f）"
                         % (dor, O["summary"]["headroom_mean"]))
    # 分组说明
    ax.text(2.0, max(means) + 2.55,
            "红：p 的边缘分布被保留，但不再指向真实干扰位置\n"
            "灰：常数或随机，本就不指向任何位置",
            ha="center", va="top", fontsize=8.0, color="#4b5563", linespacing=1.5)
    ax.legend(loc="lower left", fontsize=7.6, framealpha=0.92,
              borderpad=0.5, handlelength=2.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([MODE_CN[m] for m in modes], fontsize=8.6)
    ax.set_ylabel("Δ = R5(干预) − R4  (dB)")
    ax.set_title("(a) 推理期篡改 p：位置错 ≡ 完全剥夺信息",
                 fontsize=10.5, pad=9)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_ylim(min(means) - 1.5, max(means) + 3.6)

    # ---------------- (b) 剂量响应 ----------------
    ax = axes[1]
    if D:
        ks = D["ks"]
        bl = D["burst"]["burst_len_mean"]
        x = np.array(ks, float) / bl          # 单位：突发时长倍数
        y = np.array([D["summary"][str(k)]["d_mean"] for k in ks], float)
        e = np.array([D["summary"][str(k)]["d_sd"] for k in ks], float)
        ax.fill_between(x, y - e, y + e, color=C_BASE, alpha=0.16, zorder=1)
        ax.plot(x, y, "-o", color=C_BASE, lw=2.0, ms=6, zorder=3,
                markerfacecolor="white", markeredgewidth=1.8)
        ax.axhline(0, color="#374151", lw=1.1, zorder=1)
        ax.axhline(base_d, color=C_BASE, lw=1.0, ls="--", alpha=0.6, zorder=1)

        kh = D.get("k_half", float("nan"))
        if np.isfinite(kh):
            ax.axvline(kh / bl, color=C_LOSE, lw=1.3, ls=":", zorder=2)
            ax.text(kh / bl + 0.9, -3.62,
                    "半衰 k = %.0f 点 = %.0f ms\n= %.1f 个突发时长"
                    % (kh, 1000 * kh / D["fs"], kh / bl),
                    fontsize=8.2, color=C_LOSE, va="bottom", ha="left")
        # 标注两个关键点（错开位置，避免与曲线/彼此重叠）
        ax.annotate("1.5 个突发：只剩 30%% 增益", xy=(x[1], y[1]),
                    xytext=(7.5, 1.28), fontsize=8.0, color="#374151",
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color="#9ca3af", lw=0.9,
                                    connectionstyle="arc3,rad=0.18"))
        ax.annotate("3 个突发：已转负", xy=(x[2], y[2]),
                    xytext=(15.0, -0.62), fontsize=8.0, color="#374151",
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color="#9ca3af", lw=0.9,
                                    connectionstyle="arc3,rad=-0.18"))
        ax.set_xlim(-2.5, 54)
        ax.set_ylim(-4.1, 1.75)
        ax.set_xlabel("p 的错位量  (突发时长倍数，1 个突发 ≈ %.0f ms)"
                      % (1000 * bl / D["fs"]))
        ax.set_ylabel("Δ = R5(错位 p) − R4  (dB)")
        ax.set_title("(b) 错位剂量响应：增益在 1~3 个突发尺度内崩掉",
                     fontsize=10.5, pad=9)
        ax.grid(alpha=0.25, zorder=0)
    else:
        ax.text(0.5, 0.5, "缺 pulse_mask_dose.json", ha="center", va="center",
                transform=ax.transAxes)

    fig.suptitle("脉冲掩码干预：分支的收益来自「对齐的位置信息」，不是容量也不是 p 的统计量",
                 fontsize=11.5, y=1.005)
    fig.tight_layout()
    op = os.path.join(FIGD, "figZ_pulse_mask.png")
    fig.savefig(op, bbox_inches="tight")
    print("->", op)


if __name__ == "__main__":
    main()
