# -*- coding: utf-8 -*-
"""图 AD：SNR 轴扫描（8.19）

(a) base（α=1.5）：Δ = R5−R4 随背景 SNR 的均值（阴影 ±1 sd）与跨种子中位数，
    空心菱形为 8.1b 复现点（实测 −3.29 dB，不参与网格）
(b) base：绝对噪声抑制量 gain_imp 随背景 SNR：R4 / R5 / 谱减 / 维纳
(c) 四个 α：收益效率 Δ/gain_imp 随背景 SNR
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

C_D = "#1f4e79"
C_MD = "#0f766e"
C_R4 = "#6b7280"
C_R5 = "#b91c1c"
C_SS = "#b45309"
C_WI = "#7c3aed"
ALPHA_C = {1.2: "#b91c1c", 1.5: "#1f4e79", 1.8: "#0f766e", 2.0: "#b45309"}
ALPHA_COND = {1.2: "a12_r5", 1.5: "base", 1.8: "a18_r5", 2.0: "a20_r5"}
REPL = -3.29


def tkey(t):
    t = float(t)
    return ("%+d" % int(round(t))) if abs(t - round(t)) < 1e-9 else ("%+.2f" % t)


def main():
    S = json.load(open(os.path.join(OUTD, "snr_sweep.json"), encoding="utf-8"))
    D, C = S["delta"], S["cells"]

    def dget(cond, t, key):
        k = "%s|%s" % (cond, tkey(t))
        return D[k][key] if k in D else np.nan

    def cget(cond, t, model, key):
        k = "%s|%s|%s" % (cond, tkey(t), model)
        return C[k]["mean"][key] if k in C else np.nan

    def series(cond, model, key, drop_repl=True):
        xs, ys = [], []
        for v in C.values():
            if v["cond"] != cond or v["model"] != model:
                continue
            if drop_repl and abs(v["target_bg_snr"] - REPL) < 1e-6:
                continue
            xs.append(v["target_bg_snr"])
            ys.append(v["mean"][key])
        o = np.argsort(xs)
        return np.array(xs)[o], np.array(ys)[o]

    fig = plt.figure(figsize=(13.4, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.06, 1.0, 1.0], wspace=0.28)

    # ---------------- (a) Δ 随 SNR ----------------
    ax = fig.add_subplot(gs[0, 0])
    xs = np.array(sorted({v["target_bg_snr"] for v in D.values()
                          if v["cond"] == "base" and abs(v["target_bg_snr"] - REPL) > 1e-6}))
    d_imp = np.array([dget("base", t, "d_mean") for t in xs])
    d_med = np.array([dget("base", t, "d_med") for t in xs])
    d_sd = np.array([dget("base", t, "d_sd") for t in xs])
    ax.fill_between(xs, d_imp - d_sd, d_imp + d_sd, color=C_D, alpha=0.14, lw=0,
                    label="±1 sd（5 种子）")
    ax.plot(xs, d_imp, "-o", color=C_D, ms=4.4, lw=1.9, label="Δ 跨种子均值", zorder=4)
    ax.plot(xs, d_med, "--^", color=C_MD, ms=4.2, lw=1.4, label="Δ 跨种子中位数", zorder=4)
    ax.plot([REPL], [dget("base", REPL, "d_mean")], "D", mfc="none", mec="#111827",
            ms=7.0, mew=1.4, zorder=6, label="8.1b 复现点")
    ax.axhline(0, color="#374151", lw=1.1)
    ax.annotate("训练条件实测\n−3.29 dB", xy=(REPL, dget("base", REPL, "d_mean")),
                xytext=(12, -30), textcoords="offset points", fontsize=7.0, color="#111827",
                arrowprops=dict(arrowstyle="-", lw=0.7, color="#9ca3af"))
    ax.annotate("−15 dB 处种子离散最大\n（0.06~2.25 dB）", xy=(-15, d_imp[0]),
                xytext=(-14, 22), textcoords="offset points", fontsize=6.9, color="#6b7280")
    ax.set_xlabel("背景 SNR = Pc/Pv（dB，实测）")
    ax.set_ylabel("Δ = R5 − R4（dB）")
    ax.set_title("(a) 全程 Δ>0、12/12 点 5 种子同号，但幅度向高 SNR 压缩", fontsize=10.0, pad=8)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=7.0, framealpha=0.93)

    # ---------------- (b) 绝对增益 ----------------
    ax = fig.add_subplot(gs[0, 1])
    for m, c, lb, st, mk in (("R4", C_R4, "R4 无脉冲分支", "-", "o"),
                             ("R5", C_R5, "R5 有脉冲分支", "-", "o"),
                             ("specsub", C_SS, "谱减", "--", None),
                             ("wiener", C_WI, "维纳", "--", None)):
        x2, y2 = series("base", m, "gain_imp")
        ax.plot(x2, y2, st, color=c, marker=mk, ms=3.5, lw=1.8, label=lb)
    ax.set_xlabel("背景 SNR = Pc/Pv（dB，实测）")
    ax.set_ylabel("干扰帧噪声抑制量（dB）")
    ax.set_title("(b) 绝对抑制量几乎不随 SNR 变：不是「低 SNR 更好压」", fontsize=10.0, pad=8)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=7.0, framealpha=0.93)

    # ---------------- (c) 超额比 ----------------
    ax = fig.add_subplot(gs[0, 2])
    ts = np.array(sorted({v["target_bg_snr"] for v in D.values()
                          if v["cond"] != "base" and abs(v["target_bg_snr"] - REPL) > 1e-6}))
    for al, c in ALPHA_C.items():
        cond = ALPHA_COND[al]
        ys = [dget(cond, t, "d_mean") / cget(cond, t, "R4", "gain_imp") for t in ts]
        ax.plot(ts, ys, "-o", color=c, ms=4.2, lw=1.8, label="α=%.1f" % al)
    ax.set_xlabel("背景 SNR = Pc/Pv（dB，实测，等功率对齐）")
    ax.set_ylabel("收益效率 $e$ = Δ / gain$_{imp}$（dB/dB）")
    ax.set_title("(c) 收益效率随 SNR 下降，α=1.2 例外（反向上升）", fontsize=10.0, pad=8)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=7.0, framealpha=0.93, ncol=2)

    fig.suptitle("SNR 轴扫描：脉冲分支在 −15~+15 dB 全程为正收益，幅度随背景噪声变弱而缩小约四成"
                 "（testA，5 种子）", fontsize=11.0, y=1.02)
    op = os.path.join(FIGD, "figAD_snr_sweep.png")
    fig.savefig(op, bbox_inches="tight")
    print("->", op)


if __name__ == "__main__":
    main()
