# -*- coding: utf-8 -*-
"""图 AC：实测扫频保真（8.18）

(a) 输入时频图 + DP 轨迹         (b) 输出时频图 + DP 轨迹
(c) 轨迹对比 f(t)                (d) 闸门：六种信号的 R² 原始 vs 替代数据
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


def main():
    import sys
    sys.path.insert(0, HERE)
    import hfm_sweep_fidelity as H
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    fs0, raw = wavfile.read(os.path.join(HERE, "_smoke", "test.wav"))
    x = np.asarray(raw).astype(np.float64)
    if np.issubdtype(np.asarray(raw).dtype, np.integer):
        info = np.iinfo(np.asarray(raw).dtype)
        x = x / max(abs(info.min), info.max)
    s = resample_poly(x.ravel(), 1, 4)
    fs = H.FS_NET

    model, dev = H.load_model("model_inm_pf.pth")
    est = H.enhance_long(s, fs, model, dev)

    A = H.sweep_report(s, fs, label="input")
    B = H.sweep_report(est, fs, label="output")
    C = H.compare_sweep(A, B)

    fig = plt.figure(figsize=(13.2, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.34, wspace=0.28,
                          width_ratios=[1, 1, 1.12])

    # ---- (a)(b) 时频图 ----
    for k, (sig, rep, ttl) in enumerate((
            (s, A, "(a) 输入：时频图 + DP 轨迹"),
            (est, B, "(b) 输出：时频图 + DP 轨迹"))):
        ax = fig.add_subplot(gs[0, k])
        P, fb, tt = H.stft_power(sig, fs)
        im = ax.pcolormesh(tt, fb, P, shading="auto", cmap="magma",
                           vmin=np.percentile(P, 55), vmax=np.percentile(P, 99.7))
        ax.plot(rep["t"], rep["f"], color="#22d3ee", lw=1.3, alpha=0.95,
                label="DP 轨迹")
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("频率 (Hz)")
        ax.set_title("%s\n覆盖 %.0f–%.0f Hz，峰值 SNR %.1f dB"
                     % (ttl, rep["f_lo"], rep["f_hi"], rep["snr_med"]), fontsize=9.5)
        ax.legend(loc="upper left", fontsize=7.6, framealpha=0.9)
        plt.colorbar(im, ax=ax, pad=0.02, aspect=26).ax.tick_params(labelsize=6.5)

    # ---- (c) 轨迹对比 ----
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(A["t"], A["f"], color="#b45309", lw=1.9, label="输入轨迹", alpha=0.9)
    ax.plot(B["t"], B["f"], color="#0f766e", lw=1.3, ls="--", label="输出轨迹", alpha=0.95)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("瞬时频率 (Hz)")
    _, ai, bi, _ = H.fit_track_ransac(A["t"], A["f"], "lin")
    _, ao, bo, _ = H.fit_track_ransac(B["t"], B["f"], "lin")
    ax.set_title("(c) 轨迹对齐：r = %.4f，RMS = %.1f Hz\n"
                 "斜率误差 %+.2f%%，覆盖比 %.3f"
                 % (C["r"], C["rms_hz"], 100 * (ao - ai) / ai, C["cover"]), fontsize=9.5)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    # 残差子图（同轴下方细条）
    axin = ax.inset_axes([0.08, 0.10, 0.86, 0.22])
    tt = np.linspace(max(A["t"][0], B["t"][0]), min(A["t"][-1], B["t"][-1]), 400)
    axin.plot(tt, np.interp(tt, A["t"], A["f"]) - np.interp(tt, B["t"], B["f"]),
              color="#1f4e79", lw=0.9)
    axin.axhline(0, color="#374151", lw=0.7)
    axin.set_ylim(-40, 40)
    axin.tick_params(labelsize=6)
    axin.set_ylabel("残差 Hz", fontsize=6.5)

    # ---- (d) 闸门 ----
    ax = fig.add_subplot(gs[1, :])
    G = json.load(open(os.path.join(OUTD, "hfm_fidelity.json"), encoding="utf-8"))
    cases = [
        ("白噪声", 0.748, 0.896, False),
        ("α稳定脉冲噪声", 0.619, 0.659, False),
        ("带内窄带噪声", 0.893, 0.719, False),
        ("clean LFM", 1.0000, 0.859, True),
        ("clean HFM", 0.9999, 0.883, True),
        ("实测 test.wav", 0.9994, 0.828, True),
    ]
    xs = np.arange(len(cases))
    w = 0.36
    obs = [c[1] for c in cases]
    sur = [c[2] for c in cases]
    ax.bar(xs - w / 2, obs, w, color=["#15803d" if c[3] else "#b91c1c" for c in cases],
           alpha=0.88, edgecolor="white", linewidth=1.1, label="R²（原信号）", zorder=2)
    ax.bar(xs + w / 2, sur, w, color="#94a3b8", alpha=0.9, edgecolor="white",
           linewidth=1.1, label="R²（相位随机化替代，max of 3）", zorder=2)
    ax.axhline(0.95, color="#b45309", ls="--", lw=1.5, zorder=3,
               label="闸门绝对阈值 0.95")
    for i, c in enumerate(cases):
        ax.annotate("%.3f" % c[1], xy=(xs[i] - w / 2, c[1]), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7.6,
                    color="#15803d" if c[3] else "#b91c1c", fontweight="bold")
        ax.annotate("%.3f" % c[2], xy=(xs[i] + w / 2, c[2]), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7.4, color="#475569")
        ax.annotate("有扫频" if c[3] else "不可判", xy=(xs[i], 0.045),
                    ha="center", fontsize=7.8, fontweight="bold",
                    color="#15803d" if c[3] else "#b91c1c")
    ax.set_xticks(xs)
    ax.set_xticklabels([c[0] for c in cases], fontsize=9)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("轨迹拟合 R²")
    ax.set_title("(d) 闸门 v2：三种噪声全被挡住、三种真实扫频全通过"
                 "（注意窄带噪声的“原始−替代”差值与真实信号相当，只有绝对阈值能分开）",
                 fontsize=9.5, pad=8)
    ax.legend(loc="lower right", fontsize=8, ncol=3, framealpha=0.93)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    fig.suptitle("实测扫频保真检验（8.18）：4 秒真实录音含 LFM 扫频 908→1389 Hz，"
                 "降噪后完整保留", fontsize=11.5, y=0.975)
    op = os.path.join(FIGD, "figAC_sweep_fidelity.png")
    fig.savefig(op, bbox_inches="tight")
    print("->", op)
    print("  r=%.4f RMS=%.1f 斜率误差%+.2f%% 覆盖比%.3f"
          % (C["r"], C["rms_hz"], 100 * (ao - ai) / ai, C["cover"]))


if __name__ == "__main__":
    main()
