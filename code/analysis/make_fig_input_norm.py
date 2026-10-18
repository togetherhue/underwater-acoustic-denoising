# -*- coding: utf-8 -*-
"""输入尺度归一化 + 架构排序域内验证 的图

图 X：输入尺度敏感性对比（老 R5 vs 新 R5）—— 归一化是否解决部署风险
图 Y：base 条件 GPU 训练的架构排序（R5 / R4 / R3 / 纯时域）
"""
import os, sys, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "sens_out")
FIGD = os.path.join(HERE, "figs_pamnet")
CKPT = r"D:/project/output_pamnet"
os.makedirs(FIGD, exist_ok=True)

for cand in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"):
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [cand]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.dpi"] = 170

C_OLD = "#b91c1c"
C_NEW = "#0f766e"
C_GREY = "#9ca3af"
ARCH_CLR = ["#1f4e79", "#2f6fa8", "#6b9dc9", "#b8cfe6"]


def load_scale_inv():
    p = os.path.join(OUTD, "scale_invariance.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def rd_summary(tag, split):
    p = os.path.join(CKPT, "summary_%s.csv" % tag)
    if not os.path.exists(p):
        return None
    COLS = ["snr_in", "snr_out", "sisdr_in", "sisdr_out", "fmsnr_med_in", "fmsnr_med_out",
            "frmgain_med", "frmgain_clean", "frmgain_imp", "frmgain_imp_min",
            "fmsnr_frac_bad_out", "frac_imp"]
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.reader(f); h = next(r); i0 = h.index(COLS[0])
        for row in r:
            if row[1] == split:
                return {c: float(row[i0 + j]) for j, c in enumerate(COLS)}
    return None


def seeds(prefix, split, key="frmgain_imp"):
    out = []
    for suf in ("", "_s2", "_s3"):
        d = rd_summary(prefix + suf, split)
        if d:
            out.append(d[key])
    return out


# ---------------------------------------------------------------- 图 X
def fig_scale():
    si = load_scale_inv()
    if not si:
        print("缺 scale_invariance.json，跳过图 X"); return
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7))

    # ---- (a) 尺度-性能曲线 ----
    ax = axes[0]
    scales = si["scales"]
    def _gv(g, s):
        for k, v in g.items():
            try:
                if abs(float(k) - s) < 1e-9:
                    return v
            except (ValueError, TypeError):
                pass
        return np.nan

    for name, r in si["res"].items():
        g = r["gains"]
        ys = [_gv(g, s) for s in scales]
        is_old = "老" in name
        col = C_OLD if is_old else C_NEW
        mk = "o" if is_old else "s"
        ax.plot(scales, ys, mk + "-", color=col, ms=7, lw=2.0, label=name)
        for x, y in zip(scales, ys):
            ax.annotate("%.1f" % y, (x, y), textcoords="offset points",
                        xytext=(0, 8 if is_old else -14), ha="center",
                        fontsize=8, color=col)
    ax.axvline(1.0, color=C_GREY, ls=":", lw=1.4)
    ax.annotate("训练尺度", (1.0, ax.get_ylim()[0]), xytext=(1.06, 8),
                fontsize=8.4, color="#6b7280")
    ax.set_xscale("log", base=2)
    ax.set_xticks(scales)
    ax.set_xticklabels(["%.2f" % s for s in scales])
    ax.set_xlabel("输入整体缩放 s（不改变 SNR，只改变尺度）", fontsize=10)
    ax.set_ylabel("干扰帧降噪增益 (dB)", fontsize=10)
    ax.set_title("(a) 尺度敏感性：老 R5 差 %.1f dB，新 R5 差 %.1f dB"
                 % (si["res"][list(si["res"])[0]]["range_db"],
                    si["res"][list(si["res"])[-1]]["range_db"]),
                 fontsize=11.2, pad=10)
    ax.legend(fontsize=8.6, loc="lower left", framealpha=0.96)
    ax.grid(alpha=0.28, ls=":")

    # ---- (b) 各尺度的性能（相对 s=1 的损失）柱状 ----
    ax = axes[1]
    names = list(si["res"].keys())
    w = 0.8 / max(2, len(names))
    x = np.arange(len(scales))
    for i, nm in enumerate(names):
        g = si["res"][nm]["gains"]
        ys = np.array([_gv(g, s) for s in scales])
        rel = ys - _gv(g, 1.0)
        is_old = "老" in nm
        ax.bar(x + (i - (len(names) - 1) / 2) * w, rel, w,
               color=C_OLD if is_old else C_NEW, edgecolor="none",
               label=nm.replace("（", "\n（"))
    ax.axhline(0, color="#111827", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(["s=%.2f" % s for s in scales], fontsize=9)
    ax.set_ylabel("相对 s=1 的性能损失 (dB)", fontsize=10)
    ax.set_title("(b) 部署时若尺度不匹配，损失多大", fontsize=11.2, pad=10)
    ax.legend(fontsize=8.2, loc="lower left", framealpha=0.96)
    ax.grid(axis="y", alpha=0.28, ls=":")
    ax.annotate("s=0.25 ≈ 按 noisy rms 归一化的尺度\n（实测数据只能这样归一化）",
                xy=(0, rel.min() * 0.55), xytext=(0.55, rel.min() * 0.42),
                fontsize=8.4, color="#b91c1c",
                arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.0))

    fig.suptitle("输入尺度归一化：让模型对输入尺度不再敏感（实测数据可直接部署）",
                 fontsize=12.6, y=1.015)
    fig.tight_layout()
    out = os.path.join(FIGD, "figX_scale_invariance.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("已生成", out)


# ---------------------------------------------------------------- 图 Y
GROUPS = [("R5\n时域残差\n+频域精修\n+p注入", "devchk_pf", 14858),
          ("R4\n时域残差\n+频域精修\n(无脉冲)", "devchk_nopulse", 14367),
          ("R3\n纯频域\n掩码", "gpu_fo", 5910),
          ("纯时域\nPAM-Net\n(无脉冲)", "gpu_pn", 8457)]


def fig_arch():
    SPLITS = ["testA"]
    data_all = {}
    for sp in SPLITS:
        d = {}
        for nm, tag, npar in GROUPS:
            v = seeds(tag, sp)
            d[tag] = (nm, npar, v)
        data_all[sp] = d
    d = data_all["testA"]
    ready = [t for t, (_, _, v) in d.items() if v]
    if len(ready) < 2:
        print("GPU 训练未完成，跳过图 Y"); return

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    # ---- (a) 干净/干扰/最差 三组 ----
    ax = axes[0]
    keys = [("frmgain_clean", "干净帧", "#1f4e79"),
            ("frmgain_imp", "干扰帧", "#b91c1c"),
            ("frmgain_imp_min", "干扰最差", "#ca8a04")]
    x = np.arange(len(ready))
    w = 0.26
    for i, (key, lab, col) in enumerate(keys):
        vals = [np.mean(seeds(t, "testA", key)) for t in ready]
        ax.bar(x + (i - 1) * w, vals, w, label=lab, color=col, edgecolor="none")
        for xx, vv in zip(x + (i - 1) * w, vals):
            ax.annotate("%.2f" % vv, (xx, vv), ha="center", va="bottom",
                        fontsize=8, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([d[t][0] for t in ready], fontsize=8.6)
    ax.set_ylabel("增益 (dB)", fontsize=10)
    ax.set_title("(a) base 条件 GPU 训练：三种帧的增益", fontsize=11.2, pad=10)
    ax.legend(fontsize=8.6, framealpha=0.96)
    ax.grid(axis="y", alpha=0.28, ls=":")
    ax.set_ylim(0, 32)

    # ---- (b) 架构增益（相对 R3）+ 参数量 ----
    ax = axes[1]
    if "gpu_fo" in d and d["gpu_fo"][2]:
        ref = np.mean(seeds("gpu_fo", "testA"))
        for i, t in enumerate(ready):
            v = np.mean(seeds(t, "testA"))
            ax.barh(i, v - ref, color=ARCH_CLR[i % len(ARCH_CLR)], edgecolor="none")
            ax.annotate("%+.2f dB" % (v - ref), (v - ref, i), va="center",
                        ha="left" if v >= ref else "right", fontsize=9,
                        xytext=(4 if v >= ref else -4, 0),
                        textcoords="offset points", color="#111827")
        ax.set_yticks(range(len(ready)))
        ax.set_yticklabels(["%s（%d 参数）" % (d[t][0].replace("\n", " "), d[t][1])
                            for t in ready], fontsize=8.4)
        ax.axvline(0, color="#111827", lw=0.9)
        ax.set_xlabel("相对 R3（纯频域掩码）的干扰帧增益差 (dB)", fontsize=10)
        ax.set_title("(b) 架构贡献（域内、同设备）", fontsize=11.2, pad=10)
        ax.grid(axis="x", alpha=0.28, ls=":")
        ax.set_xlim(min(0, min(np.mean(seeds(t, "testA")) - ref
                               for t in ready) - 1),
                    max(np.mean(seeds(t, "testA")) - ref for t in ready) + 1.5)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "等 gpu_fo 训练完成", ha="center", va="center",
                fontsize=11, color="#6b7280")

    fig.suptitle("域内重训下的架构排序：R5 > R4 > R3 > 纯时域 是否保持",
                 fontsize=12.6, y=1.015)
    fig.tight_layout()
    out = os.path.join(FIGD, "figY_inarch_gpu.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("已生成", out)


if __name__ == "__main__":
    fig_scale()
    fig_arch()
