# -*- coding: utf-8 -*-
"""频域分支尺寸扫描：精度-复杂度曲线"""
import os, csv, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"D:/project/output_v3"
FIG = r"E:/workbuddy/store/2026-10-12-22-50-08/figs_v3"
os.makedirs(FIG, exist_ok=True)
C_BG, C_GRID = "#ffffff", "#e6e6e6"


def style(ax):
    ax.set_facecolor(C_BG); ax.grid(True, color=C_GRID, lw=.8)
    for s in ax.spines.values(): s.set_color("#bbbbbb")


# 旧版（重构前）写出的 summary 没有 params 等列，用已验证过的数值回退
FALLBACK = {"full": (57657, 1, 1, 1.0), "timeonly": (55745, 1, 0, 1.0),
            "freqonly": (1910, 0, 1, 1.0),
            "fch05": (702, 0, 1, 0.5), "fch2": (5910, 0, 1, 2.0),
            "fch4": (20246, 0, 1, 4.0), "fch8": (81000, 0, 1, 8.0)}


def read(tag):
    p = os.path.join(OUT, "summary_%s.csv" % tag)
    if not os.path.exists(p): return None
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if not rows: return None
    fb = FALLBACK.get(tag)
    def g(key, idx, cast):
        v = rows[0].get(key)
        return cast(v) if v not in (None, "") else cast(fb[idx])
    d = {"tag": tag, "params": g("params", 0, int),
         "use_time": g("use_time", 1, int), "use_freq": g("use_freq", 2, int),
         "fch": g("fch", 3, float)}
    for r in rows:
        key = r["testset"].split("_")[0]
        d[key] = float(r["snr_gain"])
    d["mean"] = np.mean([d[k] for k in ("testA", "testB", "testC") if k in d])
    return d


# 频域分支扫描（time=0）；按参数量排序
sweep = []
for tag in ("fch05", "freqonly", "fch2", "fch4", "fch8"):
    d = read(tag)
    if d: sweep.append(d)
sweep.sort(key=lambda x: x["params"])

print("=== 频域分支尺寸扫描（time 分支关闭）===")
print("%-10s %10s %9s %9s %9s %9s" % ("tag", "参数量", "testA", "testB", "testC", "均值"))
for d in sweep:
    print("%-10s %10d %9.2f %9.2f %9.2f %9.2f" % (
        d["tag"], d["params"], d["testA"], d["testB"], d["testC"], d["mean"]))

full = read("full")
print()
if full:
    print("完整模型（频域 fch=1 + 时域分支）: %d 参数  均值 %.2f dB" % (full["params"], full["mean"]))

# 经典基线：理想带通的【增益】(= 输出SNR - 输入SNR)，不是输出SNR
bp = None
j = os.path.join(FIG, "summary_v3.json")
if os.path.exists(j):
    S = json.load(open(j, encoding="utf-8"))
    bp = float(np.mean([S["base"][k][1] - S["base"][k][0] for k in ("testA", "testB", "testC")]))
    print("理想带通增益（三测试集均值）: %.2f dB" % bp)

# ---------------- 出图 ----------------
if sweep:
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=150)

    ax = axs[0]; style(ax)
    xs = [d["params"] for d in sweep]
    # testB 与 testC 数值几乎相同，加对数轴上的偏移 + 不同线型，避免重叠看不清
    for key, c, mk, ls, off in (("testA", "#4c78a8", "o", "-", 1.0),
                                ("testB", "#f58518", "s", "--", 0.90),
                                ("testC", "#54a24b", "^", "-", 1.11)):
        ys = [d[key] for d in sweep]
        ax.semilogx([x*off for x in xs], ys, marker=mk, color=c, lw=1.8, ms=7, ls=ls,
                    mfc="none" if key == "testB" else c,
                    label={"testA": "testA 新波形", "testB": "testB 新信道",
                           "testC": "testC 都新"}[key])
    if full:
        ax.axhline(full["mean"], color="#7f7f7f", ls="--", lw=1.4,
                   label="完整模型均值（含时域分支）")
    if bp:
        ax.axhline(bp, color="#c0392b", ls=":", lw=1.6, label="理想带通 +%.2f dB（经典基线）" % bp)
    for d in sweep:
        ax.annotate("{:,}".format(d["params"]), (d["params"], d["testB"]),
                    textcoords="offset points", xytext=(0, -15), ha="center", fontsize=8, color="#666")
    ax.set_xlabel("参数量（对数轴）"); ax.set_ylabel("SNR 提升 (dB)")
    ax.set_title("频域分支尺寸扫描：精度-复杂度曲线", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower right")

    ax = axs[1]; style(ax)
    ys = [d["mean"] for d in sweep]
    ax.semilogx(xs, ys, marker="o", color="#2ca02c", lw=2.2, ms=8)
    for d in sweep:
        ax.annotate("%.2f" % d["mean"], (d["params"], d["mean"]),
                    textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9,
                    color="#12633a", fontweight="bold")
    if bp:
        ax.axhline(bp, color="#c0392b", ls=":", lw=1.6, label="理想带通 +%.2f dB（经典基线）" % bp)
    if full:
        ax.axhline(full["mean"], color="#7f7f7f", ls="--", lw=1.4,
                   label="完整模型 %.2f dB" % full["mean"])
    ax.set_xlabel("参数量（对数轴）"); ax.set_ylabel("三测试集平均 SNR 提升 (dB)")
    ax.set_title("平均性能：拐点出现在很小参数量处", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figD_sweep.png")); plt.close(fig)
    print("\n图已保存: figs_v3/figD_sweep.png")

    # 找拐点：达到最大均值 99% 所需的最小参数量
    best = max(ys)
    for d, y in zip(sweep, ys):
        if y >= best - 0.15:                     # 允许 0.15 dB 容差
            print("\n达到最大均值 %.2f dB 的 0.15 dB 以内，最小参数量 = %d (%s, %.1f%%)"
                  % (best, d["params"], d["tag"], d["params"]/full["params"]*100 if full else 0))
            break

    with open(os.path.join(FIG, "summary_sweep.json"), "w", encoding="utf-8") as f:
        json.dump({"sweep": sweep, "full": full, "bandpass": bp}, f, ensure_ascii=False, indent=1)
    print("汇总 -> figs_v3/summary_sweep.json")
else:
    print("尚无扫描结果")
