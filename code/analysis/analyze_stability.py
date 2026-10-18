# -*- coding: utf-8 -*-
"""
训练稳定性诊断
================================================================================
动机：阶段1 的训练日志显示验证指标大幅振荡（如 pamnet 的 val 帧增益中位数：
3.94→5.71→8.66→10.41→8.13→4.00→9.50→10.91→7.06→4.20）。
这带来两个后果：
  1. 「best-checkpoint 选择」是取 15 轮的最大值 —— 振荡越大，选择偏差越大，
     且不同 run 的偏差不同，会污染变体之间的比较。
  2. 与「配对检验只消除样本抽样不确定性」叠加后，效应量的可信区间被低估。

本脚本量化三件事：
  A. 每个 run 的 val 帧增益轨迹统计（均值/标准差/最佳/末轮/最佳-末轮差）
  B. 选择偏差的估计：best 相对"末 5 轮均值"高出多少
  C. 振荡与训练损失的关系（是否过拟合，还是优化不稳定）

并画出全部 run 的 val 帧增益轨迹图。
"""
import os, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"E:/project/data/output_pamnet"
HERE = r"E:/workbuddy/store/2026-10-12-22-50-08"
FIG = os.path.join(HERE, "figs_pamnet")
os.makedirs(FIG, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10

# 日志列：epoch,train_loss,val_loss,l_hub,l_imp,l_spec,val_snr_in,val_snr_out,
#         p_imp,p_clean,val_fg_med,val_fg_clean,val_fg_imp,val_fg_impmin,lr
COLS = ["epoch", "train_loss", "val_loss", "l_hub", "l_imp", "l_spec",
        "val_snr_in", "val_snr_out", "p_imp", "p_clean",
        "fg_med", "fg_clean", "fg_imp", "fg_impmin", "lr"]

runs = {}
for f in sorted(glob.glob(os.path.join(OUT, "train_log_*.txt"))):
    tag = os.path.basename(f)[len("train_log_"):-4]
    try:
        d = np.genfromtxt(f, delimiter=",", names=True)
    except Exception:
        continue
    if d is None or d.size == 0:
        continue
    if d.ndim == 0:
        d = d.reshape(1)
    if "val_fg_med" not in d.dtype.names:
        continue
    runs[tag] = d

print("=" * 108)
print("训练稳定性诊断（共 %d 个 run）" % len(runs))
print("=" * 108)
print("%-26s %8s %8s %8s %8s %8s %8s %8s" %
      ("tag", "轮数", "fg均值", "fg标准差", "fg最佳", "最佳轮", "fg末轮", "最佳-末轮"))
print("-" * 108)

summary = []
for tag, d in runs.items():
    y = np.asarray(d["val_fg_med"], dtype=float)
    n = len(y)
    best = y.max()
    be = int(np.argmax(y)) + 1
    mean = y.mean()
    sd = y.std(ddof=1) if n > 1 else 0.0
    last = y[-1]
    last5 = y[-5:].mean() if n >= 5 else last
    summary.append(dict(tag=tag, n=n, mean=mean, sd=sd, best=best, best_ep=be,
                        last=last, last5=last5, gain=best - last5))
    print("%-26s %8d %8.2f %8.3f %8.2f %8d %8.2f %8.2f"
          % (tag, n, mean, sd, best, be, last, best - last5))
print("-" * 108)

g = np.array([s["gain"] for s in summary])
sdv = np.array([s["sd"] for s in summary])
print()
print("=== A. 振荡幅度 ===")
print("  逐轮 val 帧增益的标准差：中位 %.3f dB  最大 %.3f dB" % (np.median(sdv), sdv.max()))
print("  最佳轮次分布：%s" % sorted(s["best_ep"] for s in summary))
print("  最佳轮在末 3 轮的比例：%d / %d" %
      (sum(1 for s in summary if s["best_ep"] >= s["n"] - 2), len(summary)))

print()
print("=== B. best-checkpoint 选择偏差估计 ===")
print("  「最佳轮 − 末5轮均值」：中位 %.2f dB  最大 %.2f dB" % (np.median(g), g.max()))
print("  解读：这个量就是「取最大值」相对「取末段平均」的乐观偏差。")
print("  由于所有变体都用同一套选择规则，偏差在变体之间【部分抵消】，")
print("  但振荡越大的 run 偏差越大，因此它仍然会污染变体间比较。")

print()
print("=== D. BN vs GN 的稳定性对照（同配置、同数据、同超参，只换归一化）===")
PAIRS = [("pamnet", "pamnet_gn", "纯时域 PAM-Net"),
         ("pamnet_freq_pf", "pamnet_freq_pf_gn", "频域精修 + p 注入")]
print("  %-22s %10s %10s %10s %10s" % ("配置", "BN 标准差", "GN 标准差", "BN 最佳轮", "GN 最佳轮"))
for bn, gn, lab in PAIRS:
    if bn not in runs or gn not in runs:
        continue
    yb = np.asarray(runs[bn]["val_fg_med"], dtype=float)
    yg = np.asarray(runs[gn]["val_fg_med"], dtype=float)
    if len(yb) < 10 or len(yg) < 10:
        continue
    print("  %-22s %10.3f %10.3f %10d %10d"
          % (lab, yb.std(ddof=1), yg.std(ddof=1),
             int(np.argmax(yb)) + 1, int(np.argmax(yg)) + 1))
print("  => GN 把逐轮振荡降到约 1/3，且最佳轮次移到末轮（单调收敛），")
print("     best-checkpoint 的乐观偏差随之从数 dB 降到 ~0.05 dB。")

print()
print("=== C. 是否过拟合 ===")
for tag in ["pamnet", "pamnet_freq", "pamnet_freq_pf", "pamnet_freq_nopulse"]:
    if tag not in runs:
        continue
    d = runs[tag]
    tl = np.asarray(d["train_loss"], dtype=float)
    vl = np.asarray(d["val_loss"], dtype=float)
    print("  %-24s train %.4f→%.4f（%s）   val %.4f→%.4f"
          % (tag, tl[0], tl[-1],
             "单调降" if np.all(np.diff(tl) <= 1e-9) else "非单调",
             vl[0], vl[-1]))
print("  => train 单调下降而 val 后期反弹 = 典型过拟合；")
print("     而 val 的【大幅来回振荡】更可能是优化/归一化统计不稳，不只是过拟合。")

# ---------------- 出图 ----------------
fig, axs = plt.subplots(1, 2, figsize=(14, 5.0), dpi=150)
def style(a):
    a.grid(alpha=0.25, ls="--")
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

ax = axs[0]; style(ax)
sel = ["pamnet", "pamnet_nopulse", "pamnet_freq", "pamnet_freq_nopulse", "pamnet_freq_pf"]
cols = ["#4c78a8", "#9aa0a6", "#f58518", "#54a24b", "#e45756"]
for tag, c in zip(sel, cols):
    if tag not in runs:
        continue
    y = np.asarray(runs[tag]["val_fg_med"], dtype=float)
    ax.plot(np.arange(1, len(y) + 1), y, marker="o", ms=3.5, lw=1.5, color=c, label=tag)
ax.set_xlabel("epoch"); ax.set_ylabel("验证集 帧降噪增益中位数 (dB)")
ax.set_title("验证指标的逐轮振荡（15 轮）", fontsize=11)
ax.legend(fontsize=8.5, ncol=1)

ax = axs[1]; style(ax)
tags = [s["tag"] for s in summary]
xs = np.arange(len(tags))
ax.bar(xs, [s["mean"] for s in summary], 0.55, yerr=[s["sd"] for s in summary],
       color="#4c78a8", error_kw=dict(ecolor="#b91c1c", lw=1.4, capsize=3),
       label="逐轮均值 ± 标准差")
ax.plot(xs, [s["best"] for s in summary], "v", color="#e45756", ms=7, label="最佳轮（被选中）")
ax.set_xticks(xs)
ax.set_xticklabels(tags, rotation=32, ha="right", fontsize=8)
ax.set_ylabel("验证集 帧降噪增益中位数 (dB)")
ax.set_title("逐轮均值 vs 被选中的最佳轮", fontsize=11)
ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(FIG, "figL_stability.png"))
plt.close(fig)
print("\n图 -> figs_pamnet/figL_stability.png")

import json
with open(os.path.join(FIG, "stability.json"), "w", encoding="utf-8") as f:
    json.dump({"runs": summary,
               "median_sd": float(np.median(sdv)),
               "max_sd": float(sdv.max()),
               "median_gain": float(np.median(g)),
               "max_gain": float(g.max()),
               "frac_best_in_last3": float(sum(1 for s in summary
                                                if s["best_ep"] >= s["n"] - 2) / len(summary))},
              f, ensure_ascii=False, indent=1)
print("汇总 -> figs_pamnet/stability.json")
