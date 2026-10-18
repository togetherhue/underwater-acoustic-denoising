# -*- coding: utf-8 -*-
"""
变体间差异的配对显著性检验
================================================================================
消融的效应量只有 0.2~0.4 dB 量级，必须回答"这是真实效应还是训练/抽样波动"。

做法：两个变体在【同一批测试样本】上评估，因此可以对【逐样本】的指标差做配对检验。
    d_i = metric_A(sample_i) - metric_B(sample_i)
    t   = mean(d) / (std(d)/sqrt(n))
再配合双侧 p 值。同时报告 Cohen's d_z 效应量。

注意：配对检验只消除【样本抽样】这一项不确定性，不消除【训练随机性】。
      后者需要多随机种子才能量化（本实验未做，报告中列为局限）。
"""
import os, sys, csv
import numpy as np

OUT = r"E:/project/data/output_pamnet"
OUT0 = r"D:/project/output_pam"
COLS = ["snr_in", "snr_out", "sisdr_in", "sisdr_out", "fmsnr_med_in", "fmsnr_med_out",
        "frmgain_med", "frmgain_clean", "frmgain_imp", "frmgain_imp_min",
        "fmsnr_frac_bad_out", "frac_imp"]
IDX = {c: i for i, c in enumerate(COLS)}

# (标签, tag_A, dir_A, tag_B, dir_B)
PAIRS = [
    ("纯时域：+脉冲分支  vs  −脉冲分支", "pamnet", OUT, "pamnet_nopulse", OUT),
    ("后置频域精修：+脉冲分支 vs −脉冲分支", "pamnet_freq", OUT, "pamnet_freq_nopulse", OUT),
    ("p注入频域：+脉冲分支 vs −脉冲分支", "pamnet_freq_pf", OUT, "pamnet_freq_nopulse", OUT),
    ("p注入频域 vs 后置频域精修（均含脉冲分支）", "pamnet_freq_pf", OUT, "pamnet_freq", OUT),
    ("加频域精修 vs 不加（均含脉冲分支）", "pamnet_freq", OUT, "pamnet", OUT),
    ("PAM-Net 纯时域 vs 阶段0 纯频域", "pamnet", OUT, "pam_freq", OUT0),
    ("p注入频域 vs 阶段0 纯频域", "pamnet_freq_pf", OUT, "pam_freq", OUT0),
    ("p注入频域 vs PAM-Net 纯时域", "pamnet_freq_pf", OUT, "pamnet", OUT),
    ("损失换 L1 vs Huber（纯时域）", "pamnet_l1", OUT, "pamnet", OUT),
    ("λ2=0.5 vs λ2=0.05（纯时域）", "pamnet_l2big", OUT, "pamnet", OUT),
]
METRICS = ["frmgain_clean", "frmgain_imp", "frmgain_imp_min"]


def load(d, tag, split):
    p = os.path.join(d, "result_%s_%s.csv" % (tag, split))
    if not os.path.exists(p):
        return None
    return np.loadtxt(p, delimiter=",", skiprows=1)


def ttest_paired(a, b):
    d = a - b
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return None
    mean = d.mean()
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    t = mean / se if se > 0 else 0.0
    # 双侧 p（正态近似，n 很大时与 t 分布几乎一致）
    try:
        from scipy.stats import t as tdist
        p = 2 * tdist.sf(abs(t), df=n - 1)
    except Exception:
        from math import erfc, sqrt
        p = erfc(abs(t) / sqrt(2))
    dz = mean / sd if sd > 0 else 0.0
    return mean, se, t, p, dz, n


import json
RECD = {}
print("=" * 112)
print("配对显著性检验（testA，同一批测试样本；d = A − B，正数表示 A 更好）")
print("=" * 112)

for split in ["testA", "testC"]:
    print()
    print("### %s" % split)
    RECD[split] = {}
    for label, ta, da, tb, db in PAIRS:
        A, B = load(da, ta, split), load(db, tb, split)
        if A is None or B is None or len(A) != len(B):
            print("  %-42s (缺数据)" % label)
            continue
        print("  %s" % label)
        print("    %-14s %10s %10s %9s %12s %9s" %
              ("指标", "均值差dB", "标准误", "t", "p", "d_z"))
        rec = {}
        for m in METRICS:
            r = ttest_paired(A[:, IDX[m]], B[:, IDX[m]])
            if r is None:
                continue
            mean, se, t, p, dz, n = r
            rec[m] = dict(mean=float(mean), se=float(se), t=float(t), p=float(p),
                          dz=float(dz), n=int(n))
            star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            print("    %-14s %+10.3f %10.3f %9.2f %12.2e %9.3f  %s"
                  % (m, mean, se, t, p, dz, star))
        print("    (n=%d；*** p<0.001, ** p<0.01, * p<0.05, ns 不显著)" % len(A))
        RECD[split][label] = rec
    print("-" * 112)

OUTJ = os.path.join(r"E:/workbuddy/store/2026-10-12-22-50-08/figs_pamnet", "paired_test.json")
with open(OUTJ, "w", encoding="utf-8") as f:
    json.dump(RECD, f, ensure_ascii=False, indent=1)
print("\n配对检验汇总 -> %s" % OUTJ)
