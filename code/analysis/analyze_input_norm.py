# -*- coding: utf-8 -*-
"""输入尺度归一化重训的效果分析：inm_*（归一化）vs devchk_*（无归一化，同设备同种子）

关键论证：
  - frmgain / fmsnr 都是能量比的对数（见 pam_metrics._stats_1d），
    数学上是尺度不变量 → 归一化域里算出的指标与原尺度【直接可比】。
  - 因此 inm 与 devchk 的差异只反映"训练时是否吃归一化输入"。

用法：python analyze_input_norm.py --split testA
"""
import os, sys, csv, json, argparse, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "sens_out")
CKPT = r"E:/project/data/output_pamnet"
COLS = ["snr_in", "snr_out", "sisdr_in", "sisdr_out", "fmsnr_med_in", "fmsnr_med_out",
        "frmgain_med", "frmgain_clean", "frmgain_imp", "frmgain_imp_min",
        "fmsnr_frac_bad_out", "frac_imp"]


def read_summary(tag, split):
    p = os.path.join(CKPT, "summary_%s.csv" % tag)
    if not os.path.exists(p):
        return None
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        head = next(r)
        i0 = head.index(COLS[0])
        for row in r:
            if row[1] == split:
                return {c: float(row[i0 + j]) for j, c in enumerate(COLS)}
    return None


def seeds_of(prefix, split):
    out = []
    for suf, lbl in (("", "s1"), ("_s2", "s2"), ("_s3", "s3")):
        a = read_summary("%s_nopulse%s" % (prefix, suf), split)
        b = read_summary("%s_pf%s" % (prefix, suf), split)
        if a and b:
            out.append((lbl, a, b))
    return out


def paired(arr_a, arr_b):
    d = np.asarray(arr_b, float) - np.asarray(arr_a, float)
    n = len(d)
    if n < 2 or d.std(ddof=1) <= 0:
        return dict(n=n, mean=float(d.mean()) if n else float("nan"), sd=0.0,
                    p=float("nan"))
    from scipy import stats
    t, p = stats.ttest_rel(arr_b, arr_a)
    return dict(n=n, mean=float(d.mean()), sd=float(d.std(ddof=1)), t=float(t), p=float(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="testA")
    A = ap.parse_args()
    SPLIT = A.split

    dev = seeds_of("devchk", SPLIT)      # GPU 基线（无归一化）
    inm = seeds_of("inm", SPLIT)         # 归一化训练

    print("=" * 100)
    print("输入尺度归一化的效果（%s）  样本=跨种子均值" % SPLIT)
    print("=" * 100)
    print("口径：frmgain / fmsnr 是能量比的对数 → 尺度不变量，两组指标直接可比")
    print()

    if not inm:
        print("还没有 inm_* 结果，等训练完成"); return
    if not dev:
        print("!! 缺 devchk_* 对照组，无法判断归一化的代价")

    hdr = "%-26s %12s %12s %12s" % ("组", "R4 干扰帧", "R5 干扰帧", "Δ=R5−R4")
    print(hdr); print("-" * len(hdr))

    def show(lbl, ss):
        if not ss:
            return None
        r4 = [x[1]["frmgain_imp"] for x in ss]
        r5 = [x[2]["frmgain_imp"] for x in ss]
        d = [b - a for a, b in zip(r4, r5)]
        print("%-26s %12.3f %12.3f %+12.3f   (n=%d)"
              % (lbl, np.mean(r4), np.mean(r5), np.mean(d), len(ss)))
        return dict(r4=r4, r5=r5, d=d)

    g_dev = show("devchk（GPU 无归一化）", dev)
    g_inm = show("inm（GPU 归一化）", inm)

    # 存档 CPU 对照（仅供参考，跨设备）
    cpu_tags = ["pamnet_freq_nopulse_gn", "pamnet_freq_nopulse_gn_s2",
                "pamnet_freq_nopulse_gn_s3"]
    cpu_tags5 = ["pamnet_freq_pf_gn", "pamnet_freq_pf_gn_s2", "pamnet_freq_pf_gn_s3"]
    c4 = [read_summary(t, SPLIT)["frmgain_imp"] for t in cpu_tags
          if read_summary(t, SPLIT)]
    c5 = [read_summary(t, SPLIT)["frmgain_imp"] for t in cpu_tags5
          if read_summary(t, SPLIT)]
    if c4 and c5:
        print("%-26s %12.3f %12.3f %+12.3f   (n=%d, ⚠ 跨设备)"
              % ("存档 base（CPU）", np.mean(c4), np.mean(c5),
                 np.mean(c5) - np.mean(c4), min(len(c4), len(c5))))

    print()
    print("=" * 100)
    print("① 归一化是否损失性能（同设备配对，m=min(n_dev,n_inm)）")
    print("=" * 100)
    m = min(len(dev), len(inm))
    if m >= 2 and g_dev and g_inm:
        for nm, key in (("R4", "r4"), ("R5", "r5"), ("Δ", "d")):
            st = paired(g_dev[key][:m], g_inm[key][:m])
            print("  %-3s 归一化 − 无归一化 = %+.3f dB   (配对 n=%d, sd=%.3f, p=%.4f)"
                  % (nm, st["mean"], st["n"], st["sd"], st["p"]))
    else:
        print("  种子数不足（dev=%d, inm=%d）" % (len(dev), len(inm)))

    print()
    print("=" * 100)
    print("② 归一化是否保住脉冲感知分支收益（Δ 的绝对值与显著性）")
    print("=" * 100)
    for lbl, ss in (("无归一化", dev), ("归一化", inm)):
        if not ss:
            continue
        d = [b - a for a, b in zip([x[1]["frmgain_imp"] for x in ss],
                                   [x[2]["frmgain_imp"] for x in ss])]
        if len(d) >= 2:
            from scipy import stats
            t, p = stats.ttest_1samp(d, 0.0)
            print("  %-10s Δ = %+.3f ± %.3f dB   (n=%d, 单样本 t=%+.2f, p=%.4f) %s"
                  % (lbl, np.mean(d), np.std(d, ddof=1), len(d), t, p,
                     "显著" if p < 0.05 else "不显著"))
        else:
            print("  %-10s Δ = %+.3f dB   (n=%d, 无法检验)" % (lbl, np.mean(d), len(d)))
    print()
    print("  存档 base（CPU）Δ = +0.790（5 种子），作为量级参考")

    out = dict(split=SPLIT, devchk=g_dev, inm=g_inm)
    os.makedirs(OUTD, exist_ok=True)
    with open(os.path.join(OUTD, "input_norm_%s.json" % SPLIT), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n->", os.path.join(OUTD, "input_norm_%s.json" % SPLIT))


if __name__ == "__main__":
    main()
