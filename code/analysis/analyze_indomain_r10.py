# -*- coding: utf-8 -*-
"""
域内重训（a15_r10）结果分析：把"零样本"与"域内"并排放
================================================================================
域内重训回答的问题：零样本矩阵里看到的退化/收益，有多少是
【分布偏移造成的】、多少是【该条件下方法本身的优劣】？

做法：在 a15_r10 数据上重新训练 R4（无脉冲分支）与 R5（含脉冲分支），
各 3 个种子（与 base 的前三个种子一致，便于按种子配对），
然后与 base 上的同配置结果、以及与 a15_r10 上的零样本结果三方对比。

数据来源：E:/project/data/output_pamnet/summary_sens_r10_{nopulse,pf}[_sN].csv
          E:/project/data/output_pamnet/train_log_sens_r10_*.txt
          sens_out/sens_summary_testA.json（零样本侧）

用法：python analyze_indomain_r10.py [--split testA|testC]
"""
import os, sys, csv, json, glob
import numpy as np

WORK = r"E:/workbuddy/store/2026-10-12-22-50-08"
OUTD = os.path.join(WORK, "sens_out")
CKPT = r"E:/project/data/output_pamnet"
SPLITS = ["testA", "testC"]
COLS = ["snr_in", "snr_out", "sisdr_in", "sisdr_out",
        "fmsnr_med_in", "fmsnr_med_out", "frmgain_med", "frmgain_clean",
        "frmgain_imp", "frmgain_imp_min", "fmsnr_frac_bad_out", "frac_imp"]
I_MED, I_CLN, I_IMP, I_MIN = 6, 7, 8, 9


def read_summary(tag):
    """读 summary_<tag>.csv，返回 {split: dict(指标名->值)}"""
    p = os.path.join(CKPT, "summary_%s.csv" % tag)
    if not os.path.exists(p):
        return None
    with open(p, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        # 找 COLS 在表头中的起始位置
        i0 = hdr.index(COLS[0])
        out = {}
        for row in rd:
            sp = row[1]
            out[sp] = {c: float(row[i0 + j]) for j, c in enumerate(COLS)}
            out[sp]["_meta"] = {h: row[k] for k, h in enumerate(hdr[:i0])}
    return out


def cross_seed_test(means):
    m = np.asarray([v for v in means if np.isfinite(v)], np.float64)
    n = len(m)
    if n < 2:
        return dict(n=n, mean=float(np.nanmean(m)) if n else float("nan"),
                    sd=float("nan"), t=float("nan"), p=float("nan"))
    mu, sd = float(m.mean()), float(m.std(ddof=1))
    se = sd / np.sqrt(n)
    t = mu / se if se > 0 else float("nan")
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df=n - 1))
    except Exception:
        from math import erfc, sqrt
        p = float(erfc(abs(t) / sqrt(2.0)))
    return dict(n=n, mean=mu, sd=sd, se=float(se), t=float(t), p=p)


def main():
    ap = argparse_()
    A = ap.parse_args()
    sp = A.split

    SENS = json.load(open(os.path.join(OUTD, "sens_summary_%s.json" % sp), encoding="utf-8"))

    # 域内重训的 tag
    TAGS = {"R4_in": [], "R5_in": []}
    for si in ["", "_s2", "_s3"]:
        t4 = read_summary("sens_r10_nopulse%s" % si)
        t5 = read_summary("sens_r10_pf%s" % si)
        if t4:
            TAGS["R4_in"].append(t4)
        if t5:
            TAGS["R5_in"].append(t5)

    print("=" * 104)
    print("域内重训（a15_r10：α=1.5, rate=10/s）结果   |   split = %s" % sp)
    print("=" * 104)
    if not TAGS["R4_in"] or not TAGS["R5_in"]:
        print("重训尚未完成：R4 已有 %d 个种子，R5 已有 %d 个种子"
              % (len(TAGS["R4_in"]), len(TAGS["R5_in"])))
        print("（每个训练约 642 s/轮 × 15 轮 ≈ 2.7 h；6 个训练共约 16 h）")
        if TAGS["R4_in"] or TAGS["R5_in"]:
            for k, v in TAGS.items():
                for i, t in enumerate(v):
                    r = t.get(sp)
                    if r:
                        print("  %s seed%d  干净 %.2f  干扰 %.2f  最差 %.2f"
                              % (k, i + 1, r["frmgain_clean"], r["frmgain_imp"],
                                 r["frmgain_imp_min"]))
        return

    n4, n5 = len(TAGS["R4_in"]), len(TAGS["R5_in"])
    print("\n[1] 域内重训的绝对水平（种子均值）")
    print("-" * 104)
    print("%-34s %6s %10s %10s %10s" % ("配置", "n_seed", "干净帧", "干扰帧", "干扰最差"))
    for k, v in TAGS.items():
        for m in (I_CLN, I_IMP, I_MIN):
            pass
        cl = [t[sp]["frmgain_clean"] for t in v]
        im = [t[sp]["frmgain_imp"] for t in v]
        mn = [t[sp]["frmgain_imp_min"] for t in v]
        print("%-34s %6d %10.2f %10.2f %10.2f"
              % (k, len(v), np.mean(cl), np.mean(im), np.mean(mn)))

    print("\n[2] 三方对比（干扰帧增益，dB）")
    print("-" * 104)
    hdr = ("%-24s %22s %22s %22s" % ("配置", "base（训练条件）", "a15_r10 零样本", "a15_r10 域内重训"))
    print(hdr); print("-" * len(hdr))
    T = SENS["table"]
    for cfg_in, cfg_zs in (("R4_in", "R4_freqrefine_nopulse"),
                           ("R5_in", "R5_freqrefine_pulse")):
        b = T["base"][cfg_zs]["imp"]
        z = T["a15_r10"][cfg_zs]["imp"]
        d = np.mean([t[sp]["frmgain_imp"] for t in TAGS[cfg_in]])
        print("%-24s %22.2f %22.2f %22.2f" % (cfg_zs, b, z, d))
    print("-" * len(hdr))
    print("（零样本 = base 上训练的模型直接在 a15_r10 上推理；"
          "域内重训 = 在 a15_r10 上从头训练）")

    print("\n[3] 脉冲感知分支效应 Δ = R5 − R4，两种训练方式")
    print("-" * 104)
    for m, mn in ((I_CLN, "干净帧"), (I_IMP, "干扰帧"), (I_MIN, "干扰最差")):
        zs = SENS["pulse_eff"]["a15_r10"]
        key = {I_CLN: "clean", I_IMP: "imp", I_MIN: "imp_min"}[m]
        zin = cross_seed_test(
            [t5[sp][COLS[m]] - t4[sp][COLS[m]] for t4, t5 in zip(TAGS["R4_in"], TAGS["R5_in"])])
        print("%-10s 零样本: %+.3f ± %.3f (跨种子 p=%.3f)   域内: %+.3f ± %.3f (p=%.3f)"
              % (mn, zs[key]["mean"], zs[key]["sd_over_seed"], zs[key]["cs"]["p"],
                 zin["mean"], zin["sd"], zin["p"]))

    print("\n[4] 域内 vs 零样本：分布偏移造成了多少退化")
    print("-" * len(hdr))
    for cfg_in, cfg_zs in (("R4_in", "R4_freqrefine_nopulse"),
                           ("R5_in", "R5_freqrefine_pulse")):
        z = T["a15_r10"][cfg_zs]["imp"]
        d = np.mean([t[sp]["frmgain_imp"] for t in TAGS[cfg_in]])
        b = T["base"][cfg_zs]["imp"]
        print("%-34s 零样本相对 base: %+.2f dB   域内相对 base: %+.2f dB"
              "   → 分布偏移的代价约 %.2f dB"
              % (cfg_zs, z - b, d - b, d - z))

    out = dict(split=sp,
               indomain={k: [{kk: vv for kk, vv in t[sp].items()} for t in v]
                         for k, v in TAGS.items()},
               n_seed={"R4": n4, "R5": n5})
    with open(os.path.join(OUTD, "indomain_r10_%s.json" % sp), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n-> sens_out/indomain_r10_%s.json" % sp)


def argparse_():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="testA")
    return ap


if __name__ == "__main__":
    main()
