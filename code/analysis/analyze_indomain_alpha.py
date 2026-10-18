# -*- coding: utf-8 -*-
"""
α 轴域内重训分析：把 8.1b 的「尾形 vs 强度」分解从零样本升级为域内
================================================================================
对比三方（全部同一指标口径、同一 testA）：
  base（α=1.5，训练条件）      —— 对 α=1.5 而言这本身就是"域内"
  α=1.2/1.8 零样本（匹配功率） —— 8.1b 用的那组
  α=1.2/1.8 域内重训           —— 本脚本新增

关键判断：
  若域内 Δ(α=1.2) 回到 base 的 ~0.79  -> 8.1b 观察到的压低主要是【分布偏移】
  若域内 Δ(α=1.2) 仍接近零样本的 0.41 -> 【尾形本身】的效应是真的

设备校正：base 为 CPU 训练，α 域内为 GPU 训练；
实测设备效应对 Δ = −0.091 ± 0.031 dB（base 条件 CPU vs GPU 配对），
因此域内值需 +0.091 才与 base 可比（同时给出未校正值）。

用法：python analyze_indomain_alpha.py [--split testA]
"""
import os, csv, json, argparse, glob
import numpy as np

WORK = r"E:/workbuddy/store/2026-10-12-22-50-08"
OUTD = os.path.join(WORK, "sens_out")
CK = r"E:/project/data/output_pamnet"
COLS = ["snr_in", "snr_out", "sisdr_in", "sisdr_out", "fmsnr_med_in", "fmsnr_med_out",
        "frmgain_med", "frmgain_clean", "frmgain_imp", "frmgain_imp_min",
        "fmsnr_frac_bad_out", "frac_imp"]
CONDS = [("a12", "α=1.2（已匹配功率）", "m_a12"),
         ("a18", "α=1.8（已匹配功率）", "m_a18")]


def read(tag, split):
    p = os.path.join(CK, "summary_%s.csv" % tag)
    if not os.path.exists(p):
        return None
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        h = next(r)
        i0 = h.index(COLS[0])
        for row in r:
            if row[1] == split:
                return {c: float(row[i0 + j]) for j, c in enumerate(COLS)}
    return None


def abs_level(cond, cfg, split="testA"):
    """从 sens_*.npz 读某条件某配置的干扰帧增益（跨种子均值）"""
    import glob
    arrs = sorted(glob.glob(os.path.join(OUTD, "sens_%s__%s__s*.npz" % (cond, cfg))))
    if not arrs:
        p2 = os.path.join(OUTD, "sens_%s__%s.npz" % (cond, cfg))
        if not os.path.exists(p2):
            return float("nan")
        z = np.load(p2)
        return float(np.nanmean(z[split][:, :, 8]))
    vals = []
    for a in arrs:
        z = np.load(a)
        if split in z:
            vals.append(float(np.nanmean(z[split][:, 8])))
    return float(np.mean(vals)) if vals else float("nan")

def seeds_of(prefix, split="testA"):
    """返回 [(seed_label, R4_dict, R5_dict)]"""
    out = []
    for suf, lbl in (("", "s1"), ("_s2", "s2"), ("_s3", "s3")):
        a = read("%s_nopulse%s" % (prefix, suf), split)
        b = read("%s_pf%s" % (prefix, suf), split)
        if a and b:
            out.append((lbl, a, b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="testA")
    A = ap.parse_args()

    S = json.load(open(os.path.join(OUTD, "sens_summary_%s.json" % A.split),
                       encoding="utf-8"))
    M = json.load(open(os.path.join(OUTD, "matched_%s.json" % A.split),
                       encoding="utf-8"))
    DC = json.load(open(os.path.join(OUTD, "device_check_testA.json"),
                        encoding="utf-8"))
    dev = DC["device_diff_on_delta"]["mean"]
    dev_sd = DC["device_diff_on_delta"]["sd"]

    b4 = S["table"]["base"]["R4_freqrefine_nopulse"]["imp"]
    b5 = S["table"]["base"]["R5_freqrefine_pulse"]["imp"]
    d_base = b5 - b4

    print("=" * 92)
    print("α 轴域内重训（%s）  设备校正量 %+.3f ± %.3f dB" % (A.split, dev, dev_sd))
    print("=" * 92)
    print("%-22s %10s %10s %10s %14s %8s"
          % ("条件", "R4 干扰帧", "R5 干扰帧", "Δ", "Δ(设备校正后)", "种子"))
    print("-" * 92)
    print("%-22s %10.3f %10.3f %+10.3f %14s %8d"
          % ("base α=1.5（CPU）", b4, b5, d_base, "—（基准）",
             S["table"]["base"]["R4_freqrefine_nopulse"]["n_seed"]))
    print("-" * 92)

    res = {}
    for key, lbl, mcond in CONDS:
        z = M["pulse_eff"].get(mcond)
        z4 = z5 = float("nan")
        if z:
            z4 = abs_level(mcond, "R4_freqrefine_nopulse", A.split)
            z5 = abs_level(mcond, "R5_freqrefine_pulse", A.split)
            print("%-22s %10.3f %10.3f %+10.3f %14s %8d"
                  % (lbl + " 零样本", z4, z5, z["imp"]["mean"], "—",
                     len(glob.glob(os.path.join(OUTD, "sens_%s__R4_freqrefine_nopulse__s*.npz" % mcond)))))
        ss = seeds_of("sens_%s" % key, A.split)
        if not ss:
            print("%-22s %10s %10s %10s %14s %8s"
                  % (lbl + " 域内", "—", "—", "—", "—", 0))
            continue
        a4 = np.mean([x[1]["frmgain_imp"] for x in ss])
        a5 = np.mean([x[2]["frmgain_imp"] for x in ss])
        d = np.array([x[2]["frmgain_imp"] - x[1]["frmgain_imp"] for x in ss])
        m_, sd_ = float(d.mean()), float(d.std(ddof=1)) if len(d) > 1 else 0.0
        if len(d) > 1 and sd_ > 0:
            from scipy import stats
            t = m_ / (sd_ / np.sqrt(len(d)))
            p_ = float(2 * stats.t.sf(abs(t), df=len(d) - 1))
        else:
            p_ = float("nan")
        res[key] = dict(n=len(ss), r4=float(a4), r5=float(a5), d=m_, sd=sd_, p=p_,
                        per_seed=[float(x) for x in d], corrected=m_ - dev,
                        zero_r4=float(z4), zero_r5=float(z5))
        print("%-22s %10.3f %10.3f %+10.3f %+14.3f %8d   (sd %.3f, p=%.3f)"
              % (lbl + " 域内", a4, a5, m_, m_ - dev, len(ss), sd_, p_))
    print("-" * 92)
    print("（设备校正：α 域内为 GPU 训练，base 为 CPU；实测设备效应对 Δ = %+.3f dB，"
          "故校正值 = 域内 Δ − (%+.3f)）" % (dev, dev))

    # ---------- 判定 ----------
    print()
    print("=" * 92)
    print("判定：α 轴的收益差异是「分布偏移」还是「尾形本身」？")
    print("=" * 92)
    for key, lbl, mcond in CONDS:
        if key not in res:
            continue
        z = M["pulse_eff"][mcond]["imp"]["mean"]
        r = res[key]
        print("  %s：" % lbl)
        print("    base(α=1.5) %+.3f  →  零样本 %+.3f  →  域内 %+.3f（校正 %+.3f）"
              % (d_base, z, r["d"], r["corrected"]))
        gap_zero = z - d_base          # 零样本相对 base
        gap_in = r["corrected"] - d_base   # 域内校正后相对 base
        print("    零样本相对 base：%+.3f dB；域内校正后相对 base：%+.3f dB" % (gap_zero, gap_in))
        sig = ("-- **注意**：域内 Δ 的跨种子检验不显著（p=%.3f，sd=%.3f），"
               "该残余量级很小，不能排除为 0；"
               "两个 split 的差值方向是否一致要看 testC。")
        print("    域内 Δ 跨种子检验：p = %.3f（sd %.3f，n=%d 种子）" % (r["p"], r["sd"], r["n"]))
        if abs(gap_in) < abs(gap_zero) * 0.5:
            print("    -> **主要是分布偏移**：域内重训后收益基本回到 base 水平。")
        elif abs(gap_in) > abs(gap_zero) * 0.8:
            print("    -> **主要是尾形本身**：即使域内重训，收益仍明显偏离 base。")
        else:
            print("    -> **两者兼有**：分布偏移解释了约 %.0f%%，尾形成分解释了约 %.0f%%。"
                  % (100 * (1 - abs(gap_in) / abs(gap_zero)) if abs(gap_zero) > 1e-9 else 0,
                     100 * abs(gap_in) / abs(gap_zero) if abs(gap_zero) > 1e-9 else 0))
        sig = sig % (r["p"], r["sd"])
        if r["p"] > 0.05:
            # 不显著时不得下"主要是尾形"的强判断
            print("       " + sig)
            if abs(gap_in) > abs(gap_zero) * 0.8:
                print("       **修正判定**：虽然域内校正后相对 base 的比例仍高，但域内 Δ 本身"
                      "不显著，\n       **不能判定为'主要是尾形本身'** —— 更稳的表述是："
                      "该端存在一个小量级残余差异，但无法与 0 区分。")
        else:
            print("       域内 Δ 跨种子检验显著（p=%.3f），残余尾形成分可认为非零。" % r["p"])
        print()

    out = dict(split=A.split, device_correction=dev, d_base=d_base, conds=res)
    with open(os.path.join(OUTD, "indomain_alpha_%s.json" % A.split), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("-> sens_out/indomain_alpha_%s.json" % A.split)


if __name__ == "__main__":
    main()
