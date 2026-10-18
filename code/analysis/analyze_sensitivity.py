# -*- coding: utf-8 -*-
"""
适用边界（敏感性）实验：统计分析与出图
================================================================================
输入：sens_out/sens_<cond>__<config>__s<k>.npz（逐样本 12 列指标）
      sens_out/noise_<cond>.npz（逐样本噪声功率分解）
      sens_out/pulse_diag_sens.json（脉冲分支判别能力）
输出：sens_out/sens_summary.json / sens_table.csv
      figs_pamnet/figQ_sens_alpha.png、figR_sens_rate.png、
                  figS_sens_pulse_effect.png、figT_sens_detect.png

【统计口径】
1. 种子内逐样本配对：同一条件、同一模型配置下，不同种子看到的是同一批样本，
   但权重不同 —— 因此 R5 vs R4 的逐样本配对只能在同一"种子序号"内做，
   再跨种子汇总（先配对、后平均）。
2. 条件间逐样本配对：数据生成使用 seed+样本序号 且各条件循环顺序一致，
   故样本序号 i 在所有条件下对应同一 (chirp, 信道, SNR, 重复)；
   α 轮换时突发干扰逐位相同、rate 轮换时背景噪声逐位相同（共同随机数）。
   因此条件间的比较可以做逐样本配对，功效远高于独立样本。
"""
import os, sys, json, glob, argparse, itertools
import numpy as np

WORK = r"E:/workbuddy/store/2026-10-12-22-50-08"
OUTD = os.path.join(WORK, "sens_out")
FIGD = os.path.join(WORK, "figs_pamnet")
COLS = ["snr_in", "snr_out", "sisdr_in", "sisdr_out",
        "fmsnr_med_in", "fmsnr_med_out", "frmgain_med", "frmgain_clean",
        "frmgain_imp", "frmgain_imp_min", "fmsnr_frac_bad_out", "frac_imp"]
I_MED, I_CLN, I_IMP, I_MIN = 6, 7, 8, 9

COND_ORDER = ["a12_r5", "base", "a18_r5", "a20_r5", "a15_r25", "a15_r10", "gauss"]
COND_LABEL = {"base": "α=1.5\nrate=5", "a12_r5": "α=1.2\nrate=5", "a18_r5": "α=1.8\nrate=5",
              "a20_r5": "α=2.0\nrate=5", "a15_r25": "α=1.5\nrate=2.5",
              "a15_r10": "α=1.5\nrate=10", "gauss": "高斯\nrate=5"}
CFG_ORDER = ["R3_freqonly", "R4_freqrefine_nopulse", "R5_freqrefine_pulse",
             "P_time_pulse", "P_time_nopulse"]
CFG_SHORT = {"R3_freqonly": "R3 纯频域掩码", "R4_freqrefine_nopulse": "R4 时域+频域(无脉冲)",
             "R5_freqrefine_pulse": "R5 时域+频域+脉冲", "P_time_pulse": "纯时域 PAM-Net",
             "P_time_nopulse": "纯时域(无脉冲)"}
COLOR = {"R3_freqonly": "#4c78a8", "R4_freqrefine_nopulse": "#f58518",
         "R5_freqrefine_pulse": "#e45756", "P_time_pulse": "#54a24b",
         "P_time_nopulse": "#b279a2"}
MARK = {"R3_freqonly": "o", "R4_freqrefine_nopulse": "s", "R5_freqrefine_pulse": "^",
        "P_time_pulse": "D", "P_time_nopulse": "v"}


def load(cond, cfg, split="testA"):
    """返回 [n_seed, n_sample, n_col]；优先读聚合文件，否则由逐种子文件拼装"""
    p = os.path.join(OUTD, "sens_%s__%s.npz" % (cond, cfg))
    if os.path.exists(p):
        z = np.load(p)
        return z[split] if split in z else None
    arrs, si = [], 0
    while True:
        q = os.path.join(OUTD, "sens_%s__%s__s%d.npz" % (cond, cfg, si))
        if not os.path.exists(q):
            break
        z = np.load(q)
        if split not in z:
            break
        arrs.append(z[split]); si += 1
    return np.stack(arrs, 0) if arrs else None


def load_noise(cond):
    p = os.path.join(OUTD, "noise_%s.npz" % cond)
    return dict(np.load(p)) if os.path.exists(p) else None


def paired_stats(a, b):
    """逐样本配对差 a-b 的统计量（按样本，NaN 成对剔除）"""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    ok = np.isfinite(d)
    d = d[ok]
    n = len(d)
    if n < 5:
        return dict(n=n, mean=float("nan"), sd=float("nan"), t=float("nan"), p=float("nan"))
    m, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    t = m / se if se > 0 else float("nan")
    # 正态近似双尾 p（n>=1000 时与 t 分布几乎一致）
    from math import erfc, sqrt
    p = float(erfc(abs(t) / sqrt(2.0)))
    return dict(n=n, mean=m, sd=sd, se=float(se), t=float(t), p=p,
                dz=float(m / sd) if sd > 0 else float("nan"))


def cross_seed_test(means):
    """跨种子配对检验：对【逐种子的效应量】做单样本 t 检验（自由度 = n_seed-1）。
    这是与"种子内逐样本配对"互补的口径 —— 后者只消除样本抽样，
    前者才反映训练随机性。两者都报告，避免把 n=1080 的样本级显著性
    误当成结论的稳健性。"""
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="testA")
    A = ap.parse_args()
    sp = A.split
    conds = [c for c in COND_ORDER if os.path.exists(os.path.join(OUTD, "noise_%s.npz" % c))]
    print("可用条件:", conds)

    # ---------- 1. 主表：条件 × 配置 ----------
    tab = {}
    for c in conds:
        for cf in CFG_ORDER:
            R = load(c, cf, sp)
            if R is None:
                continue
            # R: [n_seed, n_sample, n_col]
            tab.setdefault(c, {})[cf] = dict(
                n_seed=R.shape[0], n_sample=R.shape[1],
                med=float(np.nanmean(R[:, :, I_MED])),
                clean=float(np.nanmean(R[:, :, I_CLN])),
                imp=float(np.nanmean(R[:, :, I_IMP])),
                imp_min=float(np.nanmean(R[:, :, I_MIN])),
                clean_sd=float(np.nanstd(np.nanmean(R[:, :, I_CLN], axis=1), ddof=1))
                if R.shape[0] > 1 else 0.0,
                imp_sd=float(np.nanstd(np.nanmean(R[:, :, I_IMP], axis=1), ddof=1))
                if R.shape[0] > 1 else 0.0,
                per_seed_imp=[float(v) for v in np.nanmean(R[:, :, I_IMP], axis=1)],
                per_seed_clean=[float(v) for v in np.nanmean(R[:, :, I_CLN], axis=1)])

    print()
    print("=" * 112)
    print("表 1  各条件 × 各模型 的逐帧降噪增益（%s，种子均值）" % sp)
    print("=" * 112)
    print("%-22s %-24s %8s %10s %10s %10s %8s" %
          ("条件", "模型", "n_seed", "干净帧", "干扰帧", "干扰最差", "帧中位"))
    print("-" * 112)
    for c in conds:
        for cf in CFG_ORDER:
            if cf not in tab.get(c, {}):
                continue
            r = tab[c][cf]
            print("%-22s %-24s %8d %10.2f %10.2f %10.2f %8.2f"
                  % (COND_LABEL[c].replace("\n", " "), CFG_SHORT[cf], r["n_seed"],
                     r["clean"], r["imp"], r["imp_min"], r["med"]))
        print("-" * 112)

    # ---------- 2. 脉冲分支效应（R5 - R4），逐条件 ----------
    print()
    print("=" * 112)
    print("表 2  脉冲感知分支效应 Δ = R5(含脉冲) − R4(无脉冲)，同种子内逐样本配对")
    print("=" * 112)
    print("%-22s %26s %26s %26s" %
          ("条件", "干净帧 Δ (dB)", "干扰帧 Δ (dB)", "干扰最差 Δ (dB)"))
    print("-" * 112)
    pulse_eff = {}
    for c in conds:
        A5, A4 = load(c, "R5_freqrefine_pulse", sp), load(c, "R4_freqrefine_nopulse", sp)
        if A5 is None or A4 is None:
            continue
        ns = min(A5.shape[0], A4.shape[0])
        per = {"clean": [], "imp": [], "imp_min": []}
        allstat = {}
        for k in range(ns):
            for key, idx in (("clean", I_CLN), ("imp", I_IMP), ("imp_min", I_MIN)):
                per[key].append(paired_stats(A5[k, :, idx], A4[k, :, idx]))
        for key in per:
            ms = np.array([s["mean"] for s in per[key]])
            ps = np.array([s["p"] for s in per[key]])
            ts = np.array([s["t"] for s in per[key]])
            cs = cross_seed_test(ms)
            allstat[key] = dict(mean=float(np.nanmean(ms)),
                                sd_over_seed=float(np.nanstd(ms, ddof=1)) if ns > 1 else 0.0,
                                t=float(np.nanmedian(ts)),
                                p=float(np.nanmax(ps)),          # 最保守（最大的种子内 p）
                                p_min=float(np.nanmin(ps)),
                                cs=cs,                           # 跨种子配对检验
                                dz=float(np.nanmedian([s["dz"] for s in per[key]])),
                                n=per[key][0]["n"], n_seed=ns)
        pulse_eff[c] = allstat
        print("%-22s %10.3f±%.3f (cs p=%.3f) %10.3f±%.3f (cs p=%.4f) %10.3f±%.3f (cs p=%.3f)"
              % (COND_LABEL[c].replace("\n", " "),
                 allstat["clean"]["mean"], allstat["clean"]["sd_over_seed"], allstat["clean"]["cs"]["p"],
                 allstat["imp"]["mean"], allstat["imp"]["sd_over_seed"], allstat["imp"]["cs"]["p"],
                 allstat["imp_min"]["mean"], allstat["imp_min"]["sd_over_seed"], allstat["imp_min"]["cs"]["p"]))
    print("-" * 112)
    print("（± 为种子间标准差；cs p = 对逐种子效应量做单样本 t 检验的 p，自由度 = 种子数−1）")

    # 纯时域上的脉冲分支效应（作为对照：应接近 0）
    print()
    print("表 2b 纯时域结构上的脉冲分支效应 Δ = 纯时域(含脉冲) − 纯时域(无脉冲)")
    print("-" * 112)
    for c in conds:
        A1, A0 = load(c, "P_time_pulse", sp), load(c, "P_time_nopulse", sp)
        if A1 is None or A0 is None:
            continue
        ns = min(A1.shape[0], A0.shape[0])
        ms = [paired_stats(A1[k, :, I_IMP], A0[k, :, I_IMP])["mean"] for k in range(ns)]
        cs = cross_seed_test(ms)
        print("%-22s 干扰帧 Δ %+.3f ± %.3f  跨种子 p=%.3f (n_seed=%d)  |  种子内逐样本 p 中位 %.1e"
              % (COND_LABEL[c].replace("\n", " "), cs["mean"], cs["sd"] if ns > 1 else 0.0,
                 cs["p"], ns,
                 np.nanmedian([paired_stats(A1[k, :, I_IMP], A0[k, :, I_IMP])["p"] for k in range(ns)])))
    print("（对照：种子内逐样本配对会把 n=1080 的样本级显著性放大成 p≈0；"
          "跨种子检验的自由度只有 n_seed−1，才是结论稳健性的正确口径）")

    # ---------- 2c. 架构优势（R5 − R3）：时域残差通路 + 脉冲注入相对纯频域掩码 ----------
    print()
    print("=" * 112)
    print("表 2c 架构优势 Δ = R5（时域残差+频域精修+脉冲） − R3（纯频域掩码），同种子内逐样本配对")
    print("=" * 112)
    print("%-22s %14s %14s %14s" % ("条件", "干净帧 Δ", "干扰帧 Δ", "跨种子 p"))
    print("-" * 112)
    arch_adv = {}
    for c in conds:
        A5, A3 = load(c, "R5_freqrefine_pulse", sp), load(c, "R3_freqonly", sp)
        if A5 is None or A3 is None:
            continue
        ns = min(A5.shape[0], A3.shape[0])
        rec = {}
        for key, idx in (("clean", I_CLN), ("imp", I_IMP)):
            ms = [paired_stats(A5[k, :, idx], A3[k, :, idx])["mean"] for k in range(ns)]
            rec[key] = cross_seed_test(ms)
        arch_adv[c] = rec
        print("%-22s %14s %14s %14s"
              % (COND_LABEL[c].replace("\n", " "),
                 "%+.2f±%.2f" % (rec["clean"]["mean"], rec["clean"]["sd"]),
                 "%+.2f±%.2f" % (rec["imp"]["mean"], rec["imp"]["sd"]),
                 "%.2e" % rec["imp"]["p"]))
    print("-" * 112)
    print("（架构优势随 α 增大而缩小：背景越接近高斯，频域掩码的相对优势越强）")

    # ---------- 3. 条件间比较（共同随机数，逐样本配对）----------    print()
    print("=" * 112)
    print("表 3  条件间退化：各模型在异常条件下相对 base 的干扰帧增益变化（逐样本配对，跨种子均值）")
    print("=" * 112)
    hdr = "%-24s" % "模型" + "".join("%18s" % COND_LABEL[c].replace("\n", " ") for c in conds)
    print(hdr); print("-" * len(hdr))
    # ------------------------------------------------------------------
    # 【语义警告】cond_delta[cfg][cond] = 同一配置在 cond 与 base 上的逐样本配对差
    #   = "换个条件后这个模型退化/提升多少"（分布偏移的代价），**不是**脉冲分支效应。
    #   例：cond_delta["R4_freqrefine_nopulse"]["a15_r10"] = −0.455
    #       意思是 R4 在 rate=10 上比 base 低 0.455 dB，符号为负完全合理。
    #   脉冲分支效应 Δ = R5 − R4 在 pulse_eff[cond]["imp"] 里。
    #   两者语义完全不同，混用会得到符号相反的结论（8.14 节踩过这个坑）。
    # ------------------------------------------------------------------
    cond_delta = {}
    for cf in CFG_ORDER:
        if cf not in tab.get("base", {}):
            continue
        row = "%-24s" % CFG_SHORT[cf]
        cond_delta[cf] = {}
        B = load("base", cf, sp)
        for c in conds:
            A2 = load(c, cf, sp)
            if A2 is None or B is None:
                row += "%18s" % "-"; continue
            ns = min(A2.shape[0], B.shape[0])
            if ns < 1:
                row += "%18s" % "-"; continue
            ms, ps = [], []
            for k in range(ns):
                st = paired_stats(A2[k, :, I_IMP], B[k, :, I_IMP])
                ms.append(st["mean"]); ps.append(st["p"])
            # 干扰极稀疏时（如 rate=2.5 的 testC）逐样本差的有效值可能 < 5，
            # 此时 paired_stats 返回 NaN，属正常边界情况，表格显示为 "—"
            if not ms or np.all(np.isnan(ms)):
                row += "%18s" % "—"; continue
            st0 = dict(mean=float(np.nanmean(ms)),
                       sd_over_seed=float(np.nanstd(ms, ddof=1)) if ns > 1 else 0.0,
                       p=float(np.nanmax(ps)) if np.any(np.isfinite(ps)) else float("nan"),
                       n_seed=ns,
                       t=float(paired_stats(A2[0, :, I_IMP], B[0, :, I_IMP])["t"]))
            cond_delta[cf][c] = st0
            row += "%18s" % ("%+.2f±%.2f" % (st0["mean"], st0["sd_over_seed"]))
        print(row)
    print("-" * len(hdr))
    print("（负值 = 该条件下干扰帧增益低于 base；± 为种子间标准差；n≈1080/种子）")

    # ---------- 4. 噪声统计分解 ----------
    print()
    print("=" * 112)
    print("表 4  噪声功率分解（名义 SNR 固定为 75% 分位数匹配，故有效功率 SNR 会随 α 变化）")
    print("=" * 112)
    print("%-16s %10s %12s %12s %12s %10s %10s %10s" %
          ("条件", "干扰帧占比", "有效SNR(总)", "有效SNR(背景)", "有效SNR(干扰)",
           "背景峰/rms", "背景p99.9/rms", "背景p75/rms"))
    print("-" * 112)
    noise_tab = {}
    for c in conds:
        nd = load_noise(c)
        if nd is None:
            continue
        a = nd[sp]
        # kurtosis 由 p99.9/q75 等近似不便，这里直接用分位数比刻画重尾
        rec = dict(frac_imp=float(np.mean(a[:, 6])),
                   snr_tot=float(np.median(a[:, 5])),
                   snr_bg=float(np.median(a[:, 3])),
                   snr_imp=float(np.median(a[:, 4])),
                   peak_over_rms=float(np.median(a[:, 7])),
                   p999_over_rms=float(np.median(a[:, 9])),
                   p75_over_rms=float(np.median(a[:, 11])))
        noise_tab[c] = rec
        print("%-16s %10.3f %12.2f %12.2f %12.2f %10.1f %10.2f %10.3f"
              % (COND_LABEL[c].replace("\n", " "), rec["frac_imp"], rec["snr_tot"],
                 rec["snr_bg"], rec["snr_imp"], rec["peak_over_rms"],
                 rec["p999_over_rms"], rec["p75_over_rms"]))
    print("-" * 112)
    print("p99.9/p75 越大说明背景噪声尾部越重（α 越小该比值越大）")

    # ---------- 5. 脉冲分支判别能力 ----------
    pd_path = os.path.join(OUTD, "pulse_diag_sens.json")
    pulse_diag = json.load(open(pd_path, encoding="utf-8")) if os.path.exists(pd_path) else {}
    if pulse_diag:
        print()
        print("=" * 112)
        print("表 5  脉冲感知分支判别能力（p_n 作为脉冲检测器）随条件变化")
        print("=" * 112)
        print("%-16s %-26s %10s %10s %10s %12s" %
              ("条件", "模型", "AUC均值", "AUC中位", ">0.8占比", "干扰位置p均值"))
        print("-" * 112)
        for c in conds:
            for cf in ("R5_freqrefine_pulse", "P_time_pulse"):
                d = pulse_diag.get(c, {}).get(cf, {})
                if not d:
                    continue
                k0 = sorted(d.keys())[0]
                v = d[k0]
                print("%-16s %-26s %10.4f %10.4f %10.1f%% %12.4f"
                      % (COND_LABEL[c].replace("\n", " "), CFG_SHORT[cf], v["auc_mean"],
                         v["auc_median"], 100 * v["frac_auc_gt08"], v["p_imp"]))
        print("-" * 112)

    # ---------- 保存 ----------
    out = dict(split=sp, conds=conds, table=tab, pulse_eff=pulse_eff, arch_adv=arch_adv,
               cond_delta={k: {c: v for c, v in d.items()} for k, d in cond_delta.items()},
               noise=noise_tab, pulse_diag=pulse_diag)
    with open(os.path.join(OUTD, "sens_summary_%s.json" % sp), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # 扁平 CSV：条件 × 配置 × 种子 × 三个关键指标（便于外部复核）
    rows = []
    for c in conds:
        for cf in CFG_ORDER:
            R = load(c, cf, sp)
            if R is None:
                continue
            for k in range(R.shape[0]):
                rows.append([c, cf, k, R.shape[1],
                             float(np.nanmean(R[k, :, I_CLN])),
                             float(np.nanmean(R[k, :, I_IMP])),
                             float(np.nanmean(R[k, :, I_MIN])),
                             float(np.nanmean(R[k, :, I_MED]))])
    with open(os.path.join(OUTD, "sens_table_%s.csv" % sp), "w", newline="",
              encoding="utf-8") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(["cond", "config", "seed_idx", "n_sample",
                    "frmgain_clean", "frmgain_imp", "frmgain_imp_min", "frmgain_med"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3]] + ["%.4f" % v for v in r[4:]])
    print("汇总 -> sens_out/sens_summary_%s.json" % sp)
    print("扁平表 -> sens_out/sens_table_%s.csv  (%d 行)" % (sp, len(rows)))


if __name__ == "__main__":
    main()
