# -*- coding: utf-8 -*-
"""
等有效功率 SNR 的 α 重扫：分析
================================================================================
补齐"适用边界"实验里唯一的方法论缺口 —— 把 α（尾形）与噪声强度解耦。

问题：名义 SNR 用 75% 分位数匹配定义，α<2 时方差无穷，故有效功率 SNR 随 α 漂移：
    α=1.5 : +6.71 dB   （base）
    α=1.2 : −6.36 dB   → 同一名义 SNR 下有效噪声功率高 13.1 dB
    α=1.8 : +16.71 dB  → 低 10.0 dB
因此未匹配的 α 扫描同时变了两个因素，其结论无法归因。

本脚本的匹配条件（m_a12 / m_a18）把名义 SNR 平移对应量，使有效功率 SNR 与 base
一一对应；样本序号、底层均匀随机数、突发实现全部与 base 一致，只有背景噪声尾形不同。

要回答的问题：
    未匹配时"α 越小脉冲分支收益越小"的现象，是【尾形】造成的还是【强度】造成的？

用法：python analyze_matched.py [--split testA]
"""
import os, sys, json, argparse
import numpy as np

WORK = r"E:/workbuddy/store/2026-10-12-22-50-08"
OUTD = os.path.join(WORK, "sens_out")

I_CLN, I_IMP, I_MIN = 7, 8, 9
COLS = ["snr_in", "snr_out", "sisdr_in", "sisdr_out",
        "fmsnr_med_in", "fmsnr_med_out", "frmgain_med", "frmgain_clean",
        "frmgain_imp", "frmgain_imp_min", "fmsnr_frac_bad_out", "frac_imp"]

LBL = {"base": "α=1.5（base，未匹配）",
       "a12_r5": "α=1.2（未匹配，有效噪声高 13.1 dB）",
       "a18_r5": "α=1.8（未匹配，有效噪声低 10.0 dB）",
       "m_a12": "α=1.2（已匹配有效功率 SNR）",
       "m_a18": "α=1.8（已匹配有效功率 SNR）"}
ORDER = ["a12_r5", "m_a12", "base", "m_a18", "a18_r5"]


def load(cond, cfg, split):
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


def paired_stats(a, b):
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    ok = np.isfinite(d)
    d = d[ok]
    n = len(d)
    if n < 5:
        return dict(n=n, mean=float("nan"), sd=float("nan"), t=float("nan"), p=float("nan"))
    m, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    t = m / se if se > 0 else float("nan")
    from math import erfc, sqrt
    return dict(n=n, mean=m, sd=sd, se=float(se), t=float(t),
                p=float(erfc(abs(t) / sqrt(2.0))))


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


def pulse_effect(cond, split):
    """Δ = R5 − R4（同种子内逐样本配对，再跨种子汇总）"""
    A5, A4 = load(cond, "R5_freqrefine_pulse", split), load(cond, "R4_freqrefine_nopulse", split)
    if A5 is None or A4 is None:
        return None
    ns = min(A5.shape[0], A4.shape[0])
    out = {}
    for key, idx in (("clean", I_CLN), ("imp", I_IMP), ("imp_min", I_MIN)):
        ms = [paired_stats(A5[k, :, idx], A4[k, :, idx])["mean"] for k in range(ns)]
        out[key] = cross_seed_test(ms)
    out["n_seed"] = ns
    return out


def arch_adv(cond, split):
    """Δ = R5 − R3"""
    A5, A3 = load(cond, "R5_freqrefine_pulse", split), load(cond, "R3_freqonly", split)
    if A5 is None or A3 is None:
        return None
    ns = min(A5.shape[0], A3.shape[0])
    out = {}
    for key, idx in (("clean", I_CLN), ("imp", I_IMP)):
        ms = [paired_stats(A5[k, :, idx], A3[k, :, idx])["mean"] for k in range(ns)]
        out[key] = cross_seed_test(ms)
    out["n_seed"] = ns
    return out


def noise_check(split="testA"):
    """校验匹配条件的有效功率 SNR 是否真的与 base 对齐"""
    print()
    print("=" * 100)
    print("第 1 步  校验：匹配条件的有效功率 SNR 是否与 base 对齐")
    print("=" * 100)
    print("%-40s %14s %14s %14s" % ("条件", "背景有效SNR中位", "干扰有效SNR中位", "总有效SNR中位"))
    print("-" * 100)
    base = None
    rows = {}
    for c in ["base", "a12_r5", "m_a12", "m_a18", "a18_r5"]:
        p = os.path.join(OUTD, "noise_%s.npz" % c)
        if not os.path.exists(p):
            continue
        a = np.load(p)[split]
        r = dict(bg=float(np.median(a[:, 3])), imp=float(np.median(a[:, 4])),
                 tot=float(np.median(a[:, 5])))
        rows[c] = r
        if c == "base":
            base = r
        tag = ""
        if base and c.startswith("m_"):
            tag = "   (与 base 差 %+.2f dB)" % (r["bg"] - base["bg"])
        print("%-40s %14.2f %14.2f %14.2f%s" % (LBL[c], r["bg"], r["imp"], r["tot"], tag))
    print("-" * 100)
    if base:
        for c in ["m_a12", "m_a18"]:
            if c in rows:
                d = rows[c]["bg"] - base["bg"]
                ok = "✓ 对齐" if abs(d) < 1.0 else "✗ 未对齐（差 %+.2f dB）" % d
                print("  %-12s 背景有效功率 SNR 与 base 差 %+.2f dB  -> %s"
                      % (c, d, ok))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="testA")
    A = ap.parse_args()
    sp = A.split
    rows = noise_check(sp)

    print()
    print("=" * 100)
    print("第 2 步  脉冲感知分支效应 Δ = R5 − R4：匹配 vs 未匹配（%s）" % sp)
    print("=" * 100)
    print("%-40s %20s %20s" % ("条件", "干扰帧 Δ (dB)", "跨种子 p"))
    print("-" * 100)
    res = {}
    for c in ORDER:
        pe = pulse_effect(c, sp)
        if pe is None:
            continue
        # 留一法：检查均值是否被单个种子主导
        A5, A4 = load(c, "R5_freqrefine_pulse", sp), load(c, "R4_freqrefine_nopulse", sp)
        ns = min(A5.shape[0], A4.shape[0])
        ms = [paired_stats(A5[k, :, I_IMP], A4[k, :, I_IMP])["mean"] for k in range(ns)]
        loo = [float(np.mean([v for j, v in enumerate(ms) if j != i]))
               for i in range(len(ms))]
        pe["imp"]["loo_min"] = min(loo) if loo else float("nan")
        pe["imp"]["loo_max"] = max(loo) if loo else float("nan")
        pe["imp"]["per_seed"] = ms
        res[c] = pe
        print("%-40s %20s %20s" % (LBL[c], "%+.3f ± %.3f" % (pe["imp"]["mean"], pe["imp"]["sd"]),
                                   "%.4f" % pe["imp"]["p"]))
    print("-" * 100)
    print("（留一法范围：去掉任一单个种子后的均值，用于检查是否被个别种子主导）")
    for c in ORDER:
        if c in res:
            pe = res[c]["imp"]
            print("  %-40s 均值 %+.3f  留一法 [%+.3f, %+.3f]%s"
                  % (LBL[c], pe["mean"], pe["loo_min"], pe["loo_max"],
                     "   ← 有种子影响 >1 dB" if (pe["loo_max"] - pe["loo_min"]) > 1.0 else ""))

    print()
    print("=" * 100)
    print("第 3 步  架构优势 Δ = R5 − R3：匹配 vs 未匹配（%s）" % sp)
    print("=" * 100)
    print("%-40s %20s %20s" % ("条件", "干扰帧 Δ (dB)", "跨种子 p"))
    print("-" * 100)
    resa = {}
    for c in ORDER:
        aa = arch_adv(c, sp)
        if aa is None:
            continue
        resa[c] = aa
        print("%-40s %20s %20s" % (LBL[c], "%+.3f ± %.3f" % (aa["imp"]["mean"], aa["imp"]["sd"]),
                                   "%.2e" % aa["imp"]["p"]))
    print("-" * 100)

    # ---------- 结论 ----------
    print()
    print("=" * 100)
    print("第 4 步  结论：α 的效应是【尾形】还是【强度】造成的？")
    print("=" * 100)
    if "base" in res and "m_a12" in res and "a12_r5" in res:
        b, um, m = res["base"]["imp"]["mean"], res["a12_r5"]["imp"]["mean"], res["m_a12"]["imp"]["mean"]
        print("  α=1.2 方向（未匹配时效应被压低）：")
        print("    未匹配 α=1.2 : Δ = %+.3f dB  （有效噪声比 base 高 13.1 dB）" % um)
        print("    已匹配 α=1.2 : Δ = %+.3f dB  （有效噪声与 base 相同）" % m)
        print("    base (α=1.5) : Δ = %+.3f dB" % b)
        gap_um, gap_m = um - b, m - b
        print("    -> 未匹配时相对 base 的变化：%+.3f dB；匹配后：%+.3f dB" % (gap_um, gap_m))
        print("    -> 强度成分 %+.3f dB，尾形成分 %+.3f dB"
              % (gap_um - gap_m, gap_m))
        if abs(gap_um) > 1e-9:
            print("    -> **判定：两者兼有**（强度解释 %.0f%%，尾形解释 %.0f%%）。"
                  % (100 * (1 - abs(gap_m) / abs(gap_um)), 100 * abs(gap_m) / abs(gap_um)))
    if "base" in res and "m_a18" in res and "a18_r5" in res:
        b, um, m = res["base"]["imp"]["mean"], res["a18_r5"]["imp"]["mean"], res["m_a18"]["imp"]["mean"]
        print()
        print("  α=1.8 方向（未匹配时效应被抬高）：")
        print("    未匹配 α=1.8 : Δ = %+.3f dB  （有效噪声比 base 低 10.0 dB）" % um)
        print("    已匹配 α=1.8 : Δ = %+.3f dB  （有效噪声与 base 相同）" % m)
        print("    base (α=1.5) : Δ = %+.3f dB" % b)
        print("    -> 未匹配时相对 base 的变化：%+.3f dB；匹配后：%+.3f dB" % (um - b, m - b))
        print("    -> 强度成分 %+.3f dB，尾形成分 %+.3f dB"
              % ((um - b) - (m - b), m - b))
        print("    （α=1.8 方向强度成分与尾形成分【符号相反】：匹配掉多余的有效信噪比后，"
              "尾形本身的增强作用反而更明显）")

    with open(os.path.join(OUTD, "matched_%s.json" % sp), "w", encoding="utf-8") as f:
        json.dump(dict(split=sp, noise=rows, pulse_eff=res, arch_adv=resa), f,
                  ensure_ascii=False, indent=1)
    print("\n-> sens_out/matched_%s.json" % sp)


if __name__ == "__main__":
    main()
