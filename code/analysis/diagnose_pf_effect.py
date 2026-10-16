# -*- coding: utf-8 -*-
"""
诊断：脉冲分支（p 注入频域）的效应为什么在 testA 落在干扰帧、在 testC 落在干净帧？

方法：用 5 个随机种子的逐样本 CSV 做配对分析。
  - 逐样本差值 (pf - nopulse)，先在种子间平均（降低训练随机性）
  - 再按 SNR 档、信道、chirp 分层
  - 统计胜率、截尾均值（判断是否被少数样本主导）
  - 报 Bootstrap 置信区间
"""
import os, csv, glob, re
import numpy as np
import scipy.io as sio

OUT = r"D:/project/output_pamnet"
DS = r"D:/project/dataset_pam/alphastable"
COLS = ["snr_in", "snr_out", "sisdr_in", "sisdr_out", "fmsnr_med_in", "fmsnr_med_out",
        "frmgain_med", "frmgain_clean", "frmgain_imp", "frmgain_imp_min",
        "fmsnr_frac_bad_out", "frac_imp"]
IDX = {c: i for i, c in enumerate(COLS)}


def load(tag, split):
    p = os.path.join(OUT, "result_%s_%s.csv" % (tag, split))
    if not os.path.exists(p):
        return None
    a = np.loadtxt(p, delimiter=",", skiprows=1)
    return a if a.ndim == 2 else a.reshape(1, -1)


def seeds_of(prefix):
    """返回 (seed 号列表, 每个 seed 的 tag)。seed 1 = 无后缀。"""
    tags = sorted(glob.glob(os.path.join(OUT, "summary_%s*.csv" % prefix)))
    out = []
    for t in tags:
        base = os.path.basename(t)[8:-4]
        if not base.startswith(prefix):
            continue
        suf = base[len(prefix):]
        if suf == "":
            out.append((1, base))
        elif re.fullmatch(r"_s\d+", suf):
            out.append((int(suf[2:]), base))
    return sorted(out)


def meta(split):
    mt = sio.loadmat(os.path.join(DS, "meta_%s.mat" % split),
                     squeeze_me=True, struct_as_record=False)["meta"]
    return (np.array([mt[k].chirp_id for k in range(mt.size)]),
            np.array([mt[k].chan_id for k in range(mt.size)]),
            np.array([mt[k].snr for k in range(mt.size)]))


def paired_ci(d, n_boot=4000, seed=0):
    """配对差值的均值 + Bootstrap 95% CI + t 检验"""
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return None
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, n, size=(n_boot, n))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    se = d.std(ddof=1) / np.sqrt(n)
    t = d.mean() / se if se > 0 else 0.0
    return dict(n=n, mean=float(d.mean()), se=float(se), t=float(t),
                lo=float(lo), hi=float(hi),
                med=float(np.median(d)), win=float((d > 0).mean()))


def trimmed(d, q=0.05):
    d = np.sort(d[np.isfinite(d)])
    k = int(len(d) * q)
    return float(d[k:len(d) - k].mean()) if len(d) - 2 * k > 2 else float("nan")


A, B = "pamnet_freq_pf_gn", "pamnet_freq_nopulse_gn"
print("=" * 96)
print("配对诊断：%s  −  %s   （p 注入频域  vs  同参数无脉冲分支）" % (A, B))
print("=" * 96)

for split in ["testA", "testC"]:
    sa, sb = seeds_of(A), seeds_of(B)
    common = sorted(set(x[0] for x in sa) & set(x[0] for x in sb))
    print("\n### %s   （共同种子 %s，共 %d 个）" % (split, common, len(common)))
    if not common:
        continue
    da = {s: load(t, split) for s, t in sa if s in common}
    db = {s: load(t, split) for s, t in sb if s in common}
    if any(v is None for v in da.values()) or any(v is None for v in db.values()):
        print("  (缺数据)"); continue
    n = min(min(v.shape[0] for v in da.values()), min(v.shape[0] for v in db.values()))

    # 先在种子间平均每个指标的逐样本值
    Ma = np.mean([da[s][:n] for s in common], axis=0)
    Mb = np.mean([db[s][:n] for s in common], axis=0)
    D = Ma - Mb

    ci_, ca_, sn_ = meta(split)
    ci_, ca_, sn_ = ci_[:n], ca_[:n], sn_[:n]

    print("  %-14s %9s %9s %8s %9s %20s %8s %7s" %
          ("指标", "均值差", "标准误", "t", "截尾均值", "Bootstrap 95%CI", "胜率", "中位差"))
    for m in ["frmgain_clean", "frmgain_imp", "frmgain_imp_min", "frmgain_med"]:
        d = D[:, IDX[m]]
        r = paired_ci(d)
        if r is None:
            continue
        print("  %-14s %+9.3f %9.3f %8.2f %+9.3f   [%+7.3f, %+7.3f] %7.1f%% %+8.3f"
              % (m, r["mean"], r["se"], r["t"], trimmed(d), r["lo"], r["hi"],
                 r["win"] * 100, r["med"]))

    # ---- 按 SNR 分层 ----
    print("\n  按 SNR 档分层（frmgain_imp 的均值差）")
    print("    %-8s %6s %10s %10s %10s %10s" % ("SNR", "n", "干净帧", "干扰帧", "干扰最差", "中位指标"))
    for s in sorted(set(sn_.tolist())):
        m = sn_ == s
        row = []
        for k in ["frmgain_clean", "frmgain_imp", "frmgain_imp_min", "frmgain_med"]:
            v = D[m, IDX[k]]
            row.append(np.nanmean(v))
        print("    %-8g %6d %+10.3f %+10.3f %+10.3f %+10.3f" % (s, m.sum(), *row))

    # ---- 按信道分层 ----
    print("\n  按信道分层（干扰帧 / 干净帧 的均值差）")
    print("    %-8s %6s %10s %10s" % ("chan", "n", "干净帧", "干扰帧"))
    for c in sorted(set(ca_.tolist())):
        m = ca_ == c
        print("    %-8d %6d %+10.3f %+10.3f"
              % (c, m.sum(), np.nanmean(D[m, IDX["frmgain_clean"]]),
                 np.nanmean(D[m, IDX["frmgain_imp"]])))

    # ---- 按 chirp 分层 ----
    print("\n  按 chirp 分层（干扰帧 / 干净帧 的均值差）")
    for c in sorted(set(ci_.tolist())):
        m = ci_ == c
        print("    chirp %-3d n=%4d  干净帧 %+8.3f   干扰帧 %+8.3f"
              % (c, m.sum(), np.nanmean(D[m, IDX["frmgain_clean"]]),
                 np.nanmean(D[m, IDX["frmgain_imp"]])))

    # ---- 分布形状（干扰帧）----
    d = D[:, IDX["frmgain_imp"]]
    print("\n  干扰帧差值的分位数: " +
          "  ".join("%s%%=%+.2f" % (q, np.percentile(d, q))
                    for q in [1, 5, 25, 50, 75, 95, 99]))
    print("  最差 5 个样本的差值: %s" % np.round(np.sort(d)[:5], 2))
    print("  最好 5 个样本的差值: %s" % np.round(np.sort(d)[-5:], 2))
print()
