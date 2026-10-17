# -*- coding: utf-8 -*-
"""验证 _stats_batch 与逐样本 _stats_1d 逐位等价（base testA 全量 1080 样本）。"""
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDS = [os.path.normpath(os.path.join(_HERE, "..", "python")),
          r"C:\Users\32213\PycharmProjects\project1"]
for _p in [_HERE] + [c for c in _CANDS if os.path.isdir(c)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                      # noqa: E402
import pam_metrics as PM          # noqa: E402
import overlap_intervention as OI  # noqa: E402

WORK = r"E:/workbuddy/store/2026-10-12-22-50-08"
OUTD = os.path.join(WORK, "sens_out")
KEYS = ["fmsnr_med", "frmgain_med", "fmsnr_frac_bad", "frac_imp",
        "fmsnr_imp", "fmsnr_clean", "frmgain_imp", "frmgain_imp_min", "frmgain_clean"]


def main():
    z = np.load(os.path.join(OUTD, "cache_base.npz"))
    noisy = z["testA_n"].astype(np.float32)
    clean = z["testA_c"].astype(np.float64)
    mask = z["testA_m"].astype(np.float64)
    m5 = OI.load_model(OI.R5_ARCH, OI.R5_CKPTS[0])
    e = OI.frame_gains(m5, noisy)
    est = np.stack([np.asarray(x, dtype=np.float64) for x in e])
    ref = clean
    noi = noisy.astype(np.float64)
    print("样本 %d 个（base testA，R5 推理）" % len(est))

    # 逐样本（原实现）
    ref_loop = {k: [] for k in KEYS}
    for i in range(len(est)):
        st = PM._stats_1d(est[i], ref[i], noi[i], PM.FRAME, mask[i])
        for k in KEYS:
            ref_loop[k].append(st[k])
    ref_loop = {k: np.array(v, dtype=np.float64) for k, v in ref_loop.items()}

    # 批量（新实现）
    B = 64
    bat = {k: [] for k in KEYS}
    for s in range(0, len(est), B):
        out = PM._stats_batch(est[s:s + B], ref[s:s + B], noi[s:s + B], PM.FRAME,
                              mask[s:s + B])
        for k in KEYS:
            bat[k].append(out[k])
    bat = {k: np.concatenate(v) for k, v in bat.items()}

    print()
    print("%-16s %14s %14s %14s" % ("指标", "逐样本均值", "批量均值", "最大逐样本差"))
    print("-" * 64)
    worst = 0.0
    for k in KEYS:
        a, b = ref_loop[k], bat[k]
        both_nan = np.isnan(a) & np.isnan(b)
        d = np.where(both_nan, 0.0, np.abs(a - b))
        d = np.where(np.isnan(d), np.inf, d)
        mx = float(np.nanmax(d)) if np.isfinite(d).any() else float("nan")
        worst = max(worst, mx if np.isfinite(mx) else 0.0)
        print("%-16s %14.6f %14.6f %14.3e" % (k, np.nanmean(a), np.nanmean(b), mx))
    print("-" * 64)
    print("最大逐样本差 %.3e  ->  %s" % (worst, "✓ 逐位等价" if worst == 0.0
                                        else ("✓ 在浮点容差内" if worst < 1e-12 else "✗ 有实质差异")))

    # 速度对比
    import time
    t0 = time.time()
    for i in range(len(est)):
        PM._stats_1d(est[i], ref[i], noi[i], PM.FRAME, mask[i])
    t_loop = time.time() - t0
    t0 = time.time()
    for s in range(0, len(est), B):
        PM._stats_batch(est[s:s + B], ref[s:s + B], noi[s:s + B], PM.FRAME, mask[s:s + B])
    t_bat = time.time() - t0
    print()
    print("速度（%d 样本）：逐样本 %.2fs  批量 %.2fs  加速 %.1f×"
          % (len(est), t_loop, t_bat, t_loop / t_bat))


if __name__ == "__main__":
    main()
