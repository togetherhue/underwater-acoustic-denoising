# -*- coding: utf-8 -*-
"""
输入尺度鲁棒性测试 —— 实测数据接入的前置条件
================================================================================
【为什么必须先测这个】
训练数据在 MATLAB 里做了 `clean_signal / rms(clean_signal)`，即干净信号 rms = 1。
但实测数据**没有干净参考**，无法做同样的归一化。若直接用 `noisy / rms(noisy)`，
则相当于把输入整体缩小约 rms(clean)/rms(noisy) 倍（base 条件下约 1/2.6）。

本测试回答：模型输出对输入整体尺度是否敏感？
  - 若不敏感（GroupNorm 带来尺度不变性）→ 实测数据可以直接按 noisy rms 归一化
  - 若敏感 → 必须从实测数据里估计信号 rms（如用噪声段 + 假定 SNR）

做法：取 base testA 的 120 个样本，把输入乘以一组缩放因子 s，
看逐帧降噪增益如何变化。s=1 为训练尺度。

用法：python scale_robustness.py
"""
import os, sys, json
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

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
SEG, DS = 8000, 4
NMAX = 120
# 缩放因子：1 为训练尺度；覆盖"按 noisy rms 归一化"可能造成的偏差范围
SCALES = [0.1, 0.2, 0.38, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0, 10.0]


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
    z = np.load(os.path.join(OUTD, "cache_base.npz"))
    noisy = z["testA_n"][:NMAX].astype(np.float32)
    clean = z["testA_c"][:NMAX].astype(np.float64)
    mask = z["testA_m"][:NMAX].astype(np.float64)
    print("样本 %d 个（base testA）" % len(noisy))

    # 训练时的典型尺度比：rms(noisy) / rms(clean)
    rc = np.sqrt((clean ** 2).mean(axis=1))
    rn = np.sqrt((noisy.astype(np.float64) ** 2).mean(axis=1))
    print("训练数据: rms(clean) 中位 %.3f, rms(noisy) 中位 %.3f, 比值中位 %.3f"
          % (np.median(rc), np.median(rn), np.median(rn / rc)))

    models = [OI.load_model(OI.R5_ARCH, c) for c in OI.R5_CKPTS]
    print("R5 模型 %d 个种子" % len(models))

    res = {}
    print()
    print("%-8s %14s %14s %14s %14s" % ("scale", "干净帧", "干扰帧", "干扰最差", "帧中位"))
    print("-" * 76)
    for s in SCALES:
        per = []
        for m in models:
            e = OI.frame_gains(m, noisy * np.float32(s))
            cl, im, mn, md = [], [], [], []
            for j in range(len(noisy)):
                st = PM._stats_1d(e[j], clean[j], (noisy[j] * s).astype(np.float64),
                                  PM.FRAME, mask[j])
                cl.append(st["frmgain_clean"]); im.append(st["frmgain_imp"])
                mn.append(st["frmgain_imp_min"]); md.append(st["frmgain_med"])
            per.append([np.nanmean(cl), np.nanmean(im), np.nanmean(mn), np.nanmean(md)])
        per = np.array(per)                    # [n_seed, 4]
        res["s%.2f" % s] = dict(scale=s, clean=float(per[:, 0].mean()),
                                imp=float(per[:, 1].mean()),
                                imp_min=float(per[:, 2].mean()),
                                med=float(per[:, 3].mean()),
                                sd_imp=float(per[:, 1].std(ddof=1)))
        print("%-8.2f %14.3f %14.3f %14.3f %14.3f"
              % (s, res["s%.2f" % s]["clean"], res["s%.2f" % s]["imp"],
                 res["s%.2f" % s]["imp_min"], res["s%.2f" % s]["med"]), flush=True)
    print("-" * 76)

    base = res["s1.00"]
    print()
    print("相对训练尺度 s=1 的退化（干扰帧）：")
    for s in SCALES:
        d = res["s%.2f" % s]["imp"] - base["imp"]
        flag = "" if abs(d) < 0.5 else ("  <-- 明显退化" if d < 0 else "  <-- 明显改善?")
        print("  s=%-5.2f  %+7.3f dB%s" % (s, d, flag))

    # 判定
    tol = 0.5
    ok = all(abs(res["s%.2f" % s]["imp"] - base["imp"]) < tol for s in SCALES)
    print()
    print("=" * 76)
    if ok:
        print("判定：**尺度不敏感**（全部缩放因子下退化 < %.1f dB）" % tol)
        print("→ 实测数据可以直接按 noisy 的 rms 归一化，无需估计信号 rms。")
    else:
        bad = [s for s in SCALES if abs(res["s%.2f" % s]["imp"] - base["imp"]) >= tol]
        print("判定：**尺度敏感**。退化超过 %.1f dB 的缩放因子：%s" % (tol, bad))
        print("→ 实测数据必须估计信号 rms（用噪声段 + 假定 SNR），不能简单按 noisy rms 归一化。")
    print("=" * 76)

    with open(os.path.join(OUTD, "scale_robustness.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("\n-> sens_out/scale_robustness.json")


if __name__ == "__main__":
    main()
