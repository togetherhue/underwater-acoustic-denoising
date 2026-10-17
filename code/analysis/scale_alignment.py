# -*- coding: utf-8 -*-
"""
实测数据的尺度对齐：三种可行方案的定量比较
================================================================================
【已确认的约束】
1. 模型对输入尺度极度敏感，最优点尖锐落在训练尺度（按 noisy rms 归一化损失约 16 dB）
2. 四种无参考准则动态范围仅 0.07%~0.7%，盲搜索不可行
3. 带外能量外推估计噪声功率失效 —— α 稳定噪声是脉冲性的，带外功率被尖峰主导，
   与带宽不成比例（估计的 rms 偏小约 300 倍）

【本脚本比较三种真正可行的方案】
  A. 假定 SNR 与训练一致：s = median[rms(noisy)/rms(clean)]_train / rms(noisy)
  B. 有噪声段参考：从噪声段直接估 Pn，再 Pc = P_active − Pn
     （用数据集里的真实背景噪声 v 模拟"噪声段"；这是 B 的**上界**表现）
  C. 真值 s = 1（oracle，仅作参照）

用法：python scale_alignment.py
"""
import os, sys, json, glob
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


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
    z = np.load(os.path.join(OUTD, "cache_base.npz"))
    noisy = z["testA_n"][:NMAX].astype(np.float32)
    clean = z["testA_c"][:NMAX].astype(np.float64)
    mask = z["testA_m"][:NMAX].astype(np.float64)

    # 取真实背景噪声 v（用于方案 B 的"噪声段"）
    files = sorted(glob.glob(r"D:/project/dataset_pam/alphastable/testA/*.mat"))[:NMAX]
    V = []
    for f in files:
        m = sio.loadmat(f)
        n = np.asarray(m["noisy_signal"]).squeeze().astype(np.float32)
        c = np.asarray(m["clean_signal"]).squeeze().astype(np.float32)
        imp = np.asarray(m["imp"]).squeeze().astype(np.float32)
        n = resample_poly(n, 1, DS).astype(np.float32)
        c = resample_poly(c, 1, DS).astype(np.float32)
        imp = resample_poly(imp, 1, DS).astype(np.float32)
        fx = lambda x: x[:SEG] if len(x) >= SEG else np.pad(x, (0, SEG - len(x)))
        V.append(fx(n) - fx(c) - fx(imp))
    V = np.stack(V).astype(np.float64)

    rms_c = np.sqrt((clean ** 2).mean(axis=1))
    rms_n = np.sqrt((noisy.astype(np.float64) ** 2).mean(axis=1))
    ratio_train = float(np.median(rms_n / rms_c))
    print("训练集 rms(noisy)/rms(clean) 中位 = %.3f" % ratio_train)

    m5 = OI.load_model(OI.R5_ARCH, OI.R5_CKPTS[0])

    schemes = {}
    # A. 假定 SNR 与训练一致
    schemes["A_假定训练SNR"] = ratio_train / rms_n
    # B. 噪声段参考：Pn 从 v 估计（模拟真实录音的静默段）
    Pn = (V ** 2).mean(axis=1)
    P_tot = (noisy.astype(np.float64) ** 2).mean(axis=1)
    Pc = np.maximum(P_tot - Pn, 1e-3 * P_tot)
    schemes["B_噪声段参考"] = 1.0 / np.sqrt(Pc)
    # C. oracle
    schemes["C_真值(oracle)"] = np.ones(len(noisy))

    print()
    print("%-16s %12s %12s %10s %14s" % ("方案", "s 中位", "s 四分位距", "与真值偏差", "干扰帧增益"))
    print("-" * 74)
    res = {}
    for name, s in schemes.items():
        s = np.clip(np.asarray(s, dtype=np.float64), 0.05, 20.0)
        e = OI.frame_gains(m5, noisy * s[:, None].astype(np.float32))
        im = []
        for j in range(len(noisy)):
            st = PM._stats_1d(e[j], clean[j], (noisy[j] * s[j]).astype(np.float64),
                              PM.FRAME, mask[j])
            im.append(st["frmgain_imp"])
        val = float(np.nanmean(im))
        rel = float(np.median(np.abs(s - 1.0)))
        res[name] = dict(s_median=float(np.median(s)),
                         s_iqr=float(np.percentile(s, 75) - np.percentile(s, 25)),
                         dev_from_1=rel, imp=val)
        print("%-16s %12.3f %12.3f %10.3f %14.3f"
              % (name, np.median(s), np.percentile(s, 75) - np.percentile(s, 25),
                 rel, val), flush=True)
    print("-" * 74)

    oracle = res["C_真值(oracle)"]["imp"]
    print()
    print("相对 oracle 的损失：")
    for name, r in res.items():
        print("  %-16s %+.3f dB" % (name, r["imp"] - oracle))

    print()
    print("=" * 74)
    a = abs(res["A_假定训练SNR"]["imp"] - oracle)
    b = abs(res["B_噪声段参考"]["imp"] - oracle)
    print("  判定阈值：损失 < 1.0 dB 才算可用")
    print("  方案 A 损失 %.2f dB -> %s" % (a, "可用" if a < 1.0 else "**不可用**"))
    print("  方案 B 损失 %.2f dB -> %s" % (b, "可用" if b < 1.0 else "**不可用**"))
    if a >= 1.0 and b >= 1.0:
        print()
        print("→ **两种方案都不可用。** 原因：")
        print("   A 假定训练 SNR，但实测数据的 SNR 未知，s 的离散度（四分位距 %.2f）远大于容差；" % res["A_假定训练SNR"]["s_iqr"])
        print("   B 用噪声段估 Pn 后算 Pc = P_total − Pn，但 P_total 里还含【突发干扰】能量，" )
        print("     导致 Pc 被高估、s 被低估（中位 %.2f vs 真值 1.0）。" % res["B_噪声段参考"]["s_median"])
        print("→ 结论：**模型必须先加入输入归一化并重训**，才能在实测数据上可靠使用。")
    print("=" * 74)

    with open(os.path.join(OUTD, "scale_alignment.json"), "w", encoding="utf-8") as f:
        json.dump(dict(ratio_train=ratio_train, schemes=res, oracle=oracle), f,
                  ensure_ascii=False, indent=1)
    print("\n-> sens_out/scale_alignment.json")


if __name__ == "__main__":
    main()
