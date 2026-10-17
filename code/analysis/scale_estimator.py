# -*- coding: utf-8 -*-
"""
输入尺度的物理估计器 —— 实测数据部署的关键一环
================================================================================
【问题】
scale_robustness.py：模型对输入整体尺度极度敏感（最优点尖锐落在训练尺度 s=1，
按 noisy rms 归一化会损失约 16 dB）。
scale_autocalib.py：四种无参考准则的动态范围都只有 0.07%~0.7%，
**无法**用盲搜索定位最优尺度。

【因此只能用物理先验估计】
训练数据里 rms(clean) = 1（MATLAB 侧做的归一化）。所以部署时要估计 rms(clean)，
把输入乘 1/rms_est(clean)。

信号是窄带 chirp（798–1500 Hz），背景噪声是宽带的。于是：
    带外（1600–2900 Hz）能量  →  噪声功率谱密度  →  全带噪声功率 Pn
    总功率 P_total            →  Pc = P_total − Pn
    rms_est(clean) = sqrt(Pc)

本脚本检验这个估计器的精度：把估计出的 s_est = 1/rms_est(clean) 与真值 1 比较，
并直接测"用 s_est 推理"相对"用真值 s=1 推理"损失多少 dB。

【注意】α 稳定分布的二阶矩在 α<2 时无穷，带外功率估计会低估噪声的瞬时功率。
因此同时给出"用 Pn 的中位数"与"用带外功率按带宽比外推"两种变体。

用法：python scale_estimator.py
"""
import os, sys, json
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
FS = 6000
NMAX = 120
# 信号带 798-1500；取 1600-2900 作纯噪声带（与谱减/维纳基线用的频带一致）
SIG_BAND = (798.0, 1500.0)
NOI_BAND = (1600.0, 2900.0)


def band_power(x, lo, hi):
    E = np.abs(np.fft.rfft(x)) ** 2 / len(x)
    f = np.fft.rfftfreq(len(x), 1 / FS)
    sel = (f >= lo) & (f <= hi)
    return float(E[sel].sum()), float(sel.sum())


def est_signal_rms(x, mode="extrapolate"):
    """估计信号分量的 rms（训练数据里该值为 1）"""
    P_tot = float(np.mean(x ** 2))
    p_noi, n_noi = band_power(x, *NOI_BAND)
    df = FS / 2.0 / (len(x) / 2.0 + 1)        # 每 bin 带宽
    if mode == "extrapolate":
        # 假设噪声谱在带内近似平坦：把带外功率密度按带宽外推到全带
        psd = p_noi / max(1e-30, n_noi * df)
        Pn = psd * (FS / 2.0)
    else:                                      # median：用带内噪声底估计
        Pn = p_noi / max(1e-30, n_noi) * (FS / 2.0)
    Pc = max(P_tot - Pn, 1e-6 * P_tot)
    return float(np.sqrt(Pc)), Pn, P_tot


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
    z = np.load(os.path.join(OUTD, "cache_base.npz"))
    noisy = z["testA_n"][:NMAX].astype(np.float32)
    clean = z["testA_c"][:NMAX].astype(np.float64)
    mask = z["testA_m"][:NMAX].astype(np.float64)
    m5 = OI.load_model(OI.R5_ARCH, OI.R5_CKPTS[0])

    true_rms = np.sqrt((clean ** 2).mean(axis=1))
    print("真值 rms(clean)：中位 %.3f（训练时被归一化为 1）" % np.median(true_rms))
    print()

    out = {}
    for mode in ("extrapolate", "median"):
        est, pn, pt = [], [], []
        for i in range(len(noisy)):
            r, a, b = est_signal_rms(noisy[i].astype(np.float64), mode)
            est.append(r); pn.append(a); pt.append(b)
        est = np.array(est)
        # 估计的尺度因子：要让 rms(clean) 变成 1，需乘 1/rms_est
        s_est = 1.0 / np.maximum(est, 1e-6)
        s_est = np.clip(s_est, 0.05, 20.0)
        print("=== 估计器 mode=%s ===" % mode)
        print("  rms_est 中位 %.3f  (真值中位 %.3f)  比值中位 %.3f"
              % (np.median(est), np.median(true_rms),
                 np.median(est / np.maximum(true_rms, 1e-9))))
        print("  s_est 中位 %.3f  四分位 [%.2f, %.2f]"
              % (np.median(s_est), np.percentile(s_est, 25), np.percentile(s_est, 75)))

        # 用 s_est 逐样本推理
        e = OI.frame_gains(m5, noisy * s_est[:, None].astype(np.float32))
        im, im1 = [], []
        e1 = OI.frame_gains(m5, noisy)          # 真值 s=1
        for j in range(len(noisy)):
            n_s = (noisy[j] * s_est[j]).astype(np.float64)
            st = PM._stats_1d(e[j], clean[j], n_s, PM.FRAME, mask[j])
            im.append(st["frmgain_imp"])
            st1 = PM._stats_1d(e1[j], clean[j], noisy[j].astype(np.float64),
                               PM.FRAME, mask[j])
            im1.append(st1["frmgain_imp"])
        a_, b_ = float(np.nanmean(im)), float(np.nanmean(im1))
        print("  用 s_est 推理: 干扰帧 %+.3f dB | 用真值 s=1: %+.3f dB | 损失 %.3f dB"
              % (a_, b_, b_ - a_))
        print()
        out[mode] = dict(rms_est_median=float(np.median(est)),
                         ratio_median=float(np.median(est / np.maximum(true_rms, 1e-9))),
                         s_est_median=float(np.median(s_est)),
                         s_est_q25=float(np.percentile(s_est, 25)),
                         s_est_q75=float(np.percentile(s_est, 75)),
                         imp_with_est=a_, imp_true=b_, loss=b_ - a_)

    print("=" * 78)
    print("结论")
    print("=" * 78)
    best = min(out, key=lambda k: out[k]["loss"])
    for k, v in out.items():
        print("  %-12s 估计/真值 = %.3f，用估计尺度的损失 %.2f dB" % (k, v["ratio_median"], v["loss"]))
    print()
    if out[best]["loss"] < 1.0:
        print("→ 估计器 %s 可用（损失 < 1 dB），可作为实测数据部署的默认方案。" % best)
    else:
        print("→ 估计器仍不可用（最优 %s 损失 %.2f dB）。" % (best, out[best]["loss"]))
        print("  必须改用：(a) 实测数据提供噪声段以直接估计 Pn；或")
        print("            (b) 在模型内加入输入归一化层并重训（根本解法）。")

    with open(os.path.join(OUTD, "scale_estimator.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n-> sens_out/scale_estimator.json")


if __name__ == "__main__":
    main()
