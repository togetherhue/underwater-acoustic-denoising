# -*- coding: utf-8 -*-
"""
尺度自校准：能否用【无参考指标】自动找到正确的输入尺度？
================================================================================
【背景】
scale_robustness.py 证明模型对输入整体尺度极度敏感（最优点尖锐地落在训练尺度），
而实测数据没有干净参考、无法直接算出"该乘多少"。
若存在一个**无参考**（不需要 clean）的准则，其最优点也落在正确尺度上，
就能用它做部署时的自动校准。

【候选准则】（都只用 noisy 与模型输出 y）
  1. 带内能量占比   E_inband(y) / E_total(y)      —— 去噪后应更集中到信号频带
  2. 峰度下降       kurt(noisy) − kurt(y)          —— 去噪应降低脉冲性
  3. 带外抑制       E_outband(noisy) − E_outband(y) —— 带外噪声应被压掉
  4. 残差白性       std(PSD(y−noisy)) / mean(...)   —— 残差应接近白噪声

【判据】哪个准则的峰值位置与"真实最优尺度"一致，就可以用于实测校准。

用法：python scale_autocalib.py
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
BAND = (780.0, 1520.0)
NMAX = 100
SCALES = [0.25, 0.38, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0]


def band_frac(x, lo, hi):
    E = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / FS)
    tot = E.sum()
    return float(E[(f >= lo) & (f <= hi)].sum() / tot) if tot > 0 else float("nan")


def kurt(x):
    x = x - x.mean()
    s2 = (x ** 2).mean()
    return float(((x ** 4).mean()) / (s2 ** 2 + 1e-30))


def resid_whiteness(x):
    """残差谱的平坦度（std/mean，越小越白）"""
    E = np.abs(np.fft.rfft(x)) ** 2
    E = E[1:]
    m = E.mean()
    return float(E.std() / (m + 1e-30))


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
    z = np.load(os.path.join(OUTD, "cache_base.npz"))
    noisy = z["testA_n"][:NMAX].astype(np.float32)
    clean = z["testA_c"][:NMAX].astype(np.float64)
    mask = z["testA_m"][:NMAX].astype(np.float64)
    m5 = OI.load_model(OI.R5_ARCH, OI.R5_CKPTS[0])

    print("样本 %d 个，单种子（用于准则形状分析）" % len(noisy))
    print()
    print("%-7s %10s | %-9s %-9s %-10s %-9s"
          % ("scale", "真实干扰帧", "带内占比", "峰度下降", "带外抑制", "残差白性"))
    print("-" * 74)
    rows = []
    for s in SCALES:
        e = OI.frame_gains(m5, noisy * np.float32(s))
        im, c1, c2, c3, c4 = [], [], [], [], []
        for j in range(len(noisy)):
            st = PM._stats_1d(e[j], clean[j], (noisy[j] * s).astype(np.float64),
                              PM.FRAME, mask[j])
            im.append(st["frmgain_imp"])
            y = e[j]
            n = noisy[j] * s
            c1.append(band_frac(y, *BAND))
            c2.append(kurt(n) - kurt(y))
            c3.append(band_frac(n, 1600, 2900) - band_frac(y, 1600, 2900))
            c4.append(resid_whiteness(y - n))
        r = dict(scale=s, imp=float(np.nanmean(im)),
                 band=float(np.nanmean(c1)), kurt=float(np.nanmean(c2)),
                 outband=float(np.nanmean(c3)), white=float(np.nanmean(c4)))
        rows.append(r)
        print("%-7.2f %10.3f | %-9.4f %-9.2f %-10.4f %-9.3f"
              % (s, r["imp"], r["band"], r["kurt"], r["outband"], r["white"]), flush=True)
    print("-" * 74)

    # 各准则的峰值位置
    print()
    print("=" * 74)
    print("各准则的最优尺度 vs 真实最优尺度")
    print("=" * 74)
    best_true = max(rows, key=lambda r: r["imp"])["scale"]
    print("真实最优尺度（按干扰帧增益）：s = %.2f" % best_true)
    print()
    verdict = {}
    for key, name, mode in (("band", "带内能量占比", "max"), ("kurt", "峰度下降", "max"),
                            ("outband", "带外抑制", "max"), ("white", "残差白性", "min")):
        vals = [r[key] for r in rows]
        if mode == "max":
            i = int(np.argmax(vals))
        else:
            i = int(np.argmin(vals))
        ok = abs(rows[i]["scale"] - best_true) <= 0.3
        verdict[key] = dict(best_scale=rows[i]["scale"], ok=ok, mode=mode)
        print("  %-12s 峰值在 s=%-5.2f  %s" % (name, rows[i]["scale"],
                                              "✓ 与真实最优一致" if ok else "✗ 不一致"))
    print("=" * 74)

    with open(os.path.join(OUTD, "scale_autocalib.json"), "w", encoding="utf-8") as f:
        json.dump(dict(rows=rows, best_true=best_true, verdict=verdict), f,
                  ensure_ascii=False, indent=1)
    print("\n-> sens_out/scale_autocalib.json")


if __name__ == "__main__":
    main()
