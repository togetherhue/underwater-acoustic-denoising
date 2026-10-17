# -*- coding: utf-8 -*-
"""
经典基线在全部条件上的补齐
================================================================================
目的：报告的头条结论「PAM-Net 远超全部经典方法」此前只在 base 条件上测过。
本节把同样的经典基线放到 7 个条件上，检验该结论是否跨条件成立。

【口径一致性】直接复用 sens_out/cache_*.npz —— 与神经模型评估用的是
【完全相同的输入数组、相同的降采样/定长/掩码口径、相同的指标代码】，
因此两者是严格配对的，不存在任何实现差异。

校验：先在 base 的前 200 个样本上重算，与 figs_pamnet/baseline_pam.json
（由 compare_baseline_pam.py 用 N=200 产出）对比，确认口径无误后再跑全量。

用法：
  python baseline_sens.py --verify      # 口径校验（前 200 样本 vs 存档）
  python baseline_sens.py               # 全量：7 条件 × 4 方法
"""
import os, sys, json, argparse
import numpy as np
from scipy.signal import medfilt, stft, istft

WORK = r"E:/workbuddy/store/2026-10-12-22-50-08"
OUTD = os.path.join(WORK, "sens_out")
FIGD = os.path.join(WORK, "figs_pamnet")

# ---- 路径解析：兼容仓库布局与本地开发布局 ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDS = [os.path.normpath(os.path.join(_HERE, "..", "python")),
          r"C:\Users\32213\PycharmProjects\project1"]
for _p in [_HERE] + [c for c in _CANDS if os.path.isdir(c)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pam_metrics as PM       # noqa: E402

SEG, DS, FS = 8000, 4, 6000     # 与 pam_metrics / eval_sensitivity 完全一致
CONDS = ["base", "a12_r5", "a18_r5", "a20_r5", "a15_r25", "a15_r10", "gauss",
         "m_a12", "m_a18"]
SPLITS = ["testA", "testC"]


# ============================ 经典方法（与 compare_baseline_pam.py 逐字一致）============================
def ideal_bandpass(x, lo=798, hi=1500, fs=FS):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, n=len(x))


def spec_sub(x, fs=FS, n_fft=512, hop=128, alpha=2.0, beta=0.02):
    f, t, Z = stft(x, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
    mag, ph = np.abs(Z), np.angle(Z)
    band = (f >= 1600) & (f <= 2900)          # 纯噪声频带
    nse = mag[band].mean(axis=0, keepdims=True) if band.any() else mag.min(axis=0, keepdims=True)
    nse = np.repeat(nse, mag.shape[0], axis=0)
    m = np.maximum(mag - alpha * nse, beta * mag)
    _, y = istft(m * np.exp(1j * ph), fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
    return y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


def wiener(x, fs=FS, n_fft=512, hop=128, beta=0.02):
    f, t, Z = stft(x, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
    mag, ph = np.abs(Z), np.angle(Z)
    band = (f >= 1600) & (f <= 2900)
    nse = mag[band].mean(axis=0, keepdims=True) if band.any() else mag.min(axis=0, keepdims=True)
    nse = np.repeat(nse, mag.shape[0], axis=0)
    g = np.maximum(1.0 - (nse ** 2) / (mag ** 2 + 1e-12), beta)
    _, y = istft((mag * g) * np.exp(1j * ph), fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
    return y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


def median_filt(x, k=5):
    return medfilt(x, kernel_size=k)


METHODS = [("理想带通 798-1500Hz", ideal_bandpass),
           ("谱减法（盲）", spec_sub),
           ("维纳滤波（盲）", wiener),
           ("中值滤波 k=5", median_filt)]


def stats_for(y, c, n, m):
    """与 run_eval / eval_cached 完全相同的指标计算"""
    st = PM._stats_1d(y, c, n, PM.FRAME, m)
    return st


def run_cond(cond, split, nmax=0):
    """对一个条件跑全部经典方法，返回 {方法: 指标}"""
    p = os.path.join(OUTD, "cache_%s.npz" % cond)
    if not os.path.exists(p):
        return None
    z = np.load(p)
    noisy, clean, mask = z["%s_n" % split], z["%s_c" % split], z["%s_m" % split]
    k = len(noisy) if nmax <= 0 else min(nmax, len(noisy))
    out = {}
    for name, fn in METHODS:
        cl, im, mn, med, snrs = [], [], [], [], []
        for i in range(k):
            n = noisy[i].astype(np.float64)
            c = clean[i].astype(np.float64)
            m = mask[i].astype(float)
            y = fn(n)
            st = stats_for(y, c, n, m)
            cl.append(st["frmgain_clean"]); im.append(st["frmgain_imp"])
            mn.append(st["frmgain_imp_min"]); med.append(st["frmgain_med"])
            snrs.append(float(10 * np.log10((c ** 2).sum() /
                                            (((y - c) ** 2).sum() + 1e-12))))
        out[name] = dict(snr=float(np.nanmean(snrs)),
                         frmgain_clean=float(np.nanmean(cl)),
                         frmgain_imp=float(np.nanmean(im)),
                         frmgain_imp_min=float(np.nanmean(mn)),
                         frmgain_med=float(np.nanmean(med)), n=int(k))
    return out


def verify():
    """口径校验：base 前 200 样本 vs figs_pamnet/baseline_pam.json"""
    ref_p = os.path.join(FIGD, "baseline_pam.json")
    if not os.path.exists(ref_p):
        print("无存档可比（baseline_pam.json 不存在），跳过校验")
        return True
    ref = json.load(open(ref_p, encoding="utf-8"))["testA"]
    new = run_cond("base", "testA", nmax=200)
    print("=" * 92)
    print("口径校验：base 前 200 样本，新脚本 vs 存档 compare_baseline_pam.py")
    print("=" * 92)
    print("%-22s %10s %10s %10s %10s" % ("方法", "存档干扰帧", "新算干扰帧", "差", "存档最差"))
    print("-" * 92)
    ok = True
    for name, _ in METHODS:
        if name not in ref:
            continue
        r, n2 = ref[name], new[name]
        d = abs(r["frmgain_imp"] - n2["frmgain_imp"])
        good = d < 0.05
        ok = ok and good
        print("%-22s %10.3f %10.3f %10.4f %10.3f  %s"
              % (name, r["frmgain_imp"], n2["frmgain_imp"], d, r["frmgain_imp_min"],
                 "✓" if good else "✗"))
    print("-" * 92)
    print("校验结果：%s（判据：干扰帧差 < 0.05 dB）" % ("**通过**" if ok else "**不一致**"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--split", default="testA")
    A = ap.parse_args()
    if A.verify:
        verify(); return

    sp = A.split
    res = {}
    for c in CONDS:
        r = run_cond(c, sp)
        if r is None:
            print("跳过 %s（无缓存）" % c); continue
        res[c] = r
        bp = r.get("理想带通 798-1500Hz", {})
        print("[%s] 理想带通 干扰帧 %+.2f  谱减 %+.2f  维纳 %+.2f  中值 %+.2f"
              % (c, bp.get("frmgain_imp", float("nan")),
                 r["谱减法（盲）"]["frmgain_imp"], r["维纳滤波（盲）"]["frmgain_imp"],
                 r["中值滤波 k=5"]["frmgain_imp"]), flush=True)

    with open(os.path.join(OUTD, "baseline_sens_%s.json" % sp), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("\n-> sens_out/baseline_sens_%s.json" % sp)


if __name__ == "__main__":
    main()
