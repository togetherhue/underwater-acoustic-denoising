# -*- coding: utf-8 -*-
"""
脉冲噪声数据集上的经典基线（不含任何学习成分）
================================================================================
目的：回答一个关键问题 —— PAM-Net 拿到的增益里，有多少只是"带限"带来的？

信号是窄带 chirp（798~1500 Hz），而 α 稳定背景噪声与突发脉冲都是宽带的。
因此一个固定的带通滤波器就能滤掉大量噪声。若模型打不过固定带通，
说明它的增益主要来自"学会了带限"，而非脉冲感知。

方法：
  1. 理想带通 798~1500 Hz（已知信号频带，属"作弊"上界）
  2. 谱减法（盲噪声估计，用 1500~3000 Hz 纯噪声频带估噪）
  3. 维纳滤波（盲估计）
  4. 中值滤波（对脉冲最经典的时域对策）

评估口径与 PAM-Net 完全一致（逐帧降噪增益，按干扰掩码分解）。
"""
import os, sys, glob
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly, stft, istft, medfilt

# ---- 路径解析：兼容仓库布局（库模块在 ../python）与本地开发布局 ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDS = [os.path.normpath(os.path.join(_HERE, "..", "python")),
          r"C:\Users\32213\PycharmProjects\project1"]
for _p in [_HERE] + [c for c in _CANDS if os.path.isdir(c)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pam_metrics as PM       # noqa: E402

DATA = r"E:/project/data/dataset_pam/alphastable"
SEG, DS, FS = 8000, 4, 6000
N = 200


def load(f):
    m = sio.loadmat(f)
    n = np.asarray(m["noisy_signal"]).squeeze().astype(np.float64)
    c = np.asarray(m["clean_signal"]).squeeze().astype(np.float64)
    k = np.asarray(m["imp_mask"]).squeeze().astype(np.float64)
    n, c, k = resample_poly(n, 1, DS), resample_poly(c, 1, DS), resample_poly(k, 1, DS)
    def fx(x):
        return x[:SEG] if len(x) >= SEG else np.pad(x, (0, SEG - len(x)))
    return fx(n), fx(c), (fx(k) > 0.3).astype(np.float64)


# ---------------- 基线 ----------------
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
    return (y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y))))


def wiener(x, fs=FS, n_fft=512, hop=128, beta=0.02):
    f, t, Z = stft(x, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
    mag, ph = np.abs(Z), np.angle(Z)
    band = (f >= 1600) & (f <= 2900)
    nse = mag[band].mean(axis=0, keepdims=True) if band.any() else mag.min(axis=0, keepdims=True)
    nse = np.repeat(nse, mag.shape[0], axis=0)
    g = np.maximum(1.0 - (alpha_ratio := (nse ** 2) / (mag ** 2 + 1e-12)), beta)
    _, y = istft((mag * g) * np.exp(1j * ph), fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
    return (y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y))))


def median_filt(x, k=5):
    return medfilt(x, kernel_size=k)


METHODS = [("含噪输入", lambda x: x),
           ("理想带通 798-1500Hz", ideal_bandpass),
           ("谱减法（盲）", spec_sub),
           ("维纳滤波（盲）", wiener),
           ("中值滤波 k=5", median_filt)]

import json
RESULTS = {}
for split in ["testA", "testC"]:
    files = sorted(glob.glob(os.path.join(DATA, split, "*.mat")))[:N]
    print()
    print("=" * 104)
    print("经典基线（%s，%d 个样本）—— 逐帧降噪增益，口径与 PAM-Net 完全一致" % (split, len(files)))
    print("=" * 104)
    print("%-22s %9s %11s %11s %11s %11s" %
          ("方法", "全局SNR", "帧增益中位", "干净帧", "干扰帧", "干扰帧最差"))
    print("-" * 104)
    RESULTS[split] = {}
    for name, fn in METHODS:
        acc = {"med": [], "cl": [], "im": [], "imn": []}
        snrs = []
        for f in files:
            n, c, k = load(f)
            try:
                y = fn(n)
            except Exception as e:
                print("  %-22s 失败: %s" % (name, e)); break
            st = PM._stats_1d(y, c, n, PM.FRAME, k)
            acc["med"].append(st["frmgain_med"])
            acc["cl"].append(st["frmgain_clean"])
            acc["im"].append(st["frmgain_imp"])
            acc["imn"].append(st["frmgain_imp_min"])
            snrs.append(float(10 * np.log10((c ** 2).sum() / (((y - c) ** 2).sum() + 1e-12))))
        rec = dict(snr=float(np.nanmean(snrs)), frmgain_med=float(np.nanmean(acc["med"])),
                   frmgain_clean=float(np.nanmean(acc["cl"])),
                   frmgain_imp=float(np.nanmean(acc["im"])),
                   frmgain_imp_min=float(np.nanmean(acc["imn"])))
        RESULTS[split][name] = rec
        print("%-22s %9.2f %11.2f %11.2f %11.2f %11.2f"
              % (name, rec["snr"], rec["frmgain_med"], rec["frmgain_clean"],
                 rec["frmgain_imp"], rec["frmgain_imp_min"]))
    print("-" * 104)

OUTJ = os.path.join(r"E:/workbuddy/store/2026-10-12-22-50-08/figs_pamnet", "baseline_pam.json")
os.makedirs(os.path.dirname(OUTJ), exist_ok=True)
with open(OUTJ, "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, ensure_ascii=False, indent=1)
print("\n经典基线汇总 -> %s" % OUTJ)
