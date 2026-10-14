# -*- coding: utf-8 -*-
"""验证稳健指标：证明"全局 SNR 会被单个脉冲毁掉，而逐帧中位 SNR 不会"
   用真实干净信号 + 受控噪声构造，检查指标能否正确区分模型质量。
"""
import os, sys
import numpy as np
import torch
import scipy.io as sio
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pam_metrics import snr_db, si_sdr_db, frame_snr_stats

SEG, DS = 8000, 4


def load_clean(path):
    m = sio.loadmat(path)
    c = np.asarray(m["clean_signal"]).squeeze().astype(np.float64)
    c = resample_poly(c, 1, DS)
    c = c[:SEG] if len(c) >= SEG else np.pad(c, (0, SEG - len(c)))
    return c


def rep(name, est, ref):
    st = frame_snr_stats(est, ref)
    print("%-26s snr=%8.2f  sisdr=%8.2f  fmsnr_med=%8.2f  fmsnr_p10=%8.2f  "
          "fmsnr_mean=%8.2f  bad=%.3f"
          % (name, snr_db(est, ref).mean(), si_sdr_db(est, ref).mean(),
             st["med"].mean(), st["p10"].mean(), st["mean"].mean(),
             st["frac_bad"].mean()))


src = r"D:/project/dataset_wave/LFM/testA/s_00001.mat"
c = load_clean(src)
T = torch.from_numpy(c).float().unsqueeze(0)
rms = float(T.pow(2).mean().sqrt())
print("干净信号：长度 %d，rms %.4f，峰值 %.2f" % (T.shape[-1], rms, T.abs().max()))
print()

rng = np.random.default_rng(20261014)

# ---------- 构造噪声 ----------
# 高斯噪声，使其全局 SNR = 0 dB
g = torch.from_numpy(rng.standard_normal(SEG)).float().unsqueeze(0) * rms

# 脉冲干扰：8 个尖峰，幅度 20×rms，随机位置（1 个采样点宽，最恶劣情形）
imp = torch.zeros_like(T)
pos = rng.choice(np.arange(200, SEG - 200), 8, replace=False)
imp[0, pos] = torch.from_numpy(rng.choice([-1.0, 1.0], 8)).float() * 20 * rms

N1 = T + g                 # 纯高斯
N2 = T + g + imp           # 高斯 + 脉冲
print("噪声：高斯 rms=%.4f（全局 SNR≈0 dB）；脉冲 8 个，幅度 ±%.1f×rms，位置 %s"
      % (rms, 20, sorted(pos.tolist())))
print()
print("=" * 118)
print("【一】输入信号本身")
print("=" * 118)
rep("纯高斯噪声", N1, T)
rep("高斯 + 8 个脉冲", N2, T)
print("  ↑ 注意：只加了 8 个采样点（占 0.1%）的脉冲，全局 SNR 就崩了，而 fmsnr_med 几乎不动")
print()

# ---------- 构造不同质量的"模型输出" ----------
print("=" * 118)
print("【二】模拟不同质量的模型输出（输入统一为 高斯+脉冲）")
print("=" * 118)
outs = [
    ("理想模型 est=clean", T),
    ("优秀模型(残差0.15×)", T + 0.15 * g + 0.15 * imp),
    ("好模型(残差0.30×)", T + 0.30 * g + 0.30 * imp),
    ("漏掉1个脉冲(残差0.15×)", T + 0.15 * g + 0.15 * imp
     + imp * (torch.arange(SEG) == int(pos[0])).float()),
    ("漏掉全部脉冲(残差0.15×)", T + 0.15 * g + imp),
    ("不处理(est=输入)", N2),
]
for name, est in outs:
    rep(name, est, T)

print()
print("=" * 118)
print("【三】关键对比：'漏掉1个脉冲' vs '好模型'")
print("=" * 118)
a = T + 0.30 * g + 0.30 * imp
b = T + 0.15 * g + 0.15 * imp + imp * (torch.arange(SEG) == int(pos[0])).float()
sa, sb = frame_snr_stats(a, T), frame_snr_stats(b, T)
print("  %-22s 全局SNR %8.2f   fmsnr_med %8.2f   fmsnr_p10 %8.2f"
      % ("好模型(残差0.30×)", snr_db(a, T).mean(), sa["med"].mean(), sa["p10"].mean()))
print("  %-22s 全局SNR %8.2f   fmsnr_med %8.2f   fmsnr_p10 %8.2f"
      % ("漏掉1个脉冲", snr_db(b, T).mean(), sb["med"].mean(), sb["p10"].mean()))
print()
print("  → 全局 SNR 认为'好模型'远好于'漏1个脉冲'（差距 %.1f dB），"
      % (snr_db(a, T).mean() - snr_db(b, T).mean()))
print("    但 fmsnr_med 认为两者相当（差 %.2f dB），只有 fmsnr_p10 能反映"
      % (sa["med"].mean() - sb["med"].mean()))
print("    '漏掉脉冲'这件事（差 %.2f dB）。" % (sa["p10"].mean() - sb["p10"].mean()))
print()
print("  结论：必须同时看 fmsnr_med（典型表现）和 fmsnr_p10（脉冲段表现），")
print("        单看全局 SNR 会得出错误结论。")
