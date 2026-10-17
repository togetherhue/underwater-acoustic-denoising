# -*- coding: utf-8 -*-
"""模型 vs 经典降噪基线 对比 + 补齐/长度分布检查
所有指标在同一套管线、同一批 1500 个测试样本上计算，保证可比。
"""
import os, sys, glob, time
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly, butter, filtfilt, stft, istft

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 复用大创_v2.py 的模型定义 ----------------
import importlib.util
# ---- 路径解析：兼容仓库布局（库模块在 ../python）与本地开发布局 ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDS = [os.path.normpath(os.path.join(_HERE, "..", "python")),
          r"C:\Users\32213\PycharmProjects\project1"]
for _p in [_HERE] + [c for c in _CANDS if os.path.isdir(c)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import 大创_v2 as dc

CFG = dc.CFG
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS  = CFG["fs_native"] // CFG["downsample"]     # 6000
L   = CFG["seg_len"]

files = sorted(glob.glob(os.path.join(CFG["test_dir"], "*.mat")))
print("测试样本:", len(files), " 设备:", DEV)

def fix_len(x):
    if len(x) >= L: return x[:L]
    return np.pad(x, (0, L - len(x)))

# ---------------- 载入全部测试样本 ----------------
t0 = time.time()
N = np.zeros((len(files), L), np.float32)
C = np.zeros((len(files), L), np.float32)
raw_len = np.zeros(len(files), np.int32)
for i, f in enumerate(files):
    m = sio.loadmat(f)
    n = np.asarray(m["noisy_signal"]).squeeze().astype(np.float64)
    c = np.asarray(m["clean_signal"]).squeeze().astype(np.float64)
    raw_len[i] = len(n)
    n = resample_poly(n, 1, CFG["downsample"])
    c = resample_poly(c, 1, CFG["downsample"])
    N[i] = fix_len(n.astype(np.float32))
    C[i] = fix_len(c.astype(np.float32))
print("载入耗时 %.1f s" % (time.time()-t0))

# 去掉公共零填充（只在有效区内评估，更严格）
eff = raw_len // CFG["downsample"]              # 有效长度（降采样后）
eff = np.minimum(eff, L)
print("\n=== 样本有效长度（降采样后 / 统一 8000）===")
print("  min %d  max %d  中位 %d" % (eff.min(), eff.max(), int(np.median(eff))))
print("  被零填充的比例: 中位 %.1f%%  最大 %.1f%%" % (
    (1-np.median(eff)/L)*100, (1-eff.min()/L)*100))

# ---------------- 指标（与脚本完全一致：全长求和）----------------
def snr(est, ref):
    e = ((est-ref)**2).sum(-1); s = (ref**2).sum(-1)
    return 10*np.log10(s/(e+1e-12))
def sisdr(est, ref):
    r = ref - ref.mean(-1, keepdims=True); x = est - est.mean(-1, keepdims=True)
    a = (x*r).sum(-1, keepdims=True)/((r**2).sum(-1, keepdims=True)+1e-12)
    p = a*r; nz = x-p
    return 10*np.log10((p**2).sum(-1)/((nz**2).sum(-1)+1e-12))

def report(name, est):
    si, so = snr(N, C), snr(est, C)
    di, do = sisdr(N, C), sisdr(est, C)
    print("  %-26s 输出SNR %6.2f  (提升 %+6.2f)   输出SI-SDR %6.2f (提升 %+6.2f)" % (
        name, so.mean(), (so-si).mean(), do.mean(), (do-di).mean()))
    return so.mean()

print("\n" + "="*88)
print("模型 vs 经典基线（同一批 1500 样本，指标定义完全一致）")
print("="*88)
print("  %-26s %s" % ("方法", "结果"))
base = snr(N, C).mean()
print("  %-26s 输入SNR %6.2f  (基准)" % ("含噪输入", base))

# --- 基线 1: 理想带通（信号已知在 798~1500 Hz）---
for lo, hi, tag in [(798,1500,"带通 798-1500Hz (理想)"),
                    (700,1600,"带通 700-1600Hz (放宽)")]:
    b, a = butter(6, [lo/(FS/2), hi/(FS/2)], btype="band")
    est = filtfilt(b, a, N, axis=1).astype(np.float32)
    report(tag, est)

# --- 基线 2/3: 谱减法 / 维纳滤波 ---
# 盲噪声估计：信号只在 798~1500 Hz，1500~3000 Hz 全是噪声，用该频段中位数估计噪声幅度。
def _mask_denoise(x, mode, n_fft=512, hop=128, alpha=1.0, beta=0.02):
    out = np.zeros_like(x)
    f, _, _ = stft(x[0], fs=FS, nperseg=n_fft, noverlap=n_fft-hop)
    nb = (f >= 1800) & (f <= 2900)              # 纯噪声频段
    for i in range(x.shape[0]):
        _, _, Z = stft(x[i], fs=FS, nperseg=n_fft, noverlap=n_fft-hop)
        mag, ph = np.abs(Z), np.angle(Z)
        nse = np.median(mag[nb, :], axis=0, keepdims=True)   # [1,T]
        if mode == "ss":
            m = np.maximum(1 - alpha*nse/np.maximum(mag, 1e-12), beta)
        else:                                     # 维纳
            S2 = np.maximum(mag**2 - nse**2, 0)
            m = S2/np.maximum(mag**2, 1e-12)
        _, y = istft(m*mag*np.exp(1j*ph), fs=FS, nperseg=n_fft, noverlap=n_fft-hop)
        out[i, :min(len(y), L)] = y[:L]
    return out

t1 = time.time()
report("谱减法 (盲噪声估计)", _mask_denoise(N, "ss"))
report("维纳滤波 (盲噪声估计)", _mask_denoise(N, "wiener"))
print("  （两种经典方法合计耗时 %.0f s）" % (time.time()-t1))

# --- 模型 ---
model = dc.DenoiseNet(CFG["n_fft"], CFG["hop_length"], CFG["hidden"])
model.load_state_dict(torch.load(CFG["model_save_path"], map_location=DEV))
model = model.to(DEV).eval()
EST = np.zeros_like(N)
with torch.no_grad():
    for i in range(0, len(N), 32):
        xb = torch.from_numpy(N[i:i+32]).unsqueeze(1).to(DEV)
        EST[i:i+32] = model(xb).squeeze(1).cpu().numpy()
report("本模型 (v2)", EST)

# ---------------- 只在有效区评估（排除零填充）----------------
print("\n=== 只在信号有效区评估（排除零填充，更严格）===")
mask = np.zeros((len(N), L), np.float32)
for i in range(len(N)): mask[i, :eff[i]] = 1
def snr_m(est, ref):
    e = (((est-ref)*mask)**2).sum(-1); s = ((ref*mask)**2).sum(-1)
    return 10*np.log10(s/(e+1e-12))
print("  输入           %6.2f dB" % snr_m(N, C).mean())
print("  带通 798-1500  %6.2f dB" % snr_m(filtfilt(*butter(6,[798/(FS/2),1500/(FS/2)],btype="band"), N, axis=1).astype(np.float32), C).mean())
print("  本模型 (v2)    %6.2f dB" % snr_m(EST, C).mean())

# ---------------- 按 SNR 档位比基线 ----------------
snr_in = snr(N, C)
print("\n=== 分档对比（输出 SNR, dB）===")
edges = [-5, 3, 8, 13, 17]
names = ["低 SNR (<3dB)", "中低 (3~8)", "中 (8~13)", "中高 (13~17)", "高 (>17dB)"]
bp = filtfilt(*butter(6,[798/(FS/2),1500/(FS/2)],btype="band"), N, axis=1).astype(np.float32)
print("  %-16s %5s %8s %10s %10s" % ("档位", "n", "输入", "带通", "本模型"))
for k in range(len(edges)-1):
    m = (snr_in >= edges[k]) & (snr_in < edges[k+1])
    if m.sum() == 0: continue
    print("  %-16s %5d %8.2f %10.2f %10.2f" % (
        names[k], m.sum(), snr_in[m].mean(), snr(bp,C)[m].mean(), snr(EST,C)[m].mean()))
