# -*- coding: utf-8 -*-
"""图 H：LFM 与 HFM 的时频图对比（干净 / 含噪 / 模型输出）
   直观展示模型是否保住了正确的时频轨迹（LFM 是直线，HFM 是曲线）。"""
import os, glob, sys, shutil
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly, spectrogram
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import torch
# ---- 路径解析：兼容仓库布局（库模块在 ../python）与本地开发布局 ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDS = [os.path.normpath(os.path.join(_HERE, "..", "python")),
          r"C:\Users\32213\PycharmProjects\project1"]
for _p in [_HERE] + [c for c in _CANDS if os.path.isdir(c)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import v3_model as V3

WAVE = r"D:/project/dataset_wave"
OUTW = r"D:/project/output_wave"
FIG = r"E:/workbuddy/store/2026-10-12-22-50-08/figs_wave"
SEG, DS = 8000, 4
FS = 24000 // DS
TAG = "mixed"          # 用混合训练的模型
os.makedirs(FIG, exist_ok=True)


def fix_len(x):
    return x[:SEG] if len(x) >= SEG else np.pad(x, (0, SEG - len(x)))


def pick(mod, chirp=18, chan=1, snr=0, rep=1):
    """在 testA 里挑一个指定 chirp/信道/SNR 的样本"""
    mt = sio.loadmat(os.path.join(WAVE, mod, "meta_testA.mat"),
                     squeeze_me=True, struct_as_record=False)["meta"]
    for k in range(mt.size):
        if (mt[k].chirp_id == chirp and mt[k].chan_id == chan
                and mt[k].snr == snr and rep == 1):
            return os.path.join(WAVE, mod, "testA", mt[k].file)
    raise RuntimeError("没找到样本")


def load(f):
    m = sio.loadmat(f)
    n = np.asarray(m["noisy_signal"]).squeeze().astype(np.float64)
    c = np.asarray(m["clean_signal"]).squeeze().astype(np.float64)
    return (fix_len(resample_poly(n, 1, DS)).astype(np.float32),
            fix_len(resample_poly(c, 1, DS)).astype(np.float32))


def snr(est, ref):
    return 10 * np.log10((ref ** 2).sum() / (((est - ref) ** 2).sum() + 1e-12))


# ---- 载入模型 ----
mp = os.path.join(OUTW, "model_%s.pth" % TAG)
model = V3.DenoiseNet(V3.CFG["n_fft"], V3.CFG["hop_length"], 64,
                      use_time=False, use_freq=True, fch=2.0).to(V3.DEVICE)
model.load_state_dict(torch.load(mp, map_location=V3.DEVICE))
model = model.eval()
print("模型 %s  参数量 %d" % (TAG, model.n_params()))

fig, axs = plt.subplots(2, 3, figsize=(15, 6.4), dpi=150)
info = []
for r, mod in enumerate(["LFM", "HFM"]):
    f = pick(mod)
    n, c = load(f)
    with torch.no_grad():
        e = model(torch.from_numpy(n)[None, None].to(V3.DEVICE)).cpu().numpy()[0, 0]
    si, so = snr(n, c), snr(e, c)
    info.append((mod, os.path.basename(f), si, so, so - si))

    fq, tq, Zc = spectrogram(c, fs=FS, nperseg=256, noverlap=192)
    _, _, Zn = spectrogram(n, fs=FS, nperseg=256, noverlap=192)
    _, _, Ze = spectrogram(e, fs=FS, nperseg=256, noverlap=192)
    vmax = 20 * np.log10(np.abs(Zc).max() + 1e-12)
    vmin = vmax - 45
    for k, (Z, ttl) in enumerate([
            (Zc, "干净目标"),
            (Zn, "含噪输入  (SNR %+.2f dB)" % si),
            (Ze, "模型输出  (SNR %+.2f dB, %+.2f)" % (so, so - si))]):
        ax = axs[r, k]
        ax.pcolormesh(tq, fq, 20 * np.log10(np.abs(Z) + 1e-12),
                      shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_ylim(0, 3000)
        ax.set_title("%s · %s" % (mod, ttl), fontsize=10)
        ax.set_ylabel("频率 (Hz)" if k == 0 else "")
        ax.set_xlabel("时间 (s)")
fig.suptitle("时频图：模型对 LFM（直线轨迹）与 HFM（曲线轨迹）都保住了信号结构  "
             "（模型：%s，chirp 18，信道 1，0 dB）" % TAG, fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig(os.path.join(FIG, "figH_spectrogram.png"))
plt.close(fig)

print()
print("%-6s %-14s %10s %10s %10s" % ("调制", "样本", "输入SNR", "输出SNR", "提升"))
for mod, fn, si, so, g in info:
    print("%-6s %-14s %10.2f %10.2f %+10.2f" % (mod, fn, si, so, g))
print("\n图 -> figs_wave/figH_spectrogram.png")
