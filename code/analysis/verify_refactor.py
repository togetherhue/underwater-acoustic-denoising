# -*- coding: utf-8 -*-
"""重构验证：确认 FrequencyBranch 加宽度系数后，fch=1.0 与原架构【逐位一致】

验证三件事：
 1. 三个已训练 checkpoint 能否用 strict=True 加载（键名与形状完全一致）
 2. 加载后用 freqonly 模型重算 testA，看能否复现训练时报告的 +13.9848 dB
 3. 打印各 fch 的参数量，确认扫描档位设计合理
"""
import os, sys, glob
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly
import torch
import importlib.util

spec = importlib.util.spec_from_file_location("v3", r"C:\Users\32213\PycharmProjects\project1\大创_v3.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
CFG = m.CFG
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = r"D:/project/output_v3"

print("=" * 78)
print("① 参数量 / 通道配置")
print("=" * 78)
print("%-8s %-26s %10s" % ("fch", "频域通道 (c1,c2,c3,p1,p2)", "参数量"))
for fch, ut, uf, nm in [(1.0, True, True, "full"),
                        (1.0, True, False, "timeonly"),
                        (1.0, False, True, "freqonly"),
                        (0.5, False, True, "sweep 0.5x"),
                        (2.0, False, True, "sweep 2x"),
                        (4.0, False, True, "sweep 4x")]:
    net = m.DenoiseNet(CFG["n_fft"], CFG["hop_length"], CFG["hidden"], ut, uf, fch)
    w = getattr(net, "freq", None)
    print("%-8s %-26s %10s   %s" % (fch, str(getattr(w, "widths", "-")),
                                    "{:,}".format(net.n_params()), nm))

print()
print("=" * 78)
print("② checkpoint 严格加载（strict=True）")
print("=" * 78)
ok_all = True
for tag, ut, uf in [("full", True, True), ("timeonly", True, False), ("freqonly", False, True)]:
    p = os.path.join(OUT, "model_%s.pth" % tag)
    if not os.path.exists(p):
        print("  %-10s  (无 checkpoint)" % tag); continue
    net = m.DenoiseNet(CFG["n_fft"], CFG["hop_length"], CFG["hidden"], ut, uf, 1.0)
    try:
        sd = torch.load(p, map_location="cpu")
        r = net.load_state_dict(sd, strict=True)
        print("  %-10s  ✓ strict 加载成功  键数 %d  参数量 %s" % (
            tag, len(sd), "{:,}".format(net.n_params())))
    except Exception as e:
        ok_all = False
        print("  %-10s  ✗ 加载失败: %s" % (tag, e))

print()
print("=" * 78)
print("③ 功能复现：用 freqonly 重算 testA，对比训练时报告的 +13.9848 dB")
print("=" * 78)
net = m.DenoiseNet(CFG["n_fft"], CFG["hop_length"], CFG["hidden"], False, True, 1.0)
net.load_state_dict(torch.load(os.path.join(OUT, "model_freqonly.pth"), map_location="cpu"))
net = net.to(DEV).eval()

files = sorted(glob.glob(os.path.join(CFG["test_dirs"][0][1], "*.mat")))
L = CFG["seg_len"]; DS = CFG["downsample"]
N = np.zeros((len(files), L), np.float32); C = np.zeros_like(N)
for i, p in enumerate(files):
    mm = sio.loadmat(p)
    for src, dst in ((mm["noisy_signal"], N), (mm["clean_signal"], C)):
        x = np.asarray(src).squeeze().astype(np.float32)
        x = resample_poly(x, 1, DS).astype(np.float32)
        dst[i] = x[:L] if len(x) >= L else np.pad(x, (0, L - len(x)))

EST = np.zeros_like(N)
with torch.no_grad():
    for i in range(0, len(N), 32):
        EST[i:i+32] = net(torch.from_numpy(N[i:i+32]).unsqueeze(1).to(DEV)).squeeze(1).cpu().numpy()

def snr(e, r):
    return 10*np.log10((r**2).sum(-1)/(((e-r)**2).sum(-1)+1e-12))

g = (snr(EST, C) - snr(N, C)).mean()
print("  重算 testA 的 SNR 提升 = %.4f dB" % g)
print("  训练时报告            = 13.9848 dB")
print("  差值                  = %+.6f dB" % (g - 13.984793039714848))
print("  判定: %s" % ("✓ 一致（重构未改变 fch=1 的行为）" if abs(g-13.984793039714848) < 1e-6
                     else "✗ 不一致，重构改变了行为！"))

print()
print("=" * 78)
print("总判定: %s" % ("✓ 全部通过，可以开始扫描" if (ok_all and abs(g-13.984793039714848) < 1e-6)
                     else "✗ 有失败项，不要开始扫描"))
print("=" * 78)
