# -*- coding: utf-8 -*-
"""严格的信道泛化估计：同信道 + 同 chirp，只换模型

对照设计（都用 chirp 18-20，排除波形这个变量）：
  v2 模型在 25 个信道上训练过（含 22-25）
  v3 模型在 18 个信道上训练过（22-25 未见过）

  若 v3 在【双方都见过】的信道 1-18 上 ≥ v2，
  而 v3 在【只有 v2 见过】的信道 22-25 上 < v2，
  两者之差就是纯粹的「信道新颖度」代价。
"""
import os, glob
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly, butter, filtfilt

FS, L, DSF = 6000, 8000, 4
b, a = butter(6, [798/(FS/2), 1500/(FS/2)], btype="band")


def snr(e, r):
    return 10*np.log10((r**2).sum(-1)/(((e-r)**2).sum(-1)+1e-12))


def load_set(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    N = np.zeros((len(files), L), np.float32); C = np.zeros_like(N)
    for i, p in enumerate(files):
        m = sio.loadmat(p)
        for src, dst in ((m["noisy_signal"], N), (m["clean_signal"], C)):
            x = np.asarray(src).squeeze().astype(np.float64)
            x = resample_poly(x, 1, DSF)
            dst[i] = x[:L] if len(x) >= L else np.pad(x, (0, L-len(x)))
    return N, C


def meta(folder, split):
    mt = sio.loadmat(os.path.join(folder, "meta_%s.mat" % split),
                     squeeze_me=True, struct_as_record=False)["meta"]
    return (np.array([mt[k].chirp_id for k in range(mt.size)]),
            np.array([mt[k].chan_id for k in range(mt.size)]))


# ---- v2：chirp 18-20，信道 1-25 ----
r2 = np.loadtxt(r"D:/project/output_v2/test_results.csv", delimiter=",", skiprows=1)
_, ca2 = meta(r"D:/project/dataset", "test")
g2 = r2[:, 1] - r2[:, 0]

# ---- v3 testA：chirp 18-20，信道 1-18（与 v2 是同一批 clean_signal）----
rA = np.loadtxt(r"D:/project/output_v3/result_full_testA.csv", delimiter=",", skiprows=1)
_, caA = meta(r"D:/project/dataset_v3", "testA_seench_newchirp")
gA = rA[:, 1] - rA[:, 0]

# ---- v3 testC：chirp 18-20，信道 22-25（v3 未见过）----
rC = np.loadtxt(r"D:/project/output_v3/result_full_testC.csv", delimiter=",", skiprows=1)
_, caC = meta(r"D:/project/dataset_v3", "testC_newch_newchirp")
gC = rC[:, 1] - rC[:, 0]

print("=" * 88)
print("全部使用 chirp 18-20（排除波形变量）")
print("=" * 88)
print("%-40s %10s %8s" % ("条件", "平均增益", "样本数"))
print("-" * 88)
m18 = ca2 <= 18
m2225 = ca2 >= 22
print("%-40s %10.2f %8d" % ("v2 模型 · 信道 1-18（双方都见过）", g2[m18].mean(), m18.sum()))
print("%-40s %10.2f %8d" % ("v3 模型 · 信道 1-18（双方都见过）", gA.mean(), len(gA)))
print("%-40s %10.2f %8d" % ("v2 模型 · 信道 22-25（仅 v2 见过）", g2[m2225].mean(), m2225.sum()))
print("%-40s %10.2f %8d" % ("v3 模型 · 信道 22-25（v3 未见过）", gC.mean(), len(gC)))
print("-" * 88)

d_seen = gA.mean() - g2[m18].mean()
d_new = gC.mean() - g2[m2225].mean()
print("v3 相对 v2，在【见过的信道 1-18】上: %+.2f dB  （模型自身强弱差异）" % d_seen)
print("v3 相对 v2，在【未见的信道 22-25】上: %+.2f dB  （强弱差异 + 新颖度代价）" % d_new)
print()
print("★ 纯粹的信道新颖度代价 = %.2f - (%.2f) = %+.2f dB" % (d_new, d_seen, d_new - d_seen))
print()

# ---- 噪声实现的波动量级（对照噪声底）----
N2, C2 = load_set(r"D:/project/dataset/test")
bp2 = filtfilt(b, a, N2, axis=1).astype(np.float32)
gbp2 = snr(bp2, C2) - snr(N2, C2)
NA, CA = load_set(r"D:/project/dataset_v3/testA_seench_newchirp")
bpA = filtfilt(b, a, NA, axis=1).astype(np.float32)
gbpA = snr(bpA, CA) - snr(NA, CA)
print("对照：带通（无学习）在同样信道 1-18 上，两个数据集的差异 = %+.3f dB"
      % (gbpA.mean() - gbp2[m18].mean()))
print("      => 噪声实现不同造成的波动约 %.2f dB，远小于上面的新颖度代价" %
      abs(gbpA.mean() - gbp2[m18].mean()))

print()
print("=" * 88)
print("顺带修正：模型 vs 带通（同为【增益】口径，此前我把输出SNR当增益比了）")
print("=" * 88)
NC, CC = load_set(r"D:/project/dataset_v3/testC_newch_newchirp")
bpC = filtfilt(b, a, NC, axis=1).astype(np.float32)
print("  testC: 模型 %+.2f dB   带通 %+.2f dB   领先 %+.2f dB"
      % (gC.mean(), (snr(bpC, CC) - snr(NC, CC)).mean(),
         gC.mean() - (snr(bpC, CC) - snr(NC, CC)).mean()))
print("  testB: 模型 %+.2f dB   带通 %+.2f dB"
      % (np.loadtxt(r"D:/project/output_v3/result_full_testB.csv", delimiter=",", skiprows=1)[:, 1].mean()
         - np.loadtxt(r"D:/project/output_v3/result_full_testB.csv", delimiter=",", skiprows=1)[:, 0].mean(), 6.21))
