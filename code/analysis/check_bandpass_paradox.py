# -*- coding: utf-8 -*-
"""查清「模型 vs 带通」在新信道上的矛盾：按信道逐一对比

三个对照，都在【相同信道 + 相同 chirp】上做，排除混杂：
  A. v2 模型 vs 带通，在 v2 测试集（chirp 18-20，信道 1-25）
  B. v3 模型在 testC（chirp 18-20，信道 22-25，未见过）
  C. 带通在 testC 上
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


print("=" * 90)
print("A. v2 测试集（chirp 18-20，全部 25 个信道，模型训练时【见过】这些信道）")
print("=" * 90)
r2 = np.loadtxt(r"D:/project/output_v2/test_results.csv", delimiter=",", skiprows=1)
ch2, ca2 = meta(r"D:/project/dataset", "test")
g2 = r2[:, 1] - r2[:, 0]
N2, C2 = load_set(r"D:/project/dataset/test")
bp2 = filtfilt(b, a, N2, axis=1).astype(np.float32)
gbp2 = snr(bp2, C2) - snr(N2, C2)
print("  模型整体: 输入 %.2f 输出 %.2f 提升 %.2f" % (
    snr(N2, C2).mean(), r2[:, 1].mean(), g2.mean()))
print("  带通整体: 输入 %.2f 输出 %.2f 提升 %.2f" % (
    snr(N2, C2).mean(), snr(bp2, C2).mean(), gbp2.mean()))
print()
print("  按信道（只看 22-25 与全体）：")
print("  %-8s %10s %10s %10s" % ("chan_id", "v2模型", "带通", "差"))
for c in [1, 2, 11, 12] + [22, 23, 24, 25]:
    m = ca2 == c
    print("  %-8d %10.2f %10.2f %+10.2f" % (c, g2[m].mean(), gbp2[m].mean(), g2[m].mean()-gbp2[m].mean()))
print("  %-8s %10.2f %10.2f %+10.2f" % ("全体", g2.mean(), gbp2.mean(), g2.mean()-gbp2.mean()))
print()
m2225 = np.isin(ca2, [22, 23, 24, 25])
print("  信道 22-25 汇总（chirp 18-20，模型【见过】）:")
print("    v2 模型 %.2f dB   带通 %.2f dB   差 %+.2f dB" % (
    g2[m2225].mean(), gbp2[m2225].mean(), g2[m2225].mean()-gbp2[m2225].mean()))

print()
print("=" * 90)
print("B/C. v3 testC（chirp 18-20，信道 22-25，v3 模型【未见过】这些信道）")
print("=" * 90)
rc = np.loadtxt(r"D:/project/output_v3/result_full_testC.csv", delimiter=",", skiprows=1)
chc, cac = meta(r"D:/project/dataset_v3", "testC_newch_newchirp")
gc = rc[:, 1] - rc[:, 0]
Nc, Cc = load_set(r"D:/project/dataset_v3/testC_newch_newchirp")
bpc = filtfilt(b, a, Nc, axis=1).astype(np.float32)
gbpc = snr(bpc, Cc) - snr(Nc, Cc)
print("  %-8s %10s %10s %10s" % ("chan_id", "v3模型", "带通", "差"))
for c in [22, 23, 24, 25]:
    m = cac == c
    print("  %-8d %10.2f %10.2f %+10.2f" % (c, gc[m].mean(), gbpc[m].mean(), gc[m].mean()-gbpc[m].mean()))
print("  %-8s %10.2f %10.2f %+10.2f" % ("全体", gc.mean(), gbpc.mean(), gc.mean()-gbpc.mean()))

print()
print("=" * 90)
print("★ 关键对照：同样的信道 22-25 + 同样的 chirp 18-20")
print("=" * 90)
print("  v2 模型（信道【见过】，但模型只在 25 信道上训过、7000 样本、50 轮）: %.2f dB" % g2[m2225].mean())
print("  v3 模型（信道【未见过】，18 信道、7560 样本、25 轮）:            %.2f dB" % gc.mean())
print("  带通（无学习）在 v2 测试集同样信道上:                            %.2f dB" % gbp2[m2225].mean())
print("  带通（无学习）在 v3 testC 同样信道上:                            %.2f dB" % gbpc.mean())
print()
print("  ==> 带通两次相差 %.2f dB（同一信道、同一 chirp，只是数据集不同），" % abs(gbp2[m2225].mean()-gbpc.mean()))
print("      说明噪声实现不同会带来这个量级的波动，可作为对照噪声底。")
