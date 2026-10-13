# -*- coding: utf-8 -*-
"""分析 test_results.csv —— 按 chirp / 信道 / 真实SNR 分组统计"""
import numpy as np, os, glob
import scipy.io as sio

CSV = r"D:/project/output_v2/test_results.csv"
DS  = r"D:/project/dataset/test"
r = np.loadtxt(CSV, delimiter=",", skiprows=1)
print("样本数:", r.shape)
print()

# ---- 读 meta 建立 索引 -> (chirp_id, chan_id, snr) 映射 ----
mt = sio.loadmat(r"D:/project/dataset/meta_test.mat", squeeze_me=True, struct_as_record=False)
meta = mt["meta"]
n = meta.size
chirp = np.array([meta[k].chirp_id for k in range(n)])
chan  = np.array([meta[k].chan_id  for k in range(n)])
snr   = np.array([meta[k].snr      for k in range(n)])
print("meta 数:", n, " chirp:", sorted(set(chirp.tolist())), " chan:", sorted(set(chan.tolist())),
      " snr:", sorted(set(snr.tolist())))
assert n == r.shape[0], "meta 与 csv 行数不一致！"

gain_snr = r[:,1] - r[:,0]
gain_sd  = r[:,3] - r[:,2]

# ---- 校验：第一行应该对应 chirp=18, chan=1, snr=-10 ----
print("\n[校验] 前 3 行 meta:", [(chirp[k], chan[k], snr[k]) for k in range(3)])
print("[校验] 前 3 行 snr_in:", np.round(r[:3,0],2))
print()

def block(title, keys, vals_in, vals_out, gains, extra=None):
    print(title)
    print("  %-14s %5s %9s %9s %9s" % ("组", "n", "输入SNR", "输出SNR", "提升"))
    for k in sorted(set(keys.tolist())):
        m = keys == k
        print("  %-14s %5d %9.2f %9.2f %+9.2f" % (str(k), m.sum(),
              vals_in[m].mean(), vals_out[m].mean(), gains[m].mean()))
    print()

block("=== 按真实 SNR 档位（加噪时设定值，非降采样后）===", snr, r[:,0], r[:,1], gain_snr)
block("=== 按 chirp 分组（测试集 chirp 18/19/20 从未在训练出现）===", chirp, r[:,0], r[:,1], gain_snr)

# 信道按输入SNR差异很大，所以按「提升」排序更公平
print("=== 按信道（按提升排序，仅列最好/最差各 5 个）===")
g = np.array([gain_snr[chan==c].mean() for c in sorted(set(chan.tolist()))])
cs = np.array(sorted(set(chan.tolist())))
order = np.argsort(g)
print("  最差 5 个信道:")
for i in order[:5]:
    print("    chan %2d  n=%3d  输入 %6.2f -> 输出 %6.2f  提升 %+6.2f" % (
        cs[i], (chan==cs[i]).sum(), r[chan==cs[i],0].mean(), r[chan==cs[i],1].mean(), g[i]))
print("  最好 5 个信道:")
for i in order[-5:][::-1]:
    print("    chan %2d  n=%3d  输入 %6.2f -> 输出 %6.2f  提升 %+6.2f" % (
        cs[i], (chan==cs[i]).sum(), r[chan==cs[i],0].mean(), r[chan==cs[i],1].mean(), g[i]))
print()

# ---- 分布 ----
print("=== 输出 SNR 分布 ===")
for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
    print("  %3d%% 分位: 输入 %7.2f  输出 %7.2f  提升 %+6.2f" % (
        p, np.percentile(r[:,0],p), np.percentile(r[:,1],p), np.percentile(gain_snr,p)))
print()
print("=== SI-SDR 分布 ===")
for p in [0, 5, 50, 95, 100]:
    print("  %3d%% 分位: 输入 %7.2f  输出 %7.2f  提升 %+6.2f" % (
        p, np.percentile(r[:,2],p), np.percentile(r[:,3],p), np.percentile(gain_sd,p)))
print()

# ---- 失败样本 ----
bad = gain_snr < 0
print("=== 提升为负（变差）的样本 ===")
print("  个数 %d / %d  (%.2f%%)" % (bad.sum(), len(bad), bad.sum()/len(bad)*100))
if bad.sum():
    idx = np.where(bad)[0]
    print("  它们对应的 snr 档:", sorted(set(snr[idx].tolist())))
    print("  它们对应的 chirp:", sorted(set(chirp[idx].tolist())))
    print("  最差 5 个: 输入/输出/提升")
    for k in idx[np.argsort(gain_snr[idx])][:5]:
        print("    样本 %4d  chirp=%d chan=%2d snr=%3d  %6.2f -> %6.2f  %+6.2f" % (
            k+1, chirp[k], chan[k], snr[k], r[k,0], r[k,1], gain_snr[k]))
print()

# ---- 低输出 SNR 的样本 ----
print("=== 输出 SNR < 10 dB 的样本 ===")
low = r[:,1] < 10
print("  个数 %d (%.1f%%)" % (low.sum(), low.sum()/len(low)*100))
if low.sum():
    print("  snr 档分布:", {int(s): int((snr[low]==s).sum()) for s in sorted(set(snr[low].tolist()))})
    print("  chirp 分布:", {int(c): int((chirp[low]==c).sum()) for c in sorted(set(chirp[low].tolist()))})
print()

# ---- 分组交叉：chirp x snr 的提升 ----
print("=== chirp x snr 交叉表（提升 dB）===")
print("        ", "".join("%9d" % s for s in sorted(set(snr.tolist()))))
for c in sorted(set(chirp.tolist())):
    row = []
    for s in sorted(set(snr.tolist())):
        m = (chirp==c) & (snr==s)
        row.append(gain_snr[m].mean() if m.sum() else float("nan"))
    print("  chirp %2d" % c, "".join("%+9.2f" % v for v in row))
