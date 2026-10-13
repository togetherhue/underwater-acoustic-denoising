# -*- coding: utf-8 -*-
"""实测确定 chan_id -> 信道文件的映射（不能靠猜排序），再重算信道物理量与增益的相关性"""
import numpy as np, glob, os, scipy.io as sio
from scipy.signal import resample_poly

CH = r"D:/project/信道产生/data_fixed"
fl = sorted(glob.glob(os.path.join(CH, "channel_H_*.mat")))
print("glob 顺序前 6 个:", [os.path.basename(f) for f in fl[:6]])

H, fn = [], []
for f in fl:
    H.append(np.asarray(sio.loadmat(f)["channel_H"]).squeeze())
    fn.append(os.path.basename(f))
H = np.array(H)
keep = []
for i in range(len(H)):
    if not any(np.all(np.abs(H[j]-H[i]) < 1e-30) for j in keep): keep.append(i)
H = H[keep]; fn = [fn[i] for i in keep]
print("去重后 %d 个" % len(H))

# --- 取 chirp18 的样本，用 clean_signal 反查真实信道 ---
mt = sio.loadmat(r"D:/project/dataset/meta_test.mat", squeeze_me=True, struct_as_record=False)["meta"]
cid = np.array([mt[k].chirp_id for k in range(mt.size)])
chn = np.array([mt[k].chan_id  for k in range(mt.size)])
snn = np.array([mt[k].snr      for k in range(mt.size)])

# 找出 (chirp18, snr=+10, 每个 chan 的第一个样本)
idx = {}
for k in range(mt.size):
    if cid[k] == 18 and snn[k] == 10:
        idx.setdefault(chn[k], k)

# 用 chirp_params.mat 取 chirp18 波形
cp = sio.loadmat(r"D:/project/dataset/chirp_params.mat", squeeze_me=True, struct_as_record=False)
P = cp["P"]
p18 = P[17]
print("\nchirp18 参数: f0=%.0f f1=%.0f T=%.2f env=%s" % (p18.f0, p18.f1, p18.T, p18.env))

# 复现 gen_chirp（必须用梯形积分，与 MATLAB cumtrapz 一致；用 cumsum 会引入约 0.7 rad 相位偏差）
fs0, T = 9600, float(p18.T)
t = np.arange(round(T*fs0))/fs0
fv = p18.f0 + (p18.f1-p18.f0)*t/T
integ = np.concatenate([[0.0], np.cumsum((fv[:-1]+fv[1:])/2)])
ph = 2*np.pi*integ/fs0
s18 = np.cos(ph + p18.phi)*p18.A
if str(p18.env) != "rect":
    print("  (包络非 rect: %s)" % p18.env)
s18 = resample_poly(s18, 5, 2); s18 = s18/np.max(np.abs(s18))

def corr(a, b):
    n = min(len(a), len(b)); a = a[:n]-a[:n].mean(); b = b[:n]-b[:n].mean()
    d = np.sqrt((a**2).sum()*(b**2).sum())
    return (a*b).sum()/d if d > 0 else 0

print("\n=== chan_id -> 文件名 实测映射（corr 应接近 1.000）===")
mapping = {}
for c in sorted(idx):
    m = sio.loadmat(sorted(glob.glob(os.path.join(r"D:/project/dataset/test", "*.mat")))[idx[c]])
    clean = np.asarray(m["clean_signal"]).squeeze()
    best, bk = -9, None
    for k in range(len(H)):
        cc = np.convolve(s18, H[k])
        v = corr(cc, clean)
        if v > best: best, bk = v, k
    mapping[c] = fn[bk]
    if c <= 5 or c >= 23:
        print("  chan_id %2d -> %-18s  corr=%.6f" % (c, fn[bk], best))
print("  唯一匹配文件数: %d / %d" % (len(set(mapping.values())), len(mapping)))

# 字典序假设 vs 数字序假设
lex = [fn[c-1] for c in sorted(mapping)]
ok_lex = sum(1 for c in mapping if mapping[c] == fn[c-1])
ok_num = sum(1 for c in mapping if mapping[c] == "channel_H_%d.mat" % c)
print("\n  与『glob 字典序』假设一致: %d / %d" % (ok_lex, len(mapping)))
print("  与『文件名数字序』假设一致: %d / %d" % (ok_num, len(mapping)))

# --- 重算相关性 ---
FSch = 24000
f = np.fft.rfftfreq(8000, 1/FSch)
band = (f >= 798) & (f < 1500)
stat = {}
for k in range(len(H)):
    A = np.abs(np.fft.rfft(H[k], 8000))[band]
    Adb = 20*np.log10(np.maximum(A/A.mean(), 1e-12))
    e = np.abs(H[k])**2; e = e/e.sum(); cc = np.cumsum(e)
    stat[fn[k]] = dict(dip=np.percentile(Adb,5), ripple=Adb.std(),
                       spread=(np.searchsorted(cc,0.95)-np.searchsorted(cc,0.05))/FSch*1000)

res = np.loadtxt(r"D:/project/output_v2/test_results.csv", delimiter=",", skiprows=1)
gain = res[:,1]-res[:,0]
g = np.array([gain[chn == c].mean() for c in sorted(mapping)])
sp = np.array([stat[mapping[c]]["spread"] for c in sorted(mapping)])
dp = np.array([stat[mapping[c]]["dip"]    for c in sorted(mapping)])
rp = np.array([stat[mapping[c]]["ripple"] for c in sorted(mapping)])

print("\n=== 修正映射后：信道物理量 vs 模型增益 ===")
for nm, x in [("90% 时延扩展", sp), ("带内 5% 分位凹陷", dp), ("带内起伏 std", rp)]:
    print("  %-18s r = %+.3f" % (nm, np.corrcoef(x, g)[0,1]))
o = np.argsort(rp)
print("\n  起伏最大的 4 个信道:")
for i in o[-4:][::-1]:
    print("    chan_id %2d (%s)  起伏 %5.1f dB  增益 %+6.2f" % (sorted(mapping)[i], mapping[sorted(mapping)[i]], rp[i], g[i]))
print("  起伏最小的 4 个信道:")
for i in o[:4]:
    print("    chan_id %2d (%s)  起伏 %5.1f dB  增益 %+6.2f" % (sorted(mapping)[i], mapping[sorted(mapping)[i]], rp[i], g[i]))

np.savez(r"E:/workbuddy/store/2026-10-12-22-50-08/chan_map.npz",
         cids=np.array(sorted(mapping)), files=np.array([mapping[c] for c in sorted(mapping)]),
         sp=sp, dp=dp, rp=rp, g=g)
