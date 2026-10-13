# -*- coding: utf-8 -*-
"""信道特性 vs 模型增益 —— 解释为什么不同信道差 5 dB"""
import numpy as np, glob, os
import scipy.io as sio

CH = r"D:/project/信道产生/data_fixed"
files = sorted(glob.glob(os.path.join(CH, "channel_H_*.mat")))
print("信道文件 %d 个" % len(files))

H, names, cid = [], [], []
for f in files:
    S = sio.loadmat(f); h = np.asarray(S["channel_H"]).squeeze()
    H.append(h); names.append(os.path.basename(f))
    cid.append(int(os.path.basename(f).split("_")[-1].split(".")[0]))
H = np.array(H); cid = np.array(cid)
print("文件名顺序（字典序）:", names[:4], "...")

# 内容去重（与 make_dataset 一致）
keep = []
for i in range(len(H)):
    if not any(np.all(np.abs(H[j]-H[i]) < 1e-30) for j in keep):
        keep.append(i)
H = H[keep]; names = [names[i] for i in keep]; cid = cid[keep]
print("去重后 %d 个" % len(H))

FS = 24000
print("\n%-16s %10s %10s %10s %10s" % ("文件", "峰位置", "90%时延", "rms", "峰/rms"))
rows = []
for i, h in enumerate(H):
    e = np.abs(h)**2; e = e/e.sum(); c = np.cumsum(e)
    p95 = np.searchsorted(c, 0.95); p05 = np.searchsorted(c, 0.05)
    spread_ms = (p95-p05)/FS*1000
    pk = int(np.argmax(np.abs(h)))
    rows.append((names[i], pk, spread_ms, float(np.sqrt((h**2).mean())),
                 float(np.abs(h).max()/np.sqrt((h**2).mean()))))
    if i < 8 or i >= len(H)-2:
        print("%-16s %10d %9.1fms %10.2e %10.1f" % rows[-1])

sp = np.array([r[2] for r in rows])
print("\n时延扩展分布: min %.1f  中位 %.1f  max %.1f ms" % (sp.min(), np.median(sp), sp.max()))

# 与测试结果的增益关联
res = np.loadtxt(r"D:/project/output_v2/test_results.csv", delimiter=",", skiprows=1)
mt = sio.loadmat(r"D:/project/dataset/meta_test.mat", squeeze_me=True, struct_as_record=False)["meta"]
chan = np.array([mt[k].chan_id for k in range(mt.size)])
gain = res[:,1]-res[:,0]
g = np.array([gain[chan == c].mean() if (chan == c).sum() else np.nan for c in cid])
print("\n=== 信道时延扩展 vs 模型增益 ===")
cc = np.corrcoef(sp, g)[0,1]
print("  相关系数 = %+.3f  （负相关：多径越强，增益越低）" % cc)
order = np.argsort(sp)
print("\n  %-16s %12s %10s" % ("信道", "90%时延", "增益dB"))
for i in order[:4]:
    print("  %-16s %10.1fms %+10.2f" % (rows[i][0], sp[i], g[i]))
print("  ...")
for i in order[-4:]:
    print("  %-16s %10.1fms %+10.2f" % (rows[i][0], sp[i], g[i]))

np.savez(r"E:/workbuddy/store/2026-10-12-22-50-08/ch_gain.npz",
         sp=sp, g=g, cid=cid, names=np.array(names))
