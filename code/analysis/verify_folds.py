# -*- coding: utf-8 -*-
"""独立复算全部折：不复用 train_fold.py / 大创_v3.py 的任何代码，
   自行实现数据管线（降采样+定长）与指标（SNR / SI-SDR），逐样本对照。
   用法: python verify_folds.py [fold ...]   默认 1 2 3
"""
import os, sys, glob, csv
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

ROOT = r"D:/project/dataset_folds"
OUT = r"D:/project/output_folds"
SEG, DOWNSAMPLE = 8000, 4
TAGS = {1: "lite_fold1", 2: "lite_fold2", 3: "lite_fold3"}


def pipeline(f):
    m = sio.loadmat(f)
    n = np.asarray(m["noisy_signal"]).squeeze().astype(np.float64)
    c = np.asarray(m["clean_signal"]).squeeze().astype(np.float64)
    n = resample_poly(n, 1, DOWNSAMPLE)
    c = resample_poly(c, 1, DOWNSAMPLE)

    def fix(x):
        return x[:SEG] if len(x) >= SEG else np.pad(x, (0, SEG - len(x)))
    return fix(n), fix(c)


def snr_in(noisy, clean):
    return 10 * np.log10((clean ** 2).sum() / ((noisy - clean) ** 2).sum())


def sisdr(est, ref):
    ref = ref - ref.mean(); est = est - est.mean()
    a = (est * ref).sum() / (ref ** 2).sum()
    proj = a * ref
    return 10 * np.log10((proj ** 2).sum() / ((est - proj) ** 2).sum())


folds = [int(x) for x in sys.argv[1:]] or [1, 2, 3]

print("=" * 88)
print("独立复算：数据管线与指标全部自行实现，逐样本与训练脚本记录对照")
print("=" * 88)
print("%-6s %-9s %6s %12s %12s %11s" %
      ("fold", "split", "n", "输入SNR最大差", "输入SI-SDR最大差", "代价(dB)"))

results = {}
for k in folds:
    tag = TAGS[k]
    fdir = os.path.join(ROOT, "fold%d" % k)
    rec = {}
    for split, which in (("test_seen", "seen"), ("test_unseen", "unseen")):
        csvp = os.path.join(OUT, "result_%s_%s.csv" % (tag, which))
        if not os.path.exists(csvp):
            print("  fold%d %s 结果缺失，跳过" % (k, which)); continue
        files = sorted(glob.glob(os.path.join(fdir, split, "*.mat")))
        R = np.loadtxt(csvp, delimiter=",", skiprows=1)
        if len(R) != len(files):
            print("  !! fold%d %s 样本数不符 %d vs %d" % (k, which, len(R), len(files)))
            continue
        si, sd = [], []
        for f in files:
            n_, c_ = pipeline(f)
            si.append(snr_in(n_, c_))      # est=noisy, ref=clean
            sd.append(sisdr(n_, c_))
        si, sd = np.array(si), np.array(sd)
        d1 = np.abs(si - R[:, 0]).max()
        d2 = np.abs(sd - R[:, 2]).max()
        rec[which] = dict(g_snr=(R[:, 1] - R[:, 0]).mean(),
                          g_sd=(R[:, 3] - R[:, 2]).mean(),
                          worst=min((R[:, 1] - R[:, 0]).min(), (R[:, 3] - R[:, 2]).min()),
                          bad=int(((R[:, 1] - R[:, 0]) < 0).sum() + ((R[:, 3] - R[:, 2]) < 0).sum()),
                          n=len(R), d1=d1, d2=d2)
        print("%-6d %-9s %6d %12.2e %12.2e" % (k, which, len(R), d1, d2))
    if "seen" in rec and "unseen" in rec:
        p = rec["unseen"]["g_snr"] - rec["seen"]["g_snr"]
        ps = rec["unseen"]["g_sd"] - rec["seen"]["g_sd"]
        results[k] = (p, ps, rec)
        print("%-6s %-9s %6s %12s %12s %+11.4f   (SI-SDR %+.4f)" %
              ("", "★代价", "", "", "", p, ps))

if results:
    P = np.array([v[0] for v in results.values()])
    PS = np.array([v[1] for v in results.values()])
    print()
    print("=" * 88)
    print("信道新颖度代价（独立复算）")
    for k, (p, ps, _) in results.items():
        print("  fold %d :  SNR %+.4f dB   SI-SDR %+.4f dB" % (k, p, ps))
    print("  ----")
    print("  均值   :  SNR %+.4f dB   SI-SDR %+.4f dB" % (P.mean(), PS.mean()))
    if len(P) > 1:
        print("  标准差 :  SNR  %.4f dB   SI-SDR  %.4f dB" % (P.std(ddof=1), PS.std(ddof=1)))
    print()
    print("  对照：上一版单次实验测得 0.32 dB（4 个测试信道）")
    tot_bad = sum(v[2]["seen"]["bad"] + v[2]["unseen"]["bad"] for v in results.values())
    tot_n = sum(v[2]["seen"]["n"] + v[2]["unseen"]["n"] for v in results.values())
    print("  全部 %d 个测试样本中，性能变差的共 %d 个" % (tot_n, tot_bad))
