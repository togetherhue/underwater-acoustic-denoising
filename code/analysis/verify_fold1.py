# -*- coding: utf-8 -*-
"""独立复算 fold1：不复用 train_fold.py / 大创_v3.py 的任何代码，
   自行实现数据管线（降采样+定长）与指标（SNR / SI-SDR），逐样本对照。
"""
import os, glob
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

DS = r"D:/project/dataset_folds/fold1"
OUT = r"D:/project/output_folds"
SEG, DOWNSAMPLE = 8000, 4


def pipeline(f):
    """完全独立复现：读 .mat -> resample_poly(1,4) -> 定长 8000"""
    m = sio.loadmat(f)
    n = np.asarray(m["noisy_signal"]).squeeze().astype(np.float64)
    c = np.asarray(m["clean_signal"]).squeeze().astype(np.float64)
    n = resample_poly(n, 1, DOWNSAMPLE)
    c = resample_poly(c, 1, DOWNSAMPLE)
    def fix(x):
        return x[:SEG] if len(x) >= SEG else np.pad(x, (0, SEG - len(x)))
    return fix(n), fix(c)


def snr_in(noisy, clean):
    """与 V3.snr_db(est=noisy, ref=clean) 同定义，但独立实现"""
    return 10 * np.log10((clean ** 2).sum() / ((noisy - clean) ** 2).sum())


def sisdr(est, ref):
    """与 V3.si_sdr_db 同定义，独立实现"""
    ref = ref - ref.mean(); est = est - est.mean()
    a = (est * ref).sum() / (ref ** 2).sum()
    proj = a * ref
    return 10 * np.log10((proj ** 2).sum() / ((est - proj) ** 2).sum())


print("=" * 84)
print("独立复算 fold1：数据管线与指标全部自行实现，逐样本与训练脚本记录对照")
print("=" * 84)

summary = {}
for split, tag in (("test_seen", "seen"), ("test_unseen", "unseen")):
    files = sorted(glob.glob(os.path.join(DS, split, "*.mat")))
    R = np.loadtxt(os.path.join(OUT, "result_lite_fold1_%s.csv" % tag),
                   delimiter=",", skiprows=1)
    assert len(R) == len(files), "样本数不符 %d vs %d" % (len(R), len(files))

    si, sd = [], []
    for f in files:
        n, c = pipeline(f)
        si.append(snr_in(n, c))
        sd.append(sisdr(n, c))
    si, sd = np.array(si), np.array(sd)

    print()
    print("--- %s (n=%d) ---" % (split, len(files)))
    print("  输入SNR    : 复算 %.4f   脚本 %.4f   逐样本最大差 %.3e"
          % (si.mean(), R[:, 0].mean(), np.abs(si - R[:, 0]).max()))
    print("  输入SI-SDR : 复算 %.4f   脚本 %.4f   逐样本最大差 %.3e"
          % (sd.mean(), R[:, 2].mean(), np.abs(sd - R[:, 2]).max()))
    g = R[:, 1] - R[:, 0]
    gs = R[:, 3] - R[:, 2]
    print("  模型增益   : SNR %+.4f   SI-SDR %+.4f" % (g.mean(), gs.mean()))
    print("  最差样本   : SNR %+.2f   SI-SDR %+.2f" % (g.min(), gs.min()))
    print("  变差样本   : SNR %d 个   SI-SDR %d 个" % ((g < 0).sum(), (gs < 0).sum()))
    summary[tag] = (g.mean(), gs.mean(), len(files))

print()
print("=" * 84)
(gsn, gsd, nsn) = summary["seen"]
(gun, gud, nun) = summary["unseen"]
print("★ 信道新颖度代价（独立复算）")
print("   SNR    : unseen %+.4f  -  seen %+.4f  =  %+.4f dB" % (gun, gsn, gun - gsn))
print("   SI-SDR : unseen %+.4f  -  seen %+.4f  =  %+.4f dB" % (gud, gsd, gud - gsd))
print()
print("  脚本 summary 记录 penalty_db = 0.0048")
