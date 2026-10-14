# -*- coding: utf-8 -*-
"""独立复算波形实验：不复用 train_wave.py / 大创_v3.py 的任何代码，
   自行实现数据管线（降采样+定长）与指标（SNR / SI-SDR），逐样本对照。
   同时验证 LFM / HFM 的配对性（噪声实现是否完全相同）。"""
import os, glob, csv
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

WAVE = r"D:/project/dataset_wave"
OUT = r"D:/project/output_wave"
SEG, DOWNSAMPLE = 8000, 4
SETS = [("LFM", "testA"), ("HFM", "testA"), ("LFM", "testC"), ("HFM", "testC")]
MODELS = ["lfmonly", "hfmonly", "mixed"]


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


print("=" * 92)
print("独立复算：数据管线与指标全部自行实现，逐样本与训练脚本记录对照")
print("=" * 92)

# ---------- 1. 配对性验证 ----------
print("\n[1] LFM 与 HFM 的配对性（噪声实现是否完全相同）")
for sp in ["testA", "testC"]:
    fL = sorted(glob.glob(os.path.join(WAVE, "LFM", sp, "*.mat")))
    fH = sorted(glob.glob(os.path.join(WAVE, "HFM", sp, "*.mat")))
    assert len(fL) == len(fH)
    dn, dc = 0.0, 0.0
    for a, b in zip(fL, fH):
        mL, mH = sio.loadmat(a), sio.loadmat(b)
        nL = np.asarray(mL["noisy_signal"]).squeeze(); cL = np.asarray(mL["clean_signal"]).squeeze()
        nH = np.asarray(mH["noisy_signal"]).squeeze(); cH = np.asarray(mH["clean_signal"]).squeeze()
        dn = max(dn, np.abs((nL - cL) - (nH - cH)).max())
        dc = max(dc, np.abs(cL - cH).max())
    print("    %-6s n=%4d  噪声最大差 %.2e   干净信号最大差 %.2e"
          % (sp, len(fL), dn, dc))

# ---------- 2. 输入指标独立复算 ----------
print("\n[2] 输入指标独立复算（与训练脚本记录逐样本对照）")
key = "lfmonly"
inputs = {}
for mod, split in SETS:
    files = sorted(glob.glob(os.path.join(WAVE, mod, split, "*.mat")))
    csvp = os.path.join(OUT, "result_%s_%s_%s.csv" % (key, mod, split))
    R = np.loadtxt(csvp, delimiter=",", skiprows=1)
    assert len(R) == len(files), "%s/%s 样本数不符" % (mod, split)
    si, sd = [], []
    for f in files:
        n_, c_ = pipeline(f)
        si.append(snr_in(n_, c_)); sd.append(sisdr(n_, c_))
    si, sd = np.array(si), np.array(sd)
    inputs[(mod, split)] = (si.mean(), sd.mean())
    print("    %-4s %-6s n=%4d  输入SNR 复算 %.4f / 脚本 %.4f (最大差 %.1e)"
          "   输入SI-SDR 最大差 %.1e"
          % (mod, split, len(files), si.mean(), R[:, 0].mean(),
             np.abs(si - R[:, 0]).max(), np.abs(sd - R[:, 2]).max()))

# ---------- 3. 从原始 CSV 重算增益与调制代价 ----------
print("\n[3] 从逐样本 CSV 重算调制代价 = 增益(HFM) − 增益(LFM)")
print("    %-10s %12s %12s %12s" % ("模型", "testA 代价", "testC 代价", "平均"))
print("    " + "-" * 50)
recomputed = {}
for tag in MODELS:
    vals = []
    for split in ["testA", "testC"]:
        g = {}
        for mod in ["LFM", "HFM"]:
            R = np.loadtxt(os.path.join(OUT, "result_%s_%s_%s.csv" % (tag, mod, split)),
                           delimiter=",", skiprows=1)
            g[mod] = (R[:, 1] - R[:, 0]).mean()
        vals.append(g["HFM"] - g["LFM"])
    recomputed[tag] = vals
    print("    %-10s %+12.4f %+12.4f %+12.4f" % (tag, vals[0], vals[1], np.mean(vals)))

print("\n    对照 summary CSV 记录：")
for tag in MODELS:
    p = os.path.join(OUT, "summary_%s.csv" % tag)
    d = {r["testset"]: float(r["snr_gain"]) for r in csv.DictReader(open(p, encoding="utf-8"))}
    print("      %-10s testA %+.4f  testC %+.4f" %
          (tag, d["HFM_testA"] - d["LFM_testA"], d["HFM_testC"] - d["LFM_testC"]))

# ---------- 4. 退化样本 ----------
print("\n[4] 逐样本检查：是否有样本被降噪变差")
tot = bad = 0
worst = 1e9
for tag in MODELS:
    for mod, split in SETS:
        R = np.loadtxt(os.path.join(OUT, "result_%s_%s_%s.csv" % (tag, mod, split)),
                       delimiter=",", skiprows=1)
        g = R[:, 1] - R[:, 0]
        tot += len(g); bad += int((g < 0).sum()); worst = min(worst, g.min())
print("    合计 %d 个（模型 × 测试集 × 样本）组合，变差 %d 个，最差 %+.2f dB" %
      (tot, bad, worst))
