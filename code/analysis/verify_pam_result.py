# -*- coding: utf-8 -*-
"""独立复算阶段0结果：不复用 train_pam.py 的任何代码
   ① 从 .mat 独立重算输入侧指标，与结果 CSV 对照
   ② 重新加载模型对子集推理，独立复算输出侧指标，与 CSV 对照
"""
import os, sys, glob, csv
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly
import torch

_HERE = r"C:/Users/32213/PycharmProjects/project1"
sys.path.insert(0, _HERE)
import v3_model as V3
import pam_metrics as PM

OUT = r"D:/project/output_pam"
DATA = r"D:/project/dataset_pam"
SEG, DS = 8000, 4
RUNS = [("pam_freq", "alphastable", 0, 2.0),
        ("pam_full", "alphastable", 1, 1.0),
        ("pam_freq_g", "gauss", 0, 2.0)]


def pipeline(f):
    """独立实现：读 .mat -> resample_poly(1,4) -> 定长 8000"""
    m = sio.loadmat(f)
    def rs(x):
        return resample_poly(np.asarray(x).squeeze().astype(np.float64), 1, DS)
    n = rs(m["noisy_signal"]); c = rs(m["clean_signal"])
    k = rs(np.asarray(m["imp_mask"]).squeeze().astype(np.float64))
    def fix(x):
        return x[:SEG] if len(x) >= SEG else np.pad(x, (0, SEG - len(x)))
    return fix(n), fix(c), (fix(k) > 0.3).astype(np.float64)


print("=" * 100)
print("独立复算：输入侧指标（从 .mat 直接重算，与结果 CSV 对照）")
print("=" * 100)
print("%-12s %-6s %6s %14s %14s %14s" %
      ("模型", "测试集", "n", "snr_in最大差", "sisdr_in最大差", "帧SNR入最大差"))
for tag, noise, ut, fch in RUNS:
    for ts in ["testA", "testC"]:
        csvp = os.path.join(OUT, "result_%s_%s.csv" % (tag, ts))
        R = np.loadtxt(csvp, delimiter=",", skiprows=1)
        files = sorted(glob.glob(os.path.join(DATA, noise, ts, "*.mat")))
        assert len(R) == len(files), "%s/%s 样本数不符" % (tag, ts)
        d1 = d2 = d3 = 0.0
        for i, f in enumerate(files):
            n, c, k = pipeline(f)
            d1 = max(d1, abs(PM.snr_db(torch.from_numpy(n).float(),
                                       torch.from_numpy(c).float()).item() - R[i, 0]))
            d2 = max(d2, abs(PM.si_sdr_db(torch.from_numpy(n).float(),
                                          torch.from_numpy(c).float()).item() - R[i, 2]))
            st = PM._stats_1d(n, c, n, PM.FRAME, k)
            d3 = max(d3, abs(st["fmsnr_med"] - R[i, 4]))
        print("%-12s %-6s %6d %14.2e %14.2e %14.2e" % (tag, ts, len(R), d1, d2, d3))

print()
print("=" * 100)
print("独立复算：输出侧指标（重新加载模型推理 60 个样本，与结果 CSV 对照）")
print("=" * 100)
print("%-12s %-6s %6s %14s %14s %14s %14s" %
      ("模型", "测试集", "n", "snr_out最大差", "帧增益中位差", "干净帧增益差", "干扰帧增益差"))
for tag, noise, ut, fch in RUNS:
    mp = os.path.join(OUT, "model_%s.pth" % tag)
    model = V3.DenoiseNet(V3.CFG["n_fft"], V3.CFG["hop_length"], 64,
                          use_time=bool(ut), use_freq=True, fch=fch).to(V3.DEVICE).eval()
    model.load_state_dict(torch.load(mp, map_location=V3.DEVICE))
    for ts in ["testA", "testC"]:
        R = np.loadtxt(os.path.join(OUT, "result_%s_%s.csv" % (tag, ts)),
                       delimiter=",", skiprows=1)
        files = sorted(glob.glob(os.path.join(DATA, noise, ts, "*.mat")))
        idx = np.linspace(0, len(files) - 1, 60).astype(int)
        d1 = d2 = d3 = d4 = 0.0
        with torch.no_grad():
            for i in idx:
                n, c, k = pipeline(files[i])
                nt = torch.from_numpy(n).float().unsqueeze(0).unsqueeze(0).to(V3.DEVICE)
                e = model(nt)[0, 0].cpu().numpy().astype(np.float64)
                d1 = max(d1, abs(PM.snr_db(torch.from_numpy(e).float(),
                                           torch.from_numpy(c).float()).item() - R[i, 1]))
                st = PM._stats_1d(e, c, n, PM.FRAME, k)
                d2 = max(d2, abs(st["frmgain_med"] - R[i, 6]))
                d3 = max(d3, abs(st["frmgain_clean"] - R[i, 7]))
                if not np.isnan(st["frmgain_imp"]):
                    d4 = max(d4, abs(st["frmgain_imp"] - R[i, 8]))
        print("%-12s %-6s %6d %14.2e %14.2e %14.2e %14.2e" %
              (tag, ts, len(idx), d1, d2, d3, d4))

print()
print("说明：最大差在 1e-5 量级即为 CSV 用 float32 存储造成的舍入，属正常。")
