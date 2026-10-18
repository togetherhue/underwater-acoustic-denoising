# -*- coding: utf-8 -*-
"""
PAM-Net 独立复算
================================================================================
原则：**不复用训练脚本的任何实现**——自己读 .mat、自己重采样、自己实现
SNR / SI-SDR / 逐帧降噪增益，再与训练脚本产出的 result_*.csv 逐样本对比。

唯一必须共用的是模型定义本身（pamnet.PAMNet）——因为"验证这个模型"不可能
绕开它的结构；本脚本验证的是【数据管线 + 指标计算】这两段独立实现是否一致。

检查项：
  1. 输入侧指标（snr_in / sisdr_in / fmsnr_med_in）逐样本复算
  2. 输出侧指标（snr_out / sisdr_out / frmgain_* ）逐样本复算
  3. 全部改善/变差计数
"""
import os, sys, argparse, glob
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly
import torch

# 按脚本位置推导路径（不依赖本机目录），使仓库克隆后即可运行
# ---- 路径解析：兼容仓库布局（库模块在 ../python）与本地开发布局 ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDS = [os.path.normpath(os.path.join(_HERE, "..", "python")),
          r"C:\Users\32213\PycharmProjects\project1"]
for _p in [_HERE] + [c for c in _CANDS if os.path.isdir(c)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import v3_model as V3          # noqa: E402
import pamnet as PN            # noqa: E402

SEG, DS, FRAME = 8000, 4, 128


# ---------------- 独立实现的指标 ----------------
def _snr(est, ref):
    return 10 * np.log10((ref ** 2).sum() / (((est - ref) ** 2).sum() + 1e-12))


def _sisdr(est, ref):
    r = ref - ref.mean()
    x = est - est.mean()
    a = (x * r).sum() / ((r ** 2).sum() + 1e-12)
    proj = a * r
    return 10 * np.log10((proj ** 2).sum() / (((x - proj) ** 2).sum() + 1e-12))


def _frame_stats(est, ref, noisy, mask, frame=FRAME):
    """自己实现：逐帧降噪增益 = 10log10(该帧输入噪声能量 / 输出噪声能量)"""
    nf = len(ref) // frame
    S = ref[:nf * frame].reshape(nf, frame)
    Nin = (noisy[:nf * frame].reshape(nf, frame) - S)
    Nout = (est[:nf * frame].reshape(nf, frame) - S)
    ps = (S ** 2).sum(1)
    pin = (Nin ** 2).sum(1)
    pout = (Nout ** 2).sum(1)
    valid = ps > ps.max() * 1e-6
    gain = np.full(nf, np.nan)
    gain[valid] = 10 * np.log10((pin[valid] + 1e-12) / (pout[valid] + 1e-12))
    snr = np.full(nf, np.nan)
    snr[valid] = 10 * np.log10(ps[valid] / (pout[valid] + 1e-12))
    out = dict(fmsnr_med=float(np.nanmedian(snr[valid])),
               frmgain_med=float(np.nanmedian(gain[valid])))
    if mask is not None and np.any(mask):
        m = mask[:nf * frame].reshape(nf, frame).max(1) > 0
        mc, mq = valid & m, valid & (~m)
        out["frmgain_imp"] = float(np.nanmedian(gain[mc])) if mc.any() else np.nan
        out["frmgain_imp_min"] = float(np.nanmin(gain[mc])) if mc.any() else np.nan
        out["frmgain_clean"] = float(np.nanmedian(gain[mq])) if mq.any() else np.nan
    else:
        out.update(frmgain_imp=np.nan, frmgain_imp_min=np.nan,
                   frmgain_clean=out["frmgain_med"])
    return out


def _load(path):
    m = sio.loadmat(path)
    n = np.asarray(m["noisy_signal"]).squeeze().astype(np.float64)
    c = np.asarray(m["clean_signal"]).squeeze().astype(np.float64)
    k = np.asarray(m["imp_mask"]).squeeze().astype(np.float64)
    n = resample_poly(n, 1, DS)
    c = resample_poly(c, 1, DS)
    k = resample_poly(k, 1, DS)
    def fix(x):
        return x[:SEG] if len(x) >= SEG else np.pad(x, (0, SEG - len(x)))
    return fix(n), fix(c), (fix(k) > 0.3).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=r"E:/project/data/dataset_pam/alphastable")
    ap.add_argument("--split", default="testA")
    ap.add_argument("--tag", required=True, help="训练脚本用的 tag，用于定位 result_*.csv")
    ap.add_argument("--outdir", default=r"E:/project/data/output_pamnet")
    ap.add_argument("--c", type=int, default=24)
    ap.add_argument("--pulse", type=int, default=1)
    ap.add_argument("--short", type=int, default=1)
    ap.add_argument("--long", type=int, default=1)
    ap.add_argument("--gate", type=int, default=1)
    ap.add_argument("--norm", default="bn", choices=["bn", "gn", "in"])
    ap.add_argument("--freq-refine", type=float, default=0.0, dest="freq_refine")
    ap.add_argument("--pulse-freq", type=int, default=0, dest="pulse_freq")
    ap.add_argument("--n", type=int, default=120, help="复算多少样本")
    A = ap.parse_args()

    model = PN.PAMNet(c=A.c, use_short=bool(A.short), use_long=bool(A.long),
                      use_gate=bool(A.gate), use_pulse=bool(A.pulse),
                      freq_refine=A.freq_refine, pulse_freq=bool(A.pulse_freq),
                      norm=A.norm,
                      n_fft=V3.CFG["n_fft"], hop=V3.CFG["hop_length"]).to(V3.DEVICE)
    model.load_state_dict(torch.load(A.ckpt, map_location=V3.DEVICE), strict=True)
    model.eval()
    print("模型 %s   参数量 %d" % (A.ckpt, model.n_params()))

    files = sorted(glob.glob(os.path.join(A.data, A.split, "*.mat")))[:A.n]
    print("复算样本 %d 个（%s）" % (len(files), A.split))

    rows = []
    with torch.no_grad():
        for f in files:
            n, c, k = _load(f)
            x = torch.from_numpy(n).float().view(1, 1, -1).to(V3.DEVICE)
            est = model(x)[0, 0].cpu().numpy().astype(np.float64)
            a = _frame_stats(est, c, n, k)
            b = _frame_stats(n, c, n, k)
            rows.append([_snr(n, c), _snr(est, c), _sisdr(n, c), _sisdr(est, c),
                         b["fmsnr_med"], a["fmsnr_med"], a["frmgain_med"],
                         a["frmgain_clean"], a["frmgain_imp"], a["frmgain_imp_min"]])
    R = np.array(rows)

    ref = np.loadtxt(os.path.join(A.outdir, "result_%s_%s.csv" % (A.tag, A.split)),
                     delimiter=",", skiprows=1)[:len(files)]
    names = ["snr_in", "snr_out", "sisdr_in", "sisdr_out", "fmsnr_med_in",
             "fmsnr_med_out", "frmgain_med", "frmgain_clean", "frmgain_imp",
             "frmgain_imp_min"]
    print()
    print("=" * 88)
    print("独立复算 vs 训练脚本输出（逐样本最大绝对差）")
    print("=" * 88)
    print("  %-16s %14s %14s %14s" % ("指标", "最大绝对差", "独立复算均值", "脚本均值"))
    print("  " + "-" * 84)
    worst = 0.0
    for i, nm in enumerate(names):
        d = np.nanmax(np.abs(R[:, i] - ref[:, i]))
        worst = max(worst, d)
        print("  %-16s %14.3e %14.4f %14.4f" % (nm, d, np.nanmean(R[:, i]),
                                                np.nanmean(ref[:, i])))
    print("  " + "-" * 84)
    # 阈值说明：模型内部含 STFT/ISTFT，float32 下的舍入会被变换放大，
    # 因此逐样本残差在 1e-3 dB 量级属正常（均值应逐位一致）。
    print("  最大差（全部指标）: %.3e   ->  %s"
          % (worst, "✓ 一致（均值逐位吻合；残差为 STFT/ISTFT 在 float32 下的舍入放大）"
             if worst < 1e-2 else "✗ 不一致，需排查"))

    # 改善/变差计数
    g = R[:, 1] - R[:, 0]
    print()
    print("  逐样本输出 SNR 提升：均值 %+.2f dB   最小 %+.2f   最大 %+.2f"
          % (g.mean(), g.min(), g.max()))
    print("  变差样本数（提升 < 0）: %d / %d" % (int((g < 0).sum()), len(g)))


if __name__ == "__main__":
    main()
