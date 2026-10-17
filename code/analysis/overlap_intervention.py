# -*- coding: utf-8 -*-
"""
时频重叠干预实验：把「重叠 → 脉冲分支收益」从相关变成因果
================================================================================
【为什么做】
第七节发现脉冲分支收益与「干扰-信号时频重叠率」相关（r = −0.696, p < 0.001），
但那是<b>观察性</b>证据 —— 重叠率高的样本可能同时还有其他不同，无法排除混淆。
本实验直接<b>操纵</b>重叠率，看效应是否随之变化。

【设计】
取 base 条件 testA 的<b>真实信号 s 与真实背景噪声 v</b>，只把突发干扰 i
替换为<b>载频受控</b>的合成突发。合成突发的波形类型/时长/幅度分布与
gen_burst.m 完全一致（保持训练分布内），唯一改变的变量是<b>载频位置</b>，
从而把干扰能量推入或推出信号的时频支撑。

   载频远低于信号带（~300 Hz）  -> 重叠率 ≈ 0
   载频位于信号带内（~1150 Hz） -> 重叠率高

噪声 = s + v + i_synthetic  （s、v 为真实，仅 i 受控）

【重叠率定义（与第七节一致）】
   信号支撑   = 干净信号 STFT 中 |S| > 阈值 的时频单元
   干扰支撑   = 突发干扰 STFT 中 |I| > 阈值 的时频单元
   重叠率     = |交集| / |干扰支撑|

【预测】若机制成立，重叠率越高，脉冲分支收益 Δ = R5 − R4 越小。

用法：python overlap_intervention.py [--nmax 300]
"""
import os, sys, json, argparse, glob
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly, butter, lfilter, stft

# ---- 路径解析：兼容仓库布局与本地开发布局 ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDS = [os.path.normpath(os.path.join(_HERE, "..", "python")),
          r"C:\Users\32213\PycharmProjects\project1"]
for _p in [_HERE] + [c for c in _CANDS if os.path.isdir(c)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                      # noqa: E402
import v3_model as V3             # noqa: E402
import pamnet as PN               # noqa: E402
import pam_metrics as PM          # noqa: E402

WORK = r"E:/workbuddy/store/2026-10-12-22-50-08"
OUTD = os.path.join(WORK, "sens_out")
CKPT = r"D:/project/output_pamnet"
SEG, DS, FS = 8000, 4, 6000

# 合成突发参数：与 gen_burst.m / make_dataset_pam.m 的 cfg.burst 完全一致
BURST = dict(rate=5, dur_range=(0.002, 0.030), amp_range=(4, 25),
             type_prob=(0.25, 0.30, 0.25, 0.20),
             freq_range=(200, 2600), bw_frac=(0.15, 0.60))

R4_ARCH = dict(use_short=True, use_long=True, use_gate=True, use_pulse=False,
               freq_refine=2.0, pulse_freq=False)
R5_ARCH = dict(use_short=True, use_long=True, use_gate=True, use_pulse=True,
               freq_refine=2.0, pulse_freq=True)
R4_CKPTS = ["model_pamnet_freq_nopulse_gn.pth", "model_pamnet_freq_nopulse_gn_s2.pth",
            "model_pamnet_freq_nopulse_gn_s3.pth", "model_pamnet_freq_nopulse_gn_s4.pth",
            "model_pamnet_freq_nopulse_gn_s5.pth"]
R5_CKPTS = ["model_pamnet_freq_pf_gn.pth", "model_pamnet_freq_pf_gn_s2.pth",
            "model_pamnet_freq_pf_gn_s3.pth", "model_pamnet_freq_pf_gn_s4.pth",
            "model_pamnet_freq_pf_gn_s5.pth"]

# 载频档位：从远低于信号带到信号带内部
FC_LEVELS = [300, 500, 700, 900, 1100, 1300, 1500]


# ============================ 合成突发（gen_burst.m 的 Python 复现）============================
def poissrnd(lam, rng):
    if lam <= 0:
        return 0
    if lam < 30:                      # Knuth
        L, k, p = np.exp(-lam), 0, 1.0
        while True:
            k += 1
            p *= rng.random()
            if p <= L:
                return k - 1
    return max(0, int(round(lam + np.sqrt(lam) * rng.standard_normal())))


def gen_burst_py(N, fs, rng, fc_override=None):
    """复现 gen_burst.m；fc_override 非 None 时强制所有突发的载频（用于控制重叠率）。
    返回 (i, mask)"""
    T = N / fs
    i = np.zeros(N)
    mask = np.zeros(N, dtype=bool)
    n_burst = poissrnd(BURST["rate"] * T, rng)
    for _ in range(n_burst):
        tau = rng.random() * T
        d = BURST["dur_range"][0] + (BURST["dur_range"][1] - BURST["dur_range"][0]) * rng.random()
        A = BURST["amp_range"][0] + (BURST["amp_range"][1] - BURST["amp_range"][0]) * rng.random()
        r = rng.random()
        c = np.cumsum(np.array(BURST["type_prob"]) / sum(BURST["type_prob"]))
        ty = int(np.searchsorted(c, r, side="left")) + 1
        if ty > 4:
            ty = 4
        n0 = max(0, int(round(tau * fs)))
        if ty == 1:                                    # 尖峰 1~3 点
            ns = 1 + int(rng.random() * 3)
            w = np.ones(ns) * (2 * (rng.random() > 0.5) - 1)
        elif ty == 2:                                  # 衰减振荡
            ns = max(3, int(round(d * fs)))
            fc = fc_override if fc_override else (
                BURST["freq_range"][0] + (BURST["freq_range"][1] - BURST["freq_range"][0]) * rng.random())
            tt = np.arange(ns) / fs
            tau_d = d / (2 + 3 * rng.random())
            w = np.exp(-tt / tau_d) * np.sin(2 * np.pi * fc * tt + 2 * np.pi * rng.random())
            w = w / (np.max(np.abs(w)) + 1e-12)
        elif ty == 3:                                  # 限带噪声突发
            ns = max(3, int(round(d * fs)))
            fc = fc_override if fc_override else (
                BURST["freq_range"][0] + (BURST["freq_range"][1] - BURST["freq_range"][0]) * rng.random())
            bw = fc * (BURST["bw_frac"][0] + (BURST["bw_frac"][1] - BURST["bw_frac"][0]) * rng.random())
            lo = max(fc - bw / 2, 20) / (fs / 2)
            hi = min(fc + bw / 2, fs / 2 * 0.98) / (fs / 2)
            if hi <= lo:
                hi = min(lo * 1.5, 0.98)
            b, a = butter(4, [lo, hi], btype="bandpass")
            w = lfilter(b, a, rng.standard_normal(ns))   # 对应 MATLAB 的 filter（因果 IIR）
            w = w / (np.max(np.abs(w)) + 1e-12)
        else:                                          # 矩形包络窄带
            ns = max(3, int(round(d * fs)))
            fc = fc_override if fc_override else (
                BURST["freq_range"][0] + (BURST["freq_range"][1] - BURST["freq_range"][0]) * rng.random())
            tt = np.arange(ns) / fs
            w = np.sin(2 * np.pi * fc * tt + 2 * np.pi * rng.random())
        idx = np.arange(n0, min(N, n0 + len(w)))
        if len(idx) == 0:
            continue
        i[idx] += A * w[:len(idx)]
        mask[idx] = True
    return i, mask


# ============================ 时频支撑与重叠率 ============================
def tf_support(x, n_fft=512, hop=128, thr_db=-25.0):
    """返回 STFT 幅度超过 (max - thr_db) 的布尔支撑"""
    f, t, Z = stft(x, fs=FS, nperseg=n_fft, noverlap=n_fft - hop)
    mag = np.abs(Z)
    mx = mag.max()
    if mx <= 0:
        return np.zeros_like(mag, dtype=bool), mag
    return mag > mx * (10 ** (thr_db / 20.0)), mag


def inband_frac(i_syn, lo=780.0, hi=1520.0):
    """突发干扰落在信号频带 [780,1520] Hz 内的能量占比（动态范围大，易于解读）"""
    E = np.abs(np.fft.rfft(i_syn)) ** 2
    f = np.fft.rfftfreq(len(i_syn), 1 / FS)
    tot = E.sum()
    if tot <= 0:
        return float("nan")
    return float(E[(f >= lo) & (f <= hi)].sum() / tot)


def overlap_rate(sig_sup, int_sup):
    """|交集| / |干扰支撑|"""
    den = int_sup.sum()
    if den == 0:
        return float("nan")
    return float((sig_sup & int_sup).sum()) / float(den)


# ============================ 模型推理 ============================
def load_model(arch, ckpt):
    m = PN.PAMNet(c=24, n_fft=V3.CFG["n_fft"], hop=V3.CFG["hop_length"],
                  norm="gn", **arch).to(V3.DEVICE)
    m.load_state_dict(torch.load(os.path.join(CKPT, ckpt), map_location=V3.DEVICE),
                      strict=True)
    return m.to(V3.DEVICE).eval()


@torch.no_grad()
def frame_gains(model, noisy, bs=32):
    """批量前向，返回每个样本的增强输出（float64 numpy 列表）。
    自动跟随模型所在设备（CPU 或 CUDA）。"""
    dev = next(model.parameters()).device
    outs = []
    for s in range(0, len(noisy), bs):
        nb = torch.from_numpy(np.asarray(noisy[s:s + bs])).unsqueeze(1).to(dev)
        e = model(nb)
        for k in range(e.shape[0]):
            outs.append(e[k, 0].detach().cpu().numpy().astype(np.float64))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=300)
    ap.add_argument("--seed", type=int, default=777)
    A = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

    # ---- 1. 取真实信号 s 与真实背景噪声 v ----
    files = sorted(glob.glob(os.path.join(r"D:/project/dataset_pam/alphastable",
                                          "testA", "*.mat")))
    k = min(A.nmax, len(files))
    S, V, MASK = [], [], []
    for f in files[:k]:
        m = sio.loadmat(f)
        n = np.asarray(m["noisy_signal"]).squeeze().astype(np.float32)
        c = np.asarray(m["clean_signal"]).squeeze().astype(np.float32)
        imp = np.asarray(m["imp"]).squeeze().astype(np.float32)
        n = resample_poly(n, 1, DS).astype(np.float32)
        c = resample_poly(c, 1, DS).astype(np.float32)
        imp = resample_poly(imp, 1, DS).astype(np.float32)

        def fx(x):
            return x[:SEG] if len(x) >= SEG else np.pad(x, (0, SEG - len(x)))
        n, c, imp = fx(n), fx(c), fx(imp)
        S.append(c); V.append((n - c - imp).astype(np.float32))
        MASK.append(np.asarray(m["imp_mask"]).squeeze().astype(np.float64))
    S = np.stack(S); V = np.stack(V)
    print("取真实样本 %d 个（信号 + 背景噪声均为真实，仅突发干扰受控）" % len(S))

    # ---- 2. 加载模型（前 3 个种子）----
    m4 = [load_model(R4_ARCH, c) for c in R4_CKPTS]
    m5 = [load_model(R5_ARCH, c) for c in R5_CKPTS]
    print("已加载模型: R4 = %d 种子, R5 = %d 种子" % (len(m4), len(m5)))

    # ---- 3. 逐载频档位做干预 ----
    res = {}
    for fc in FC_LEVELS:
        rng = np.random.default_rng(A.seed + fc)
        NOISY, OVS, MASKS, IB = [], [], [], []
        for i in range(len(S)):
            i_syn, mk = gen_burst_py(SEG, FS, rng, fc_override=fc)
            i_syn = i_syn * float(np.sqrt(np.mean(S[i] ** 2)))   # 幅度以信号 rms 为单位
            NOISY.append((S[i] + V[i] + i_syn).astype(np.float32))
            sig_sup, _ = tf_support(S[i].astype(np.float64))
            int_sup, _ = tf_support(i_syn)
            OVS.append(overlap_rate(sig_sup, int_sup))
            IB.append(inband_frac(i_syn))
            MASKS.append(mk.astype(float))
        NOISY = np.stack(NOISY); OVS = np.array(OVS)
        MASKS = np.stack(MASKS); IB = np.array(IB)
        ov_med = float(np.nanmedian(OVS))
        ib_med = float(np.nanmedian(IB))

        # 逐种子算 Δ = R5 − R4
        ds = []
        for si in range(len(m4)):
            e4 = frame_gains(m4[si], NOISY)
            e5 = frame_gains(m5[si], NOISY)
            per = []
            for j in range(len(NOISY)):
                a = PM._stats_1d(e5[j], S[j].astype(np.float64),
                                 NOISY[j].astype(np.float64), PM.FRAME, MASKS[j])
                b = PM._stats_1d(e4[j], S[j].astype(np.float64),
                                 NOISY[j].astype(np.float64), PM.FRAME, MASKS[j])
                for key in ("frmgain_med", "frmgain_clean", "frmgain_imp",
                            "frmgain_imp_min"):
                    pass
                per.append((a["frmgain_imp"] - b["frmgain_imp"],
                            a["frmgain_clean"] - b["frmgain_clean"]))
            per = np.array(per)                      # [n, 2]
            ds.append(per)
        ds = np.stack(ds)                            # [n_seed, n, 2]
        # 干扰帧指标在「该样本没有任何有效干扰帧」时无定义（NaN），
        # 这类样本应剔除而不是让 NaN 毒化均值
        fin = np.isfinite(ds).all(axis=(0, 2))       # [n] 该样本在所有种子、两个指标下都有定义
        n_valid = int(fin.sum())
        d_imp = ds[:, fin, 0].mean(axis=1)           # 各种子均值
        d_cln = ds[:, fin, 1].mean(axis=1)
        # 保存逐样本（种子平均）数据，供跨档位合并的逐样本相关分析
        per_sample = dict(ov=OVS.tolist(), ib=IB.tolist(),
                          d_imp=ds[:, :, 0].mean(axis=0).tolist(),
                          d_cln=ds[:, :, 1].mean(axis=0).tolist(),
                          valid=fin.tolist())
        res["fc%d" % fc] = dict(fc=fc, ov_median=ov_med,
                                ov_mean=float(np.nanmean(OVS)),
                                d_imp_mean=float(d_imp.mean()),
                                d_imp_sd=float(d_imp.std(ddof=1)) if len(d_imp) > 1 else 0.0,
                                d_cln_mean=float(d_cln.mean()),
                                d_cln_sd=float(d_cln.std(ddof=1)) if len(d_cln) > 1 else 0.0,
                                n_seed=len(d_imp), n_sample=len(NOISY),
                                n_valid=n_valid, ib_median=ib_med,
                                ib_mean=float(np.nanmean(IB)),
                                per_sample=per_sample)
        print("[fc=%4d Hz] 带内占比中位 %.3f | Δ干扰帧 %+.3f ± %.3f dB | Δ干净帧 %+.3f | 有效样本 %d/%d"
              % (fc, ib_med, res["fc%d" % fc]["d_imp_mean"],
                 res["fc%d" % fc]["d_imp_sd"], res["fc%d" % fc]["d_cln_mean"],
                 res["fc%d" % fc]["n_valid"], len(NOISY)),
              flush=True)

    # ---- 4. 重叠率 vs 收益 的相关（跨载频档位）----
    print()
    print("=" * 92)
    print("重叠率（受控）→ 脉冲分支收益 Δ：干预实验")
    print("=" * 92)
    ovs = np.array([res[k]["ib_median"] for k in sorted(res, key=lambda x: int(x[2:]))])
    dimp = np.array([res[k]["d_imp_mean"] for k in sorted(res, key=lambda x: int(x[2:]))])
    fcs = np.array([res[k]["fc"] for k in sorted(res, key=lambda x: int(x[2:]))])
    order = np.argsort(ovs)
    print("%-12s %-14s %-12s %-16s" % ("载频(Hz)", "带内占比中位", "TF重叠率中位", "Δ干扰帧(dB)"))
    for i in order:
        k = "fc%d" % fcs[i]
        print("%-12d %-14.3f %-12.3f %-16s" % (fcs[i], res[k]["ib_median"],
              res[k]["ov_median"],
              "%+.3f ± %.3f" % (res[k]["d_imp_mean"], res[k]["d_imp_sd"])))
    print("-" * 92)
    if len(ovs) >= 3 and ovs.std() > 0:
        r = float(np.corrcoef(ovs, dimp)[0, 1])
        n = len(ovs)
        t = r * np.sqrt((n - 2) / max(1e-12, 1 - r ** 2))
        from math import erfc, sqrt
        p = float(erfc(abs(t) / sqrt(2.0)))
        print("带内能量占比 与 Δ干扰帧 的相关：r = %+.3f   (n=%d 个载频档位, t=%+.2f, p=%.4f)"
              % (r, n, t, p))
        print("方向：%s" % ("与第七节的观察一致（重叠率越高，收益越小）"
                            if r < 0 else "与第七节的观察【相反】"))
    # ---- 逐样本合并：跨全部载频档位，按带内占比分箱看 Δ 的趋势 ----
    allo = np.concatenate([[res[k]["per_sample"]["ib"][j]
                            for k in res for j in range(res[k]["n_sample"])]])
    alld = np.concatenate([[res[k]["per_sample"]["d_imp"][j]
                            for k in res for j in range(res[k]["n_sample"])]])
    alv = np.concatenate([[res[k]["per_sample"]["valid"][j]
                           for k in res for j in range(res[k]["n_sample"])]])
    allo, alld = allo[alv], alld[alv]
    okf = np.isfinite(allo) & np.isfinite(alld)
    allo, alld = allo[okf], alld[okf]
    print()
    print("=" * 92)
    print("逐样本合并（n=%d，跨全部载频档位）：按带内能量占比分箱" % len(allo))
    print("=" * 92)
    edges = [0, 0.05, 0.3, 0.7, 1.0001]
    labs = ["<0.05", "0.05-0.3", "0.3-0.7", ">0.7"]
    print("%-12s %8s %10s %12s" % ("带内占比", "n", "Δ干扰帧", "种子间SD"))
    for i in range(len(labs)):
        sel = (allo >= edges[i]) & (allo < edges[i + 1])
        if sel.sum() < 5:
            continue
        print("%-12s %8d %10.3f %12.3f" % (labs[i], int(sel.sum()),
              float(np.mean(alld[sel])), float(np.std(alld[sel]) / np.sqrt(max(1, sel.sum())))))
    print("-" * 92)
    if allo.std() > 0:
        r = float(np.corrcoef(allo, alld)[0, 1])
        n = len(allo)
        t = r * np.sqrt((n - 2) / max(1e-12, 1 - r ** 2))
        from math import erfc, sqrt
        p = float(erfc(abs(t) / sqrt(2.0)))
        print("逐样本相关：r = %+.4f  (n=%d, t=%+.2f, p=%.2e)" % (r, n, t, p))
        print("（注意：逐样本 Δ 噪声很大，此相关只作参考；档位级的相关见上表）")

    with open(os.path.join(OUTD, "overlap_intervention.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("\n-> sens_out/overlap_intervention.json")


if __name__ == "__main__":
    main()
