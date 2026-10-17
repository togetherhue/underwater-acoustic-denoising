# -*- coding: utf-8 -*-
"""
时频对齐干预实验（决定性版本）
================================================================================
【为什么需要这一步】
第一版干预只改变了突发干扰的【载频】（带内 vs 带外），结果是**零效应**
（档位级 r = +0.40, p = 0.33；方向还与第七节相反）。

原因很清楚：信号是**扫频 chirp**（800→1500 Hz），"载频落在信号带内"并不等于
"与信号的瞬时频率对齐"。一个 1100 Hz 的突发，只有当 chirp 恰好扫到 1100 Hz
的那一小段时间里才真正重叠 —— 实测 TF 支撑交集率只有 0.073。

所以真正该操纵的自变量是【突发时刻与 chirp 扫频的瞬时对齐程度】。

【本实验设计】
对每个样本，从干净信号的 STFT 逐帧主频提取瞬时频率曲线 f_inst(t)。
选一个载频 fc（在信号扫频范围内随机），生成一个固定载频的突发，然后：

  对齐组  ：把突发放在 f_inst(t) ≈ fc 的时刻  -> TF 重叠率高
  错开组  ：把突发放在 f_inst(t) 远离 fc 的时刻 -> TF 重叠率低

两组用**同一个突发波形**（同 seed、同幅度/时长/带宽），唯一差别是放置时刻。
这是真正的单因素操纵：波形、信号、背景噪声、突发数量全部相同。

【预测】若第七节的机制成立，对齐组的脉冲分支收益 Δ 应显著小于错开组。

用法：python tf_alignment_intervention.py [--nmax 200]
"""
import os, sys, json, argparse, glob
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly, stft

_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDS = [os.path.normpath(os.path.join(_HERE, "..", "python")),
          r"C:\Users\32213\PycharmProjects\project1"]
for _p in [_HERE] + [c for c in _CANDS if os.path.isdir(c)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                      # noqa: E402
import pam_metrics as PM          # noqa: E402
import overlap_intervention as OI  # noqa: E402 复用模型加载与突发生成

WORK = r"E:/workbuddy/store/2026-10-12-22-50-08"
OUTD = os.path.join(WORK, "sens_out")
FS, SEG = 6000, 8000
NFFT, HOP = 512, 128


def inst_freq(clean):
    """逐帧瞬时主频（STFT 幅度峰值所在频率）"""
    f, t, Z = stft(clean, fs=FS, nperseg=NFFT, noverlap=NFFT - HOP)
    mag = np.abs(Z)
    return f, t, f[np.argmax(mag, axis=0)], mag


def place_burst(seg_len, fc, rng, target_t=None):
    """生成一个固定载频的突发，可指定放置的采样时刻 target_t（None = 随机）"""
    # 单突发：类型 3（限带噪声），时长与带宽按训练分布
    d = 0.002 + (0.030 - 0.002) * rng.random()
    ns = max(3, int(round(d * FS)))
    bw = fc * (0.15 + (0.60 - 0.15) * rng.random())
    lo = max(fc - bw / 2, 20) / (FS / 2)
    hi = min(fc + bw / 2, FS / 2 * 0.98) / (FS / 2)
    if hi <= lo:
        hi = min(lo * 1.5, 0.98)
    from scipy.signal import butter, lfilter
    b, a = butter(4, [lo, hi], btype="bandpass")
    w = lfilter(b, a, rng.standard_normal(ns))
    w = w / (np.max(np.abs(w)) + 1e-12)
    A = 4 + (25 - 4) * rng.random()
    n0 = int(target_t) if target_t is not None else int(rng.random() * (seg_len - ns))
    n0 = max(0, min(n0, seg_len - ns - 1))
    i = np.zeros(seg_len)
    mk = np.zeros(seg_len, dtype=bool)
    i[n0:n0 + ns] = A * w
    mk[n0:n0 + ns] = True
    return i, mk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=200)
    ap.add_argument("--seed", type=int, default=4242)
    A = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

    files = sorted(glob.glob(os.path.join(r"D:/project/dataset_pam/alphastable",
                                          "testA", "*.mat")))[:A.nmax]
    S, V = [], []
    for f in files:
        m = sio.loadmat(f)
        n = np.asarray(m["noisy_signal"]).squeeze().astype(np.float32)
        c = np.asarray(m["clean_signal"]).squeeze().astype(np.float32)
        imp = np.asarray(m["imp"]).squeeze().astype(np.float32)
        n = resample_poly(n, 1, 4).astype(np.float32)
        c = resample_poly(c, 1, 4).astype(np.float32)
        imp = resample_poly(imp, 1, 4).astype(np.float32)
        fx = lambda x: x[:SEG] if len(x) >= SEG else np.pad(x, (0, SEG - len(x)))
        c, n, imp = fx(c), fx(n), fx(imp)
        S.append(c); V.append((n - c - imp).astype(np.float32))
    S = np.stack(S); V = np.stack(V)
    print("样本 %d 个" % len(S))

    m4 = [OI.load_model(OI.R4_ARCH, c) for c in OI.R4_CKPTS]
    m5 = [OI.load_model(OI.R5_ARCH, c) for c in OI.R5_CKPTS]
    print("模型: R4 %d 种子, R5 %d 种子" % (len(m4), len(m5)))

    # ---- 构造两组：对齐 / 错开 ----
    rng = np.random.default_rng(A.seed)
    NOISY, MASKS, OVS, GRP, FC = [], [], [], [], []
    for i in range(len(S)):
        f, t, f_inst, mag = inst_freq(S[i].astype(np.float64))
        # 只在信号能量足够强的帧里选（避开零填充区）
        e = mag.sum(axis=0)
        good = e > e.max() * 1e-3
        if good.sum() < 8:
            continue
        fr = np.where(good)[0]
        # 随机选一个载频 = 某帧的瞬时频率
        j0 = fr[rng.integers(len(fr))]
        fc = float(f_inst[j0])
        if not (300 < fc < 2500):
            continue
        # 对齐：f_inst 接近 fc 的帧；错开：f_inst 远离 fc 的帧
        dist = np.abs(f_inst - fc)
        dist_good = np.where(good)[0]
        al = dist_good[np.argmin(dist[dist_good])]
        un = dist_good[np.argmax(dist[dist_good])]
        for grp, fr_idx in (("aligned", al), ("offset", un)):
            tt = t[fr_idx] * FS                      # 帧中心对应的采样点
            i_syn, mk = place_burst(SEG, fc, np.random.default_rng(A.seed + i),
                                    target_t=tt)
            i_syn = i_syn * float(np.sqrt(np.mean(S[i] ** 2)))
            NOISY.append((S[i] + V[i] + i_syn).astype(np.float32))
            MASKS.append(mk.astype(float))
            sig_sup, _ = OI.tf_support(S[i].astype(np.float64))
            int_sup, _ = OI.tf_support(i_syn)
            OVS.append(OI.overlap_rate(sig_sup, int_sup))
            GRP.append(grp); FC.append(fc)
    NOISY = np.stack(NOISY); MASKS = np.stack(MASKS)
    OVS = np.array(OVS); GRP = np.array(GRP); FC = np.array(FC)
    print("构造 %d 个测试样本（对齐 %d / 错开 %d）"
          % (len(NOISY), (GRP == "aligned").sum(), (GRP == "offset").sum()))

    # ---- 推理 ----
    ds = []
    for si in range(len(m4)):
        e4 = OI.frame_gains(m4[si], NOISY)
        e5 = OI.frame_gains(m5[si], NOISY)
        per = []
        for j in range(len(NOISY)):
            ref = S[j % len(S)].astype(np.float64) if len(S) != len(NOISY) else S[j].astype(np.float64)
            a = PM._stats_1d(e5[j], ref, NOISY[j].astype(np.float64), PM.FRAME, MASKS[j])
            b = PM._stats_1d(e4[j], ref, NOISY[j].astype(np.float64), PM.FRAME, MASKS[j])
            per.append(a["frmgain_imp"] - b["frmgain_imp"])
        ds.append(np.array(per))
        print("  种子 %d/%d 完成" % (si + 1, len(m4)), flush=True)
    ds = np.stack(ds)                       # [n_seed, n]

    print()
    print("=" * 92)
    print("时频对齐干预：同一突发波形，只改放置时刻")
    print("=" * 92)
    print("%-10s %8s %12s %14s %12s" % ("组", "n", "TF重叠率中位", "Δ干扰帧(dB)", "种子间SD"))
    out = {}
    for g in ("offset", "aligned"):
        sel = GRP == g
        v = ds[:, sel]
        fin = np.isfinite(v).all(axis=0)
        v = v[:, fin]
        out[g] = dict(n=int(fin.sum()), ov_median=float(np.nanmedian(OVS[sel])),
                      d_mean=float(v.mean()), d_sd=float(v.mean(axis=1).std(ddof=1)),
                      per_seed=[float(x) for x in v.mean(axis=1)])
        print("%-10s %8d %12.3f %14.3f %12.3f"
              % (g, out[g]["n"], out[g]["ov_median"], out[g]["d_mean"], out[g]["d_sd"]))

    if "aligned" in out and "offset" in out:
        diff = out["aligned"]["d_mean"] - out["offset"]["d_mean"]
        print("-" * 92)
        print("对齐 − 错开 = %+.3f dB" % diff)
        # 逐样本配对检验（同一样本的对齐/错开是配对的）
        na, no = (GRP == "aligned"), (GRP == "offset")
        k = min(na.sum(), no.sum())
        d_seed = []
        for si in range(ds.shape[0]):
            va, vo = ds[si, na][:k], ds[si, no][:k]
            ok = np.isfinite(va) & np.isfinite(vo)
            if ok.sum() > 5:
                d_seed.append(float(np.mean(va[ok] - vo[ok])))
        if len(d_seed) >= 2:
            m_ = float(np.mean(d_seed)); s_ = float(np.std(d_seed, ddof=1))
            t_ = m_ / (s_ / np.sqrt(len(d_seed))) if s_ > 0 else float("nan")
            from scipy import stats
            p_ = float(2 * stats.t.sf(abs(t_), df=len(d_seed) - 1)) if np.isfinite(t_) else float("nan")
            print("逐样本配对的差值（跨种子）：%+.3f ± %.3f dB，跨种子 p = %.4f (n_seed=%d)"
                  % (m_, s_, p_, len(d_seed)))
            out["paired_diff"] = dict(mean=m_, sd=s_, p=p_, n_seed=len(d_seed))
        print()
        if diff < 0:
            print("方向：对齐组收益【更小】—— 与第七节的机制一致（重叠越高收益越小）")
        else:
            print("方向：对齐组收益【更大】—— 与第七节的机制【相反】")

    with open(os.path.join(OUTD, "tf_alignment_intervention.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n-> sens_out/tf_alignment_intervention.json")


if __name__ == "__main__":
    main()
