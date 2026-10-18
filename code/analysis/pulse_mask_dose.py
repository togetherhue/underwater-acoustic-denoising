# -*- coding: utf-8 -*-
"""脉冲掩码的错位剂量响应：增益随「p 与真实干扰的对齐程度」如何衰减

动机
----
pulse_mask_intervention.py 用的是二值干预（identity / shuffle / zero ...）。
二值只能回答"信息重不重要"，回答不了"多重要的什么"。

本实验沿【单一连续轴】做剂量响应：
  把 p 在样本内循环平移 k 个采样点，k ∈ {0, 125, 250, 500, 1000, 2000, 4000}
  · 边缘分布、方差、稀疏度【严格不变】（循环平移是置换）
  · 只有"p 指向的位置"与"真实干扰所在位置"的对齐关系随 k 退化
因此曲线形状直接反映"分支依赖对齐到什么尺度"。

额外给出物理刻度：从真实干扰掩码 M 估计平均突发时长，把 k 换算成
"错位了多少个突发时长"，这样横轴就有物理意义。

用法：python pulse_mask_dose.py --seeds 1,2,3,4,5
"""
import os, sys, json, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_CANDS = [os.path.normpath(os.path.join(HERE, "..", "python")),
          r"C:\Users\32213\PycharmProjects\project1"]
for _p in [HERE] + [c for c in _CANDS if os.path.isdir(c)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pam_metrics as PM
import v3_model as V3
import pamnet as PN

OUTD = os.path.join(HERE, "sens_out")
CKPT = r"E:/project/data/output_pamnet"
NPYD = r"E:/project/data/dataset_pam/alphastable/_npy"
FS = 6000                      # 数据采样率（Hz），用于把 k 换算成 ms

R5_ARCH = dict(use_short=True, use_long=True, use_gate=True, use_pulse=True,
               freq_refine=2.0, pulse_freq=True)
R4_ARCH = dict(use_short=True, use_long=True, use_gate=True, use_pulse=False,
               freq_refine=2.0, pulse_freq=False)
SEED_SUF = {1: "", 2: "_s2", 3: "_s3", 4: "_s4", 5: "_s5"}
KS = [0, 125, 250, 500, 1000, 2000, 4000]


def load_npy(split):
    p = [os.path.join(NPYD, "%s_%s.npy" % (split, t)) for t in ("n", "c", "m")]
    if not all(os.path.exists(x) for x in p):
        return None
    a = [np.load(x, mmap_mode="r") for x in p]
    return np.array(a[0], np.float32), np.array(a[1], np.float32), np.array(a[2], np.float32)


def build(arch, ckpt):
    m = PN.PAMNet(c=24, n_fft=V3.CFG["n_fft"], hop=V3.CFG["hop_length"],
                  norm="gn", **arch)
    m.load_state_dict(torch.load(os.path.join(CKPT, ckpt), map_location="cpu"),
                      strict=True)
    return m.to(V3.DEVICE).eval()


@torch.no_grad()
def gains_shift(model, N, C, M, k, bs=48):
    """把 p 循环平移 k（k=0 即 identity）。k 固定，无需多次重复。"""
    h = None
    if k:
        def hook(mod, inp, out):
            return torch.roll(out, shifts=k, dims=-1)
        h = model.pulse.register_forward_hook(hook)
    gains = []
    try:
        for i in range(0, len(N), bs):
            xb = torch.from_numpy(N[i:i + bs]).unsqueeze(1).to(V3.DEVICE)
            e = model(xb).squeeze(1).cpu().numpy().astype(np.float64)
            for j in range(e.shape[0]):
                g = i + j
                st = PM._stats_1d(e[j], C[g].astype(np.float64),
                                  N[g].astype(np.float64),
                                  PM.FRAME, M[g].astype(np.float64))
                gains.append(st["frmgain_imp"])
    finally:
        if h is not None:
            h.remove()
    return np.array(gains, dtype=np.float64)


def burst_stats(M, nmax=300):
    """从真实干扰掩码估计平均突发时长（采样点）与占空比"""
    lens, duties = [], []
    for i in range(min(len(M), nmax)):
        m = (M[i] > 0.5).astype(np.int8)
        if m.sum() == 0:
            continue
        d = np.diff(np.concatenate([[0], m, [0]]))
        st, en = np.where(d == 1)[0], np.where(d == -1)[0]
        ln = en - st
        if len(ln):
            lens.append(ln.mean())
            duties.append(m.mean())
    return dict(burst_len_mean=float(np.mean(lens)) if lens else float("nan"),
                burst_len_med=float(np.median(lens)) if lens else float("nan"),
                duty=float(np.mean(duties)) if duties else float("nan"),
                n=int(len(lens)))


def t1(d):
    d = np.asarray([x for x in d if np.isfinite(x)], float)
    if len(d) < 2:
        return float("nan"), float("nan")
    from scipy import stats
    t, p = stats.ttest_1samp(d, 0.0)
    return float(t), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=0)
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--split", default="testA")
    ap.add_argument("--tag", default="pulse_mask_dose")
    A = ap.parse_args()
    seeds = [int(x) for x in A.seeds.split(",") if x.strip()]

    d = load_npy(A.split)
    if d is None:
        print("缺 _npy 缓存"); return
    N, C, M = d
    nm = len(N) if A.nmax <= 0 else min(A.nmax, len(N))
    N, C, M = N[:nm], C[:nm], M[:nm]
    print("split=%s 样本 %d 种子 %s" % (A.split, nm, seeds))

    bs = burst_stats(M)
    print("真实干扰掩码：平均突发 %.0f 采样点 = %.1f ms，中位 %.0f，占空比 %.1f%%"
          % (bs["burst_len_mean"], 1000 * bs["burst_len_mean"] / FS,
             bs["burst_len_med"], 100 * bs["duty"]))
    print()

    res = dict(split=A.split, n=nm, seeds=seeds, ks=KS, fs=FS,
               burst=bs, per_seed={}, delta={}, curves={})
    for sd in seeds:
        suf = SEED_SUF[sd]
        ck4 = "model_pamnet_freq_nopulse_gn%s.pth" % suf
        ck5 = "model_pamnet_freq_pf_gn%s.pth" % suf
        if not (os.path.exists(os.path.join(CKPT, ck4))
                and os.path.exists(os.path.join(CKPT, ck5))):
            print("跳过 seed%d：缺 checkpoint" % sd); continue

        m4 = build(R4_ARCH, ck4)
        g4 = gains_shift(m4, N, C, M, 0)
        r4m = float(np.nanmean(g4[np.isfinite(g4)]))
        del m4; torch.cuda.empty_cache()

        m5 = build(R5_ARCH, ck5)
        print("seed%d  R4 = %.3f" % (sd, r4m))
        res["per_seed"].setdefault(str(sd), {})["R4"] = r4m
        curve = []
        for k in KS:
            g = gains_shift(m5, N, C, M, k)
            gm = float(np.nanmean(g[np.isfinite(g)]))
            curve.append(gm)
            res["delta"].setdefault(str(k), {})[str(sd)] = gm - r4m
            res["per_seed"][str(sd)]["k%d" % k] = gm
            print("  k=%5d (%5.1f ms, %4.2f 个突发时长)  R5=%7.3f  Δ=%+.3f"
                  % (k, 1000 * k / FS, k / bs["burst_len_mean"] if bs["burst_len_mean"] else float("nan"),
                     gm, gm - r4m))
        res["curves"][str(sd)] = curve
        del m5; torch.cuda.empty_cache()
        print()

    print("=" * 92)
    print("剂量响应汇总（跨种子）")
    print("=" * 92)
    print("%8s %10s %10s %12s %12s %10s" %
          ("k(点)", "k(ms)", "k/突发", "R5 均值", "Δ 均值", "sd"))
    print("-" * 92)
    summ = {}
    for k in KS:
        v = np.array(list(res["delta"].get(str(k), {}).values()), float)
        v = v[np.isfinite(v)]
        r5 = np.array([res["per_seed"][s]["k%d" % k] for s in res["per_seed"]], float)
        summ[k] = dict(r5_mean=float(r5.mean()), d_mean=float(v.mean()),
                       d_sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0, n=int(len(v)))
        print("%8d %10.1f %10.2f %12.3f %+12.3f %10.3f"
              % (k, 1000 * k / FS,
                 k / bs["burst_len_mean"] if bs["burst_len_mean"] else float("nan"),
                 r5.mean(), v.mean(), summ[k]["d_sd"]))
    res["summary"] = {str(k): v for k, v in summ.items()}

    # 半衰点：Δ 衰减到 (Δ(0) − Δ(max))/2 的 k（线性插值）
    d0, dmax = summ[KS[0]]["d_mean"], summ[KS[-1]]["d_mean"]
    half = (d0 + dmax) / 2.0
    khalf = float("nan")
    for i in range(1, len(KS)):
        a, b = summ[KS[i - 1]]["d_mean"], summ[KS[i]]["d_mean"]
        if (a - half) * (b - half) <= 0 and abs(a - b) > 1e-9:
            khalf = KS[i - 1] + (a - half) / (a - b) * (KS[i] - KS[i - 1])
            break
    res["k_half"] = khalf
    print()
    print("Δ(0)=%+.3f → Δ(%d)=%+.3f，半衰点 k ≈ %.0f 采样点 = %.1f ms = %.2f 个突发时长"
          % (d0, KS[-1], dmax, khalf, 1000 * khalf / FS,
             khalf / bs["burst_len_mean"] if bs["burst_len_mean"] else float("nan")))

    os.makedirs(OUTD, exist_ok=True)
    op = os.path.join(OUTD, A.tag + ".json")
    with open(op, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("\n->", op)


if __name__ == "__main__":
    main()
