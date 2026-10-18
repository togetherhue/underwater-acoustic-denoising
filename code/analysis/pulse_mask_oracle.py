# -*- coding: utf-8 -*-
"""Oracle p：把分支输出的 p 换成【真实干扰掩码】，量化 p 估计误差吃掉多少收益

p 的占空比约 4%（mean 0.04），真实干扰掩码占空比 4.9%，两者量级一致，
所以喂真实掩码基本不算分布外输入 —— 可以当作"p 估计完美"的上界。

  identity  p = 分支估计
  oracle    p = 真实掩码 M（上界）
  R4        无脉冲分支（下界参照）

若 oracle 明显高于 identity ⇒ 分支的瓶颈在 p 的估计精度，不在下游利用；
若 oracle ≈ identity     ⇒ p 已接近最优，提升空间在别处。

用法：python pulse_mask_oracle.py --seeds 1,2,3,4,5
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

R5_ARCH = dict(use_short=True, use_long=True, use_gate=True, use_pulse=True,
               freq_refine=2.0, pulse_freq=True)
R4_ARCH = dict(use_short=True, use_long=True, use_gate=True, use_pulse=False,
               freq_refine=2.0, pulse_freq=False)
SEED_SUF = {1: "", 2: "_s2", 3: "_s3", 4: "_s4", 5: "_s5"}


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
def gains(model, N, C, M, oracle=False, bs=48):
    h = None
    cur = [0]
    if oracle:
        def hook(mod, inp, out):
            # out: [B,1,L]；用真实掩码替换（M 已与 out 同长度）
            mb = torch.from_numpy(M[cur[0]:cur[0] + out.shape[0]]).to(out.device)
            return mb.unsqueeze(1).to(out.dtype)
        h = model.pulse.register_forward_hook(hook)
    g = []
    try:
        for i in range(0, len(N), bs):
            cur[0] = i
            xb = torch.from_numpy(N[i:i + bs]).unsqueeze(1).to(V3.DEVICE)
            e = model(xb).squeeze(1).cpu().numpy().astype(np.float64)
            for j in range(e.shape[0]):
                k = i + j
                st = PM._stats_1d(e[j], C[k].astype(np.float64),
                                  N[k].astype(np.float64),
                                  PM.FRAME, M[k].astype(np.float64))
                g.append(st["frmgain_imp"])
    finally:
        if h is not None:
            h.remove()
    return np.array(g, dtype=np.float64)


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
    A = ap.parse_args()
    seeds = [int(x) for x in A.seeds.split(",") if x.strip()]

    d = load_npy(A.split)
    if d is None:
        print("缺 _npy 缓存"); return
    N, C, M = d
    nm = len(N) if A.nmax <= 0 else min(A.nmax, len(N))
    N, C, M = N[:nm], C[:nm], M[:nm]
    print("split=%s 样本 %d 种子 %s" % (A.split, nm, seeds))
    print("真实掩码占空比 %.2f%%" % (100 * float((M > 0.5).mean())))
    print()

    res = dict(split=A.split, n=nm, seeds=seeds,
               duty=100 * float((M > 0.5).mean()), per_seed={})
    for sd in seeds:
        suf = SEED_SUF[sd]
        ck4 = "model_pamnet_freq_nopulse_gn%s.pth" % suf
        ck5 = "model_pamnet_freq_pf_gn%s.pth" % suf
        if not (os.path.exists(os.path.join(CKPT, ck4))
                and os.path.exists(os.path.join(CKPT, ck5))):
            print("跳过 seed%d" % sd); continue
        m4 = build(R4_ARCH, ck4)
        g4 = gains(m4, N, C, M)
        del m4; torch.cuda.empty_cache()
        m5 = build(R5_ARCH, ck5)
        gi = gains(m5, N, C, M)
        go = gains(m5, N, C, M, oracle=True)
        del m5; torch.cuda.empty_cache()

        f = lambda g: float(np.nanmean(g[np.isfinite(g)]))
        r = dict(R4=f(g4), R5_est=f(gi), R5_oracle=f(go))
        r["d_est"] = r["R5_est"] - r["R4"]
        r["d_oracle"] = r["R5_oracle"] - r["R4"]
        r["headroom"] = r["d_oracle"] - r["d_est"]
        res["per_seed"][str(sd)] = r
        print("seed%d  R4=%.3f  R5(估计 p)=%.3f Δ=%+.3f  R5(oracle)=%.3f Δ=%+.3f  余量 %+.3f"
              % (sd, r["R4"], r["R5_est"], r["d_est"], r["R5_oracle"],
                 r["d_oracle"], r["headroom"]))

    ps = res["per_seed"]
    if ps:
        de = np.array([ps[k]["d_est"] for k in ps], float)
        do = np.array([ps[k]["d_oracle"] for k in ps], float)
        hd = do - de
        t, p = t1(hd)
        print()
        print("=" * 84)
        print("Δ(估计 p) = %+.3f ± %.3f   Δ(oracle p) = %+.3f ± %.3f"
              % (de.mean(), de.std(ddof=1), do.mean(), do.std(ddof=1)))
        print("p 估计误差的代价（oracle − est）= %+.3f ± %.3f dB  n=%d  p=%.4f"
              % (hd.mean(), hd.std(ddof=1), len(hd), p))
        res["summary"] = dict(d_est_mean=float(de.mean()),
                              d_est_sd=float(de.std(ddof=1)),
                              d_oracle_mean=float(do.mean()),
                              d_oracle_sd=float(do.std(ddof=1)),
                              headroom_mean=float(hd.mean()),
                              headroom_sd=float(hd.std(ddof=1)),
                              headroom_p=float(p), n=int(len(hd)))
        os.makedirs(OUTD, exist_ok=True)
        op = os.path.join(OUTD, "pulse_mask_oracle.json")
        with open(op, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print("\n->", op)


if __name__ == "__main__":
    main()
