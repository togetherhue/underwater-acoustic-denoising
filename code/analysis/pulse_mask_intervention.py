# -*- coding: utf-8 -*-
"""脉冲掩码干预：R5 的收益到底来自「知道脉冲在哪」，还是来自多出来的容量？

背景
----
PAM-Net 的核心主张是「p 注入让模型知道干扰位置，所以降噪更好」。
但 7.x 的两个干预实验（载频扫描、时频对齐）都没有复现「重叠率→收益」的负相关，
那条因果链已被降级为「不能作为机制结论引用」。于是全文最关键的设计主张
目前只剩相关性证据。

本实验直接在推理期篡改 p，不重训、不改模型：
  model.pulse 输出的 p（sigmoid，[B,1,L]）同时进入 resid 与 freq，
  用 forward hook 替换它，即可精确控制「分支拿到什么脉冲信息」。

干预模式
--------
  identity  原样（基线，用于确认 hook 通路无扰动）
  shuffle   换成同批另一样本的 p —— 【统计量完全相同、位置错】，最干净的控制
  shift     p 在样本内循环平移 L/4~3L/4 —— 边缘分布不变、位置错
  flip      p 时间反转 —— 边缘分布不变、位置镜像
  zero      p≡0（宣称"无干扰"）—— 完全剥夺信息
  one       p≡1（宣称"全是干扰"）—— 完全剥夺信息
  noise     U(0,1) 随机 —— 同取值范围、无结构

判读
----
  · shuffle/shift/flip 若明显掉分 ⇒ 分支确实在用【位置】信息（边缘分布没变，只有对齐关系变了）
  · zero 若 ≈ R4 ⇒ 容量本身没贡献，R5 的全部收益来自 p；
    zero 若 > R4  ⇒ 多出来的参数自己也有贡献，可据此把 Δ 拆成「容量」+「信息」两部分

用法
----
  python pulse_mask_intervention.py --nmax 96 --seeds 1     # 冒烟
  python pulse_mask_intervention.py --seeds 1,2,3,4,5       # 全量 5 种子
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

MODES = ["identity", "shuffle", "shift", "flip", "zero", "one", "noise"]
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


def corrupt(p, mode, rng):
    """按模式替换 p（[B,1,L]），返回新张量。"""
    B, _, L = p.shape
    if mode == "identity":
        return p
    if mode == "zero":
        return torch.zeros_like(p)
    if mode == "one":
        return torch.ones_like(p)
    if mode == "noise":
        return torch.rand_like(p)
    if mode == "flip":
        return torch.flip(p, dims=[-1])
    if mode == "shuffle":
        # 循环移位是一种保证【无不动点】的随机重排：每个样本拿到的是另一个
        # 样本的 p，边缘分布严格不变。比 randperm 更好，因为 randperm 会有
        # 约 1 个不动点（B=48 时），把效应稀释掉。
        k = int(rng.integers(1, B)) if B > 1 else 0
        return torch.roll(p, shifts=k, dims=0)
    if mode == "shift":
        out = torch.empty_like(p)
        for b in range(B):
            k = int(rng.integers(L // 4, max(L // 4 + 1, 3 * L // 4)))
            out[b] = torch.roll(p[b], shifts=k, dims=-1)
        return out
    raise ValueError(mode)


@torch.no_grad()
def run_gains(model, N, C, M, mode=None, bs=48, seed=0, want_p=False):
    """返回 (frmgain_imp 数组, p 诊断字典或 None)

    注意：与 scale_invariance_test.py 一样，est/ref/noisy 必须同一尺度。
    这里不做缩放，三个量都在原始尺度，无需额外处理。
    """
    rng = np.random.default_rng(seed)
    h = None
    captured = []

    if mode is not None:
        def hook(mod, inp, out):
            if want_p:
                captured.append(out.detach().cpu().numpy()[:, 0, :].astype(np.float32))
            return corrupt(out, mode, rng)
        h = model.pulse.register_forward_hook(hook)

    gains = []
    try:
        for i in range(0, len(N), bs):
            xb = torch.from_numpy(N[i:i + bs]).unsqueeze(1).to(V3.DEVICE)
            e = model(xb).squeeze(1).cpu().numpy().astype(np.float64)
            for k in range(e.shape[0]):
                j = i + k
                st = PM._stats_1d(e[k], C[j].astype(np.float64),
                                  N[j].astype(np.float64),
                                  PM.FRAME, M[j].astype(np.float64))
                gains.append(st["frmgain_imp"])
    finally:
        if h is not None:
            h.remove()

    g = np.array(gains, dtype=np.float64)
    pd = None
    if want_p and captured:
        P = np.concatenate(captured, axis=0)[:len(N)]
        pd = p_diag(P, M[:len(P)])
    return g, pd


def auc_score(score, label):
    """Mann-Whitney U 形式的 AUC（不依赖 sklearn）"""
    pos = label > 0.5
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def p_diag(P, M, nmax=200):
    """p 的信息量诊断：值域、样本内波动、与真实干扰掩码的 AUC / 相关"""
    n = min(len(P), nmax)
    aucs, stds, corrs = [], [], []
    for i in range(n):
        p, m = P[i], (M[i] > 0.5).astype(np.float64)
        a = auc_score(p, m)
        if np.isfinite(a):
            aucs.append(a)
        stds.append(float(np.std(p)))
        if m.std() > 1e-9 and p.std() > 1e-9:
            corrs.append(float(np.corrcoef(p, m)[0, 1]))
    return dict(mean=float(np.mean(P)), std_within=float(np.mean(stds)),
                auc_mean=float(np.mean(aucs)) if aucs else float("nan"),
                auc_p05=float(np.percentile(aucs, 5)) if aucs else float("nan"),
                auc_p95=float(np.percentile(aucs, 95)) if aucs else float("nan"),
                corr_mean=float(np.mean(corrs)) if corrs else float("nan"),
                n_eval=int(n))


def t1(d):
    d = np.asarray([x for x in d if np.isfinite(x)], float)
    if len(d) < 2:
        return float("nan"), float("nan")
    from scipy import stats
    t, p = stats.ttest_1samp(d, 0.0)
    return float(t), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=0, help="样本上限，<=0 全量")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--split", default="testA")
    ap.add_argument("--passes", type=int, default=3,
                    help="shuffle/shift/noise 是随机的，每个种子重复几次取均值")
    ap.add_argument("--tag", default="pulse_mask_intervention")
    A = ap.parse_args()
    seeds = [int(x) for x in A.seeds.split(",") if x.strip()]

    d = load_npy(A.split)
    if d is None:
        print("缺 _npy 缓存"); return
    N, C, M = d
    nm = len(N) if A.nmax <= 0 else min(A.nmax, len(N))
    N, C, M = N[:nm], C[:nm], M[:nm]
    print("split=%s  样本 %d  种子 %s  随机模式重复 %d 次"
          % (A.split, nm, seeds, A.passes))
    print()

    res = dict(split=A.split, n=nm, modes=MODES, seeds=seeds, passes=A.passes,
               r4={}, r5={}, delta={}, p_diag={})

    for sd in seeds:
        suf = SEED_SUF[sd]
        ck4 = "model_pamnet_freq_nopulse_gn%s.pth" % suf
        ck5 = "model_pamnet_freq_pf_gn%s.pth" % suf
        if not (os.path.exists(os.path.join(CKPT, ck4))
                and os.path.exists(os.path.join(CKPT, ck5))):
            print("跳过 seed%d：缺 checkpoint" % sd); continue

        # ---- R4：无脉冲分支，不受 hook 影响 ----
        m4 = build(R4_ARCH, ck4)
        g4, _ = run_gains(m4, N, C, M)
        del m4
        torch.cuda.empty_cache()

        g4f = g4[np.isfinite(g4)]
        # ---- R5：7 种干预 ----
        m5 = build(R5_ARCH, ck5)
        print("seed%d  R4 = %.3f" % (sd, np.nanmean(g4f)))
        res["r4"][str(sd)] = dict(mean=float(np.nanmean(g4f)),
                                  median=float(np.median(g4f)))

        for mode in MODES:
            acc, pd = [], None
            npas = A.passes if mode in ("shuffle", "shift", "noise") else 1
            for k in range(npas):
                g, pd_ = run_gains(m5, N, C, M, mode=mode,
                                   seed=1000 * sd + k,
                                   want_p=(mode == "identity" and k == 0))
                acc.append(g)
                if pd_ is not None:
                    pd = pd_
            g = np.mean(np.vstack(acc), axis=0)
            gf = g[np.isfinite(g)]
            res["r5"].setdefault(str(sd), {})[mode] = dict(
                mean=float(np.nanmean(gf)), median=float(np.median(gf)),
                frac_better_r4=float(np.mean(g > g4)) if len(g) == len(g4) else float("nan"))
            res["delta"].setdefault(mode, {})[str(sd)] = float(
                np.nanmean(gf) - np.nanmean(g4f))
            if pd is not None:
                res["p_diag"][str(sd)] = pd
            print("  %-9s R5 = %7.3f   Δ = %+.3f"
                  % (mode, np.nanmean(gf), res["delta"][mode][str(sd)]))
        del m5
        torch.cuda.empty_cache()
        print()

    # ---------------- 汇总 ----------------
    print("=" * 88)
    print("汇总：Δ = R5(某模式) − R4，跨种子均值 ± sd，配对 t 检验")
    print("=" * 88)
    print("%-10s %12s %10s %8s %10s" % ("模式", "Δ 均值(dB)", "sd", "n", "p(≠0)"))
    print("-" * 88)
    summ = {}
    for mode in MODES:
        v = np.array([x for x in res["delta"].get(mode, {}).values()], float)
        v = v[np.isfinite(v)]
        t, p = t1(v)
        summ[mode] = dict(mean=float(v.mean()) if len(v) else float("nan"),
                          sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                          n=int(len(v)), t=t, p=p)
        print("%-10s %+12.3f %10.3f %8d %10.4f"
              % (mode, summ[mode]["mean"], summ[mode]["sd"], len(v), p))
    res["summary"] = summ

    # ---- 关键对比：相对 identity 的损失 ----
    print()
    print("=" * 88)
    print("相对 identity 的损失（同一模型、同一种子，配对）")
    print("=" * 88)
    base = res["delta"].get("identity", {})
    rel = {}
    for mode in MODES:
        if mode == "identity":
            continue
        d_ = res["delta"].get(mode, {})
        common = sorted(set(base) & set(d_), key=int)
        if not common:
            continue
        diff = np.array([d_[k] - base[k] for k in common], float)
        t, p = t1(diff)
        rel[mode] = dict(mean=float(diff.mean()), sd=float(diff.std(ddof=1))
                         if len(diff) > 1 else 0.0, n=len(diff), t=t, p=p,
                         per_seed=[float(x) for x in diff])
        print("  %-9s %+8.3f ± %.3f dB   n=%d  p=%.4f"
              % (mode, rel[mode]["mean"], rel[mode]["sd"], len(diff), p))
    res["rel_identity"] = rel

    # ---- p 的信息量诊断 ----
    if res["p_diag"]:
        print()
        print("=" * 88)
        print("p 的信息量诊断（identity 模式下捕获的 p 与真实干扰掩码对比）")
        print("=" * 88)
        for k, v in sorted(res["p_diag"].items(), key=lambda x: int(x[0])):
            print("  seed%s  mean=%.3f  within-std=%.3f  AUC=%.3f [%.3f, %.3f]  r=%.3f"
                  % (k, v["mean"], v["std_within"], v["auc_mean"],
                     v["auc_p05"], v["auc_p95"], v["corr_mean"]))

    os.makedirs(OUTD, exist_ok=True)
    op = os.path.join(OUTD, A.tag + ".json")
    with open(op, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("\n->", op)


if __name__ == "__main__":
    main()
