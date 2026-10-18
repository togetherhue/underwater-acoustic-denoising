# -*- coding: utf-8 -*-
"""跨条件的脉冲掩码干预：把 Δ 分解成「容量项」+「位置信息项」，并判断
α 轴 / rate 轴上的收益变化到底来自哪一项。

为什么这个实验值得做
--------------------
8.15 只在 base 条件上证明了「分支依赖逐点对齐」。而此前所有跨条件的 Δ 变化
（α 轴 0.2~1.7 dB、rate 轴 0.55~0.98 dB）都只被解释成"分支价值变了"，
没有任何分解。文献里这类解释通常是循环论证：收益小 → 说"分支没用"。

本实验给出一个可测的分解。对每个条件 c：
    Δ_id(c)   = R5(自己的 p) − R4        ---- 分支的实际净贡献
    Δ_zero(c) = R5(p≡0)      − R4        ---- 【无信息基线】：被剥夺位置信息、
                                              且被告知"处处无干扰"时的净效应
                                              注意它不是"纯容量"：p≡0 是训练时
                                              没见过的常数输入，因此这一项同时
                                              包含"参数冗余"与"错误先验"两种代价
    位置项(c) = Δ_id(c) − Δ_zero(c)      ---- 【位置信息的价值】：从"错误先验"
                                              换成"正确位置"带来的净提升
    Δ_orc(c)  = R5(真实掩码) − R4        ---- p 完美时的上界

于是"某条件下分支收益小"就有了两种完全不同的成因，且可区分：
    (A) 位置项小  ⇒ 此刻位置信息本身不值钱（例如干扰本来就好抑制）—— 真实的条件效应
    (B) 位置项大但 Δ_id 小 ⇒ 位置值钱，但被容量项的负贡献抵消（下游没利用好）—— 可改进
    (C) Δ_orc ≫ Δ_id ⇒ p 估计太差，位置信息的"可得量"没拿到 —— 换估计器有用

关键设计：**模型固定，只换数据**
所有条件用同一批 base checkpoint（model_pamnet_freq_*_gn），因此条件间的
任何差异都只能归因于数据分布，不存在"每个条件各训一个模型"的混杂。

用法：python pulse_mask_crosscond.py --seeds 1,2,3
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

R5_ARCH = dict(use_short=True, use_long=True, use_gate=True, use_pulse=True,
               freq_refine=2.0, pulse_freq=True)
R4_ARCH = dict(use_short=True, use_long=True, use_gate=True, use_pulse=False,
               freq_refine=2.0, pulse_freq=False)
SEED_SUF = {1: "", 2: "_s2", 3: "_s3", 4: "_s4", 5: "_s5"}

# 条件 → 数据根目录。base 为 alphastable（训练分布），其余为敏感性矩阵
CONDS = {
    "base":    (r"E:/project/data/dataset_pam/alphastable", "base（α=1.5, 5/s）"),
    "a12_r5":  (r"E:/project/data/ds_sens/a12_r5",  "α=1.2"),
    "a18_r5":  (r"E:/project/data/ds_sens/a18_r5",  "α=1.8"),
    "a20_r5":  (r"E:/project/data/ds_sens/a20_r5",  "α=2.0"),
    "a15_r25": (r"E:/project/data/ds_sens/a15_r25", "rate=2.5/s"),
    "a15_r10": (r"E:/project/data/ds_sens/a15_r10", "rate=10/s"),
}
ORDER = ["a12_r5", "base", "a18_r5", "a20_r5", "a15_r25", "a15_r10"]


def load_npy(root, split="testA"):
    p = [os.path.join(root, "_npy", "%s_%s.npy" % (split, t)) for t in ("n", "c", "m")]
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
def gains(model, N, C, M, mode=None, bs=48, seed=0, want_p=False):
    """mode: None(不干预) / identity / shuffle / zero / oracle"""
    rng = np.random.default_rng(seed)
    h, cap, cur = None, [], [0]

    if mode is not None:
        def hook(mod, inp, out):
            if want_p:
                cap.append(out.detach().cpu().numpy()[:, 0, :].astype(np.float32))
            if mode == "identity":
                return out
            if mode == "zero":
                return torch.zeros_like(out)
            if mode == "shuffle":
                B = out.shape[0]
                k = int(rng.integers(1, B)) if B > 1 else 0
                return torch.roll(out, shifts=k, dims=0)
            if mode == "oracle":
                mb = torch.from_numpy(M[cur[0]:cur[0] + out.shape[0]]).to(out.device)
                return mb.unsqueeze(1).to(out.dtype)
            raise ValueError(mode)
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

    out = np.array(g, dtype=np.float64)
    pd = None
    if want_p and cap:
        P = np.concatenate(cap, axis=0)[:len(N)]
        pd = p_diag(P, M[:len(P)])
    return out, pd


def auc_score(score, label):
    pos = label > 0.5
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def p_diag(P, M, nmax=200):
    n = min(len(P), nmax)
    aucs, corrs = [], []
    for i in range(n):
        p, m = P[i], (M[i] > 0.5).astype(np.float64)
        a = auc_score(p, m)
        if np.isfinite(a):
            aucs.append(a)
        if m.std() > 1e-9 and p.std() > 1e-9:
            corrs.append(float(np.corrcoef(p, m)[0, 1]))
    return dict(auc_mean=float(np.mean(aucs)) if aucs else float("nan"),
                auc_p05=float(np.percentile(aucs, 5)) if aucs else float("nan"),
                corr_mean=float(np.mean(corrs)) if corrs else float("nan"),
                p_mean=float(np.mean(P)), n_eval=int(n))


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
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--split", default="testA")
    ap.add_argument("--tag", default="pulse_mask_crosscond")
    ap.add_argument("--conds", default="", help="逗号分隔，默认全部")
    A = ap.parse_args()
    seeds = [int(x) for x in A.seeds.split(",") if x.strip()]
    conds = [c.strip() for c in A.conds.split(",") if c.strip()] or ORDER

    res = dict(split=A.split, seeds=seeds, conds={}, delta={}, decomp={}, p_diag={})
    for c in conds:
        if c not in CONDS:
            print("未知条件 %s" % c); continue
        root, lbl = CONDS[c]
        d = load_npy(root, A.split)
        if d is None:
            print("[跳过] %s：缺 _npy 缓存" % c); continue
        N, C, M = d
        nm = len(N) if A.nmax <= 0 else min(A.nmax, len(N))
        N, C, M = N[:nm], C[:nm], M[:nm]
        print("=" * 88)
        print("%s  样本 %d  占空比 %.2f%%" % (lbl, nm, 100 * float((M > 0.5).mean())))
        print("=" * 88)

        acc = {m: [] for m in ("identity", "shuffle", "zero", "oracle")}
        r4a, pds = [], []
        for sd in seeds:
            suf = SEED_SUF[sd]
            ck4 = "model_pamnet_freq_nopulse_gn%s.pth" % suf
            ck5 = "model_pamnet_freq_pf_gn%s.pth" % suf
            if not (os.path.exists(os.path.join(CKPT, ck4))
                    and os.path.exists(os.path.join(CKPT, ck5))):
                continue
            m4 = build(R4_ARCH, ck4)
            g4, _ = gains(m4, N, C, M)
            del m4; torch.cuda.empty_cache()
            r4a.append(float(np.nanmean(g4[np.isfinite(g4)])))

            m5 = build(R5_ARCH, ck5)
            for mode in acc:
                g, pd = gains(m5, N, C, M, mode=mode, seed=1000 * sd,
                              want_p=(mode == "identity"))
                acc[mode].append(float(np.nanmean(g[np.isfinite(g)])))
                if pd is not None:
                    pds.append(pd)
            del m5; torch.cuda.empty_cache()

        if not r4a:
            print("  没有可用 checkpoint"); continue
        r4m = float(np.mean(r4a))
        f = lambda v: float(np.mean(v)) if v else float("nan")
        s = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        r = dict(n=nm, n_seed=len(r4a), R4=r4m, R4_sd=s(r4a),
                 duty=100 * float((M > 0.5).mean()))
        for mode in acc:
            r["R5_" + mode] = f(acc[mode])
            r["R5_" + mode + "_sd"] = s(acc[mode])
            r["d_" + mode] = r["R5_" + mode] - r4m
        # 分解：容量项 / 位置项
        r["capacity"] = r["d_zero"]
        r["position"] = r["d_identity"] - r["d_zero"]
        r["oracle_gain"] = r["d_oracle"] - r["d_identity"]
        res["conds"][c] = r
        res["delta"][c] = {m: [x - r4m for x in acc[m]] for m in acc}
        if pds:
            res["p_diag"][c] = dict(
                auc_mean=float(np.mean([x["auc_mean"] for x in pds])),
                auc_p05=float(np.mean([x["auc_p05"] for x in pds])),
                corr_mean=float(np.mean([x["corr_mean"] for x in pds])),
                p_mean=float(np.mean([x["p_mean"] for x in pds])))

        print("  R4 = %.3f ± %.3f" % (r4m, r["R4_sd"]))
        print("  %-10s %8s %8s" % ("模式", "R5", "Δ"))
        for mode in acc:
            print("  %-10s %8.3f %+8.3f" % (mode, r["R5_" + mode], r["d_" + mode]))
        print("  --> 分解：无信息基线 %+.3f  位置信息价值 %+.3f  (= Δ_id %+.3f)"
              % (r["capacity"], r["position"], r["d_identity"]))
        if c in res["p_diag"]:
            print("      p 的 AUC = %.3f，占空比 %.2f%%，p 均值 %.3f"
                  % (res["p_diag"][c]["auc_mean"], r["duty"], res["p_diag"][c]["p_mean"]))
        print()

    # ---------------- 汇总表 ----------------
    print("=" * 100)
    print("跨条件分解（同一批 base checkpoint，只换数据）")
    print("=" * 100)
    print("%-16s %8s %8s %9s %9s %9s %9s %7s" %
          ("条件", "R4", "Δ_id", "无信息基线", "位置价值", "Δ_oracle", "oracle余量", "p AUC"))
    print("-" * 100)
    for c in ORDER:
        if c not in res["conds"]:
            continue
        r = res["conds"][c]
        auc = res.get("p_diag", {}).get(c, {}).get("auc_mean", float("nan"))
        print("%-16s %8.3f %+8.3f %+9.3f %+9.3f %+9.3f %+9.3f %7.3f"
              % (CONDS[c][1], r["R4"], r["d_identity"], r["capacity"],
                 r["position"], r["d_oracle"], r["oracle_gain"], auc))

    pos = [res["conds"][c]["position"] for c in ORDER if c in res["conds"]]
    cap = [res["conds"][c]["capacity"] for c in ORDER if c in res["conds"]]
    if pos:
        print()
        print("位置价值 范围 [%.3f, %.3f] 极差 %.3f dB；无信息基线 范围 [%.3f, %.3f] 极差 %.3f dB"
              % (min(pos), max(pos), max(pos) - min(pos),
                 min(cap), max(cap), max(cap) - min(cap)))
        if (max(pos) - min(pos)) > (max(cap) - min(cap)):
            print("-> 跨条件的变化主要来自【位置价值】（位置信息值多少钱在变）")
        else:
            print("-> 跨条件的变化主要来自【无信息基线】（错误先验的代价在变）")

    os.makedirs(OUTD, exist_ok=True)
    op = os.path.join(OUTD, A.tag + ".json")
    with open(op, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("\n->", op)


if __name__ == "__main__":
    main()
