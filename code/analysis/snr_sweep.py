# -*- coding: utf-8 -*-
"""SNR 轴扫描：单一因子（背景噪声电平）对脉冲分支价值的影响

【为什么需要这一节】
全报告只有 8.1 定性讨论了「名义 SNR 固定 ≠ 有效 SNR 固定」，但没有任何一条
Δ(SNR) 曲线。最自然的追问是：低 SNR 下这个分支还有用吗？高 SNR 下它是不是
在空转？本脚本补上这条轴。

【设计：只动一个因子】
固定 α=1.5、rate=5/s（base 条件），只改背景噪声电平。
- 数据集的 `snr` 参数只控制【背景噪声 v】的电平：生成器里
  Pn = Pc/10^(snr/10) 作用在 v 上，而突发干扰幅度固定为 imp = imp_full*rms_c
  （与 SNR 无关）。所以"SNR 轴"的正确含义是【背景 SNR = Pc/Pv】，不是总 SNR。
  本脚本按这个定义扫，并把实测总 SNR（Pc/(Pv+Pi)）一并输出。
- 物理意义：突发干扰强度不变，背景噪声变强/变弱。

【重混而不是重生成】
v 无法直接读出，但 noisy = clean + v + imp，故 v = noisy − clean − imp
（resample_poly 是线性的，等式在重采样域上严格成立）。任意目标 SNR 只需对 v
乘逐样本增益 g = sqrt(Pc / (Pv·10^(t/10)))，新混合 = clean + g·v + imp。
这比重跑 MATLAB 生成器更严格：噪声实现、突发时刻、信道全部不变，只变电平，
是严格的【配对】比较。

【两道闸门（--verify）】
1. 令 g = 1（即还原到原始电平），重混数组与 cache_*.npz 的 noisy 相对误差
   必须 < 1e-6；模型指标与 sens_*.npz 存档差必须 < 1e-3 dB。
2. 向量化指标 stats_batch 与 PM._stats_1d 逐样本结果差必须 < 1e-9。
任一不过就不允许往下跑。

用法：
  python snr_sweep.py --prep
  python snr_sweep.py --verify
  python snr_sweep.py --bench
  python snr_sweep.py --run [--seeds 5] [--skip-inter]
  python snr_sweep.py --agg
"""
import os, sys, json, time, glob, argparse
import numpy as np

WORK = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(WORK, "sens_out")
CKPT = r"E:/project/data/output_pamnet"
CONDS = {
    "base":   r"E:/project/data/dataset_pam/alphastable",
    "a12_r5": r"E:/project/data/ds_sens/a12_r5",
    "a18_r5": r"E:/project/data/ds_sens/a18_r5",
    "a20_r5": r"E:/project/data/ds_sens/a20_r5",
}
ALPHA = {"base": 1.5, "a12_r5": 1.2, "a18_r5": 1.8, "a20_r5": 2.0}
SPLIT = "testA"
SUF = ""          # testC 时置 "_testC"；npz/json 聚合产物用它区分
STEM = "snrcell"   # 单元文件词干；testC 用 snrcellTC，避免被前缀 glob 误吞


def set_split(name):
    """切换评估划分。

    单元文件名用不同词干（snrcell_ / snrcellTC_）而非后缀，因为聚合是按
    `snrcell_*` 前缀 glob 的，加后缀会被 testA 的聚合误吞（踩过一次）。
    """
    global SPLIT, SUF, STEM
    SPLIT = name
    SUF = "" if name == "testA" else "_" + name
    STEM = "snrcell" if name == "testA" else "snrcellTC"

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in [_HERE, r"C:\Users\32213\PycharmProjects\project1"]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import scipy.io as sio                                    # noqa: E402
from scipy.signal import resample_poly                     # noqa: E402
import pam_metrics as PM                                   # noqa: E402

SEG, DS = 8000, 4
FRAME = PM.FRAME
SNR_GRID = [-15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15]
INTER_SNR = [-10, 0, 10]
INTER_CONDS = ["a12_r5", "a18_r5", "a20_r5"]
# base 的实测背景 SNR 中位数（= Pc/Pv，见 --prep 输出）。8.1b 的"等功率匹配"
# 就是把这个电平对齐；重混可以在不改数据集的前提下精确复现该点。
T_REPL = -3.29


def tkey(t):
    """统一的 SNR 标签：整数用 +3/-3，非整数用 +3.29。"""
    return ("%+d" % t) if float(t) == int(t) else ("%+.2f" % t)


# ============================ 1. 噪声分解缓存 ============================
def prep(cond):
    """把 v（背景）与 i（突发）从 .mat 拆出来缓存。

    口径与 eval_sensitivity.prep_cond 一致（先 float32 再 resample_poly），
    分解本身用 float64（v 比 clean 大 1~2 个数量级，float32 减法会掉精度）。
    """
    root = CONDS[cond]
    files = sorted(glob.glob(os.path.join(root, SPLIT, "*.mat")))
    N = len(files)
    v = np.zeros((N, SEG), np.float64)
    i = np.zeros((N, SEG), np.float64)
    n32 = np.zeros((N, SEG), np.float32)
    c32 = np.zeros((N, SEG), np.float32)
    mask = np.zeros((N, SEG), np.uint8)
    snr_nom = np.zeros(N, np.float64)

    def fx(x):
        return x[:SEG] if len(x) >= SEG else np.pad(x, (0, SEG - len(x)))

    for k, f in enumerate(files):
        m = sio.loadmat(f)
        ns = np.asarray(m["noisy_signal"]).squeeze().astype(np.float32)
        cs = np.asarray(m["clean_signal"]).squeeze().astype(np.float32)
        im = np.asarray(m["imp"]).squeeze().astype(np.float64)
        km = np.asarray(m["imp_mask"]).squeeze().astype(np.float64)
        snr_nom[k] = float(np.asarray(m["true_snr"]).squeeze())

        n32[k] = fx(resample_poly(ns, 1, DS).astype(np.float32))
        c32[k] = fx(resample_poly(cs, 1, DS).astype(np.float32))
        mask[k] = (fx(resample_poly(km, 1, DS)) > 0.3).astype(np.uint8)

        nd = fx(resample_poly(ns.astype(np.float64), 1, DS))
        cd = fx(resample_poly(cs.astype(np.float64), 1, DS))
        idd = fx(resample_poly(im, 1, DS))
        i[k] = idd
        v[k] = nd - cd - idd

    Pc = np.mean(c32.astype(np.float64) ** 2, axis=1)
    Pv = np.mean(v ** 2, axis=1)
    Pi = np.mean(i ** 2, axis=1)
    p = os.path.join(OUTD, "snr_%s%s.npz" % (cond, SUF))
    np.savez_compressed(p, v=v.astype(np.float32), i=i.astype(np.float32),
                        n=n32, c=c32, mask=mask, snr_nom=snr_nom,
                        Pc=Pc, Pv=Pv, Pi=Pi)
    u, ct = np.unique(snr_nom, return_counts=True)
    print("  %-8s N=%4d  背景SNR中位 %+6.2f  总SNR中位 %+6.2f  干扰SNR中位 %+6.2f"
          % (cond, N, np.median(10 * np.log10(Pc / Pv)),
             np.median(10 * np.log10(Pc / (Pv + Pi))),
             np.median(10 * np.log10(Pc / Pi))), flush=True)
    print("           名义 SNR: %s" % dict(zip(u.astype(int).tolist(), ct.tolist())))


# ============================ 2. 重混 ============================
def remix_gain(cond, g, d=None):
    """按逐样本增益 g 重混：mix = clean + g·v + imp。g 可为标量或 [N]。"""
    if d is None:
        d = np.load(os.path.join(OUTD, "snr_%s%s.npz" % (cond, SUF)))
    c = d["c"].astype(np.float64)
    v = d["v"].astype(np.float64)
    i = d["i"].astype(np.float64)
    g = np.asarray(g, np.float64)
    if g.ndim == 0:
        g = np.full(len(c), float(g))
    mix = (c + v * g[:, None] + i).astype(np.float32)
    return mix, g, d["Pv"] * g ** 2, d["Pi"]


def remix(cond, target_bg_snr, d=None):
    """背景噪声缩放到目标【背景 SNR】= Pc/Pv，突发干扰保持不动。

    注意：α<2 时 q75 匹配会让实际 Pc/Pv 偏离名义值（8.1 的混杂），
    所以这里的 g 一般不恰好等于 1；要还原原始电平请用 remix_gain(cond, 1.0)。
    """
    if d is None:
        d = np.load(os.path.join(OUTD, "snr_%s%s.npz" % (cond, SUF)))
    g = np.sqrt(d["Pc"] / (d["Pv"] * 10 ** (target_bg_snr / 10.0)))
    return remix_gain(cond, g, d)


# ============================ 3. 指标（向量化，与 PM._stats_1d 等价）============================
def stats_batch(est, ref, noisy, mask):
    """批量算 PM._stats_1d 的全部量，逐样本 median，返回逐样本数组。"""
    N, L = ref.shape
    nf = L // FRAME
    # 必须先升到 float64 再算能量：float32 的平方和相对误差 ~1e-7，
    # 经 10log10 后就是 1e-7 dB 量级，与 PM._stats_1d（内部 float64）对不上。
    ref = np.asarray(ref, np.float64)
    noisy = np.asarray(noisy, np.float64)
    est = np.asarray(est, np.float64)

    def fr(x):
        return x[:, :nf * FRAME].reshape(N, nf, FRAME)

    s = fr(ref)
    ps = (s ** 2).sum(2)
    pn_in = ((fr(noisy) - s) ** 2).sum(2)
    pn_out = ((fr(est) - s) ** 2).sum(2)
    valid = ps > ps.max(1, keepdims=True) * PM.MIN_ENERGY_FRAC
    snr = np.full((N, nf), np.nan)
    gain = np.full((N, nf), np.nan)
    snr[valid] = 10 * np.log10(ps[valid] / (pn_out[valid] + 1e-12))
    gain[valid] = 10 * np.log10((pn_in[valid] + 1e-12) / (pn_out[valid] + 1e-12))
    mf = fr(mask.astype(np.float32)).max(2) > 0
    mc, mq = valid & mf, valid & (~mf)

    def nanmed(a, m):
        return np.nanmedian(np.where(m, a, np.nan), axis=1)

    return dict(snr_out=nanmed(snr, valid), gain_med=nanmed(gain, valid),
                gain_clean=nanmed(gain, mq), gain_imp=nanmed(gain, mc),
                snr_imp=nanmed(snr, mc),
                frac_imp=mf.sum(1) / np.maximum(valid.sum(1), 1))


def snr_in_batch(noisy, ref):
    return 10 * np.log10((ref ** 2).sum(1) / ((noisy - ref) ** 2).sum(1))


# ============================ 4. 模型 ============================
MODELS = {
    "R4": dict(arch=dict(use_short=True, use_long=True, use_gate=True, use_pulse=False,
                         freq_refine=2.0, pulse_freq=False),
               ck=["model_pamnet_freq_nopulse_gn.pth", "model_pamnet_freq_nopulse_gn_s2.pth",
                   "model_pamnet_freq_nopulse_gn_s3.pth", "model_pamnet_freq_nopulse_gn_s4.pth",
                   "model_pamnet_freq_nopulse_gn_s5.pth"]),
    "R5": dict(arch=dict(use_short=True, use_long=True, use_gate=True, use_pulse=True,
                         freq_refine=2.0, pulse_freq=True),
               ck=["model_pamnet_freq_pf_gn.pth", "model_pamnet_freq_pf_gn_s2.pth",
                   "model_pamnet_freq_pf_gn_s3.pth", "model_pamnet_freq_pf_gn_s4.pth",
                   "model_pamnet_freq_pf_gn_s5.pth"]),
}


def build(name, seed_idx):
    import torch
    import v3_model as V3
    import pamnet as PN
    mdl = PN.PAMNet(c=24, n_fft=V3.CFG["n_fft"], hop=V3.CFG["hop_length"],
                    norm="gn", **MODELS[name]["arch"]).to(V3.DEVICE)
    mdl.load_state_dict(torch.load(os.path.join(CKPT, MODELS[name]["ck"][seed_idx]),
                                   map_location=V3.DEVICE), strict=True)
    return mdl.to(V3.DEVICE).eval()


def run_model(mdl, mix, bs=64):
    import torch
    import v3_model as V3
    outs = []
    with torch.no_grad():
        for s in range(0, len(mix), bs):
            xb = torch.from_numpy(np.ascontiguousarray(mix[s:s + bs])).unsqueeze(1).to(V3.DEVICE)
            outs.append(mdl(xb)[:, 0].cpu().numpy())
    return np.concatenate(outs, 0)


# ============================ 5. 经典基线（与 baseline_sens.py 逐字一致）============================
def _spec_sub(x, fs=6000, n_fft=512, hop=128, alpha=2.0, beta=0.02):
    from scipy.signal import stft, istft
    f, t, Z = stft(x, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
    mag, ph = np.abs(Z), np.angle(Z)
    band = (f >= 1600) & (f <= 2900)
    nse = mag[band].mean(axis=0, keepdims=True) if band.any() else mag.min(axis=0, keepdims=True)
    nse = np.repeat(nse, mag.shape[0], axis=0)
    m = np.maximum(mag - alpha * nse, beta * mag)
    _, y = istft(m * np.exp(1j * ph), fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
    return y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


def _wiener(x, fs=6000, n_fft=512, hop=128, beta=0.02):
    from scipy.signal import stft, istft
    f, t, Z = stft(x, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
    mag, ph = np.abs(Z), np.angle(Z)
    band = (f >= 1600) & (f <= 2900)
    nse = mag[band].mean(axis=0, keepdims=True) if band.any() else mag.min(axis=0, keepdims=True)
    nse = np.repeat(nse, mag.shape[0], axis=0)
    m = np.maximum(1.0 - (nse ** 2) / (mag ** 2 + 1e-12), beta)
    _, y = istft(m * mag * np.exp(1j * ph), fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
    return y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


BASELINES = {"specsub": _spec_sub, "wiener": _wiener}


def run_baseline(name, mix):
    fn = BASELINES[name]
    return np.stack([fn(x.astype(np.float64)).astype(np.float32) for x in mix], 0)


# ============================ 6. 闸门 ============================
def verify():
    ok = True
    print("=" * 96)
    print("闸门 1：向量化指标 vs PM._stats_1d")
    print("=" * 96)
    d = np.load(os.path.join(OUTD, "snr_base.npz"))
    n, c, mk = d["n"][:16], d["c"][:16], d["mask"][:16]
    est = n * 0.9 + 0.1 * c                     # 任意"估计"，只为比对两条实现
    B = stats_batch(est, c, n, mk)
    worst = 0.0
    for k in range(len(n)):
        r = PM._stats_1d(est[k].astype(np.float64), c[k].astype(np.float64),
                         n[k].astype(np.float64), FRAME, mk[k])
        pairs = [("snr_out", "fmsnr_med"), ("gain_med", "frmgain_med"),
                 ("gain_clean", "frmgain_clean"), ("gain_imp", "frmgain_imp"),
                 ("snr_imp", "fmsnr_imp"), ("frac_imp", "frac_imp")]
        for a, b in pairs:
            worst = max(worst, abs(float(B[a][k]) - float(r[b])))
    good = worst < 1e-9
    ok = ok and good
    print("  6 个指标 × 16 样本 最大差 %.3e  %s" % (worst, "✓" if good else "✗"))

    print()
    print("=" * 96)
    print("闸门 2：g=1 重混 vs 原始缓存（逐位）")
    print("=" * 96)
    cache = np.load(os.path.join(OUTD, "cache_base.npz"))
    mix_all, g_all, _, _ = remix_gain("base", 1.0, d)
    rel = float(np.abs(mix_all - d["n"]).max() / np.abs(d["n"]).max())
    dmax = float(np.abs(mix_all - d["n"]).max())
    good2 = rel < 1e-6
    ok = ok and good2
    print("  重混(clean+v+imp) vs 缓存 noisy：最大绝对差 %.3e，相对 %.3e  %s"
          % (dmax, rel, "✓" if good2 else "✗"))
    print("  与 cache_base.npz 的 noisy 最大差：%.3e  %s"
          % (float(np.abs(cache["testA_n"] - d["n"]).max()),
             "✓ 同源" if np.abs(cache["testA_n"] - d["n"]).max() < 1e-6 else "✗ 非同源"))

    print()
    print("=" * 96)
    print("闸门 3：重算指标 vs sens_*.npz 存档（≤1e-3 dB）")
    print("=" * 96)
    for mname, fn in (("R4", "R4_freqrefine_nopulse"), ("R5", "R5_freqrefine_pulse")):
        p = os.path.join(OUTD, "sens_base__%s__s0.npz" % fn)
        if not os.path.exists(p):
            print("  跳过 %s（无存档）" % mname); continue
        ref = np.load(p)["testA"]
        mdl = build(mname, 0)
        est = run_model(mdl, mix_all)
        B = stats_batch(est, d["c"], mix_all, d["mask"])
        cols = list(np.load(p)["cols"])
        diffs = []
        for a, b in (("gain_imp", "frmgain_imp"), ("gain_clean", "frmgain_clean"),
                     ("snr_out", "fmsnr_med_out")):
            rv = float(np.nanmean(ref[:, cols.index(b)]))
            nv = float(np.nanmean(B[a]))
            diffs.append((b, rv, nv, abs(rv - nv)))
        dd = max(x[3] for x in diffs)
        good3 = dd < 1e-3
        ok = ok and good3
        print("  %s" % mname)
        for nm, rv, nv, x in diffs:
            print("     %-12s 存档 %9.4f  重算 %9.4f  差 %.2e" % (nm, rv, nv, x))
    print()
    print("=" * 96)
    print("闸门总判定：%s" % ("**通过**" if ok else "**不通过，禁止继续**"))
    print("=" * 96)
    return ok


# ============================ 7. 运行 ============================
def one_cell(cond, t, model, seed_idx, d=None, mdl=None):
    mix, g, Pv_new, Pi = remix(cond, t, d)
    if model in MODELS:
        mdl = mdl if mdl is not None else build(model, seed_idx)
        est = run_model(mdl, mix)
    else:
        est = run_baseline(model, mix)
    B = stats_batch(est, d["c"], mix, d["mask"])
    row = {k: float(np.nanmean(v)) for k, v in B.items()}
    row.update(snr_in=float(np.nanmean(snr_in_batch(mix, d["c"]))),
               bg_snr_meas=float(np.nanmean(10 * np.log10(d["Pc"] / Pv_new))),
               tot_snr_meas=float(np.nanmean(10 * np.log10(d["Pc"] / (Pv_new + Pi)))),
               g_mean=float(np.mean(g)), n=int(len(mix)))
    return row


def run(seeds=5, skip_inter=False, replicate=False, extra_pts=(), pts_all=()):
    jobs = []
    if pts_all:
        # 所有 α 条件 + base 在同一批 SNR 点上各跑一遍，用于 α×SNR 的共同网格
        for t in pts_all:
            for cond in ["base"] + INTER_CONDS:
                for mname in ("R4", "R5"):
                    for si in range(min(seeds, len(MODELS[mname]["ck"]))):
                        jobs.append((cond, t, mname, si))
                for bname in BASELINES:
                    jobs.append((cond, t, bname, 0))
        print("共同网格点 %s：%d 个任务" % (list(pts_all), len(jobs)), flush=True)
    if extra_pts:
        for t in extra_pts:
            for mname in ("R4", "R5"):
                for si in range(min(seeds, len(MODELS[mname]["ck"]))):
                    jobs.append(("base", t, mname, si))
            for bname in BASELINES:
                jobs.append(("base", t, bname, 0))
        print("附加点 %s：%d 个任务" % (list(extra_pts), len(jobs)), flush=True)
    for t in SNR_GRID:
        for mname in ("R4", "R5"):
            for si in range(min(seeds, len(MODELS[mname]["ck"]))):
                jobs.append(("base", t, mname, si))
        for bname in BASELINES:
            jobs.append(("base", t, bname, 0))
    if not skip_inter:
        for cond in INTER_CONDS:
            ns = min(seeds, 5)
            for t in INTER_SNR:
                for mname in ("R4", "R5"):
                    for si in range(ns):
                        jobs.append((cond, t, mname, si))
                for bname in BASELINES:
                    jobs.append((cond, t, bname, 0))
    if replicate:
        # 8.1b 的复现点：所有 α 都对齐到 base 的实测背景 SNR
        for cond in ["base"] + INTER_CONDS:
            for mname in ("R4", "R5"):
                for si in range(min(seeds, len(MODELS[mname]["ck"]))):
                    jobs.append((cond, T_REPL, mname, si))
    print("总任务 %d" % len(jobs), flush=True)

    cache_d = {}
    cur_mdl = {}
    t00 = time.time()
    for k, (cond, t, mname, si) in enumerate(jobs):
        tag = "%s_%s__t%s__%s__s%d.json" % (
            STEM, cond, tkey(t).replace("+", "p").replace("-", "m"), mname, si)
        p = os.path.join(OUTD, tag)
        if os.path.exists(p):
            continue
        if cond not in cache_d:
            cache_d[cond] = np.load(os.path.join(OUTD, "snr_%s%s.npz" % (cond, SUF)))
        mdl = None
        if mname in MODELS:
            key = (mname, si)
            if key not in cur_mdl:
                cur_mdl[key] = build(mname, si)
            mdl = cur_mdl[key]
        t0 = time.time()
        row = one_cell(cond, t, mname, si, cache_d[cond], mdl)
        row.update(cond=cond, alpha=ALPHA[cond], target_bg_snr=t, model=mname, seed=si)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=1)
        print("[%3d/%3d] %-7s t=%+3d %-8s s%d  增益(干扰帧) %6.2f  干净帧 %6.2f  总SNR %+5.2f  (%.1fs, 累计 %.0fs)"
              % (k + 1, len(jobs), cond, t, mname, si, row["gain_imp"], row["gain_clean"],
                 row["tot_snr_meas"], time.time() - t0, time.time() - t00), flush=True)
    print("完成，用时 %.1f 分钟" % ((time.time() - t00) / 60), flush=True)


# ============================ 8. 汇总 ============================
def agg():
    files = sorted(glob.glob(os.path.join(OUTD, STEM + "_*.json")))
    rows = [json.load(open(f, encoding="utf-8")) for f in files]
    if not rows:
        print("没有结果"); return
    # (cond, target, model) -> 逐种子列表
    from collections import defaultdict
    G = defaultdict(lambda: defaultdict(list))
    for r in rows:
        G[(r["cond"], r["target_bg_snr"], r["model"])][r["seed"]].append(r)
    out = {"grid": SNR_GRID, "inter_snr": INTER_SNR, "n_cell": len(rows), "cells": {}}
    for (cond, t, mname), byseed in sorted(G.items()):
        per_seed = {str(s): {k: float(np.mean([x[k] for x in v]))
                             for k in ("gain_imp", "gain_clean", "gain_med", "snr_out", "snr_imp")}
                    for s, v in sorted(byseed.items())}
        keys = ("gain_imp", "gain_clean", "gain_med", "snr_out", "snr_imp")
        mean = {k: float(np.mean([p[k] for p in per_seed.values()])) for k in keys}
        sd = {k: float(np.std([p[k] for p in per_seed.values()], ddof=1))
              if len(per_seed) > 1 else 0.0 for k in keys}
        r0 = list(byseed.values())[0][0]
        out["cells"]["%s|%s|%s" % (cond, tkey(t), mname)] = dict(
            cond=cond, alpha=ALPHA[cond], target_bg_snr=t, model=mname,
            n_seed=len(per_seed), mean=mean, sd=sd, per_seed=per_seed,
            tot_snr_meas=r0["tot_snr_meas"], bg_snr_meas=r0["bg_snr_meas"],
            snr_in=r0["snr_in"], n=r0["n"])
    # Δ = R5 − R4，逐种子配对
    deltas = {}
    for cond in ["base"] + INTER_CONDS:
        for t in sorted({c["target_bg_snr"] for c in out["cells"].values() if c["cond"] == cond}):
            a = out["cells"].get("%s|%s|R4" % (cond, tkey(t)))
            b = out["cells"].get("%s|%s|R5" % (cond, tkey(t)))
            if not a or not b:
                continue
            seeds = sorted(set(a["per_seed"]) & set(b["per_seed"]))
            d = [b["per_seed"][s]["gain_imp"] - a["per_seed"][s]["gain_imp"] for s in seeds]
            dc = [b["per_seed"][s]["gain_clean"] - a["per_seed"][s]["gain_clean"] for s in seeds]
            deltas["%s|%s" % (cond, tkey(t))] = dict(
                cond=cond, target_bg_snr=t, n_seed=len(seeds), per_seed=d,
                d_mean=float(np.mean(d)), d_med=float(np.median(d)),
                d_sd=float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
                d_clean_mean=float(np.mean(dc)),
                n_pos=int(sum(1 for x in d if x > 0)),
                tot_snr_meas=a["tot_snr_meas"])
    out["delta"] = deltas
    p = os.path.join(OUTD, "snr_sweep%s.json" % SUF)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("->", p)
    print()
    print("%-8s %8s %10s %10s %10s %8s" % ("条件", "目标SNR", "Δ(干扰帧)", "Δ(干净帧)", "实测总SNR", "同号"))
    for k, v in deltas.items():
        print("%-8s %+8.2f %+10.3f %+10.3f %10.2f %6s"
              % (v["cond"], v["target_bg_snr"], v["d_mean"], v["d_clean_mean"],
                 v["tot_snr_meas"], "%d/%d" % (v["n_pos"], v["n_seed"])))


def diag(points=(("base", -15), ("base", -3), ("base", 0), ("base", 15))):
    """逐样本诊断：Δ 是普遍抬高还是少数样本拉出来的？

    聚合指标是"逐样本 median 帧增益 -> 对样本取均值"，均值对离群样本敏感。
    这里把 seed0 的逐样本 gain_imp 存下来，给出
      · 配对差的均值 / 中位 / 正号比例
      · 去掉 |Δ| 最大的 5% 样本后的均值（截尾均值）
      · 贡献最大的 5 个样本
    """
    d_all = {}
    for cond, t in points:
        d = np.load(os.path.join(OUTD, "snr_%s%s.npz" % (cond, SUF)))
        mix, g, Pv_new, Pi = remix(cond, t, d)
        per = {}
        for mname in ("R4", "R5"):
            mdl = build(mname, 0)
            est = run_model(mdl, mix)
            B = stats_batch(est, d["c"], mix, d["mask"])
            per[mname] = B
        d4, d5 = per["R4"]["gain_imp"], per["R5"]["gain_imp"]
        ok = np.isfinite(d4) & np.isfinite(d5)
        diff = (d5 - d4)[ok]
        idx = np.where(ok)[0]
        n = len(diff)
        cut = max(1, int(round(0.05 * n)))
        order = np.argsort(-np.abs(diff))
        trimmed = float(np.mean(np.sort(diff)[cut:n - cut]))
        print("-" * 88)
        print("%s  t=%+.2f  有效样本 %d（无干扰帧 %d）"
              % (cond, t, n, len(d4) - n))
        print("  R4 均值 %.3f  R5 均值 %.3f  Δ均值 %+.3f  中位 %+.3f  正号 %d/%d"
              % (np.mean(d4[ok]), np.mean(d5[ok]), np.mean(diff), np.median(diff),
                 int((diff > 0).sum()), n))
        print("  截尾均值（去掉 |Δ| 最大 5%%）%+.3f  ⇒ 离群贡献 %+.3f"
              % (trimmed, np.mean(diff) - trimmed))
        print("  最大 5 个 |Δ| 样本：%s"
              % ", ".join("#%d(%+.1f)" % (idx[i], diff[i]) for i in order[:5]))
        d_all["%s|%s" % (cond, tkey(t))] = dict(diff=diff.tolist(), idx=idx.tolist(),
                                                g=float(np.mean(g)))
    p = os.path.join(OUTD, "snr_diag.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d_all, f)
    print("->", p)


def detect():
    """机制量：突发在【输入】里还有多容易被看出来（不需要模型）。

    定义：把输入切成 128 点帧，算每帧能量（dB），
        det_in = median(E_burst帧) − median(E_clean帧)   （dB）
    即"突发帧比非突发帧响多少"。突发幅度固定（imp = imp_full·rms_c，
    与 SNR 无关），所以背景噪声越强，这个差越小 —— 突发越"看不见"。
    若 Δ 与该量单调负相关，则支持"分支在替代一个越来越难的检测"这一机制。
    同时在【输出】上算同一个量：det_out 小说明该模型把突发压下去了。
    """
    rows = {}
    # 点集直接从已聚合结果里取，避免与 SNR_GRID / INTER_SNR 的常量漂移
    _S = json.load(open(os.path.join(OUTD, "snr_sweep%s.json" % SUF), encoding="utf-8"))
    grid = [(v["cond"], v["target_bg_snr"]) for v in _S["delta"].values()]
    grid.append(("base", T_REPL))
    grid += [(c, T_REPL) for c in INTER_CONDS]
    for cond, t in grid:
        d = np.load(os.path.join(OUTD, "snr_%s%s.npz" % (cond, SUF)))
        mix, g, Pv_new, Pi = remix(cond, t, d)
        x = mix.astype(np.float64)
        N, L = x.shape
        nf = L // FRAME
        E = 10 * np.log10((x[:, :nf * FRAME].reshape(N, nf, FRAME) ** 2).sum(2) + 1e-20)
        mf = (d["mask"][:, :nf * FRAME].reshape(N, nf, FRAME).max(2) > 0)
        ok = mf.any(1) & (~mf).any(1)
        det, auc = [], []
        for i in np.where(ok)[0]:
            eb, ec = E[i][mf[i]], E[i][~mf[i]]
            det.append(np.median(eb) - np.median(ec))
            # AUC = P(突发帧能量 > 干净帧能量)，用秩统计（Mann-Whitney）
            # 只用中位数差会在跨 α 比较时失真：重尾背景的干净帧中位数很低，
            # 但它的高分位尾巴照样能冒充突发。AUC 对尾部敏感，跨 α 可比。
            r = np.argsort(np.argsort(np.concatenate([eb, ec]))) + 1
            rb = r[:len(eb)].sum()
            auc.append((rb - len(eb) * (len(eb) + 1) / 2.0) / (len(eb) * len(ec)))
        rows["%s|%s" % (cond, tkey(t))] = dict(
            cond=cond, t=t, det_in=float(np.median(det)), det_auc=float(np.mean(auc)),
            n=int(ok.sum()),
            tot_snr_meas=float(np.nanmean(10 * np.log10(d["Pc"] / (Pv_new + Pi)))))
        print("  %-7s t=%+7.2f  帧能量差 %6.2f dB  AUC %.3f  总SNR %+6.2f"
              % (cond, t, rows["%s|%s" % (cond, tkey(t))]["det_in"],
                 rows["%s|%s" % (cond, tkey(t))]["det_auc"],
                 rows["%s|%s" % (cond, tkey(t))]["tot_snr_meas"]), flush=True)
    p = os.path.join(OUTD, "snr_detect.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("->", p)


def bench():
    d = np.load(os.path.join(OUTD, "snr_base.npz"))
    mix, g, _, _ = remix("base", 0.0, d)
    for mname in ("R4", "R5"):
        mdl = build(mname, 0)
        t0 = time.time(); est = run_model(mdl, mix); t1 = time.time()
        B = stats_batch(est, d["c"], mix, d["mask"]); t2 = time.time()
        print("%s: 前向 %.1fs  指标 %.1fs  合计 %.1fs / %d 样本"
              % (mname, t1 - t0, t2 - t1, t2 - t0, len(mix)), flush=True)
    for b in BASELINES:
        t0 = time.time(); est = run_baseline(b, mix); t1 = time.time()
        B = stats_batch(est, d["c"], mix, d["mask"]); t2 = time.time()
        print("%s: 处理 %.1fs  指标 %.1fs  合计 %.1fs" % (b, t1 - t0, t2 - t1, t2 - t0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--agg", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--skip-inter", action="store_true")
    ap.add_argument("--replicate", action="store_true")
    ap.add_argument("--extra-pts", nargs="*", type=float, default=[])
    ap.add_argument("--pts-all", nargs="*", type=float, default=[])
    ap.add_argument("--split", default="testA", choices=["testA", "testC"])
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--detect", action="store_true")
    ap.add_argument("--conds", nargs="*", default=[])
    A = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)
    set_split(A.split)
    if A.split != "testA":
        print("[split] %s（产物后缀 %s）" % (A.split, SUF), flush=True)
    if A.prep:
        for c in (A.conds or list(CONDS)):
            print("[prep] %s" % c, flush=True)
            prep(c)
    if A.verify:
        if SPLIT != "testA":
            print("[verify] 闸门以 testA 存档为参照，跳过（%s 无对应存档）" % SPLIT)
        else:
            verify()
    if A.bench:
        bench()
    if A.run:
        if A.pts_all:
            run(A.seeds, True, False, (), A.pts_all)
        elif A.extra_pts:
            run(A.seeds, True, False, A.extra_pts)
        else:
            run(A.seeds, A.skip_inter, A.replicate)
    if A.diag:
        diag()
    if A.detect:
        detect()
    if A.agg:
        agg()


if __name__ == "__main__":
    main()
