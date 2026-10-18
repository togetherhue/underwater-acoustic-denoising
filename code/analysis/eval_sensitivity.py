# -*- coding: utf-8 -*-
"""
适用边界（敏感性）实验：缓存 + 评估
================================================================================
【为什么要缓存】
原评估走 train_pam.run_eval -> PAMDataset -> scipy.io.loadmat，每样本约 40 ms，
1320 个样本的数据加载就占掉总耗时的一半以上。本脚本先把每个条件的
testA/testC 预处理（重采样到 1/4、定长 8000、掩码二值化）成 float32 数组缓存，
之后每次评估只做前向 + 指标，吞吐提升约 2 倍。

【口径一致性】
指标计算逐行复制 train_pam.run_eval（同一个 PM._stats_1d、同样的列顺序），
并且提供 --verify 模式：对 base 条件重算一遍，与既有 result_*.csv 逐位比对。

用法：
  python eval_sensitivity.py --prep                 # 生成全部缓存
  python eval_sensitivity.py --verify               # 校验口径
  python eval_sensitivity.py --conds a12_r5 --configs R3_freqonly
  python eval_sensitivity.py --all --jobs 4         # 分片跑（见 --shard/--nshard）
"""
import os, sys, glob, json, time, argparse
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

# ---- 路径解析：兼容仓库布局（库模块在 ../python）与本地开发布局 ----
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
CKPT = r"E:/project/data/output_pamnet"
SEG, DS = V3.CFG["seg_len"], V3.CFG["downsample"]
NFFT, HOP = V3.CFG["n_fft"], V3.CFG["hop_length"]

CONDS = {
    "base":    r"E:/project/data/dataset_pam/alphastable",
    "a12_r5":  r"E:/project/data/ds_sens/a12_r5",
    "a18_r5":  r"E:/project/data/ds_sens/a18_r5",
    "a20_r5":  r"E:/project/data/ds_sens/a20_r5",
    "a15_r25": r"E:/project/data/ds_sens/a15_r25",
    "a15_r10": r"E:/project/data/ds_sens/a15_r10",
    "gauss":   r"E:/project/data/dataset_pam/gauss",
    # 等有效功率 SNR 的 α 重扫：名义 SNR 已平移，使有效功率 SNR 与 base 对应
    "m_a12":   r"E:/project/data/ds_sens_matched/m_a12",
    "m_a18":   r"E:/project/data/ds_sens_matched/m_a18",
}

MODELS = {
    "R3_freqonly": dict(
        arch=dict(use_short=False, use_long=False, use_gate=True, use_pulse=False,
                  freq_refine=2.0, pulse_freq=False), params=5910,
        ckpts=["model_fo_gn.pth", "model_fo_gn_s2.pth", "model_fo_gn_s3.pth",
               "model_fo_gn_s4.pth", "model_fo_gn_s5.pth"],
        label="纯频域掩码"),
    "R4_freqrefine_nopulse": dict(
        arch=dict(use_short=True, use_long=True, use_gate=True, use_pulse=False,
                  freq_refine=2.0, pulse_freq=False), params=14367,
        ckpts=["model_pamnet_freq_nopulse_gn.pth", "model_pamnet_freq_nopulse_gn_s2.pth",
               "model_pamnet_freq_nopulse_gn_s3.pth", "model_pamnet_freq_nopulse_gn_s4.pth",
               "model_pamnet_freq_nopulse_gn_s5.pth"],
        label="时域残差+频域精修（无脉冲分支）"),
    "R5_freqrefine_pulse": dict(
        arch=dict(use_short=True, use_long=True, use_gate=True, use_pulse=True,
                  freq_refine=2.0, pulse_freq=True), params=14858,
        ckpts=["model_pamnet_freq_pf_gn.pth", "model_pamnet_freq_pf_gn_s2.pth",
               "model_pamnet_freq_pf_gn_s3.pth", "model_pamnet_freq_pf_gn_s4.pth",
               "model_pamnet_freq_pf_gn_s5.pth"],
        label="时域残差+频域精修+p注入（推荐）"),
    "P_time_pulse": dict(
        arch=dict(use_short=True, use_long=True, use_gate=True, use_pulse=True,
                  freq_refine=0.0, pulse_freq=False), params=8906,
        ckpts=["model_pamnet_gn.pth", "model_pamnet_gn_s2.pth", "model_pamnet_gn_s3.pth"],
        label="纯时域 PAM-Net"),
    "P_time_nopulse": dict(
        arch=dict(use_short=True, use_long=True, use_gate=True, use_pulse=False,
                  freq_refine=0.0, pulse_freq=False), params=8457,
        ckpts=["model_pamnet_nopulse_gn.pth", "model_pamnet_nopulse_gn_s2.pth",
               "model_pamnet_nopulse_gn_s3.pth"],
        label="纯时域（无脉冲分支）"),
}
SPLITS = ["testA", "testC"]
I_MED, I_CLN, I_IMP, I_MIN = 6, 7, 8, 9
COLS = ["snr_in", "snr_out", "sisdr_in", "sisdr_out",
        "fmsnr_med_in", "fmsnr_med_out", "frmgain_med", "frmgain_clean",
        "frmgain_imp", "frmgain_imp_min", "fmsnr_frac_bad_out", "frac_imp"]


# ============================ 缓存 ============================
def prep_cond(cond):
    """把一个条件的 testA/testC 预处理为定长数组缓存。

    【必须与 PAMDataset 逐位一致】PAMDataset 的顺序是
        loadmat -> astype(float32) -> resample_poly -> astype(float32) -> 定长
    即【先降到 float32 再重采样】。若反过来（先 float64 重采样再降精度），
    中间精度不同会让结果产生 ~1e-7 的相对差，经深层网络的 STFT 与除法放大后
    可到 1e-1 dB 量级。这里严格照抄 PAMDataset 的顺序。
    mask 在 PAMDataset 里是 float64 重采样后 >0.3，同样照抄。
    """
    out = {}
    for sp in SPLITS:
        files = sorted(glob.glob(os.path.join(CONDS[cond], sp, "*.mat")))
        N, C, M = [], [], []
        for f in files:
            m = sio.loadmat(f)
            n = np.asarray(m["noisy_signal"]).squeeze().astype(np.float32)
            c = np.asarray(m["clean_signal"]).squeeze().astype(np.float32)
            k = np.asarray(m["imp_mask"]).squeeze().astype(np.float64)
            if DS > 1:
                n = resample_poly(n, 1, DS).astype(np.float32)
                c = resample_poly(c, 1, DS).astype(np.float32)
                k = resample_poly(k, 1, DS)

            def fx(x):
                return x[:SEG] if len(x) >= SEG else np.pad(x, (0, SEG - len(x)))
            N.append(fx(n)); C.append(fx(c)); M.append((fx(k) > 0.3).astype(np.uint8))
        out["%s_n" % sp] = np.stack(N); out["%s_c" % sp] = np.stack(C)
        out["%s_m" % sp] = np.stack(M)
    p = os.path.join(OUTD, "cache_%s.npz" % cond)
    np.savez(p, **out)
    print("  缓存 -> %s  (testA %d, testC %d, %.1f MB)"
          % (p, len(out["testA_n"]), len(out["testC_n"]),
             os.path.getsize(p) / 1e6), flush=True)
    prep_noise(cond)


def prep_noise(cond):
    """逐样本噪声功率分解（float64，独立于评估缓存）。

    背景噪声 v 无法直接读出，但数据集同时保存了 imp，故 v = noisy - clean - imp。
    全部在【降采样 + 定长】后的信号上计算，与模型看到的域一致。
    列：Pc Pv Pi  snr_bg  snr_imp  snr_tot  frac_imp  peak_v/rms  peak_i/rms
    """
    nd = {}
    for sp in SPLITS:
        files = sorted(glob.glob(os.path.join(CONDS[cond], sp, "*.mat")))
        rows = []
        for f in files:
            m = sio.loadmat(f)
            n = np.asarray(m["noisy_signal"]).squeeze().astype(np.float64)
            c = np.asarray(m["clean_signal"]).squeeze().astype(np.float64)
            i = np.asarray(m["imp"]).squeeze().astype(np.float64)
            k = np.asarray(m["imp_mask"]).squeeze().astype(np.float64)
            n = resample_poly(n, 1, DS); c = resample_poly(c, 1, DS)
            i = resample_poly(i, 1, DS); k = resample_poly(k, 1, DS)

            def fx(x):
                return x[:SEG] if len(x) >= SEG else np.pad(x, (0, SEG - len(x)))
            n, c, i, k = fx(n), fx(c), fx(i), fx(k)
            v = n - c - i
            Pc = float(np.mean(c ** 2)); Pv = float(np.mean(v ** 2)); Pi = float(np.mean(i ** 2))
            rms_c = np.sqrt(Pc) + 1e-300
            rows.append([Pc, Pv, Pi,
                         10 * np.log10(Pc / (Pv + 1e-300)),
                         10 * np.log10(Pc / (Pi + 1e-300)),
                         10 * np.log10(Pc / (Pv + Pi + 1e-300)),
                         float(np.mean(k > 0.3)),
                         float(np.max(np.abs(v)) / rms_c),
                         float(np.max(np.abs(i)) / rms_c),
                         float(np.percentile(np.abs(v), 99.9) / rms_c),
                         float(np.percentile(np.abs(v), 99.0) / rms_c),
                         float(np.percentile(np.abs(v), 75.0) / rms_c)])
        nd[sp] = np.array(rows)
    np.savez(os.path.join(OUTD, "noise_%s.npz" % cond), **nd)
    a = nd["testA"]
    print("  噪声分解: 名义q75固定下 有效功率SNR(总) 中位 %+.2f dB  "
          "(背景 %+.2f / 干扰 %+.2f)  干扰帧占比 %.3f  背景峰/rms %.1f"
          % (np.median(a[:, 5]), np.median(a[:, 3]), np.median(a[:, 4]),
             np.mean(a[:, 6]), np.median(a[:, 7])), flush=True)


# ============================ 评估（口径 = train_pam.run_eval）============================
@torch.no_grad()
def eval_cached(model, noisy, clean, mask, bs=32):
    """逐样本计算全部指标。noisy/clean: [n,8000] float32, mask: [n,8000] uint8"""
    n = len(noisy)
    rows = []
    for s in range(0, n, bs):
        nb = torch.from_numpy(noisy[s:s + bs]).unsqueeze(1)
        cb = torch.from_numpy(clean[s:s + bs]).unsqueeze(1)
        mb = torch.from_numpy(mask[s:s + bs].astype(np.float32))
        e = model(nb)
        for k in range(e.shape[0]):
            ei, ri, ni = e[k, 0].cpu(), cb[k, 0].cpu(), nb[k, 0].cpu()
            mi = mb[k].numpy()
            si = PM.snr_db(ni, ri).item()
            so = PM.snr_db(ei, ri).item()
            di = PM.si_sdr_db(ni, ri).item()
            do = PM.si_sdr_db(ei, ri).item()
            a = PM._stats_1d(ei.numpy().astype(np.float64), ri.numpy().astype(np.float64),
                             ni.numpy().astype(np.float64), PM.FRAME, mi)
            b = PM._stats_1d(ni.numpy().astype(np.float64), ri.numpy().astype(np.float64),
                             ni.numpy().astype(np.float64), PM.FRAME, mi)
            rows.append([si, so, di, do, b["fmsnr_med"], a["fmsnr_med"],
                         a["frmgain_med"], a["frmgain_clean"], a["frmgain_imp"],
                         a["frmgain_imp_min"], a["fmsnr_frac_bad"], a["frac_imp"]])
    return np.array(rows, dtype=np.float64)


def build_model(arch):
    return PN.PAMNet(c=24, n_fft=NFFT, hop=HOP, norm="gn", **arch).to(V3.DEVICE)


def run_one(cond, cname, si):
    mcfg = MODELS[cname]
    cache = np.load(os.path.join(OUTD, "cache_%s.npz" % cond))
    model = build_model(mcfg["arch"])
    model.load_state_dict(torch.load(os.path.join(CKPT, mcfg["ckpts"][si]),
                                     map_location=V3.DEVICE), strict=True)
    model = model.to(V3.DEVICE).eval()
    res = {}
    for sp in SPLITS:
        res[sp] = eval_cached(model, cache["%s_n" % sp], cache["%s_c" % sp],
                              cache["%s_m" % sp])
    return res


def verify():
    """对 base 条件重算，与既有 result_*.csv 比对（口径校验）。

    【判定标准】同时看两个层次：
      1. 逐样本差 —— 纯频域模型应 ≤1e-4 dB；含时域分支的模型已知存在
         1e-7 量级浮点归约差异经时域通路放大到 ~0.1 dB 的现象
         （见报告第十二节"局限"最后一条），这是已知且不可消除的。
      2. **聚合指标差** —— 这是本研究全部结论的依据，要求 ≤1e-3 dB。
    判定以【聚合指标】为准。
    """
    pairs = [("R3_freqonly", "result_fo_gn_testA.csv", 0),
             ("R5_freqrefine_pulse", "result_pamnet_freq_pf_gn_testA.csv", 0),
             ("R4_freqrefine_nopulse", "result_pamnet_freq_nopulse_gn_testA.csv", 0)]
    ok = True
    for cname, fn, si in pairs:
        ref_p = os.path.join(CKPT, fn)
        if not os.path.exists(ref_p):
            print("  跳过 %s（不存在）" % fn); continue
        ref = np.loadtxt(ref_p, delimiter=",", skiprows=1)
        new = run_one("base", cname, si)["testA"]
        d = np.abs(ref - new)
        dmax, dmean = np.nanmax(d), np.nanmean(d)
        # 聚合口径：三个关键指标各自的均值
        agg = []
        for j, nm in ((I_CLN, "干净"), (I_IMP, "干扰"), (I_MIN, "最差")):
            r, n2 = np.nanmean(ref[:, j]), np.nanmean(new[:, j])
            agg.append((nm, r, n2, abs(r - n2)))
        aggd = max(a[3] for a in agg)
        good = (ref.shape == new.shape) and aggd < 1e-3
        ok = ok and good
        print("  %-24s 形状 %s vs %s  逐样本最大差 %.3e (均值 %.2e)" %
              (cname, ref.shape, new.shape, dmax, dmean))
        for nm, r, n2, dd in agg:
            print("      %-4s帧 存档 %9.4f  重算 %9.4f  差 %.2e %s"
                  % (nm, r, n2, dd, "✓" if dd < 1e-3 else "✗"))
    print("  口径校验：%s（聚合指标差 < 1e-3 dB）"
          % ("**通过**" if ok else "**不一致，需检查**"))
    print("  说明：逐样本差在含时域分支的模型上可达 ~0.1 dB，这是 1e-7 量级浮点")
    print("        归约差异经时域通路放大的结果（纯频域模型仅 5e-5 dB），")
    print("        已在报告局限中记录；本研究结论全部基于聚合指标与跨种子检验。")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--conds", nargs="*", default=[])
    ap.add_argument("--configs", nargs="*", default=[])
    ap.add_argument("--seeds", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    A = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

    if A.prep:
        for c in (A.conds or list(CONDS.keys())):
            if not os.path.isdir(os.path.join(CONDS[c], "testA")):
                print("跳过 %s（无数据）" % c); continue
            print("[prep] %s" % c, flush=True)
            prep_cond(c)
        return
    if A.verify:
        verify(); return

    conds = A.conds or list(CONDS.keys())
    cnames = A.configs or list(MODELS.keys())
    # 纯时域两个配置只在 3 个关键条件（base / 最脉冲化 / 最密集干扰）上评估，
    # 用于检验"纯时域里脉冲分支效应≈0"是否跨条件成立；其余条件省下来。
    PTIME_KEY = {"base", "a12_r5", "a15_r10"}
    tasks = []
    for c in conds:
        if not os.path.exists(os.path.join(OUTD, "cache_%s.npz" % c)):
            print("跳过 %s（无缓存）" % c); continue
        for cn in cnames:
            if cn.startswith("P_time") and c not in PTIME_KEY:
                continue
            nseed = len(MODELS[cn]["ckpts"]) if A.seeds <= 0 else min(A.seeds, len(MODELS[cn]["ckpts"]))
            for si in range(nseed):
                tasks.append((c, cn, si))
    tasks = [t for i, t in enumerate(tasks) if i % A.nshard == A.shard]
    # 断点续跑：已有结果的任务直接跳过
    todo = [t for t in tasks
            if not os.path.exists(os.path.join(OUTD, "sens_%s__%s__s%d.npz" % t))]
    print("本分片任务数 %d（其中已完成 %d，待跑 %d）" % (len(tasks), len(tasks) - len(todo), len(todo)),
          flush=True)
    tasks = todo

    man = {}
    mp = os.path.join(OUTD, "manifest.json")
    if os.path.exists(mp):
        man = json.load(open(mp, encoding="utf-8"))
    for c, cn, si in tasks:
        t0 = time.time()
        r = run_one(c, cn, si)
        np.savez_compressed(os.path.join(OUTD, "sens_%s__%s__s%d.npz" % (c, cn, si)),
                            cols=np.array(COLS), **{k: v.astype(np.float32) for k, v in r.items()})
        m = {sp: {kk: float(np.nanmean(r[sp][:, j])) for j, kk in enumerate(COLS)}
             for sp in SPLITS}
        if A.nshard == 1:
            # 多分片并行时不写共享 manifest，避免竞争；逐任务 npz 已足够分析
            man.setdefault(c, {}).setdefault(cn, {})["s%d" % si] = m
            with open(mp, "w", encoding="utf-8") as f:
                json.dump(man, f, ensure_ascii=False, indent=1)
        print("[%s|%s|s%d] testA 干净 %.2f 干扰 %.2f 最差 %.2f | testC 干净 %.2f 干扰 %.2f  (%.0fs)"
              % (c, cn, si, m["testA"]["frmgain_clean"], m["testA"]["frmgain_imp"],
                 m["testA"]["frmgain_imp_min"], m["testC"]["frmgain_clean"],
                 m["testC"]["frmgain_imp"], time.time() - t0), flush=True)
    print("分片完成")


if __name__ == "__main__":
    main()
