# -*- coding: utf-8 -*-
"""
机制分析：脉冲感知分支的收益由什么决定？
================================================================================
起因：逐样本诊断发现，p 注入频域的效应在三个测试波形间极不均匀
（chirp 19 贡献了 testA 干扰帧效应的 2/3，chirp 18 几乎为零），
且两个信道完全不重叠的测试集给出同一排序 —— 说明这是【波形相关】的真实交互。

本脚本检验一个物理假设：
    H1  带宽越宽 → 信号与宽带干扰的时频重叠越多 → 脉冲感知分支越有价值
以及一个更直接的数据驱动量：
    H2  干扰落在信号频带内的能量占比越高 → 分支越有价值

方法：
  1. 用 make_dataset_pam 生成"全 chirp 扫描集"（chirp 1-20 × 信道 1-8 × 5 SNR × 2 噪声）
  2. 对每个样本计算干扰的【带内能量占比】（带内 = 该 chirp 的 [f_lo, f_hi] ± 边距）
     以及基于干净信号时频谱支撑的【时频重叠率】
  3. 用已训练好的 pf / nopulse 各 5 个种子在扫描集上推理，得到逐样本指标
  4. 按 chirp 聚合，做效应 vs 波形参数/重叠量的相关分析

注意：chirp 1-14 模型训练时见过，15-20 未见过；两者分开报告。
"""
import os, sys, csv, glob, re, json, argparse
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, r"C:\Users\32213\PycharmProjects\project1"):
    if p not in sys.path:
        sys.path.insert(0, p)

import v3_model as V3          # noqa: E402
import pamnet as PN            # noqa: E402
import pam_metrics as PM       # noqa: E402
from train_pam import PAMDataset, build_loader, COLS   # noqa: E402

OUT = r"D:/project/output_pamnet"
SCAN = r"D:/project/ds_chirpscan/chirpscan"
CTAB = r"D:/project/chirp_table.csv"
FIG = os.path.join(_HERE, "figs_pamnet")
os.makedirs(FIG, exist_ok=True)

N_FFT, HOP = V3.CFG["n_fft"], V3.CFG["hop_length"]
DS = V3.CFG["downsample"]
SEG = V3.CFG["seg_len"]


# ---------------------------------------------------------------- 工具
def chirp_table():
    d = {}
    for r in csv.DictReader(open(CTAB, encoding="utf-8")):
        d[int(r["chirp_id"])] = {k: (float(r[k]) if k not in ("env",) else r[k])
                                 for k in r}
    return d


def seeds_of(prefix):
    out = []
    for t in sorted(glob.glob(os.path.join(OUT, "summary_%s*.csv" % prefix))):
        base = os.path.basename(t)[8:-4]
        if not base.startswith(prefix):
            continue
        suf = base[len(prefix):]
        if suf == "":
            out.append((1, base))
        elif re.fullmatch(r"_s\d+", suf):
            out.append((int(suf[2:]), base))
    return sorted(out)


def model_of(kind, norm="gn"):
    kw = dict(c=24, n_fft=N_FFT, hop=HOP, norm=norm, freq_refine=2.0)
    if kind == "pf":
        kw["pulse_freq"] = True
    else:
        kw["use_pulse"] = False
    return PN.PAMNet(**kw)


@torch.no_grad()
def eval_dir(model, folder, bs=16, workers=3):
    model = model.to(V3.DEVICE).eval()
    loader = build_loader(folder, False, bs, workers)
    rows = []
    for noisy, clean, mask in loader:
        n = noisy.unsqueeze(1).to(V3.DEVICE)
        c = clean.unsqueeze(1).to(V3.DEVICE)
        e = model(n)
        for k in range(n.shape[0]):
            st = PM._stats_1d(e[k, 0].cpu().numpy().astype(np.float64),
                              c[k, 0].cpu().numpy().astype(np.float64),
                              n[k, 0].cpu().numpy().astype(np.float64),
                              PM.FRAME, mask[k].numpy())
            rows.append([st["frmgain_med"], st["frmgain_clean"],
                         st["frmgain_imp"], st["frmgain_imp_min"], st["frac_imp"]])
    return np.array(rows, dtype=np.float64)


def tf_features(files, bands):
    """逐样本计算干扰的带内能量占比 + 时频重叠率。

    带内 = 该 chirp 的 [f_lo, f_hi] ± 10% 边距。
    时频重叠率 = 干扰能量中落在"干净信号时频支撑"内的比例
                （支撑 = 干净信号 STFT 幅度超过其最大值 1% 的 bin）。
    """
    f_axis = np.fft.rfftfreq(N_FFT, 1.0 / (V3.CFG["fs_native"] / DS))
    out = []
    for f in files:
        m = sio.loadmat(f)
        c = np.asarray(m["clean_signal"]).squeeze().astype(np.float64)
        i = np.asarray(m["imp"]).squeeze().astype(np.float64)
        c = resample_poly(c, 1, DS)[:SEG]
        i = resample_poly(i, 1, DS)[:SEG]
        c = np.pad(c, (0, SEG - len(c))); i = np.pad(i, (0, SEG - len(i)))
        win = np.hanning(N_FFT)
        C = np.abs(np.array([np.fft.rfft(c[k:k + N_FFT] * win)
                             for k in range(0, SEG - N_FFT + 1, HOP)]))
        I = np.abs(np.array([np.fft.rfft(i[k:k + N_FFT] * win)
                             for k in range(0, SEG - N_FFT + 1, HOP)]))
        Pi = (I ** 2).sum(0)                      # 干扰的功率谱（时间求和）
        Pc = (C ** 2).sum(0)
        # (a) 带内占比
        b = bands[f]
        lo, hi = b["f_lo"] * 0.9, min(b["f_hi"] * 1.1, f_axis[-1])
        msk = (f_axis >= lo) & (f_axis <= hi)
        inband = float(Pi[msk].sum() / (Pi.sum() + 1e-30))
        # (b) 时频重叠率
        sup = Pc > 1e-4 * Pc.max()
        overlap = float(((Pi * sup).sum()) / (Pi.sum() + 1e-30))
        out.append([inband, overlap, float(Pi[msk].sum() / (Pc[msk].sum() + 1e-30))])
    return np.array(out)


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    t = r * np.sqrt((n - 2) / max(1e-12, 1 - r ** 2))
    return r, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="限制样本数（0=全部）")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--use-cache", type=int, default=1,
                    help="1=若缓存存在则直接复用（跳过 10 次推理，约 8 分钟）")
    ap.add_argument("--rebuild", type=int, default=0, help="1=强制重算缓存")
    ap.add_argument("--scan", default=SCAN, help="扫描集目录")
    ap.add_argument("--cache-tag", default="", dest="cache_tag",
                    help="缓存文件后缀，用于区分不同扫描集")
    A = ap.parse_args()
    scan = A.scan
    meta_p = os.path.join(os.path.dirname(scan), "meta_chirpscan.mat")
    CACHE = os.path.join(FIG, "chirp_band_cache%s.npz" % A.cache_tag)

    ctab = chirp_table()
    files = sorted(glob.glob(os.path.join(scan, "*.mat")))
    mt = sio.loadmat(meta_p, squeeze_me=True, struct_as_record=False)["meta"]
    ci = np.array([mt[k].chirp_id for k in range(mt.size)])
    ca = np.array([mt[k].chan_id for k in range(mt.size)])
    sn = np.array([mt[k].snr for k in range(mt.size)])
    assert len(files) == len(ci) == mt.size, "文件数与 meta 不一致"
    # 顺序对齐校验：meta 的 file 字段必须与 sorted(glob) 逐个对应，
    # 否则 chirp_id 会与文件错配（本项目此前在信道映射上踩过同类坑）。
    nm = [str(mt[k].file) for k in range(mt.size)]
    nb = [os.path.basename(f) for f in files]
    bad = [i for i in range(len(nb)) if nm[i] not in nb[i]]
    assert not bad, "meta 顺序与文件顺序不一致（前 3 处：%s）" % bad[:3]
    print("  ✓ meta 顺序与文件顺序逐条对应（%d 条）" % len(nb))

    if A.n:
        idx = np.arange(A.n)
        files = [files[k] for k in idx]
        ci, ca, sn = ci[idx], ca[idx], sn[idx]

    print("=" * 100)
    print("chirp 机制分析：脉冲感知分支的收益由什么决定")
    print("=" * 100)
    print("扫描集 %s   样本 %d   chirp %d 个   信道 %d 个" %
          (scan, len(files), len(set(ci.tolist())), len(set(ca.tolist()))))
    print()

    # ---- 1. 模型评估（带缓存）----
    # 说明：把"模型评估"放在最前，是因为它最慢（10 次推理 ≈ 8 分钟），
    # 且缓存后反复重算只涉及 CPU 侧的时频特征与统计，迭代很快。
    print("[1] 模型评估（pf / nopulse 各 5 个种子，种子间平均）")
    per_model = {}
    if A.use_cache and not A.rebuild and os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        ci, ca, sn = z["ci"], z["ca"], z["sn"]
        files = [str(x) for x in z["files"]]
        per_model = {"pf": z["pf"], "nopulse": z["nopulse"]}
        print("    复用缓存 %s（%d 样本）" % (CACHE, len(files)))
    else:
        for kind in ["pf", "nopulse"]:
            sd_list = seeds_of("pamnet_freq_%s_gn" % kind)
            if not sd_list:
                print("  !! 找不到 %s 的种子" % kind); continue
            accs = []
            for sd, tag in sd_list:
                ck = os.path.join(OUT, "model_%s.pth" % tag)
                if not os.path.exists(ck):
                    continue
                m = model_of(kind)
                m.load_state_dict(torch.load(ck, map_location="cpu"), strict=True)
                print("    评估 %-9s seed=%-2d ..." % (kind, sd), flush=True)
                accs.append(eval_dir(m, scan, A.bs, A.workers))
            if accs:
                per_model[kind] = np.mean(accs, axis=0)
                print("    %s：%d 个种子平均完成" % (kind, len(accs)))
        np.savez_compressed(CACHE, pf=per_model["pf"], nopulse=per_model["nopulse"],
                            ci=ci, ca=ca, sn=sn,
                            files=np.array(files, dtype=object))
        print("    缓存 -> %s" % CACHE)
    print()

    # ---- 2. 波形参数与干扰重叠 ----
    print("[2] 逐 chirp 波形参数与干扰的频带重叠")
    print("    计算干扰带内占比 / 时频重叠率 ...", flush=True)
    TF = tf_features(files, {f: ctab[int(c)] for f, c in zip(files, ci)})
    print("    完成。")
    print()
    print("  %-5s %7s %7s %6s %-6s %9s %10s %10s | %8s %8s %8s" %
          ("chirp", "f_lo", "f_hi", "BW", "env", "带内占比", "时频重叠", "干扰/信号",
           "Δ干净", "Δ干扰", "Δ最差"))
    print()

    if "pf" not in per_model or "nopulse" not in per_model:
        print("!! 缺少模型结果，退出"); return

    P, N = per_model["pf"], per_model["nopulse"]
    D = P - N
    # NaN 说明：部分样本的干扰帧全部落在零填充区，该样本的干扰帧指标不可算
    # （与训练脚本的 run_eval 同一处理）。带内平均一律用 nanmean。
    nn_p = int(np.isnan(P[:, 2]).sum())
    nn_n = int(np.isnan(N[:, 2]).sum())
    print("  干扰帧指标不可算的样本：pf %d 个 / nopulse %d 个（共 %d）"
          % (nn_p, nn_n, len(P)))
    print()

    # ---- 3. 逐 chirp 聚合 ----
    rows = []
    for c in sorted(set(ci.tolist())):
        m = ci == c
        b = ctab[c]
        rows.append(dict(
            chirp=c, bw=b["bw"], f_lo=b["f_lo"], f_hi=b["f_hi"], T=b["T"],
            env=b["env"], A=b["A"],
            inband=float(np.nanmean(TF[m, 0])), overlap=float(np.nanmean(TF[m, 1])),
            ratio=float(np.nanmean(TF[m, 2])), n=int(m.sum()),
            clean_pf=float(np.nanmean(P[m, 1])), clean_np=float(np.nanmean(N[m, 1])),
            imp_pf=float(np.nanmean(P[m, 2])), imp_np=float(np.nanmean(N[m, 2])),
            d_clean=float(np.nanmean(D[m, 1])), d_imp=float(np.nanmean(D[m, 2])),
            d_impmin=float(np.nanmean(D[m, 3])),
            seen=(c <= 14),
        ))
        r = rows[-1]
        print("  %-5d %7.1f %7.1f %6.1f %-6s %9.3f %10.3f %10.2f | %+7.3f %+7.3f %+7.3f   %s"
              % (r["chirp"], r["f_lo"], r["f_hi"], r["bw"], r["env"],
                 r["inband"], r["overlap"], r["ratio"],
                 r["d_clean"], r["d_imp"], r["d_impmin"],
                 "" if r["seen"] else "(未见过)"))

    with open(os.path.join(FIG, "chirp_band%s.csv" % A.cache_tag), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- 4. 相关分析 ----
    print()
    print("=" * 100)
    print("[2] 效应 vs 波形参数 / 干扰重叠   （效应 = pf − nopulse，dB）")
    print("=" * 100)
    for sub, lab in [(rows, "全部 20 个 chirp"),
                     ([r for r in rows if r["seen"]], "见过的 chirp 1-14"),
                     ([r for r in rows if not r["seen"]], "未见的 chirp 15-20")]:
        if len(sub) < 4:
            continue
        print("\n  %s（n=%d）" % (lab, len(sub)))
        print("    %-22s %10s %10s %10s %8s" %
              ("自变量", "vs 干净帧", "vs 干扰帧", "vs 干扰最差", "t(干扰帧)"))
        for key, nm in [("bw", "带宽 BW (Hz)"), ("f_lo", "下截止 f_lo"),
                        ("f_hi", "上截止 f_hi"), ("T", "时长 T (s)"),
                        ("A", "幅度 A"), ("inband", "干扰带内占比"),
                        ("overlap", "时频重叠率"), ("ratio", "带内干扰/信号比")]:
            x = [r[key] for r in sub]
            rc, _ = pearson(x, [r["d_clean"] for r in sub])
            ri, ti = pearson(x, [r["d_imp"] for r in sub])
            rm, _ = pearson(x, [r["d_impmin"] for r in sub])
            star = ""
            if abs(ti) > 3.0: star = " ***"
            elif abs(ti) > 2.1: star = " **"
            elif abs(ti) > 1.7: star = " *"
            print("    %-22s %+10.3f %+10.3f %+10.3f %8.2f%s"
                  % (nm, rc, ri, rm, ti, star))

    # 环境（包络）分组
    print("\n  按包络分组（干扰帧效应均值）")
    for e in ["rect", "hann", "gauss"]:
        v = [r["d_imp"] for r in rows if r["env"] == e]
        if v:
            print("    %-6s n=%2d   %+.3f dB" % (e, len(v), np.mean(v)))

    # ---- 5. 方差分解：波形效应 vs 干扰实现波动 ----
    print()
    print("=" * 100)
    print("[3] 方差分解：逐 chirp 差异是【波形效应】还是【干扰实现的波动】？")
    print("=" * 100)
    vardec = {}
    for col, nm in [(1, "干净帧"), (2, "干扰帧"), (3, "干扰最差")]:
        d = D[:, col]
        mk = np.isfinite(d)
        dd, cc = d[mk], ci[mk]
        ks = sorted(set(cc.tolist()))
        g = [dd[cc == k] for k in ks]
        n_tot, kk = len(dd), len(ks)
        ssb = sum(len(x) * (x.mean() - dd.mean()) ** 2 for x in g)
        ssw = sum(((x - x.mean()) ** 2).sum() for x in g)
        msb, msw = ssb / (kk - 1), ssw / (n_tot - kk)
        F = msb / msw if msw > 0 else float("nan")
        npc = n_tot / kk
        sd_real = float(np.sqrt(msw))                       # 逐样本（含实现波动）
        sd_wave = float(np.sqrt(max(0.0, (msb - msw) / npc)))  # 波形成分
        gm = np.array([x.mean() for x in g])
        print("  %s：" % nm)
        print("    逐样本标准差（实现波动为主）  %.3f dB" % sd_real)
        print("    逐 chirp 均值标准差（实测）   %.3f dB" % gm.std(ddof=1))
        print("    若纯属实现波动，期望          %.3f dB" % (sd_real / np.sqrt(npc)))
        print("    => 分解出的【波形】成分       %.3f dB" % sd_wave)
        print("    F(%d,%d) = %.2f   %s"
              % (kk - 1, n_tot - kk, F,
                 "波形效应显著（p<0.01）" if F > 2.0 else "与实现波动不可区分"))
        print("    波形成分占总方差            %.1f%%"
              % (100 * sd_wave ** 2 / (sd_wave ** 2 + sd_real ** 2)))
        print()
        vardec[nm] = dict(sd_real=sd_real, sd_wave=sd_wave, F=float(F),
                          n_chirp=kk, n_tot=n_tot)

    # 按信道 / SNR 分解（看变化的来源在哪一层）
    print("  分层均值（干扰帧效应，dB）")
    for key, lab in [(ca, "信道"), (sn, "SNR 档")]:
        vals = {}
        for v in sorted(set(key.tolist())):
            mk = (key == v) & np.isfinite(D[:, 2])
            vals[v] = float(np.nanmean(D[mk, 2]))
        arr = np.array(list(vals.values()))
        print("    %-8s 共 %d 组  范围 %+.3f ~ %+.3f  标准差 %.3f"
              % (lab, len(vals), arr.min(), arr.max(), arr.std(ddof=1)))
    print()

    # ---- 6. 出图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=150)
    for ax, key, xl in [(axs[0], "bw", "chirp 带宽 BW (Hz)"),
                        (axs[1], "overlap", "干扰与信号的时频重叠率")]:
        for seen, col, mk, lab in [(True, "#4c78a8", "o", "见过的 chirp 1-14"),
                                   (False, "#f58518", "^", "未见的 chirp 15-20")]:
            sub = [r for r in rows if r["seen"] == seen]
            ax.scatter([r[key] for r in sub], [r["d_imp"] for r in sub],
                       s=52, c=col, marker=mk, label=lab, zorder=3,
                       edgecolors="white", linewidths=0.6)
        r, t = pearson([r[key] for r in rows], [r["d_imp"] for r in rows])
        z = np.polyfit([r[key] for r in rows], [r["d_imp"] for r in rows], 1)
        xs = np.linspace(min(r[key] for r in rows), max(r[key] for r in rows), 20)
        ax.plot(xs, z[0] * xs + z[1], "--", color="#7f7f7f", lw=1.4,
                label="线性拟合 r=%+.3f" % r)
        ax.axhline(0, color="#333", lw=1)
        ax.set_xlabel(xl); ax.set_ylabel("干扰帧增益差：pf − nopulse (dB)")
        ax.set_title("r = %+.3f   (n=%d chirp)" % (r, len(rows)), fontsize=10)
        ax.grid(alpha=0.25, ls=":")
        ax.legend(fontsize=8.5)
    fig.suptitle("脉冲感知分支的收益 vs 波形带宽 / 干扰与信号的时频重叠率"
                 "（干扰实现已在不同波形间配对固定）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figO_chirp_band%s.png" % A.cache_tag))
    plt.close(fig)

    # ---- 6. 汇总 JSON ----
    with open(os.path.join(FIG, "chirp_band%s.json" % A.cache_tag), "w", encoding="utf-8") as f:
        json.dump({
            "scan": scan, "paired": bool(A.cache_tag),
            "n_samples": int(len(files)),
            "rows": rows,
            "corr_all": {k: pearson([r[k] for r in rows], [r["d_imp"] for r in rows])[0]
                         for k in ("bw", "f_lo", "f_hi", "T", "A", "inband",
                                   "overlap", "ratio")},
            "corr_clean": {k: pearson([r[k] for r in rows], [r["d_clean"] for r in rows])[0]
                           for k in ("bw", "inband", "overlap")},
            "vardec": vardec,
            "seen_mean": {k: float(np.mean([r[k] for r in rows if r["seen"]]))
                          for k in ("d_clean", "d_imp", "d_impmin")},
            "unseen_mean": {k: float(np.mean([r[k] for r in rows if not r["seen"]]))
                            for k in ("d_clean", "d_imp", "d_impmin")},
        }, f, ensure_ascii=False, indent=1)
    print("\n图 -> %s" % os.path.join(FIG, "figO_chirp_band%s.png" % A.cache_tag))
    print("表 -> %s" % os.path.join(FIG, "chirp_band%s.csv" % A.cache_tag))


def eval_dir_on_files(model, files, bs, workers):
    """在任意文件列表上评估（用于 --n 限制时）。"""
    model = model.to(V3.DEVICE).eval()
    ds = PAMDataset.__new__(PAMDataset)
    ds.files = list(files); ds.seg_len = SEG; ds.ds = DS
    loader = torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=False, num_workers=workers, drop_last=False)
    rows = []
    with torch.no_grad():
        for noisy, clean, mask in loader:
            n = noisy.unsqueeze(1).to(V3.DEVICE)
            c = clean.unsqueeze(1).to(V3.DEVICE)
            e = model(n)
            for k in range(n.shape[0]):
                st = PM._stats_1d(e[k, 0].cpu().numpy().astype(np.float64),
                                  c[k, 0].cpu().numpy().astype(np.float64),
                                  n[k, 0].cpu().numpy().astype(np.float64),
                                  PM.FRAME, mask[k].numpy())
                rows.append([st["frmgain_med"], st["frmgain_clean"],
                             st["frmgain_imp"], st["frmgain_imp_min"], st["frac_imp"]])
    return np.array(rows, dtype=np.float64)


if __name__ == "__main__":
    main()
