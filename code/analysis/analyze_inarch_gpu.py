# -*- coding: utf-8 -*-
"""base 条件 GPU 训练下的架构排序验证（域内）

回答：R5 > R4 > R3 > 纯时域 这个排序，在 GPU 同设备、同种子的条件下是否保持？
零样本矩阵里高斯背景上 R4 曾被 R3 反超过，所以必须验证。

用法：python analyze_inarch_gpu.py --split testA
"""
import os, sys, csv, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "sens_out")
CKPT = r"D:/project/output_pamnet"
COLS = ["snr_in", "snr_out", "sisdr_in", "sisdr_out", "fmsnr_med_in", "fmsnr_med_out",
        "frmgain_med", "frmgain_clean", "frmgain_imp", "frmgain_imp_min",
        "fmsnr_frac_bad_out", "frac_imp"]

GROUPS = [
    ("R5 时域残差+频域精修+p注入", "devchk_pf",   14858),
    ("R4 时域残差+频域精修(无脉冲)", "devchk_nopulse", 14367),
    ("R3 纯频域掩码",               "gpu_fo",      5910),
    ("纯时域 PAM-Net(无脉冲)",       "gpu_pn",      8457),
]


def read_summary(tag, split):
    p = os.path.join(CKPT, "summary_%s.csv" % tag)
    if not os.path.exists(p):
        return None
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        head = next(r)
        i0 = head.index(COLS[0])
        for row in r:
            if row[1] == split:
                return {c: float(row[i0 + j]) for j, c in enumerate(COLS)}
    return None


def seeds_of(prefix, split):
    out = []
    for suf, lbl in (("", "s1"), ("_s2", "s2"), ("_s3", "s3")):
        d = read_summary(prefix + suf, split)
        if d:
            out.append((lbl, d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="testA")
    A = ap.parse_args()
    SPLIT = A.split

    print("=" * 104)
    print("base 条件 GPU 训练下的架构排序（%s）" % SPLIT)
    print("=" * 104)
    print("（全部同设备；这与零样本 CPU 矩阵的差别见下表末行）")
    print()

    data = {}
    hdr = "%-30s %10s %12s %12s %12s %10s" % ("模型", "参数", "干净帧", "干扰帧", "干扰最差", "种子")
    print(hdr); print("-" * len(hdr))
    for name, tag, npar in GROUPS:
        ss = seeds_of(tag, SPLIT)
        if not ss:
            print("%-30s %10d %12s %12s %12s %10s"
                  % (name, npar, "—", "—", "—", "未完成"))
            continue
        cl = [x[1]["frmgain_clean"] for x in ss]
        im = [x[1]["frmgain_imp"] for x in ss]
        mn = [x[1]["frmgain_imp_min"] for x in ss]
        data[tag] = dict(name=name, npar=npar, n=len(ss), clean=cl, imp=im, mn=mn)
        print("%-30s %10d %12.3f %12.3f %12.3f %10d"
              % (name, npar, np.mean(cl), np.mean(im), np.mean(mn), len(ss)))
    print("-" * len(hdr))
    if len(data) >= 2:
        order = sorted(data.items(), key=lambda kv: -np.mean(kv[1]["imp"]))
        print("干扰帧排序：", " > ".join("%s(%.2f)" % (v["name"].split(" ")[0], np.mean(v["imp"]))
                                        for _, v in order))
    print()

    # ---- 排序是否与零样本一致 ----
    print("=" * 104)
    print("逐种子配对检验（跨种子，df=n-1）")
    print("=" * 104)
    tests = [("R5 vs R4（脉冲分支效应）", "devchk_pf", "devchk_nopulse"),
             ("R5 vs R3（架构优势）", "devchk_pf", "gpu_fo"),
             ("R4 vs R3", "devchk_nopulse", "gpu_fo"),
             ("R4 vs 纯时域（频域精修贡献）", "devchk_nopulse", "gpu_pn")]
    res = {}
    for lbl, ta, tb in tests:
        if ta not in data or tb not in data:
            continue
        n = min(data[ta]["n"], data[tb]["n"])
        a = np.array(data[ta]["imp"][:n])
        b = np.array(data[tb]["imp"][:n])
        d = a - b
        if n >= 2 and d.std(ddof=1) > 0:
            from scipy import stats
            t, p = stats.ttest_rel(a, b)
            st = dict(n=n, mean=float(d.mean()), sd=float(d.std(ddof=1)),
                      t=float(t), p=float(p))
        else:
            st = dict(n=n, mean=float(d.mean()) if n else float("nan"), sd=0.0,
                      t=float("nan"), p=float("nan"))
        res[lbl] = st
        print("  %-28s %+8.3f ± %.3f dB  (n=%d, p=%.4f) %s"
              % (lbl, st["mean"], st["sd"], st["n"], st["p"],
                 "✓ 前者更高" if st["mean"] > 0 else "✗ 后者更高"))

    print()
    print("=" * 104)
    print("与零样本 CPU 排序的一致性")
    print("=" * 104)
    S = json.load(open(os.path.join(OUTD, "sens_summary_%s.json" % SPLIT),
                       encoding="utf-8"))
    t = S["table"]["base"]
    pairs = [("R5 vs R4", "R5_freqrefine_pulse", "R4_freqrefine_nopulse"),
             ("R5 vs R3", "R5_freqrefine_pulse", "R3_freqonly"),
             ("R4 vs R3", "R4_freqrefine_nopulse", "R3_freqonly"),
             ("R4 vs 纯时域", "R4_freqrefine_nopulse", "P_time_pulse")]
    KEYMAP = {"R5 vs R4": "R5 vs R4（脉冲分支效应）",
             "R5 vs R3": "R5 vs R3（架构优势）",
             "R4 vs R3": "R4 vs R3",
             "R4 vs 纯时域": "R4 vs 纯时域（频域精修贡献）"}
    for lbl, ka, kb in pairs:
        if ka in t and kb in t:
            z = t[ka]["imp"] - t[kb]["imp"]
            g = res.get(KEYMAP.get(lbl, lbl))
            gm = g["mean"] if g else float("nan")
            agree = "—"
            if g and np.isfinite(gm):
                agree = "方向一致" if gm * z > 0 else "**方向相反**"
            print("  %-14s 零样本 %+7.3f dB   GPU域内 %+8.3f dB   %s"
                  % (lbl, z, gm, agree))

    os.makedirs(OUTD, exist_ok=True)
    with open(os.path.join(OUTD, "inarch_gpu_%s.json" % SPLIT), "w",
              encoding="utf-8") as f:
        json.dump(dict(split=SPLIT, groups={k: {kk: vv for kk, vv in v.items()
                                               if kk != "name"} | {"name": v["name"]}
                                           for k, v in data.items()}, tests=res),
                      f, ensure_ascii=False, indent=1)
    print("\n->", os.path.join(OUTD, "inarch_gpu_%s.json" % SPLIT))


if __name__ == "__main__":
    main()
