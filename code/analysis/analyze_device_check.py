# -*- coding: utf-8 -*-
"""
设备效应对「脉冲分支效应 Δ」的影响
================================================================================
在 base 条件上，同一配置、同一种子，CPU 与 GPU 各训练一次，比较
    Δ = R5(干扰帧) − R4(干扰帧)
的设备差异。

CPU 侧：summary_pamnet_freq_{pf,nopulse}_gn[_sN].csv（报告正文用的那批）
GPU 侧：summary_devchk_{pf,nopulse}[_sN].csv（run_device_check.sh 产出）

若 |Δ_gpu − Δ_cpu| 与「域内 Δ=0.32 vs base Δ=0.79」的差距同量级，
则域内与 base 的对比无法区分"条件效应"与"设备效应"。

用法：python analyze_device_check.py
"""
import os, csv, json
import numpy as np

WORK = r"E:/workbuddy/store/2026-10-12-22-50-08"
OUTD = os.path.join(WORK, "sens_out")
CK = r"E:/project/data/output_pamnet"
COLS = ["snr_in", "snr_out", "sisdr_in", "sisdr_out", "fmsnr_med_in", "fmsnr_med_out",
        "frmgain_med", "frmgain_clean", "frmgain_imp", "frmgain_imp_min",
        "fmsnr_frac_bad_out", "frac_imp"]


def read(tag, split="testA"):
    p = os.path.join(CK, "summary_%s.csv" % tag)
    if not os.path.exists(p):
        return None
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        h = next(r)
        i0 = h.index(COLS[0])
        for row in r:
            if row[1] == split:
                return {c: float(row[i0 + j]) for j, c in enumerate(COLS)}
    return None


def main():
    print("=" * 86)
    print("设备效应对 Δ = R5 − R4 的影响（base 条件，testA）")
    print("=" * 86)
    print("%-6s %14s %14s %14s | %14s %14s %14s | %12s"
          % ("seed", "CPU R4", "CPU R5", "CPU Δ", "GPU R4", "GPU R5", "GPU Δ",
             "设备差(Δ)"))
    print("-" * 86)
    ds = []
    for suf, nm in (("", "s1"), ("_s2", "s2"), ("_s3", "s3"), ("_s4", "s4"),
                    ("_s5", "s5")):
        c4 = read("pamnet_freq_nopulse_gn" + suf)
        c5 = read("pamnet_freq_pf_gn" + suf)
        g4 = read("devchk_nopulse" + suf)
        g5 = read("devchk_pf" + suf)
        if c4 and c5:
            d_cpu = c5["frmgain_imp"] - c4["frmgain_imp"]
        else:
            d_cpu = None
        if g4 and g5:
            d_gpu = g5["frmgain_imp"] - g4["frmgain_imp"]
        else:
            d_gpu = None
        if d_cpu is None and d_gpu is None:
            continue
        diff = (d_gpu - d_cpu) if (d_cpu is not None and d_gpu is not None) else None
        if diff is not None:
            ds.append(diff)
        print("%-6s %14s %14s %14s | %14s %14s %14s | %12s"
              % (nm,
                 "%.3f" % c4["frmgain_imp"] if c4 else "—",
                 "%.3f" % c5["frmgain_imp"] if c5 else "—",
                 "%+.3f" % d_cpu if d_cpu is not None else "—",
                 "%.3f" % g4["frmgain_imp"] if g4 else "—",
                 "%.3f" % g5["frmgain_imp"] if g5 else "—",
                 "%+.3f" % d_gpu if d_gpu is not None else "—",
                 "%+.3f" % diff if diff is not None else "—"))
    print("-" * 86)

    out = {}
    if ds:
        m, s = float(np.mean(ds)), float(np.std(ds, ddof=1)) if len(ds) > 1 else 0.0
        out["device_diff_on_delta"] = dict(mean=m, sd=s, n=len(ds), per_seed=ds)
        print("设备效应对 Δ：%+.3f ± %.3f dB（n=%d 个配对种子）" % (m, s, len(ds)))
        print()
        # 与域内对比
        p = os.path.join(OUTD, "indomain_r10_testA.json")
        if os.path.exists(p):
            ind = json.load(open(p, encoding="utf-8"))
            v = ind["indomain"]
            d_in = (np.mean([x["frmgain_imp"] for x in v["R5_in"]])
                    - np.mean([x["frmgain_imp"] for x in v["R4_in"]]))
            S = json.load(open(os.path.join(OUTD, "sens_summary_testA.json"),
                               encoding="utf-8"))
            d_base = S["pulse_eff"]["base"]["imp"]["mean"]
            print("=" * 86)
            print("关键判断")
            print("=" * 86)
            print("  base 的 Δ（CPU，5 种子）        : %+.3f dB" % d_base)
            print("  域内 rate=10 的 Δ（GPU，3 种子）: %+.3f dB" % d_in)
            print("  两者之差                        : %+.3f dB" % (d_in - d_base))
            print("  设备效应对 Δ 的量级             : %+.3f dB" % m)
            print()
            if abs(m) >= abs(d_in - d_base) * 0.6:
                print("  → **设备效应与条件差异同量级，无法区分。**")
                print("     要干净地比较 base 与 rate=10 的分支收益，必须在同一设备上重跑 base。")
            else:
                print("  → 设备效应远小于条件差异，可以认为「rate=10 下分支收益确实更小」。")
                print("     校正设备效应后，rate=10 的 Δ ≈ %+.3f dB（仍显著低于 base 的 %+.3f）。"
                      % (d_in - m, d_base))
            out["d_base_cpu"] = d_base
            out["d_indomain_gpu"] = d_in
            out["gap"] = d_in - d_base
            out["verdict"] = ("device_confounded" if abs(m) >= abs(d_in - d_base) * 0.6
                              else "condition_effect")

    with open(os.path.join(OUTD, "device_check_testA.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print()
    print("-> sens_out/device_check_testA.json")


if __name__ == "__main__":
    main()
