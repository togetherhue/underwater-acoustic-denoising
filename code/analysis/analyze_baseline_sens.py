# -*- coding: utf-8 -*-
"""
经典基线 vs 神经模型：跨条件对比分析
================================================================================
回答：报告头条结论「PAM-Net 远超全部经典方法」是否跨条件成立？

对比对象：
  经典最好 = max(理想带通, 谱减法, 维纳, 中值)   —— 注意理想带通是"作弊"上界
            （已知信号频带），所以同时给出"最好的盲方法"作为更严格的参照
  神经模型 = R5（推荐，14,858 参数）

用法：python analyze_baseline_sens.py [--split testA]
"""
import os, sys, json, argparse
import numpy as np

WORK = r"E:/workbuddy/store/2026-10-12-22-50-08"
OUTD = os.path.join(WORK, "sens_out")

LBL = {"base": "α=1.5 / rate=5（base）", "a12_r5": "α=1.2 / rate=5",
       "a18_r5": "α=1.8 / rate=5", "a20_r5": "α=2.0 / rate=5",
       "a15_r25": "α=1.5 / rate=2.5", "a15_r10": "α=1.5 / rate=10",
       "gauss": "独立高斯 / rate=5", "m_a12": "α=1.2 已匹配功率", "m_a18": "α=1.8 已匹配功率"}
ORDER = ["base", "a12_r5", "a18_r5", "a20_r5", "a15_r25", "a15_r10", "gauss",
         "m_a12", "m_a18"]
# 理想带通是"已知频带"的作弊上界，不算盲方法
BLIND = ["谱减法（盲）", "维纳滤波（盲）", "中值滤波 k=5"]


def load(cond, cfg, split):
    p = os.path.join(OUTD, "sens_%s__%s.npz" % (cond, cfg))
    if os.path.exists(p):
        z = np.load(p)
        return z[split] if split in z else None
    arrs, si = [], 0
    while True:
        q = os.path.join(OUTD, "sens_%s__%s__s%d.npz" % (cond, cfg, si))
        if not os.path.exists(q):
            break
        z = np.load(q)
        if split not in z:
            break
        arrs.append(z[split]); si += 1
    return np.stack(arrs, 0) if arrs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="testA")
    A = ap.parse_args()
    sp = A.split

    BL = json.load(open(os.path.join(OUTD, "baseline_sens_%s.json" % sp), encoding="utf-8"))
    conds = [c for c in ORDER if c in BL]

    print("=" * 118)
    print("经典基线 vs 神经模型（%s，干扰帧逐帧降噪增益 dB）" % sp)
    print("=" * 118)
    hdr = ("%-24s %10s %10s %10s %10s | %10s %10s %10s"
           % ("条件", "理想带通*", "谱减(盲)", "维纳(盲)", "中值(盲)", "最好盲方法", "R5 神经", "R5−最好盲"))
    print(hdr); print("-" * len(hdr))

    rows = []
    for c in conds:
        b = BL[c]
        bp = b["理想带通 798-1500Hz"]["frmgain_imp"]
        blind_vals = {k: b[k]["frmgain_imp"] for k in BLIND}
        best_blind_name = max(blind_vals, key=blind_vals.get)
        best_blind = blind_vals[best_blind_name]
        R5 = load(c, "R5_freqrefine_pulse", sp)
        if R5 is None:
            continue
        r5 = float(np.nanmean(R5[:, :, 8]))
        r5_cln = float(np.nanmean(R5[:, :, 7]))
        gap = r5 - best_blind
        rows.append(dict(cond=c, bp=bp, best_blind=best_blind,
                         best_blind_name=best_blind_name, r5=r5, r5_cln=r5_cln, gap=gap,
                         **{k: blind_vals[k] for k in BLIND}))
        print("%-24s %10.2f %10.2f %10.2f %10.2f | %10.2f %10.2f %10.2f"
              % (LBL[c], bp, blind_vals["谱减法（盲）"], blind_vals["维纳滤波（盲）"],
                 blind_vals["中值滤波 k=5"], best_blind, r5, gap))
    print("-" * len(hdr))
    print("* 理想带通是【已知信号频带】的作弊上界，不是可用的盲方法。")
    print("  「最好盲方法」= 谱减/维纳/中值 三者中的最大值。")

    # ---------- 干净帧 ----------
    print()
    print("=" * 118)
    print("干净帧（无干扰）逐帧降噪增益 dB —— 检验经典方法在低难度下的表现")
    print("=" * 118)
    hdr2 = ("%-24s %10s %10s %10s %10s | %10s %10s"
            % ("条件", "理想带通*", "谱减(盲)", "维纳(盲)", "中值(盲)", "R5 神经", "R5−带通"))
    print(hdr2); print("-" * len(hdr2))
    for c in conds:
        b = BL[c]
        R5 = load(c, "R5_freqrefine_pulse", sp)
        if R5 is None:
            continue
        r5 = float(np.nanmean(R5[:, :, 7]))
        bp = b["理想带通 798-1500Hz"]["frmgain_clean"]
        print("%-24s %10.2f %10.2f %10.2f %10.2f | %10.2f %10.2f"
              % (LBL[c], bp, b["谱减法（盲）"]["frmgain_clean"],
                 b["维纳滤波（盲）"]["frmgain_clean"], b["中值滤波 k=5"]["frmgain_clean"],
                 r5, r5 - bp))
    print("-" * len(hdr2))

    # ---------- 结论 ----------
    print()
    print("=" * 118)
    print("结论")
    print("=" * 118)
    if rows:
        gaps = [r["gap"] for r in rows]
        print("R5 相对【最好的盲方法】的优势（干扰帧）：")
        print("  范围 %+.2f ~ %+.2f dB，中位 %+.2f dB"
              % (min(gaps), max(gaps), float(np.median(gaps))))
        n_pos = sum(1 for g in gaps if g > 0)
        print("  在 %d/%d 个条件上为正" % (n_pos, len(gaps)))
        # 哪个条件优势最小
        w = min(rows, key=lambda r: r["gap"])
        print("  优势最小的条件：%s（%+.2f dB，该处最好盲方法是 %s %+.2f）"
              % (LBL[w["cond"]], w["gap"], w["best_blind_name"], w["best_blind"]))
        # 与理想带通（作弊上界）比
        gbp = [r["r5"] - r["bp"] for r in rows]
        print()
        print("R5 相对【理想带通（作弊上界）】：")
        print("  范围 %+.2f ~ %+.2f dB，中位 %+.2f dB"
              % (min(gbp), max(gbp), float(np.median(gbp))))
        n_bp = sum(1 for g in gbp if g > 0)
        print("  在 %d/%d 个条件上仍高于带通上界" % (n_bp, len(gbp)))
        if n_bp < len(gbp):
            bad = [(LBL[r["cond"]], r["r5"] - r["bp"]) for r in rows if r["r5"] - r["bp"] <= 0]
            for nm, g in bad:
                print("    %s：R5 低于带通 %+.2f dB" % (nm, g))

    with open(os.path.join(OUTD, "baseline_vs_neural_%s.json" % sp), "w",
              encoding="utf-8") as f:
        json.dump(dict(split=sp, rows=rows), f, ensure_ascii=False, indent=1)
    print("\n-> sens_out/baseline_vs_neural_%s.json" % sp)


if __name__ == "__main__":
    main()
