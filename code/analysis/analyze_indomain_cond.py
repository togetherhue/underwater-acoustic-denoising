# -*- coding: utf-8 -*-
"""通用域内重训分析：任意条件（rate / α）的「零样本 vs 域内」三方对照

用法：
  python analyze_indomain_cond.py --cond a15_r25 --tag sens_r25 --label "rate=2.5/s"
  python analyze_indomain_cond.py --cond a20_r5  --tag sens_a20 --label "α=2.0"

输出：零样本 Δ / 域内 Δ / 设备校正后 Δ / 相对 base，以及显著性。
设备校正量沿用 8.6 的实测值（base 条件 CPU vs GPU 重训，对 Δ 的影响）。
"""
import os, csv, json, glob, argparse
import numpy as np

WORK = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(WORK, "sens_out")
CKPT = r"E:/project/data/output_pamnet"
COLS = ["snr_in", "snr_out", "sisdr_in", "sisdr_out",
        "fmsnr_med_in", "fmsnr_med_out", "frmgain_med", "frmgain_clean",
        "frmgain_imp", "frmgain_imp_min", "fmsnr_frac_bad_out", "frac_imp"]


def read_summary(tag, split):
    p = os.path.join(CKPT, "summary_%s.csv" % tag)
    if not os.path.exists(p):
        return None
    with open(p, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        i0 = hdr.index(COLS[0])
        for row in rd:
            if row[1] == split:
                return {c: float(row[i0 + j]) for j, c in enumerate(COLS)}
    return None


def seeds(tagpre, split):
    """返回 [(label, R4_dict, R5_dict)]，只取两者都存在的种子"""
    out = []
    # 支持到 5 个种子：缺的后缀会被 if a and b 跳过，对只有 3 种子的条件安全
    for suf in ("", "_s2", "_s3", "_s4", "_s5"):
        a = read_summary("%s_nopulse%s" % (tagpre, suf), split)
        b = read_summary("%s_pf%s" % (tagpre, suf), split)
        if a and b:
            out.append(("s" + (suf[2:] if suf else "1"), a, b))
    return out


def t1(d):
    """单样本 t 检验：Δ 是否显著异于 0"""
    d = np.asarray([x for x in d if np.isfinite(x)], float)
    if len(d) < 2:
        return float("nan"), float("nan")
    from scipy import stats
    t, p = stats.ttest_1samp(d, 0.0)
    return float(t), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, help="数据集条件名，如 a15_r25")
    ap.add_argument("--tag", required=True, help="tag 前缀，如 sens_r25")
    ap.add_argument("--label", default="", help="人类可读标签")
    ap.add_argument("--split", default="testA")
    ap.add_argument("--dev", type=float, default=-0.091,
                    help="设备校正量（域内 GPU − base CPU），默认 8.6 实测值")
    A = ap.parse_args()
    lbl = A.label or A.cond

    # ---- 零样本侧 ----
    # 注意：必须用 pulse_eff（= R5−R4 的脉冲分支效应），
    # 不能用 cond_delta —— 后者是"同一模型换条件后的退化量"，语义完全不同
    # （a15_r10 的 cond_delta 是 −0.455，那是 R4 在 rate=10 上比 base 差多少，
    #  不是分支效应；用它当 Δ 会得到符号相反的结论）。
    S = json.load(open(os.path.join(OUTD, "sens_summary_%s.json" % A.split),
                       encoding="utf-8"))
    tb = S["table"]["base"]
    base_d = tb["R5_freqrefine_pulse"]["imp"] - tb["R4_freqrefine_nopulse"]["imp"]
    pe = S.get("pulse_eff", {}).get(A.cond)
    z_d = pe["imp"]["mean"] if pe else float("nan")
    z_p = pe["imp"].get("p") if pe else float("nan")
    z4 = S["table"].get(A.cond, {}).get("R4_freqrefine_nopulse", {}).get("imp", float("nan"))
    z5 = S["table"].get(A.cond, {}).get("R5_freqrefine_pulse", {}).get("imp", float("nan"))

    ss = seeds(A.tag, A.split)

    print("=" * 92)
    print("域内重训：%s（%s）  split=%s" % (lbl, A.cond, A.split))
    print("=" * 92)
    print("设备校正量（域内 GPU − base CPU，对 Δ）= %+.3f dB" % A.dev)
    print()
    print("%-26s %12s %12s %12s %14s %8s" %
          ("配置", "R4 干扰帧", "R5 干扰帧", "Δ", "Δ(校正后)", "种子"))
    print("-" * 92)
    print("%-26s %12.3f %12.3f %+12.3f %14s %8d" %
          ("base α=1.5（CPU 存档）", tb["R4_freqrefine_nopulse"]["imp"],
           tb["R5_freqrefine_pulse"]["imp"], base_d, "—（基准）",
           tb["R4_freqrefine_nopulse"]["n_seed"]))
    print("%-26s %12.3f %12.3f %+12.3f %14s %8d" %
          (lbl + " 零样本", z4, z5, z_d, "—", 5))
    if not ss:
        print("%-26s %12s %12s %12s %14s %8s" % (lbl + " 域内", "—", "—", "—", "—", 0))
        print("\n（域内重训尚未完成）")
        return

    r4 = [x[1]["frmgain_imp"] for x in ss]
    r5 = [x[2]["frmgain_imp"] for x in ss]
    d = [b - a for a, b in zip(r4, r5)]
    m, sd = float(np.mean(d)), float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    t, p = t1(d)
    print("%-26s %12.3f %12.3f %+12.3f %+14.3f %8d" %
          (lbl + " 域内", np.mean(r4), np.mean(r5), m, m - A.dev, len(ss)))
    print("-" * 92)
    print("  逐种子 Δ: %s" % " / ".join("%+.3f" % x for x in d))
    print("  逐种子 R4: %s（sd=%.3f）" % (" / ".join("%.3f" % x for x in r4),
                                     np.std(r4, ddof=1) if len(r4) > 1 else 0.0))
    print("  逐种子 R5: %s（sd=%.3f）" % (" / ".join("%.3f" % x for x in r5),
                                     np.std(r5, ddof=1) if len(r5) > 1 else 0.0))
    # 符号一致性 + 稳健离散度：区分"不显著"与"符号冲突"两种完全不同的情况
    n_pos = int(sum(1 for x in d if x > 0))
    n_neg = len(d) - n_pos
    from scipy.stats import median_abs_deviation
    med = float(np.median(d))
    mad = float(median_abs_deviation(d, scale="normal"))
    print("  符号一致性: %d 正 / %d 负（同号率 %.0f%%）；中位 %+.3f，MAD %.3f"
          % (n_pos, n_neg, 100.0 * max(n_pos, n_neg) / len(d), med, mad))
    if n_pos and n_neg:
        print("  **注意**：Δ 在种子间符号不一致（%+.3f vs %+.3f）。" % (min(d), max(d)))
        print("            R4 跨种子极差 %.3f dB，R5 为 %.3f dB。"
              % (max(r4) - min(r4), max(r5) - min(r5)))
        # 5 个中 4 个同号 ⇒ 不该再叫"不可判"，而是"方向一致但统计不显著"
        if max(n_pos, n_neg) / len(d) >= 0.8 and len(d) >= 5:
            print("            但 %d/%d 个种子同号且中位 %+.3f、MAD 仅 %.3f ⇒"
                  % (max(n_pos, n_neg), len(d), med, mad))
            print("            应读作「方向一致为正、统计上仍不能排除 0」，"
                  "而非「完全不可判」。反向的那个种子需单独说明，不能丢弃。")
        else:
            print("            其 Δ 不能用来判断分支价值（先看 R4 的种子波动，再看 Δ）。")
    print("  跨种子: sd=%.3f, t=%+.2f, p=%.4f（%s）" %
          (sd, t, p, "显著" if p < 0.05 else "不显著"))
    if len(d) > 1:
        loo = [float(np.delete(np.array(d), i).mean()) for i in range(len(d))]
        print("  留一法: %s（范围 [%.3f, %.3f]）" % (
            " / ".join("%+.3f" % x for x in loo), min(loo), max(loo)))

    # ---- 判定 ----
    print()
    print("=" * 92)
    print("判定：%s 的收益差异是「分布偏移」还是「条件本身」？" % lbl)
    print("=" * 92)
    print("  base(α=1.5, 5/s) %+.3f  →  零样本 %+.3f  →  域内 %+.3f（校正 %+.3f）"
          % (base_d, z_d, m, m - A.dev))
    gz, gi = z_d - base_d, (m - A.dev) - base_d
    print("  零样本相对 base: %+.3f dB；域内校正后相对 base: %+.3f dB" % (gz, gi))
    if abs(gi) < abs(gz) * 0.5:
        print("  -> **主要是分布偏移**：域内重训后收益基本回到 base 水平。")
    elif abs(gi) > abs(gz) * 0.8:
        print("  -> **主要是条件本身**：域内重训后仍明显偏离 base。")
    else:
        print("  -> **两者兼有**：分布偏移解释约 %.0f%%，条件本身解释约 %.0f%%。"
              % (100 * (1 - abs(gi) / abs(gz)) if abs(gz) > 1e-9 else 0,
                 100 * abs(gi) / abs(gz) if abs(gz) > 1e-9 else 0))
    if p > 0.05:
        print("  **注意**：域内 Δ 跨种子检验不显著（p=%.3f），残余量不能排除为 0。" % p)

    out = dict(cond=A.cond, split=A.split, base_d=base_d, zero_d=z_d,
               indomain_d=m, indomain_d_corrected=m - A.dev, sd=sd, p=p,
               n=len(ss), per_seed=[float(x) for x in d],
               r4=float(np.mean(r4)), r5=float(np.mean(r5)),
               # 逐种子的 R4/R5 绝对值与种子标签：用于判断"该条件本身训练是否稳定"
               # （若 R4 在种子间差 1 dB 以上，说明域内 Δ 的波动主要来自训练不稳定，
               #   而不是分支效应 —— a20 端点就出现过符号相反的情况）
               seed_labels=[x[0] for x in ss],
               per_seed_r4=[float(x) for x in r4],
               per_seed_r5=[float(x) for x in r5],
               sd_r4=float(np.std(r4, ddof=1)) if len(r4) > 1 else 0.0,
               sd_r5=float(np.std(r5, ddof=1)) if len(r5) > 1 else 0.0)
    op = os.path.join(OUTD, "indomain_%s_%s.json" % (A.tag.replace("sens_", ""), A.split))
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n->", op)


if __name__ == "__main__":
    main()
