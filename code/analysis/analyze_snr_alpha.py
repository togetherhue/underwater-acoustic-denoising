# -*- coding: utf-8 -*-
"""A：α=1.2 的"收益效率反向上升"是真效应还是种子噪声？

不引入新指标，只换统计口径：
1. 四个 α 条件在同一批 SNR 点（−10/−5/0/+5/+10）上都有结果；
2. 趋势不看"均值连成的线"，而看**每个种子自己的斜率**（逐样本配对，同噪声实现）；
   5 个种子给出 5 个斜率 → 配对 t 检验（df=4）+ 同号计数；
3. 端点配对检验：Δ(+10) − Δ(−10)；
4. 拆分分母：绝对抑制量 gain_imp 随 t 的变动，检查 α=1.2 的 −10 点是否"分母被抬高"。
"""
import json, os
import numpy as np
from scipy.stats import spearmanr, ttest_1samp, ttest_rel

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "sens_out")
SUF = os.environ.get("SNR_SPLIT", "")

CONDS = ["a12_r5", "base", "a18_r5", "a20_r5"]
LBL = {"a12_r5": "α=1.2", "base": "α=1.5", "a18_r5": "α=1.8", "a20_r5": "α=2.0"}
COMMON = [-10, -5, 0, 5, 10]
BASE_GRID = [-15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15]


def tk(t):
    return ("%+d" % int(round(t))) if abs(t - round(t)) < 1e-9 else ("%+.2f" % t)


def load_cells():
    cells = {}
    for f in os.listdir(OUTD):
        if not f.startswith("snrcell" + SUF) or not f.endswith(".json"):
            continue
        d = json.load(open(os.path.join(OUTD, f), encoding="utf-8"))
        cells["%s|%s|%s|%d" % (d["cond"], tk(d["target_bg_snr"]), d["model"], d["seed"])] = d
    return cells


def per_seed(C, cond, ts):
    """返回 [n_pt, n_seed] 的 Δ 矩阵与 gain_imp(R4) 矩阵；缺任一模型则跳过该点。"""
    seeds = sorted({int(k.split("|")[3]) for k in C
                    if k.startswith(cond + "|") and "|R4|" in k})
    ts_ok, D, G = [], [], []
    for t in ts:
        d, g = [], []
        for s in seeds:
            a = C.get("%s|%s|R4|%d" % (cond, tk(t), s))
            b = C.get("%s|%s|R5|%d" % (cond, tk(t), s))
            if a is None or b is None:
                continue
            d.append(b["gain_imp"] - a["gain_imp"])
            g.append(a["gain_imp"])
        if len(d) == len(seeds):
            ts_ok.append(t); D.append(d); G.append(g)
    return np.array(ts_ok), np.array(D), np.array(G)


def trend(ts, M):
    """逐种子斜率 + 逐种子 Spearman。M: [n_pt, n_seed]"""
    sl = np.array([np.polyfit(ts, M[:, i], 1)[0] for i in range(M.shape[1])])
    rho = np.array([spearmanr(ts, M[:, i])[0] for i in range(M.shape[1])])
    rho = np.where(np.isfinite(rho), rho, 0.0)
    return sl, rho


def main():
    C = load_cells()
    print("=" * 80)
    print("A. Δ 随 SNR 的趋势：逐种子检验%s" % ("（testC）" if SUF else "（testA）"))
    print("=" * 80)
    out = {}
    for cond in CONDS:
        ts, D, G = per_seed(C, cond, BASE_GRID if cond == "base" else COMMON)
        if len(ts) < 3:
            print("%-7s 数据不足（%d 点）" % (LBL[cond], len(ts)))
            continue
        sl, rho = trend(ts, D)
        n_seed = D.shape[1]
        n_neg = int((sl < 0).sum())
        tt = ttest_1samp(sl, 0.0)
        ep = ttest_rel(D[-1], D[0])
        dmean = D.mean(1)
        e = dmean / G.mean(1)
        out[cond] = dict(pts=ts.tolist(), d_mean=dmean.tolist(),
                         d_sd=D.std(1, ddof=1).tolist(), g4=G.mean(1).tolist(),
                         e=e.tolist(), slopes=sl.tolist(), rhos=rho.tolist(),
                         n_neg=n_neg, n_seed=n_seed,
                         slope_mean=float(sl.mean()), slope_p=float(tt.pvalue),
                         ep_diff=float((D[-1] - D[0]).mean()), ep_p=float(ep.pvalue),
                         per_seed_ep=(D[-1] - D[0]).tolist())
        print("\n%-7s（%d 点，%d 种子）" % (LBL[cond], len(ts), n_seed))
        print("   Δ(t)      : %s" % "  ".join("%+0.3f" % x for x in dmean))
        print("   种子 sd   : %s" % "  ".join("%0.3f" % x for x in D.std(1, ddof=1)))
        print("   逐种子斜率: %s  平均 %+.4f dB/dB  t=%.2f p=%.3f  下降 %d/%d" %
              ("  ".join("%+.4f" % x for x in sl), sl.mean(), tt.statistic, tt.pvalue,
               n_neg, n_seed))
        print("   逐种子 ρ  : %s" % "  ".join("%+.2f" % x for x in rho))
        print("   端点配对  : %s  平均 %+.3f  p=%.3f" %
              ("  ".join("%+.2f" % x for x in (D[-1] - D[0])),
               (D[-1] - D[0]).mean(), ep.pvalue))
        print("   e         : %s" % "  ".join("%.4f" % x for x in e))
        print("   gain_imp  : %s  （变动 %+.2f dB）" %
              ("  ".join("%.2f" % x for x in G.mean(1)), G.mean(1)[-1] - G.mean(1)[0]))

    print("\n" + "=" * 80)
    print("判决规则：下降需 p<0.05 且 5/5 种子同向；否则记为「方向一致但统计不显著」")
    for c, v in out.items():
        if v["n_neg"] == v["n_seed"] and v["slope_p"] < 0.05:
            s = "显著下降"
        elif v["n_neg"] == v["n_seed"]:
            s = "方向一致但不显著"
        elif v["n_neg"] <= 1:
            s = "未下降（趋势方向相反）"
        else:
            s = "符号混杂，不可判"
        print("  %-7s 下降 %d/%d  p=%.3f  ⇒ %s" % (LBL[c], v["n_neg"], v["n_seed"],
                                                 v["slope_p"], s))
    p = os.path.join(OUTD, "snr_alpha_trend%s.json" % SUF)
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n->", p)


if __name__ == "__main__":
    main()
