# -*- coding: utf-8 -*-
"""B：testA vs testC —— 8.19 的 SNR 轴结论是否跨测试集成立

输出：
1. 两个划分的主曲线（11 点）逐点对照：Δ 差、同号、相关；
2. 每个划分的峰值/最小/降幅/同号计数；
3. 跨 α 的收益效率 e 在两个划分上是否同序（α=1.2 < 1.5 < 1.8 < 2.0）；
4. 一致性闸门：符号一致率、逐点差的绝对值中位数。
"""
import json, os
import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "sens_out")
GRID = [-15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15]
CONDS = ["a12_r5", "base", "a18_r5", "a20_r5"]
LBL = {"a12_r5": "α=1.2", "base": "α=1.5", "a18_r5": "α=1.8", "a20_r5": "α=2.0"}


def tk(t):
    t = float(t)
    return ("%+d" % int(round(t))) if abs(t - round(t)) < 1e-9 else ("%+.2f" % t)


def load(suf):
    p = os.path.join(OUTD, "snr_sweep%s.json" % suf)
    return json.load(open(p, encoding="utf-8"))


def curve(S, cond="base"):
    D = S["delta"]
    ts, dm, ds, g4, np_ = [], [], [], [], []
    for t in GRID:
        k = "%s|%s" % (cond, tk(t))
        if k not in D:
            continue
        v = D[k]
        ts.append(t); dm.append(v["d_mean"]); ds.append(v["d_sd"]); np_.append(v["n_pos"])
        g4.append(S["cells"]["%s|%s|R4" % (cond, tk(t))]["mean"]["gain_imp"])
    return map(np.array, (ts, dm, ds, g4, np_))


def eff(S, cond, pts=(-10, 5, 10)):
    D, C = S["delta"], S["cells"]
    out = []
    for t in pts:
        k = "%s|%s" % (cond, tk(t))
        if k in D:
            out.append(D[k]["d_mean"] / C["%s|%s|R4" % (cond, tk(t))]["mean"]["gain_imp"])
    return np.array(out)


def main():
    A, Cc = load(""), load("_testC")
    print("=" * 82)
    print("B. testA vs testC：SNR 轴主结论的跨集复现")
    print("=" * 82)

    ta, da, sa, ga, na = curve(A)
    tc, dc, sc, gc, nc = curve(Cc)
    n = min(len(ta), len(tc))
    ta, da, tc, dc = ta[:n], da[:n], tc[:n], dc[:n]

    print("\n[1] 主曲线（base，11 点）")
    print("  %6s | %8s %8s %6s | %8s %8s %6s" %
          ("SNR", "testA Δ", "sd", "同号", "testC Δ", "sd", "同号"))
    for i, t in enumerate(ta):
        print("  %+6.0f | %+8.3f %8.3f %4d/5 | %+8.3f %8.3f %4d/5" %
              (t, da[i], sa[i], na[i], dc[i], sc[i], nc[i]))

    r = pearsonr(da, dc)
    rho = spearmanr(da, dc)[0]
    dif = dc - da
    print("\n  逐点差（testC − testA）：均值 %+.3f  中位 %+.3f  最大 |差| %.3f  RMS %.3f"
          % (dif.mean(), np.median(dif), np.abs(dif).max(), np.sqrt((dif ** 2).mean())))
    print("  两曲线相关：Pearson r=%+.4f (p=%.1e)  Spearman ρ=%+.4f" % (r[0], r[1], rho))
    print("  符号一致：%d/%d 点" % (int((np.sign(da) == np.sign(dc)).sum()), n))

    def summary(x):
        i = int(np.argmax(x))
        return dict(peak=float(x[i]), t_peak=float(ta[i]), hi=float(x[-1]),
                    drop=float(x[i] - x[-1]), pct=float(100 * (x[i] - x[-1]) / x[i]),
                    n_pos=int(sum(1 for v in x if v > 0)))

    SA, SC = summary(da), summary(dc)
    print("\n[2] 汇总")
    for nm, s in (("testA", SA), ("testC", SC)):
        print("  %-6s 峰值 %+.3f dB(@%+.0f)  +15dB %+.3f  降 %+.3f dB(%.1f%%)  "
              "全正 %d/%d" % (nm, s["peak"], s["t_peak"], s["hi"],
                              s["drop"], s["pct"], s["n_pos"], n))

    print("\n[3] 收益效率 e=Δ/gain_imp（−10/5/+10 dB）")
    EA = {c: eff(A, c) for c in CONDS}
    EC = {c: eff(Cc, c) for c in CONDS}
    for c in CONDS:
        print("  %-7s testA %s | testC %s" %
              (LBL[c], "  ".join("%.4f" % x for x in EA[c]),
               "  ".join("%.4f" % x for x in EC[c])))
    ordA = [c for c in CONDS]
    okA = all(EA[CONDS[i]][0] < EA[CONDS[i + 1]][0] for i in range(3))
    okC = all(EC[CONDS[i]][0] < EC[CONDS[i + 1]][0] for i in range(3))
    print("  −10 dB 上 e 随 α 单调递增：testA %s   testC %s" %
          ("是" if okA else "否", "是" if okC else "否"))

    out = dict(n_pts=n, snr=[float(x) for x in ta],
               testA=dict(d=da.tolist(), sd=sa.tolist(), n_pos=na.tolist(), **SA),
               testC=dict(d=dc.tolist(), sd=sc.tolist(), n_pos=nc.tolist(), **SC),
               diff=dict(mean=float(dif.mean()), med=float(np.median(dif)),
                         max_abs=float(np.abs(dif).max()),
                         rms=float(np.sqrt((dif ** 2).mean()))),
               pearson=dict(r=float(r[0]), p=float(r[1])), spearman=float(rho),
               sign_agree=int((np.sign(da) == np.sign(dc)).sum()),
               eff_testA={c: EA[c].tolist() for c in CONDS},
               eff_testC={c: EC[c].tolist() for c in CONDS},
               eff_alpha_order=dict(testA=bool(okA), testC=bool(okC)))
    p = os.path.join(OUTD, "snr_ac_compare.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n->", p)


if __name__ == "__main__":
    main()
