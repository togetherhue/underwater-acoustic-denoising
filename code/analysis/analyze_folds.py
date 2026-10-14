# -*- coding: utf-8 -*-
"""3 折信道留出实验的结果分析：把「信道泛化代价」做成有误差棒的结论"""
import os, csv, json, glob
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"D:/project/output_folds"
DS = r"D:/project/dataset_folds"
FIG = r"E:/workbuddy/store/2026-10-12-22-50-08/figs_folds"
os.makedirs(FIG, exist_ok=True)
C_BG, C_GRID = "#ffffff", "#e6e6e6"


def style(ax):
    ax.set_facecolor(C_BG); ax.grid(True, color=C_GRID, lw=.8, axis="y")
    for s in ax.spines.values(): s.set_color("#bbbbbb")


def read_sum(tag):
    p = os.path.join(OUT, "summary_%s.csv" % tag)
    if not os.path.exists(p): return None
    r = list(csv.DictReader(open(p, encoding="utf-8")))
    return r[0] if r else None


def read_res(tag, which):
    p = os.path.join(OUT, "result_%s_%s.csv" % (tag, which))
    return np.loadtxt(p, delimiter=",", skiprows=1) if os.path.exists(p) else None


def meta(split):
    p = os.path.join(DS, "fold%d" % FOLD, "meta_%s.mat" % split)
    mt = sio.loadmat(p, squeeze_me=True, struct_as_record=False)["meta"]
    return (np.array([mt[k].chirp_id for k in range(mt.size)]),
            np.array([mt[k].chan_id for k in range(mt.size)]))


print("=" * 92)
print("3 折信道留出实验：同一批测试波形（chirp 18-20）下，见过的信道 vs 没见过的信道")
print("=" * 92)
print("%-8s %8s %8s %10s %10s %10s %10s" %
      ("fold", "n_seen", "n_unseen", "增益(seen)", "增益(unseen)", "代价(dB)", "SI-SDR代价"))

rows = []
for k in (1, 2, 3):
    r = read_sum("lite_fold%d" % k)
    if r is None:
        print("fold %d  (无结果)" % k); continue
    rs, ru = read_res("lite_fold%d" % k, "seen"), read_res("lite_fold%d" % k, "unseen")
    ds = (ru[:, 3] - ru[:, 2]).mean() - (rs[:, 3] - rs[:, 2]).mean()   # unseen - seen
    pen = float(r["penalty_db"])
    rows.append(dict(fold=k, n_seen=int(r["n_seen"]), n_unseen=int(r["n_unseen"]),
                     gain_seen=float(r["gain_seen"]), gain_unseen=float(r["gain_unseen"]),
                     penalty=pen, sisdr_penalty=ds))
    print("%-8d %8s %8s %10.2f %10.2f %+10.2f %+10.2f" %
          (k, r["n_seen"], r["n_unseen"], float(r["gain_seen"]),
           float(r["gain_unseen"]), pen, ds))

print("-" * 92)
if rows:
    P = np.array([r["penalty"] for r in rows])
    print("信道新颖度代价： %s" % "  ".join("%+.2f" % v for v in P))
    print("  均值 %+.2f dB   标准差 %.2f dB   范围 %+.2f ~ %+.2f dB"
          % (P.mean(), P.std(ddof=1) if len(P) > 1 else 0, P.min(), P.max()))
    tot_seen = sum(r["n_seen"] for r in rows)
    tot_unseen = sum(r["n_unseen"] for r in rows)
    # 加权合并（按样本数）
    gs = sum(r["gain_seen"] * r["n_seen"] for r in rows) / tot_seen
    gu = sum(r["gain_unseen"] * r["n_unseen"] for r in rows) / tot_unseen
    print("  按样本加权合并： 见过 %.2f  →  未见 %.2f   代价 %+.2f dB"
          % (gs, gu, gu - gs))
    print("  合计样本：见过 %d / 未见 %d（比 v3 单次实验的 1080/480 大 %.1f 倍）"
          % (tot_seen, tot_unseen, (tot_seen + tot_unseen) / 1560))

# ---------------- 逐信道明细 ----------------
print()
print("=" * 92)
print("逐信道明细（每折内，见过的信道 vs 该折留出的信道）")
print("=" * 92)
detail = []
for k in (1, 2, 3):
    FOLD = k
    rs, ru = read_res("lite_fold%d" % k, "seen"), read_res("lite_fold%d" % k, "unseen")
    if rs is None: continue
    cs, cas = meta("test_seen")
    cu, cau = meta("test_unseen")
    gs = rs[:, 1] - rs[:, 0]; gu = ru[:, 1] - ru[:, 0]
    print("\nfold %d  留出信道: %s" % (k, sorted(set(cau.tolist()))))
    print("  %-9s %8s   %-9s %8s" % ("seen chan", "增益", "unseen chan", "增益"))
    for c in sorted(set(cas.tolist())):
        m = cas == c
        print("  %-9d %+8.2f" % (c, gs[m].mean()), end="")
        detail.append(dict(fold=k, kind="seen", chan=c, gain=float(gs[m].mean())))
        # 同行显示一个 unseen 信道（数量不同，简单配对展示）
        print()
    for c in sorted(set(cau.tolist())):
        m = cau == c
        print("  %-9s %8s   %-9d %+8.2f" % ("", "", c, gu[m].mean()))
        detail.append(dict(fold=k, kind="unseen", chan=c, gain=float(gu[m].mean())))

# ---------------- 出图 ----------------
if rows:
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=150)
    xs = np.arange(len(rows)); w = 0.34
    ax = axs[0]; style(ax)
    b1 = ax.bar(xs - w/2, [r["gain_seen"] for r in rows], w, label="见过的信道", color="#4c78a8")
    b2 = ax.bar(xs + w/2, [r["gain_unseen"] for r in rows], w, label="没见过的信道", color="#f58518")
    for bb in (b1, b2):
        for r_ in bb:
            ax.text(r_.get_x()+r_.get_width()/2, r_.get_height()+.15,
                    "%.2f" % r_.get_height(), ha="center", fontsize=9)
    for i, r in enumerate(rows):
        ax.text(xs[i], max(r["gain_seen"], r["gain_unseen"])+1.0,
                "%+.2f dB" % r["penalty"], ha="center", fontsize=10,
                color="#12633a", fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(["fold %d" % r["fold"] for r in rows])
    ax.set_ylabel("SNR 提升 (dB)"); ax.set_ylim(0, max(r["gain_seen"] for r in rows)*1.28)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("每折：同一批波形下，换信道带来的差异极小", fontsize=11)

    ax = axs[1]; style(ax)
    P = [r["penalty"] for r in rows]
    ax.bar(xs, P, 0.5, color=["#54a24b" if v < 0 else "#e45756" for v in P])
    lo = min(0.0, min(P)); hi = max(0.0, max(P))
    pad = max(0.08, (hi - lo) * 0.28)
    for i, v in enumerate(P):
        y = v - pad * 0.45 if v < 0 else v + pad * 0.30
        ax.text(xs[i], y, "%+.2f" % v, ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="#12633a" if v < 0 else "#9c2b1f")
    ax.axhline(np.mean(P), color="#7f7f7f", ls="--", lw=1.5,
               label="均值 %+.2f dB" % np.mean(P))
    ax.axhline(0, color="#333", lw=1)
    ax.set_ylim(lo - pad * 1.7, hi + pad * 1.7)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xticks(xs); ax.set_xticklabels(["fold %d" % r["fold"] for r in rows])
    ax.set_ylabel("信道新颖度代价 (dB)")
    ax.set_title("代价 = 未见信道增益 − 见过信道增益（负值表示未见反而略好）", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "figE_folds.png")); plt.close(fig)
    print("\n图已保存: figs_folds/figE_folds.png")

    with open(os.path.join(FIG, "summary_folds.json"), "w", encoding="utf-8") as f:
        json.dump({"folds": rows, "mean": float(np.mean(P)),
                   "std": float(np.std(P, ddof=1)) if len(P) > 1 else 0.0,
                   "detail": detail}, f, ensure_ascii=False, indent=1)
    print("汇总 -> figs_folds/summary_folds.json")
