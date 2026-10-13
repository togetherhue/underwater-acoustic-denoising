# -*- coding: utf-8 -*-
"""2x2 泛化实验 + 消融 的出图脚本"""
import os, glob
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly, butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"D:/project/output_v3"
DS  = r"D:/project/dataset_v3"
FIG = r"E:/workbuddy/store/2026-10-12-22-50-08/figs_v3"
os.makedirs(FIG, exist_ok=True)

TESTS = [("testA", "testA_seench_newchirp", "testA\n新波形\n见信道"),
         ("testB", "testB_newch_seenchirp", "testB\n见波形\n新信道"),
         ("testC", "testC_newch_newchirp",  "testC\n新波形\n新信道")]
FS, L, DS_FACTOR = 6000, 8000, 4

C_BG, C_GRID = "#ffffff", "#e6e6e6"
def style(ax):
    ax.set_facecolor(C_BG); ax.grid(True, color=C_GRID, lw=.8, axis="y")
    for s in ax.spines.values(): s.set_color("#bbbbbb")


def snr(e, r):
    return 10*np.log10((r**2).sum(-1)/(((e-r)**2).sum(-1)+1e-12))


def baseline(key, split):
    """算该测试集的输入 SNR 与理想带通 SNR"""
    files = sorted(glob.glob(os.path.join(DS, split, "*.mat")))
    N = np.zeros((len(files), L), np.float32); C = np.zeros_like(N)
    for i, p in enumerate(files):
        m = sio.loadmat(p)
        for src, dst in ((m["noisy_signal"], N), (m["clean_signal"], C)):
            x = np.asarray(src).squeeze().astype(np.float64)
            x = resample_poly(x, 1, DS_FACTOR)
            dst[i] = x[:L] if len(x) >= L else np.pad(x, (0, L-len(x)))
    b, a = butter(6, [798/(FS/2), 1500/(FS/2)], btype="band")
    bp = filtfilt(b, a, N, axis=1).astype(np.float32)
    return snr(N, C).mean(), snr(bp, C).mean()


def load(tag, key):
    p = os.path.join(OUT, "result_%s_%s.csv" % (tag, key))
    return np.loadtxt(p, delimiter=",", skiprows=1) if os.path.exists(p) else None


# ---------------- 图 A: 2x2 泛化 ----------------
base = {}
print("计算基线...")
for key, split, _ in TESTS:
    base[key] = baseline(key, split)
    print("  %-6s 输入 %.2f  带通 %.2f" % (key, base[key][0], base[key][1]))

xs = np.arange(3); w = 0.26
inp = [base[k][0] for k, _, _ in TESTS]
bpf = [base[k][1] for k, _, _ in TESTS]
mdl = []
for key, _, _ in TESTS:
    r = load("full", key)
    mdl.append(r[:, 1].mean() if r is not None else np.nan)

fig, ax = plt.subplots(figsize=(8.6, 5), dpi=150); style(ax)
b1 = ax.bar(xs-w, inp, w, label="含噪输入", color="#7f7f7f")
b2 = ax.bar(xs,   bpf, w, label="理想带通 798-1500 Hz", color="#ffbb78")
b3 = ax.bar(xs+w, mdl, w, label="本模型", color="#2ca02c")
for bars, vals in ((b1, inp), (b2, bpf), (b3, mdl)):
    for r_, v in zip(bars, vals):
        ax.text(r_.get_x()+r_.get_width()/2, v+.4, "%.1f" % v, ha="center", fontsize=9)
for i in range(3):
    ax.text(xs[i]+w, mdl[i]+1.6, "%+.2f" % (mdl[i]-inp[i]), ha="center",
            fontsize=10, color="#12633a", fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels([t[2] for t in TESTS])
ax.set_ylabel("输出 SNR (dB)"); ax.set_ylim(0, max(mdl)*1.38)
ax.legend(fontsize=9, loc="upper right")
ax.set_title("2×2 泛化实验：模型对「新信道」的泛化与「新波形」相当", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "figA_2x2.png")); plt.close(fig)

# ---------------- 图 B: 消融 ----------------
tags = [t for t in ("full", "timeonly", "freqonly") if load(t, "testA") is not None]
if len(tags) > 1:
    names = {"full": "时域+频域", "timeonly": "仅时域", "freqonly": "仅频域"}
    cols = {"full": "#2ca02c", "timeonly": "#4c78a8", "freqonly": "#e45756"}
    xs2 = np.arange(3); w2 = 0.8/len(tags)
    fig, ax = plt.subplots(figsize=(8.6, 5), dpi=150); style(ax)
    for j, t in enumerate(tags):
        vals = []
        for key, _, _ in TESTS:
            r = load(t, key)
            vals.append((r[:, 1]-r[:, 0]).mean() if r is not None else np.nan)
        off = (j - (len(tags)-1)/2) * w2
        bb = ax.bar(xs2+off, vals, w2*0.92, label=names.get(t, t), color=cols.get(t, "#888"))
        for r_, v in zip(bb, vals):
            ax.text(r_.get_x()+r_.get_width()/2, v+.25, "%.2f" % v, ha="center", fontsize=8.5)
    ax.set_xticks(xs2); ax.set_xticklabels([t[2] for t in TESTS])
    ax.set_ylabel("SNR 提升 (dB)"); ax.set_ylim(0, max(vals)*1.22); ax.legend(fontsize=9, loc="upper center", ncol=3)
    ax.set_title("消融实验：两个分支各自的贡献", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "figB_ablation.png")); plt.close(fig)

# ---------------- 图 C: 训练曲线 ----------------
fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150); style(ax)
for t, c in (("full", "#2ca02c"), ("timeonly", "#4c78a8"), ("freqonly", "#e45756")):
    p = os.path.join(OUT, "train_log_%s.txt" % t)
    if not os.path.exists(p): continue
    lg = np.loadtxt(p, delimiter=",", skiprows=1)
    if lg.ndim == 1: lg = lg[None, :]
    ax.plot(lg[:, 0], lg[:, 4]-lg[:, 3], color=c, lw=2,
            label="%s (验证集 SNR 提升)" % {"full": "时域+频域", "timeonly": "仅时域", "freqonly": "仅频域"}[t])
ax.set_xlabel("轮次 epoch"); ax.set_ylabel("验证集 SNR 提升 (dB)")
ax.legend(fontsize=9); ax.set_title("三个变体的收敛曲线", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "figC_curves.png")); plt.close(fig)

print("图已保存到", FIG)
for f in sorted(os.listdir(FIG)): print("  ", f)

# ---------------- 保存汇总，供 build_report_v3.py 使用 ----------------
import json
res = {"base": {k: [float(x) for x in v] for k, v in base.items()},
       "sets": {}, "ablation": {}}
for key, _, _ in TESTS:
    r = load("full", key)
    if r is None: continue
    gain = r[:, 1] - r[:, 0]
    res["sets"][key] = {
        "n": int(len(r)), "snr_in": float(r[:, 0].mean()), "snr_out": float(r[:, 1].mean()),
        "gain": float(gain.mean()), "sisdr_gain": float((r[:, 3]-r[:, 2]).mean()),
        "worst": float(gain.min()), "p5": float(np.percentile(gain, 5)),
        "median": float(np.percentile(gain, 50)), "p95": float(np.percentile(gain, 95)),
        "best": float(gain.max()), "neg": int((gain < 0).sum()),
    }
for t in ("full", "timeonly", "freqonly"):
    row = {}
    for key, _, _ in TESTS:
        r = load(t, key)
        if r is not None:
            row[key] = float((r[:, 1]-r[:, 0]).mean())
    if row: res["ablation"][t] = row
with open(os.path.join(FIG, "summary_v3.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("汇总 -> summary_v3.json")

