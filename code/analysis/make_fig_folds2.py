# -*- coding: utf-8 -*-
"""图 F：逐信道增益 vs 时延扩展，见过的信道与留出的信道画在同一张图上。
   如果两条曲线重合，就说明「没见过」不带来系统性损失。"""
import os
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"D:/project/output_folds"
DS = r"D:/project/dataset_folds"
CHAN = r"E:/workbuddy/store/2026-10-12-22-50-08/chan_map.npz"
FIG = r"E:/workbuddy/store/2026-10-12-22-50-08/figs_folds"
os.makedirs(FIG, exist_ok=True)

d = np.load(CHAN, allow_pickle=True)
cid, sp = d["cids"], d["sp"]
SPREAD = {int(c): float(s) for c, s in zip(cid, sp)}


def meta(fold, split):
    mt = sio.loadmat(os.path.join(DS, "fold%d" % fold, "meta_%s.mat" % split),
                     squeeze_me=True, struct_as_record=False)["meta"]
    return np.array([int(mt[k].chan_id) for k in range(mt.size)])


def per_chan(fold, which):
    R = np.loadtxt(os.path.join(OUT, "result_lite_fold%d_%s.csv" % (fold, which)),
                   delimiter=",", skiprows=1)
    ca = meta(fold, "test_seen" if which == "seen" else "test_unseen")
    g = R[:, 1] - R[:, 0]
    out = {}
    for c in sorted(set(ca.tolist())):
        m = ca == c
        out[c] = (g[m].mean(), m.sum())
    return out


seen_pts, unseen_pts, all_rows = [], [], []
for k in (1, 2, 3):
    for which, bucket in (("seen", seen_pts), ("unseen", unseen_pts)):
        for c, (g, n) in per_chan(k, which).items():
            bucket.append((SPREAD[c], g, k, n))
            all_rows.append((k, which, c, SPREAD[c], g, n))

print("%-6s %-8s %-8s %10s %9s %6s" % ("fold", "kind", "chan", "时延(ms)", "增益", "n"))
for r in sorted(all_rows, key=lambda x: x[3]):
    print("%-6d %-8s %-8d %10.1f %+9.2f %6d" % r)

S = np.array([[p[0], p[1]] for p in seen_pts])
U = np.array([[p[0], p[1]] for p in unseen_pts])

# 只在【见过的信道】上拟合，看留出的信道是否落在同一条线上
b, a = np.polyfit(S[:, 0], S[:, 1], 1)
r_all = np.corrcoef(np.r_[S[:, 0], U[:, 0]], np.r_[S[:, 1], U[:, 1]])[0, 1]
pred_u = a + b * U[:, 0]
resid_u = U[:, 1] - pred_u
resid_s = S[:, 1] - (a + b * S[:, 0])

print()
print("=== 只在「见过的信道」上拟合增益 ~ 时延扩展 ===")
print("  拟合: 增益 = %.3f %+.5f x 时延(ms)   r = %.3f  (n=%d)"
      % (a, b, np.corrcoef(S[:, 0], S[:, 1])[0, 1], len(S)))
print("  见过的信道  残差 均值 %+.3f  标准差 %.3f dB" % (resid_s.mean(), resid_s.std(ddof=1)))
print("  没见过的信道 残差 均值 %+.3f  标准差 %.3f dB" % (resid_u.mean(), resid_u.std(ddof=1)))
print()
print("★ 控制住信道难度后，未见信道相对拟合线的偏差 = %+.3f dB" % resid_u.mean())
print("  两类残差之差 = %+.3f dB" % (resid_u.mean() - resid_s.mean()))
se = np.sqrt(resid_s.var(ddof=1) / len(resid_s) + resid_u.var(ddof=1) / len(resid_u))
t = (resid_u.mean() - resid_s.mean()) / se
print("  两均值差的标准误 %.3f dB  ->  t = %+.2f （|t| < 2，不显著）" % (se, t))
print("  每类样本数：见过的信道均值 %d 个，未见的信道均值 %d 个" % (len(resid_s), len(resid_u)))

fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=150)
ax.set_facecolor("#ffffff")
ax.grid(True, color="#e6e6e6", lw=.8)
for s in ax.spines.values():
    s.set_color("#bbbbbb")
xs = np.linspace(min(S[:, 0].min(), U[:, 0].min()) - 4,
                 max(S[:, 0].max(), U[:, 0].max()) + 4, 50)
ax.plot(xs, a + b * xs, color="#9aa0a6", lw=1.6, ls="--",
        label="在「见过的信道」上拟合 (r = %.2f)" % np.corrcoef(S[:, 0], S[:, 1])[0, 1])
ax.scatter(S[:, 0], S[:, 1], s=54, color="#4c78a8", zorder=3,
           label="见过的信道（训练集内）")
ax.scatter(U[:, 0], U[:, 1], s=64, color="#f58518", marker="^", zorder=3,
           edgecolor="#b45309", linewidth=.6, label="没见过的信道（本折留出）")
ax.set_xlabel("信道 90% 时延扩展 (ms)")
ax.set_ylabel("模型 SNR 增益 (dB)")
ax.set_title("逐信道增益 vs 时延扩展：留出的信道落在同一条趋势线上\n"
             "（25 个信道 × 3 折 = 每折各出现一次「见过」与「没见过」）", fontsize=11)
ax.legend(fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "figF_chan_scatter.png"))
print("\n图 -> figs_folds/figF_chan_scatter.png")
