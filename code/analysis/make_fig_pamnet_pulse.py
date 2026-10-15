# -*- coding: utf-8 -*-
"""
脉冲感知分支诊断：p_n 是否真的学到了"哪里是脉冲"
================================================================================
导师方案的创新点落在「是否针对脉冲干扰与有效水声波形之间的差异形成判别能力」。
最直接的证据就是脉冲感知分支的输出 p_n 与真实干扰位置掩码 m_n 的对应关系。

本脚本对训练好的模型做三件事：
  1. 统计 p 在【干扰位置】与【干净位置】的分布（直方图 + 均值/分位数）
  2. 把 p 当作脉冲检测器，算 AUC（秩统计量，不依赖阈值）
  3. 画一个具体样本：含噪波形 / 真实掩码 / 模型预测 p，直观看对齐程度

【为什么 AUC 合适】p 是有界连续量、m 是 0/1 标签，AUC 不需要选阈值就能
衡量"p 在干扰位置是否系统性高于干净位置"。AUC=0.5 表示毫无判别力，
AUC=1.0 表示完美分离。
"""
import os, sys, argparse, json
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = r"C:\Users\32213\PycharmProjects\project1"
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
import v3_model as V3          # noqa: E402
import pamnet as PN            # noqa: E402
from train_pam import PAMDataset   # noqa: E402

HERE = r"E:/workbuddy/store/2026-10-12-22-50-08"
FIG = os.path.join(HERE, "figs_pamnet")
os.makedirs(FIG, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10


def auc_rank(score, label):
    """秩统计量 AUC（Mann-Whitney U / (n1*n0)），支持并列值取平均秩"""
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    # 并列取平均秩
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=r"D:/project/dataset_pam/alphastable")
    ap.add_argument("--split", default="testA")
    ap.add_argument("--c", type=int, default=24)
    ap.add_argument("--norm", default="bn", choices=["bn", "gn", "in"])
    ap.add_argument("--freq-refine", type=float, default=0.0, dest="freq_refine")
    ap.add_argument("--pulse-freq", type=int, default=0, dest="pulse_freq")
    ap.add_argument("--max", type=int, default=400, help="最多用多少个样本统计")
    A = ap.parse_args()

    model = PN.PAMNet(c=A.c, n_fft=V3.CFG["n_fft"], hop=V3.CFG["hop_length"],
                      freq_refine=A.freq_refine, norm=A.norm,
                      pulse_freq=bool(A.pulse_freq)).to(V3.DEVICE)
    sd = torch.load(A.ckpt, map_location=V3.DEVICE)
    missing = model.load_state_dict(sd, strict=True)
    model.eval()
    print("已加载 %s   参数量 %d" % (A.ckpt, model.n_params()))

    ds = PAMDataset(os.path.join(A.data, A.split), V3.CFG["seg_len"], V3.CFG["downsample"])
    n_use = min(A.max, len(ds))

    p_imp, p_cln = [], []
    aucs = []
    example = None
    for i in range(n_use):
        n, c, m = ds[i]
        with torch.no_grad():
            est, p = model(n.unsqueeze(0).unsqueeze(0).to(V3.DEVICE), return_p=True)
        pp = p[0, 0].cpu().numpy()
        mm = m.numpy().astype(bool)
        if mm.sum() == 0:
            continue
        p_imp.append(pp[mm])
        p_cln.append(pp[~mm])
        aucs.append(auc_rank(pp, mm))
        if example is None and mm.sum() >= 3:
            example = (n.numpy(), c.numpy(), mm, pp)

    p_imp = np.concatenate(p_imp)
    p_cln = np.concatenate(p_cln)
    aucs = np.array(aucs)

    print()
    print("=" * 88)
    print("脉冲感知分支 p_n vs 真实干扰掩码 m_n   （%s，%d 个样本）" % (A.split, len(aucs)))
    print("=" * 88)
    print("  %-14s %8s %8s %8s %8s %8s" % ("", "均值", "中位", "p10", "p90", "占比"))
    print("  %-14s %8.4f %8.4f %8.4f %8.4f %7.2f%%"
          % ("干扰位置", p_imp.mean(), np.median(p_imp),
             np.percentile(p_imp, 10), np.percentile(p_imp, 90), 100.0))
    print("  %-14s %8.4f %8.4f %8.4f %8.4f %7.2f%%"
          % ("干净位置", p_cln.mean(), np.median(p_cln),
             np.percentile(p_cln, 10), np.percentile(p_cln, 90), 100.0))
    print("  %-14s %8.4f" % ("均值差", p_imp.mean() - p_cln.mean()))
    print()
    print("  AUC（p 作为脉冲检测器，逐样本计算）：")
    print("    均值 %.4f   中位 %.4f   p10 %.4f   p90 %.4f   最差 %.4f"
          % (aucs.mean(), np.median(aucs), np.percentile(aucs, 10),
             np.percentile(aucs, 90), aucs.min()))
    print("    （0.5 = 无判别力；>0.9 表示能可靠区分脉冲与干净位置）")
    print("  逐样本 AUC > 0.8 的比例：%.1f%%" % (100.0 * (aucs > 0.8).mean()))

    # ---------------- 出图 ----------------
    fig = plt.figure(figsize=(14, 5.4), dpi=150)
    ax = fig.add_subplot(1, 2, 1)
    bins = np.linspace(0, 1, 60)
    ax.hist(p_cln, bins=bins, density=True, alpha=0.62, color="#4c78a8",
            label="干净位置 (%.2f%% 的采样点)" % (100.0 * len(p_cln) / (len(p_cln) + len(p_imp))))
    ax.hist(p_imp, bins=bins, density=True, alpha=0.62, color="#e45756",
            label="干扰位置 (%.2f%%)" % (100.0 * len(p_imp) / (len(p_cln) + len(p_imp))))
    ax.axvline(p_cln.mean(), color="#1f4e79", ls="--", lw=1.6,
               label="干净位置均值 %.3f" % p_cln.mean())
    ax.axvline(p_imp.mean(), color="#9c2b1f", ls="--", lw=1.6,
               label="干扰位置均值 %.3f" % p_imp.mean())
    ax.set_xlabel("脉冲感知分支输出 $p_n$")
    ax.set_ylabel("概率密度")
    ax.set_title("$p_n$ 分布：干扰位置显著右移\n逐样本 AUC 均值 %.3f" % aucs.mean(), fontsize=11)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, ls="--")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax2 = fig.add_subplot(1, 2, 2)
    n, c, mm, pp = example
    L = len(n)
    # 找一段含干扰的窗口
    idx = np.where(mm)[0]
    ctr = int(idx[len(idx) // 2])
    a, b = max(0, ctr - 400), min(L, ctr + 400)
    t = (np.arange(a, b)) / (V3.CFG["fs_native"] / V3.CFG["downsample"]) * 1000.0
    ax2.plot(t, n[a:b], color="#9aa0a6", lw=0.9, label="含噪信号")
    ax2.plot(t, c[a:b], color="#4c78a8", lw=1.3, label="干净信号")
    ax2.set_xlabel("时间 (ms)")
    ax2.set_ylabel("幅度")
    ax2.set_title("样本示例：脉冲位置与 $p_n$ 的对齐", fontsize=11)
    ax2.grid(alpha=0.25, ls="--")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax3 = ax2.twinx()
    ax3.fill_between(t, 0, mm[a:b], color="#e45756", alpha=0.18, step="mid",
                     label="真实干扰掩码 $m_n$")
    ax3.plot(t, pp[a:b], color="#e45756", lw=1.6, label="预测 $p_n$")
    ax3.set_ylabel("$p_n$ / 掩码", color="#9c2b1f")
    ax3.set_ylim(0, 1.05)
    ax3.tick_params(axis="y", colors="#9c2b1f")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper right")

    fig.tight_layout()
    out = os.path.join(FIG, "figK_pulse_aware.png")
    fig.savefig(out)
    plt.close(fig)
    print("\n图 -> %s" % out)

    with open(os.path.join(FIG, "pulse_diag.json"), "w", encoding="utf-8") as f:
        json.dump({"n_samples": int(len(aucs)),
                   "p_imp_mean": float(p_imp.mean()), "p_clean_mean": float(p_cln.mean()),
                   "p_imp_median": float(np.median(p_imp)),
                   "p_clean_median": float(np.median(p_cln)),
                   "auc_mean": float(aucs.mean()), "auc_median": float(np.median(aucs)),
                   "auc_p10": float(np.percentile(aucs, 10)),
                   "frac_auc_gt_08": float((aucs > 0.8).mean())}, f,
                  ensure_ascii=False, indent=1)
    print("汇总 -> figs_pamnet/pulse_diag.json")


if __name__ == "__main__":
    main()
