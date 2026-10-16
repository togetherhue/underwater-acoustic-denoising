# -*- coding: utf-8 -*-
"""
适用边界实验：出图
================================================================================
figQ_sens_alpha.png   α 轴（rate=5 固定）：干净帧 / 干扰帧增益 vs α
figR_sens_rate.png    rate 轴（α=1.5 固定）：干净帧 / 干扰帧增益 vs 突发到达率
figS_sens_pulse.png   脉冲感知分支效应 Δ=R5−R4 随条件变化（含有效噪声功率对照）
figT_sens_detect.png  判别能力(AUC) vs 性能收益(Δ)  —— "能识别 ≠ 能提升"的跨条件检验
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = r"E:/workbuddy/store/2026-10-12-22-50-08"
OUTD = os.path.join(WORK, "sens_out")
FIGD = os.path.join(WORK, "figs_pamnet")
os.makedirs(FIGD, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10.5
plt.rcParams["axes.edgecolor"] = "#8a8f98"
plt.rcParams["axes.linewidth"] = 0.9

I_MED, I_CLN, I_IMP, I_MIN = 6, 7, 8, 9
CFG_ORDER = ["R3_freqonly", "R4_freqrefine_nopulse", "R5_freqrefine_pulse",
             "P_time_pulse", "P_time_nopulse"]
LBL = {"R3_freqonly": "R3 纯频域掩码 (5,910)", "R4_freqrefine_nopulse": "R4 时域残差+频域精修 (14,367)",
       "R5_freqrefine_pulse": "R5 R4 + 脉冲感知分支 (14,858)", "P_time_pulse": "纯时域 PAM-Net (8,906)",
       "P_time_nopulse": "纯时域（无脉冲分支）(8,457)"}
CLR = {"R3_freqonly": "#4c78a8", "R4_freqrefine_nopulse": "#f58518",
       "R5_freqrefine_pulse": "#e45756", "P_time_pulse": "#54a24b",
       "P_time_nopulse": "#b279a2"}
MK = {"R3_freqonly": "o", "R4_freqrefine_nopulse": "s", "R5_freqrefine_pulse": "^",
      "P_time_pulse": "D", "P_time_nopulse": "v"}
LS = {"R3_freqonly": "-", "R4_freqrefine_nopulse": "--", "R5_freqrefine_pulse": "-",
      "P_time_pulse": ":", "P_time_nopulse": "-."}


def load(cond, cfg, split="testA"):
    p = os.path.join(OUTD, "sens_%s__%s.npz" % (cond, cfg))
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z[split] if split in z else None


def load_seed(cond, cfg, si, split="testA"):
    p = os.path.join(OUTD, "sens_%s__%s__s%d.npz" % (cond, cfg, si))
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z[split] if split in z else None


def agg(cond, cfg, idx, split="testA"):
    """返回 (逐种子均值数组, 全部样本池化后的逐样本数组)"""
    p = os.path.join(OUTD, "sens_%s__%s.npz" % (cond, cfg))
    if os.path.exists(p):
        R = np.load(p)[split]                    # [n_seed, n, 12]
        return np.nanmean(R[:, :, idx], axis=1), R[:, :, idx].ravel()
    per, pool = [], []
    si = 0
    while True:
        r = load_seed(cond, cfg, si, split)
        if r is None:
            break
        per.append(float(np.nanmean(r[:, idx]))); pool.append(r[:, idx]); si += 1
    if not per:
        return None, None
    return np.array(per), np.concatenate(pool)


def axis_panel(ax, xvals, conds, idx, title, ylabel, show_effsnr=True, noise=None):
    for cf in CFG_ORDER:
        xs, ys, es = [], [], []
        for x, c in zip(xvals, conds):
            if c is None:
                continue
            per, _ = agg(c, cf, idx)
            if per is None:
                continue
            xs.append(x); ys.append(per.mean()); es.append(per.std(ddof=1) if len(per) > 1 else 0.0)
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=es, color=CLR[cf], marker=MK[cf], ls=LS[cf], lw=2.0,
                    ms=6.5, capsize=3.5, label=LBL[cf], alpha=0.95)
    ax.set_title(title, fontsize=11.5)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, ls="--")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if show_effsnr and noise is not None:
        ax2 = ax.twinx()
        ev = [noise[c]["snr_bg"] for c in conds if c]
        ev_x = [x for x, c in zip(xvals, conds) if c]
        ax2.plot(ev_x, ev, color="#9aa0a6", lw=1.3, ls=":", marker="x", ms=6,
                 label="背景噪声有效功率 SNR（右轴）")
        ax2.set_ylabel("背景噪声有效功率 SNR (dB)", color="#6b7280", fontsize=9.5)
        ax2.tick_params(axis="y", colors="#6b7280", labelsize=9)
        for s in ("top",):
            ax2.spines[s].set_visible(False)


def main():
    sm = json.load(open(os.path.join(OUTD, "sens_summary_testA.json"), encoding="utf-8"))
    noise = sm["noise"]
    split = "testA"

    # ---------------- figQ：α 轴 ----------------
    conds = ["a12_r5", "base", "a18_r5", "a20_r5"]
    xv = [1.2, 1.5, 1.8, 2.0]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0), dpi=150)
    axis_panel(axes[0], xv, conds, I_CLN, "干净帧（无干扰）", "逐帧降噪增益 (dB)", noise=noise)
    axis_panel(axes[1], xv, conds, I_IMP, "干扰帧", "逐帧降噪增益 (dB)", noise=noise)
    for ax in axes:
        ax.set_xlabel("α 稳定分布特征指数 α（rate=5/s 固定）")
        ax.set_xticks(xv)
    h1, l1 = axes[1].get_legend_handles_labels()
    fig.legend(h1, l1, loc="lower center", ncol=5, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("图 Q  适用边界（一）：背景噪声尾指数 α 的影响"
                 "（α=1.5 为训练条件，其余为零样本跨条件评估；误差棒 = 种子间标准差）",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig.savefig(os.path.join(FIGD, "figQ_sens_alpha.png")); plt.close(fig)

    # ---------------- figR：rate 轴 ----------------
    conds = ["a15_r25", "base", "a15_r10"]
    xv = [2.5, 5.0, 10.0]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0), dpi=150)
    axis_panel(axes[0], xv, conds, I_CLN, "干净帧（无干扰）", "逐帧降噪增益 (dB)", noise=noise)
    axis_panel(axes[1], xv, conds, I_IMP, "干扰帧", "逐帧降噪增益 (dB)", noise=noise)
    for ax in axes:
        ax.set_xlabel("突发干扰到达率 (次/秒)（α=1.5 固定）")
        ax.set_xticks(xv)
    h1, l1 = axes[1].get_legend_handles_labels()
    fig.legend(h1, l1, loc="lower center", ncol=5, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("图 R  适用边界（二）：突发干扰密度的影响"
                 "（rate=5/s 为训练条件；误差棒 = 种子间标准差）", fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig.savefig(os.path.join(FIGD, "figR_sens_rate.png")); plt.close(fig)

    # ---------------- figS：脉冲分支效应 ----------------
    pe = sm["pulse_eff"]
    order = ["a12_r5", "a15_r25", "base", "a15_r10", "a18_r5", "a20_r5", "gauss"]
    order = [c for c in order if c in pe]
    keys = [("clean", "干净帧"), ("imp", "干扰帧"), ("imp_min", "干扰最差帧")]
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2), dpi=150,
                             gridspec_kw={"width_ratios": [2.1, 1]})
    ax = axes[0]
    x = np.arange(len(order)); w = 0.26
    cols = ["#7fb3d5", "#e45756", "#8c6bb1"]
    for i, (k, lab) in enumerate(keys):
        v = [pe[c][k]["mean"] for c in order]
        e = [pe[c][k]["sd_over_seed"] for c in order]
        ax.bar(x + (i - 1) * w, v, w, yerr=e, capsize=3, color=cols[i], label=lab,
               edgecolor="white", lw=0.6)
    ax.axhline(0, color="#333", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in order], fontsize=9)
    ax.set_ylabel("Δ = R5（含脉冲分支）− R4（无脉冲分支）  (dB)")
    ax.set_title("脉冲感知分支的净效应随条件变化", fontsize=11.5)
    ax.legend(fontsize=9.5, frameon=False)
    ax.grid(alpha=0.25, ls="--", axis="y")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # 右侧：条件强度的两个轴
    ax2 = axes[1]
    ax2b = ax2.twinx()
    xs = np.arange(len(order))
    ax2.plot(xs, [noise[c]["snr_bg"] for c in order], "o-", color="#4c78a8",
             label="背景有效功率 SNR (dB)")
    ax2.set_ylabel("背景噪声有效功率 SNR (dB)", color="#4c78a8", fontsize=9.5)
    ax2.tick_params(axis="y", colors="#4c78a8", labelsize=9)
    ax2b.plot(xs, [noise[c]["snr_imp"] for c in order], "s--", color="#e45756",
              label="干扰有效功率 SNR (dB)")
    ax2b.set_ylabel("干扰有效功率 SNR (dB)", color="#e45756", fontsize=9.5)
    ax2b.tick_params(axis="y", colors="#e45756", labelsize=9)
    ax2.set_xticks(xs); ax2.set_xticklabels([c.replace("_", "\n") for c in order], fontsize=8.5)
    ax2.set_title("各条件的噪声强度", fontsize=11)
    ax2.grid(alpha=0.25, ls="--")
    for s in ("top",):
        ax2.spines[s].set_visible(False); ax2b.spines[s].set_visible(False)
    fig.suptitle("图 S  脉冲感知分支在什么条件下才有价值？"
                 "（Δ 由同种子内逐样本配对得到，误差棒 = 种子间标准差）", fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIGD, "figS_sens_pulse.png")); plt.close(fig)

    # ---------------- figT：判别能力 vs 收益 ----------------
    pd = sm.get("pulse_diag", {})
    fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=150)
    if pd:
        for cf, mk, cl in (("R5_freqrefine_pulse", "^", "#e45756"),
                           ("P_time_pulse", "D", "#54a24b")):
            xs, ys, lb = [], [], []
            for c in order:
                d = pd.get(c, {}).get(cf, {})
                if not d or c not in pe:
                    continue
                k0 = sorted(d.keys())[0]
                xs.append(d[k0]["auc_mean"]); ys.append(pe[c]["imp"]["mean"]); lb.append(c)
            ax.scatter(xs, ys, s=95, marker=mk, color=cl, zorder=3,
                       label=("R5（时域+频域+脉冲）" if cf.startswith("R5") else "纯时域 PAM-Net"))
            for xx, yy, ll in zip(xs, ys, lb):
                ax.annotate(ll, (xx, yy), textcoords="offset points", xytext=(7, 5),
                            fontsize=8.5, color="#333")
    ax.axhline(0, color="#333", lw=1.0, ls="--")
    ax.set_xlabel("脉冲感知分支的判别能力（逐样本 AUC 均值，testA）")
    ax.set_ylabel("该分支带来的干扰帧增益 Δ (dB)")
    ax.set_title("图 T  「能识别」与「能提升性能」的跨条件关系\n"
                 "横轴越高表示越能准确标出脉冲位置；纵轴越高表示对降噪越有实际贡献",
                 fontsize=11.5)
    ax.grid(alpha=0.25, ls="--")
    ax.legend(fontsize=9.5, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGD, "figT_sens_detect.png")); plt.close(fig)

    print("图已输出：figQ_sens_alpha / figR_sens_rate / figS_sens_pulse / figT_sens_detect")


if __name__ == "__main__":
    main()
