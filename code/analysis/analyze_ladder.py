# -*- coding: utf-8 -*-
"""
受控阶梯分析：把"训练协议贡献"与"架构贡献"分离
================================================================================
问题：报告此前把 PAM-Net（18.27 / 27.38 / 17.12）与阶段0 纯频域
      （14.13 / 25.09 / 13.54）直接对比，并把这 ~4 dB 归因于架构。
      但两者的【归一化方式、损失函数、模型选择准则、训练脚本】全都不同，
      这个对比是混杂的，不能单独归因。

本脚本用一条"每步只改一个因素"的阶梯把它拆开：

  R0  阶段0 纯频域          5,910 参数   BN + L1+0.5·谱L1 + 按 val_loss 选模型
                                          （train_pam.py，v3 的 DenoiseNet）
  R1  纯频域 · 同协议复现    5,910 参数   同上，但走 PAM 训练管线
                                          → 检验两条管线是否等价（应 ≈ R0）
  R2  纯频域 + 新损失/选择   5,910 参数   BN + Huber+0.05·谱L1 + 按帧增益选模型
                                          → 单独分离"损失 + 模型选择准则"
  R3  纯频域 + 新归一化      5,910 参数   GN + Huber+0.05·谱L1 + 帧增益
                                          → 单独分离"归一化方式"
  R4  + 时域残差通路        14,367 参数   GN + Huber + 帧增益
                                          → 单独分离"时域残差估计通路"
  R5  + 脉冲感知分支        14,858 参数   同上 + p 注入频域掩码
                                          → 单独分离"脉冲感知分支"

前置条件（已由 test_pamnet_freqonly.py 验证）：
  纯频域模式的 PAMNet 与 v3.DenoiseNet(use_time=0, use_freq=1, fch=2.0)
  【架构逐位等价】（参数量 5,910 相同、权重复制后前向输出最大差 0.000e+00）。
"""
import os, csv, glob, re, json
import numpy as np

OUT = r"D:/project/output_pamnet"
OUT0 = r"D:/project/output_pam"
HERE = r"E:/workbuddy/store/2026-10-12-22-50-08"
FIG = os.path.join(HERE, "figs_pamnet")
os.makedirs(FIG, exist_ok=True)

METRICS = ["frmgain_clean", "frmgain_imp", "frmgain_imp_min"]
MLAB = {"frmgain_clean": "干净帧增益", "frmgain_imp": "干扰帧增益",
        "frmgain_imp_min": "干扰帧最差"}

LADDER = [
    # (rung, 显示名, 目录, 前缀, 说明)
    ("R0", "阶段0 纯频域\nBN + L1+0.5谱\n按 val_loss 选", OUT0, "pam_freq",
     "train_pam.py，v3.DenoiseNet(fch=2.0)，5,910 参数"),
    ("R1", "纯频域\n同协议走 PAM 管线", OUT, "fo_bn_l1",
     "协议与 R0 相同，换训练管线 —— 检验两管线是否等价"),
    ("R2", "+ 新损失/选择准则", OUT, "fo_bn_huber",
     "Huber(δ=0.2)+0.05·谱L1，按验证帧增益中位选模型"),
    ("R3", "+ 归一化 GN", OUT, "fo_gn",
     "GroupNorm 替代 BatchNorm"),
    ("R4", "+ 时域残差通路", OUT, "pamnet_freq_nopulse_gn",
     "PAM-Net 时域多尺度分支（短时/长时/门控）+ 频域精修，无脉冲分支"),
    ("R5", "+ 脉冲感知分支", OUT, "pamnet_freq_pf_gn",
     "p_n 作为附加通道注入频域掩码 —— 导师方案的创新点"),
]


def seed_tags(folder, prefix):
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "summary_%s*.csv" % prefix))):
        tag = os.path.basename(p)[len("summary_"):-4]
        if tag == prefix:
            out.append((1, tag))
        else:
            m = re.fullmatch(re.escape(prefix) + r"_s(\d+)", tag)
            if m:
                out.append((int(m.group(1)), tag))
    return sorted(out)


def read(folder, tag, split):
    p = os.path.join(folder, "summary_%s.csv" % tag)
    if not os.path.exists(p):
        return None
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["testset"] == split:
            d = {m: float(r[m]) for m in METRICS}
            d["params"] = int(r["params"])
            d["snr_out"] = float(r["snr_out"])
            d["norm"] = r.get("norm", "bn")
            d["loss"] = r.get("loss", "l1")
            d["select"] = r.get("select", "val_loss")
            return d
    return None


ALL = {}
for split in ["testA", "testC"]:
    print("=" * 118)
    print("受控阶梯（%s）  每步只改一个因素" % split)
    print("=" * 118)
    print("%-5s %-24s %-9s %8s %14s %14s %14s" %
          ("档", "配置", "种子数", "参数量", "干净帧增益", "干扰帧增益", "干扰帧最差"))
    print("-" * 118)
    rows = []
    for rung, name, folder, prefix, note in LADDER:
        st = seed_tags(folder, prefix)
        vals = []
        for sd, tag in st:
            d = read(folder, tag, split)
            if d is not None:
                vals.append(d)
        if not vals:
            print("%-5s %-24s  (缺数据)" % (rung, name.replace("\n", " ")))
            continue
        rec = {"rung": rung, "name": name, "note": note, "n_seed": len(vals),
               "params": vals[0]["params"], "norm": vals[0]["norm"],
               "loss": vals[0]["loss"], "select": vals[0]["select"]}
        for m in METRICS:
            v = np.array([d[m] for d in vals])
            rec[m] = float(v.mean())
            rec[m + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            rec[m + "_vals"] = [float(x) for x in v]
        rows.append(rec)
        print("%-5s %-24s %-9s %8d %14s %14s %14s"
              % (rung, name.replace("\n", " "), "n=%d" % len(vals), rec["params"],
                 "%.2f%s" % (rec["frmgain_clean"],
                             "" if len(vals) == 1 else "±%.2f" % rec["frmgain_clean_sd"]),
                 "%.2f%s" % (rec["frmgain_imp"],
                             "" if len(vals) == 1 else "±%.2f" % rec["frmgain_imp_sd"]),
                 "%.2f%s" % (rec["frmgain_imp_min"],
                             "" if len(vals) == 1 else "±%.2f" % rec["frmgain_imp_min_sd"])))
    print("-" * 118)

    # ---- 逐步增量 ----
    print()
    print("逐步增量（正数 = 该改动有正收益）")
    print("-" * 118)
    print("%-46s %14s %14s %14s" % ("改动", "Δ干净帧", "Δ干扰帧", "Δ干扰最差"))
    steps = []
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        lab = "%s→%s  %s" % (a["rung"], b["rung"], b["name"].replace("\n", " "))
        d = {}
        for m in METRICS:
            d[m] = b[m] - a[m]
            # 若两端都有多种子，给出去掉种子均值后的粗略不确定度
            sd = np.sqrt((a[m + "_sd"] ** 2) / max(a["n_seed"], 1)
                         + (b[m + "_sd"] ** 2) / max(b["n_seed"], 1))
            d[m + "_se"] = float(sd)
        steps.append(dict(label=lab, **d))
        print("%-46s %+9.2f±%.2f %+9.2f±%.2f %+9.2f±%.2f"
              % (lab, d["frmgain_clean"], d["frmgain_clean_se"],
                 d["frmgain_imp"], d["frmgain_imp_se"],
                 d["frmgain_imp_min"], d["frmgain_imp_min_se"]))
    print("-" * 118)

    # ---- 累计分解 ----
    if len(rows) >= 2:
        tot = {m: rows[-1][m] - rows[0][m] for m in METRICS}
        print()
        print("累计分解：%s → %s 共提升  干净帧 %+.2f / 干扰帧 %+.2f / 干扰最差 %+.2f dB"
              % (rows[0]["rung"], rows[-1]["rung"], tot["frmgain_clean"],
                 tot["frmgain_imp"], tot["frmgain_imp_min"]))
        for s in steps:
            share = (s["frmgain_imp"] / tot["frmgain_imp"] * 100
                     if abs(tot["frmgain_imp"]) > 1e-9 else float("nan"))
            print("    %-46s 贡献干扰帧 %+6.2f dB  (%.0f%%)" %
                  (s["label"], s["frmgain_imp"], share))

    ALL[split] = {"rows": rows, "steps": steps}

    # ---- 出图 ----
    if split == "testA":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, axs = plt.subplots(1, 2, figsize=(14.0, 5.4), dpi=150)
        xs = np.arange(len(rows))
        SHORT = {"R0": "阶段0 纯频域", "R1": "同协议·换管线", "R2": "+新损失/选择",
                 "R3": "+归一化 GN", "R4": "+时域残差通路", "R5": "+脉冲感知分支"}
        labs = ["%s\n%s\n%d 参数" % (r["rung"], SHORT.get(r["rung"], ""), r["params"])
                for r in rows]

        ax = axs[0]
        for m, c, mk in [("frmgain_clean", "#4c78a8", "o"),
                         ("frmgain_imp", "#f58518", "s"),
                         ("frmgain_imp_min", "#54a24b", "^")]:
            y = [r[m] for r in rows]
            e = [r[m + "_sd"] for r in rows]
            ax.errorbar(xs, y, yerr=e, marker=mk, color=c, lw=1.9, ms=8,
                        capsize=3.5, label=MLAB[m])
        lo = min(r["frmgain_imp_min"] - r["frmgain_imp_min_sd"] for r in rows)
        hi = max(r["frmgain_imp"] + r["frmgain_imp_sd"] for r in rows)
        ax.set_ylim(lo - 1.4, hi + 0.9)
        ax.set_xticks(xs)
        ax.set_xticklabels(labs, fontsize=8.2)
        ax.set_ylabel("逐帧降噪增益 (dB)")
        ax.set_title("受控阶梯：每步只改一个因素（testA）", fontsize=11)
        ax.grid(alpha=0.25, ls=":")
        ax.legend(fontsize=9, loc="lower right")

        ax = axs[1]
        STEP_SHORT = {"R0→R1": "换训练管线", "R1→R2": "损失+选择准则",
                      "R2→R3": "归一化 GN", "R1→R3": "损失+选择\n+归一化",
                      "R3→R4": "加时域\n残差通路", "R4→R5": "加脉冲\n感知分支"}
        pairs = [s["label"].split("  ")[0] for s in steps]
        xs2 = np.arange(len(steps)); w = 0.27
        for k, (m, c) in enumerate([("frmgain_clean", "#4c78a8"),
                                    ("frmgain_imp", "#f58518"),
                                    ("frmgain_imp_min", "#54a24b")]):
            y = [s[m] for s in steps]
            e = [s[m + "_se"] for s in steps]
            ax.bar(xs2 + (k - 1) * w, y, w, color=c, label=MLAB[m],
                   yerr=e, capsize=2, error_kw=dict(lw=0.8))
            for x, v in zip(xs2 + (k - 1) * w, y):
                ax.text(x, v + (0.10 if v >= 0 else -0.26), "%+.2f" % v,
                        ha="center", fontsize=7.4)
        ax.axhline(0, color="#333", lw=1)
        ax.set_xticks(xs2)
        ax.set_xticklabels(["%s\n%s" % (p, STEP_SHORT.get(p, "")) for p in pairs],
                           fontsize=8.2)
        ax.set_ylim(min(0.0, min(s["frmgain_imp"] for s in steps) - 0.6),
                    max(s["frmgain_clean"] for s in steps) * 1.22)
        ax.set_ylabel("该步带来的增益变化 (dB)")
        ax.set_title("逐步增量：协议 vs 架构", fontsize=11)
        ax.grid(alpha=0.25, ls=":", axis="y")
        ax.legend(fontsize=9, loc="upper left")

        fig.suptitle("把「训练协议贡献」与「架构贡献」分离（α稳定+脉冲，testA）",
                     fontsize=12.5)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "figP_ladder.png"))
        plt.close(fig)
        print("\n图 -> %s" % os.path.join(FIG, "figP_ladder.png"))

with open(os.path.join(FIG, "ladder.json"), "w", encoding="utf-8") as f:
    json.dump(ALL, f, ensure_ascii=False, indent=1, default=float)
print("汇总 -> figs_pamnet/ladder.json")
