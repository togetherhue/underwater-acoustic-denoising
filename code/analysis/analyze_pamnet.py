# -*- coding: utf-8 -*-
"""
阶段1（PAM-Net）结果汇总与出图
================================================================================
读 E:/project/data/output_pamnet/ 下所有 summary_*.csv，与阶段0 基线对照，
输出表格 + 图 + JSON 汇总。

阶段0 基线（同一份 α稳定+脉冲 数据集，同一套评估口径）：
  pam_freq     纯频域 fch=2.0，5,910 参数 —— 干净帧 14.13 / 干扰帧 25.09 / 干扰最差 13.54
  pam_full     时域+频域，57,657 参数
  pam_freq_g   高斯背景 + 同款脉冲（对照）
"""
import os, sys, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"E:/project/data/output_pamnet"
PAM0 = r"D:/project/output_pam"
HERE = r"E:/workbuddy/store/2026-10-12-22-50-08"
FIG = os.path.join(HERE, "figs_pamnet")
os.makedirs(FIG, exist_ok=True)

# 中文
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10

# 变体顺序与显示名
VARIANTS = [
    ("pamnet", "PAM-Net 完整（纯时域，BN）"),
    ("pamnet_nopulse", "纯时域 − 脉冲感知分支（BN）"),
    ("pamnet_freq", "PAM-Net + 频域精修（BN）"),
    ("pamnet_freq_nopulse", "频域精修，− 脉冲分支（BN）"),
    ("pamnet_freq_pf", "频域精修 + p 注入频域（BN）"),
    ("pamnet_gn", "PAM-Net 纯时域（GN）"),
    ("pamnet_nopulse_gn", "纯时域 − 脉冲分支（GN）"),
    ("pamnet_freq_pf_gn", "频域精修 + p 注入频域（GN）"),
    ("pamnet_freq_nopulse_gn", "频域精修，− 脉冲分支（GN）"),
    ("pamnet_nogate", "− 门控（改简单平均）"),
    ("pamnet_short", "仅短时分支"),
    ("pamnet_long", "仅长时分支"),
    ("pamnet_l1", "损失换 L1（去 Huber）"),
    ("pamnet_l2big", "λ2=0.5（谱项主导）"),
]
REF0 = [
    ("pam_freq", "【阶段0】纯频域 5,910"),
    ("pam_full", "【阶段0】时域+频域 57,657"),
    ("pam_freq_g", "【阶段0】高斯背景+脉冲"),
]

METRICS = ["frmgain_clean", "frmgain_imp", "frmgain_imp_min", "snr_out"]


NUMCOLS = ["n", "snr_in", "snr_out", "sisdr_in", "sisdr_out",
           "fmsnr_med_in", "fmsnr_med_out", "frmgain_med", "frmgain_clean",
           "frmgain_imp", "frmgain_imp_min", "fmsnr_frac_bad_out", "frac_imp"]


def read_sum(folder, tag):
    p = os.path.join(folder, "summary_%s.csv" % tag)
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if not rows:
        return None
    d = {}
    for r in rows:
        d[r["testset"]] = {k: float(r[k]) for k in NUMCOLS if k in r}
    return {"tag": tag, "params": int(rows[0]["params"]), "sets": d,
            "meta": {k: rows[0][k] for k in
                     ("c", "use_short", "use_long", "use_gate", "use_pulse",
                      "freq_refine", "pulse_freq", "loss", "delta",
                      "lam_imp", "lam_spec") if k in rows[0]}}


def collect(folder, variants):
    got = []
    for tag, label in variants:
        s = read_sum(folder, tag)
        if s is None:
            print("  (缺) %-18s %s" % (tag, os.path.join(folder, "summary_%s.csv" % tag)))
            continue
        s["label"] = label
        got.append(s)
    return got


print("=" * 110)
print("阶段1 汇总：PAM-Net 及其消融")
print("=" * 110)

res = collect(OUT, VARIANTS)
refs = collect(PAM0, REF0)
if not res:
    print("没有找到任何 PAM-Net 结果，退出")
    sys.exit(1)

print("\n%-26s %7s | %-28s | %-28s" % ("", "", "testA", "testC"))
print("%-26s %7s | %9s %9s %9s | %9s %9s %9s"
      % ("模型", "参数", "干净帧", "干扰帧", "干扰最差", "干净帧", "干扰帧", "干扰最差"))
print("-" * 110)


def show(rows, mark=""):
    out = []
    for s in rows:
        a = s["sets"].get("testA")
        c = s["sets"].get("testC")
        if a is None or c is None:
            continue
        out.append(dict(label=s["label"], tag=s["tag"], params=s["params"],
                        A_clean=a["frmgain_clean"], A_imp=a["frmgain_imp"],
                        A_impmin=a["frmgain_imp_min"], A_snr=a["snr_out"],
                        C_clean=c["frmgain_clean"], C_imp=c["frmgain_imp"],
                        C_impmin=c["frmgain_imp_min"], C_snr=c["snr_out"]))
        print("%-26s %7d | %9.2f %9.2f %9.2f | %9.2f %9.2f %9.2f %s"
              % (s["label"][:26], s["params"], a["frmgain_clean"], a["frmgain_imp"],
                 a["frmgain_imp_min"], c["frmgain_clean"], c["frmgain_imp"],
                 c["frmgain_imp_min"], mark))
    return out


rows = show(res)
print("-" * 110)
print("【阶段0 参照】")
show(refs)
print("-" * 110)

# ---------------------------------------------------------------- 与基线对比
base = next((r for r in refs if r["tag"] == "pam_freq"), None)
full = next((r for r in rows if r["tag"] == "pamnet"), None)
if base is not None and full is not None:
    print("\n=== PAM-Net 完整 vs 阶段0 纯频域基线（testA）===")
    b = base["sets"]["testA"]
    for k, nm in [("frmgain_clean", "干净帧增益"), ("frmgain_imp", "干扰帧增益"),
                  ("frmgain_imp_min", "干扰帧最差"), ("snr_out", "输出全局SNR")]:
        print("  %-12s 基线 %7.2f   PAM-Net %7.2f   差 %+6.2f dB"
              % (nm, b[k], full["A_" + ("clean" if k == "frmgain_clean" else
                                        "imp" if k == "frmgain_imp" else
                                        "impmin" if k == "frmgain_imp_min" else "snr")],
                 full["A_" + ("clean" if k == "frmgain_clean" else
                              "imp" if k == "frmgain_imp" else
                              "impmin" if k == "frmgain_imp_min" else "snr")] - b[k]))

# ---------------------------------------------------------------- 出图
def style(ax):
    ax.grid(alpha=0.25, ls="--")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


if rows:
    # --- 图 J：消融柱状图（testA）---
    labels = [r["label"] for r in rows]
    xs = np.arange(len(rows))
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 5.2), dpi=150)
    for ax, key, title in zip(axs,
                              ["A_clean", "A_imp", "A_impmin"],
                              ["干净帧降噪增益 (dB)", "干扰帧降噪增益 (dB)", "干扰帧最差增益 (dB)"]):
        style(ax)
        vals = [r[key] for r in rows]
        cols = ["#4c78a8" if r["tag"] == "pamnet" else "#9aa0a6" for r in rows]
        b = ax.bar(xs, vals, 0.62, color=cols)
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.25, "%.1f" % v,
                    ha="center", fontsize=8.5)
        if base is not None:
            bk = {"A_clean": "frmgain_clean", "A_imp": "frmgain_imp",
                  "A_impmin": "frmgain_imp_min"}[key]
            ax.axhline(base["sets"]["testA"][bk], color="#e45756", ls="--", lw=1.6,
                       label="阶段0 纯频域基线 %.2f" % base["sets"]["testA"][bk])
            ax.legend(fontsize=8.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8.5)
        ax.set_ylabel(title)
        ax.set_ylim(0, max(vals) * 1.22)
    fig.suptitle("PAM-Net 消融（testA：新波形 + 见过的信道）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figJ_pamnet_ablation.png"))
    plt.close(fig)
    print("\n图 -> figs_pamnet/figJ_pamnet_ablation.png")

# ---------------------------------------------------------------- JSON
summary = {"variants": rows, "refs": [
    dict(label=r["label"], tag=r["tag"], params=r["params"],
         A_clean=r["sets"]["testA"]["frmgain_clean"],
         A_imp=r["sets"]["testA"]["frmgain_imp"],
         A_impmin=r["sets"]["testA"]["frmgain_imp_min"],
         A_snr=r["sets"]["testA"]["snr_out"],
         C_clean=r["sets"]["testC"]["frmgain_clean"],
         C_imp=r["sets"]["testC"]["frmgain_imp"],
         C_impmin=r["sets"]["testC"]["frmgain_imp_min"],
         C_snr=r["sets"]["testC"]["snr_out"]) for r in refs]}
with open(os.path.join(FIG, "summary_pamnet.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)
print("汇总 -> figs_pamnet/summary_pamnet.json")
