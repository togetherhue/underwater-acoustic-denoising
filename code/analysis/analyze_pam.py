# -*- coding: utf-8 -*-
"""阶段0 结果汇总：脉冲噪声下现有模型的表现 + 时域分支复检

对比三组（噪声模型见 make_dataset_pam.m）：
  pam_freq     α稳定背景 + 突发干扰，纯频域 fch=2.0（5,910 参数）
  pam_full     α稳定背景 + 突发干扰，时域+频域（57,657 参数）← 看 w_time
  pam_freq_g   高斯背景 + 同样突发干扰，纯频域 fch=2.0 ← 分离背景分布的影响
"""
import os, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"D:/project/output_pam"
FIG = r"E:/workbuddy/store/2026-10-12-22-50-08/figs_pam"
os.makedirs(FIG, exist_ok=True)

MODELS = [("pam_freq", "α稳定+脉冲\n纯频域 5,910", "#4c78a8"),
          ("pam_full", "α稳定+脉冲\n时域+频域 57,657", "#f58518"),
          ("pam_freq_g", "高斯+脉冲\n纯频域 5,910", "#54a24b")]
TESTS = ["testA", "testC"]
C_BG, C_GRID = "#ffffff", "#e6e6e6"


def style(ax):
    ax.set_facecolor(C_BG); ax.grid(True, color=C_GRID, lw=.8, axis="y")
    for s in ax.spines.values():
        s.set_color("#bbbbbb")


def read_sum(tag):
    p = os.path.join(OUT, "summary_%s.csv" % tag)
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if not rows:
        return None
    return rows


res = {}
for tag, label, col in MODELS:
    r = read_sum(tag)
    if r is None:
        print("  %-12s 无结果，跳过" % tag); continue
    res[tag] = dict(label=label, color=col, params=int(r[0]["params"]),
                    w_time=float(r[0]["w_time"]), fch=float(r[0]["fch"]),
                    arch_time=int(r[0]["arch_time"]),
                    d={x["testset"]: {k: float(x[k]) for k in x
                                      if k not in ("testset", "tag")} for x in r})

if not res:
    raise SystemExit("没有任何结果")

# ---------------- 主表 ----------------
print("=" * 118)
print("阶段0：α稳定 + 突发干扰噪声下，现有架构的表现")
print("=" * 118)
print("%-26s %5s %10s %10s %10s %10s %10s %10s"
      % ("模型", "参数", "全局SNR", "SI-SDR", "帧增益中位", "增益·干净帧", "增益·干扰帧", "干扰帧最差"))
print("-" * 118)
for tag, d in res.items():
    for ts in TESTS:
        if ts not in d["d"]:
            continue
        m = d["d"][ts]
        print("%-26s %5d %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f   [%s]"
              % (d["label"].replace("\n", " "), d["params"],
                 m["snr_out"], m["sisdr_out"], m["frmgain_med"],
                 m["frmgain_clean"], m["frmgain_imp"], m["frmgain_imp_min"], ts))

# ---------------- 时域分支复检 ----------------
print()
print("=" * 118)
print("★ 时域分支复检（高斯噪声下 w_time 从 0.297 一路衰减到 0.006，模型自己弃用了它）")
print("=" * 118)
for tag, d in res.items():
    if d["arch_time"] == 1 and d["arch_time"] and "pam_full" == tag:
        print("  脉冲噪声下 pam_full 的最终 w_time = %.4f" % d["w_time"])
        print("  对照：高斯噪声下同一架构的最终 w_time = 0.006（v3 尺寸扫描实验）")

# ---------------- 关键对比 ----------------
print()
print("=" * 118)
print("关键对比")
print("=" * 118)
if "pam_freq" in res and "pam_freq_g" in res:
    a = res["pam_freq"]["d"]["testA"]; b = res["pam_freq_g"]["d"]["testA"]
    print("  ① 背景分布的影响（α稳定 vs 高斯，突发干扰完全相同）[testA]")
    for k, nm in [("frmgain_clean", "干净帧增益"), ("frmgain_imp", "干扰帧增益"),
                  ("frmgain_imp_min", "干扰帧最差"), ("snr_out", "全局输出SNR")]:
        print("     %-12s  α稳定 %7.2f   高斯 %7.2f   差 %+7.2f dB" % (nm, a[k], b[k], a[k] - b[k]))
if "pam_freq" in res and "pam_full" in res:
    a = res["pam_freq"]["d"]["testA"]; b = res["pam_full"]["d"]["testA"]
    print("  ② 加时域分支的收益（5,910 -> 57,657 参数）[testA]")
    for k, nm in [("frmgain_med", "帧增益中位"), ("frmgain_clean", "干净帧增益"),
                  ("frmgain_imp", "干扰帧增益"), ("frmgain_imp_min", "干扰帧最差")]:
        print("     %-12s  纯频域 %7.2f   含时域 %7.2f   差 %+7.2f dB" % (nm, a[k], b[k], b[k] - a[k]))

# ---------------- 出图 ----------------
tags = [t for t, _, _ in MODELS if t in res]
xs = np.arange(len(tags)); w = 0.36
fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), dpi=150)

ax = axs[0]; style(ax)
gc = [res[t]["d"]["testA"]["frmgain_clean"] for t in tags]
gi = [res[t]["d"]["testA"]["frmgain_imp"] for t in tags]
b1 = ax.bar(xs - w/2, gc, w, label="干净帧（无干扰）", color="#4c78a8")
b2 = ax.bar(xs + w/2, gi, w, label="干扰帧（含突发脉冲）", color="#e45756")
for bb in (b1, b2):
    for r in bb:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.4,
                "%.1f" % r.get_height(), ha="center", fontsize=9)
ax.set_xticks(xs); ax.set_xticklabels([res[t]["label"] for t in tags], fontsize=9)
ax.set_ylabel("逐帧降噪增益 (dB)")
ax.set_ylim(0, max(gc + gi) * 1.22)
ax.legend(fontsize=9)
ax.set_title("testA：干净帧 vs 干扰帧的降噪增益", fontsize=11)

ax = axs[1]; style(ax)
snr_g = [res[t]["d"]["testA"]["snr_out"] - res[t]["d"]["testA"]["snr_in"] for t in tags]
fms_g = [res[t]["d"]["testA"]["fmsnr_med_out"] - res[t]["d"]["testA"]["fmsnr_med_in"] for t in tags]
fg_m = [res[t]["d"]["testA"]["frmgain_med"] for t in tags]
w3 = 0.26
b1 = ax.bar(xs - w3, snr_g, w3, label="全局 SNR 提升", color="#9aa0a6")
b2 = ax.bar(xs, fms_g, w3, label="帧 SNR 中位提升", color="#4c78a8")
b3 = ax.bar(xs + w3, fg_m, w3, label="帧增益中位", color="#54a24b")
for bb in (b1, b2, b3):
    for r in bb:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.4,
                "%.1f" % r.get_height(), ha="center", fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels([res[t]["label"] for t in tags], fontsize=9)
ax.set_ylabel("提升 (dB)")
ax.set_ylim(0, max(snr_g + fms_g + fg_m) * 1.22)
ax.legend(fontsize=9)
ax.set_title("三种口径的差异：全局 SNR 明显偏大", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "figI_pam_stage0.png")); plt.close(fig)
print("\n图 -> figs_pam/figI_pam_stage0.png")

with open(os.path.join(FIG, "summary_pam.json"), "w", encoding="utf-8") as f:
    json.dump({t: {"label": res[t]["label"], "params": res[t]["params"],
                   "w_time": res[t]["w_time"], "fch": res[t]["fch"],
                   "arch_time": res[t]["arch_time"], "d": res[t]["d"]} for t in res},
              f, ensure_ascii=False, indent=1)
print("汇总 -> figs_pam/summary_pam.json")
