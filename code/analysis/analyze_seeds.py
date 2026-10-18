# -*- coding: utf-8 -*-
"""
多种子聚合分析
================================================================================
目的：把「训练随机性」这一项不确定性从效应量里分离出来。

`paired_test_pamnet.py` 只在【同一批测试样本】上做配对检验，消除的是
"样本抽样"不确定性；但同一配置换个随机种子重训结果会变多少，之前没测过。
而 BN 协议下训练振荡的逐轮标准差高达 2~3 dB、best-checkpoint 的乐观偏差
中位 2.5 dB —— 所以这一步必须补。

本脚本报告两类不确定性：
    within-seed  （样本抽样，paired_test 给出，n=1080）
    between-seed （训练随机性，本脚本给出，n=种子数）
配置按【前缀】自动匹配 tag（`prefix` 或 `prefix_s<k>`），新种子跑完自动纳入。
"""
import os, csv, glob, json, re
import numpy as np

OUT = r"E:/project/data/output_pamnet"
HERE = r"E:/workbuddy/store/2026-10-12-22-50-08"
FIG = os.path.join(HERE, "figs_pamnet")

NUMCOLS = ["n", "snr_in", "snr_out", "sisdr_in", "sisdr_out",
           "fmsnr_med_in", "fmsnr_med_out", "frmgain_med", "frmgain_clean",
           "frmgain_imp", "frmgain_imp_min", "fmsnr_frac_bad_out", "frac_imp"]
METRICS = ["frmgain_clean", "frmgain_imp", "frmgain_imp_min", "snr_out"]

CONFIGS = {
    # ---- 纯频域阶梯（用于把"训练协议贡献"与"架构贡献"分离）----
    "纯频域 BN/L1+0.5谱 (val_loss)": "fo_bn_l1",
    "纯频域 BN/Huber+0.05谱 (fg)": "fo_bn_huber",
    "纯频域 GN/Huber+0.05谱 (fg)": "fo_gn",
    # ---- 完整 PAM-Net ----
    "纯时域 +脉冲分支（GN）": "pamnet_gn",
    "纯时域 −脉冲分支（GN）": "pamnet_nopulse_gn",
    "频域精修 +p注入（GN）": "pamnet_freq_pf_gn",
    "频域精修 −脉冲分支（GN）": "pamnet_freq_nopulse_gn",
    "纯时域 +脉冲分支（BN）": "pamnet",
    "纯时域 −脉冲分支（BN）": "pamnet_nopulse",
    "频域精修 +p注入（BN）": "pamnet_freq_pf",
    "频域精修 −脉冲分支（BN）": "pamnet_freq_nopulse",
}


def seedkey(tag, prefix):
    if tag == prefix:
        return "1"
    m = re.match(re.escape(prefix) + r"_s(\d+)$", tag)
    return m.group(1) if m else None


def load(tag, split="testA"):
    p = os.path.join(OUT, "summary_%s.csv" % tag)
    if not os.path.exists(p):
        return None
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["testset"] == split:
            d = {k: float(r[k]) for k in NUMCOLS if k in r}
            d["params"] = float(r["params"])
            d["norm"] = r.get("norm", "bn")
            return d
    return None


def collect(split="testA"):
    got = {}
    for name, prefix in CONFIGS.items():
        rows = {}
        for p in glob.glob(os.path.join(OUT, "summary_*.csv")):
            tag = os.path.basename(p)[len("summary_"):-4]
            sk = seedkey(tag, prefix)
            if sk is None:
                continue
            d = load(tag, split)
            if d is not None:
                rows[sk] = d
        if rows:
            got[name] = rows
    return got


def ttest1(diffs):
    n = len(diffs)
    if n < 2:
        return None
    se = diffs.std(ddof=1) / np.sqrt(n)
    t = diffs.mean() / se if se > 0 else 0.0
    try:
        from scipy.stats import t as td
        p = 2 * td.sf(abs(t), df=n - 1)
    except Exception:
        p = float("nan")
    return se, t, p


ALL = {}
for split in ["testA", "testC"]:
    got = collect(split)
    print("=" * 120)
    print("多种子聚合（%s）" % split)
    print("=" * 120)
    print("%-28s %-8s %10s %17s %17s %17s" %
          ("配置", "种子", "参数", "干净帧增益", "干扰帧增益", "干扰帧最差"))
    print("-" * 120)
    for name, rows in got.items():
        keys = sorted(rows, key=lambda x: int(x))
        for i, sk in enumerate(keys):
            d = rows[sk]
            print("%-28s %-8s %10d %17.2f %17.2f %17.2f"
                  % (name if i == 0 else "", sk, int(d["params"]),
                     d["frmgain_clean"], d["frmgain_imp"], d["frmgain_imp_min"]))
        if len(keys) > 1:
            v = {m: np.array([rows[s][m] for s in keys]) for m in METRICS}
            print("%-28s %-8s %10s %17s %17s %17s"
                  % ("  均值 ± 标准差", "n=%d" % len(keys), "",
                     "%.2f ± %.2f" % (v["frmgain_clean"].mean(), v["frmgain_clean"].std(ddof=1)),
                     "%.2f ± %.2f" % (v["frmgain_imp"].mean(), v["frmgain_imp"].std(ddof=1)),
                     "%.2f ± %.2f" % (v["frmgain_imp_min"].mean(), v["frmgain_imp_min"].std(ddof=1))))
        print("-" * 120)

    print()
    print("跨种子比较（按种子配对，n = 种子数）  正数 = A 更好")
    print("-" * 120)
    PAIRS = [
        ("纯时域：+脉冲分支 − (−脉冲分支) [GN]", "纯时域 +脉冲分支（GN）", "纯时域 −脉冲分支（GN）"),
        ("频域精修：+p注入 − (−脉冲分支) [GN]", "频域精修 +p注入（GN）", "频域精修 −脉冲分支（GN）"),
        ("[BN 对照] 纯时域：+ − −", "纯时域 +脉冲分支（BN）", "纯时域 −脉冲分支（BN）"),
        ("[BN 对照] 频域：+ − −", "频域精修 +p注入（BN）", "频域精修 −脉冲分支（BN）"),
        # ---- 受控阶梯：每一步只改一个因素 ----
        ("阶梯①损失+选择准则（BN 固定）", "纯频域 BN/Huber+0.05谱 (fg)", "纯频域 BN/L1+0.5谱 (val_loss)"),
        ("阶梯②归一化（损失固定）", "纯频域 GN/Huber+0.05谱 (fg)", "纯频域 BN/Huber+0.05谱 (fg)"),
        ("阶梯③加时域残差通路", "频域精修 −脉冲分支（GN）", "纯频域 GN/Huber+0.05谱 (fg)"),
        ("阶梯④加脉冲感知分支", "频域精修 +p注入（GN）", "频域精修 −脉冲分支（GN）"),
    ]
    out = {}
    for label, a, b in PAIRS:
        if a not in got or b not in got:
            print("  %-44s (缺数据)" % label)
            continue
        da, db = got[a], got[b]
        common = sorted(set(da) & set(db), key=int)
        if not common:
            print("  %-44s (无公共种子)" % label)
            continue
        print("  %s   （公共种子 %s，n=%d）" % (label, common, len(common)))
        rec = {}
        for m in METRICS:
            diffs = np.array([da[s][m] - db[s][m] for s in common])
            r = ttest1(diffs)
            if r is None:
                print("    %-14s %+8.3f dB   （仅 1 个种子，无法估计训练随机性）"
                      % (m, diffs[0]))
                rec[m] = dict(mean=float(diffs[0]), se=None, n=1,
                              diffs=[float(x) for x in diffs])
                continue
            se, t, p = r
            star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            print("    %-14s %+8.3f ± %.3f dB   t=%5.2f  p=%.4f  %s"
                  % (m, diffs.mean(), se, t, p, star))
            rec[m] = dict(mean=float(diffs.mean()), se=float(se), t=float(t),
                          p=float(p), n=int(len(diffs)), diffs=[float(x) for x in diffs])
        out[label] = rec
    print("-" * 120)
    ALL[split] = {"configs": {k: dict(v) for k, v in got.items()}, "pairs": out}

with open(os.path.join(FIG, "seed_agg.json"), "w", encoding="utf-8") as f:
    json.dump(ALL, f, ensure_ascii=False, indent=1, default=float)
print("\n汇总 -> figs_pamnet/seed_agg.json")
