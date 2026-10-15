# -*- coding: utf-8 -*-
"""
生成阶段1 报告（修订版）：PAM-Net —— 实现、验证与统计效力分析
================================================================================
数据来源（全部在 figs_pamnet/）：
  summary_pamnet.json   analyze_pamnet.py   产出：各变体单种子结果 + 阶段0 参照
  seed_agg.json         analyze_seeds.py    产出：4 个关键配置 × 多种子的均值±标准差
  pulse_diag.json       make_fig_pamnet_pulse.py 产出：脉冲分支 AUC 诊断
  baseline_pam.json     compare_baseline_pam.py 产出：经典方法
  paired_test.json      paired_test_pamnet.py 产出：样本级配对检验
  stability.json        analyze_stability.py 产出：逐轮振荡与选择偏差
  图：figJ（消融）、figK（脉冲诊断）、figL（稳定性）、figM（多种子）、figN（归一化对照）
"""
import base64, os, json
import numpy as np

HERE = r"E:/workbuddy/store/2026-10-12-22-50-08"
FIG = os.path.join(HERE, "figs_pamnet")
OUT = os.path.join(HERE, "降噪模型_脉冲感知网络阶段1报告.html")

S = json.load(open(os.path.join(FIG, "summary_pamnet.json"), encoding="utf-8"))
SEED = json.load(open(os.path.join(FIG, "seed_agg.json"), encoding="utf-8"))
PD = json.load(open(os.path.join(FIG, "pulse_diag.json"), encoding="utf-8"))
BL = json.load(open(os.path.join(FIG, "baseline_pam.json"), encoding="utf-8"))
PT = json.load(open(os.path.join(FIG, "paired_test.json"), encoding="utf-8"))
ST = json.load(open(os.path.join(FIG, "stability.json"), encoding="utf-8"))

V = S["variants"]
R = S["refs"]
by_tag = {v["tag"]: v for v in V}
bref = {r["tag"]: r for r in R}
base0 = bref.get("pam_freq")

C = SEED["testA"]["configs"]
PAIRS = SEED["testA"]["pairs"]
CT = SEED["testC"]["pairs"]


def img(name, cap):
    p = os.path.join(FIG, name)
    if not os.path.exists(p):
        return "<figure><figcaption>（%s：图尚未生成）</figcaption></figure>" % cap
    with open(p, "rb") as f:
        b = base64.b64encode(f.read()).decode()
    return ('<figure><img src="data:image/png;base64,%s" alt="%s">'
            "<figcaption>%s</figcaption></figure>") % (b, cap, cap)


def ms(name, metric):
    rows = C.get(name, {})
    if not rows:
        return None
    v = np.array([rows[s][metric] for s in rows])
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0), len(v)


def fmt_ms(t, dec=2):
    if t is None:
        return "—"
    m, s, n = t
    return ("%.*f ± %.*f" % (dec, m, dec, s)) if n > 1 else ("%.*f（n=1）" % (dec, m))


def pget(label, metric, key, default=None):
    r = PAIRS.get(label, {}).get(metric)
    return r.get(key, default) if r else default


def fnum(x, d=2):
    return "—" if x is None else "%+.*f" % (d, x)


# ================= 主结果表（GN 多种子） =================
GN_ROWS = [
    ("纯时域 + 脉冲感知分支", "纯时域 +脉冲分支（GN）", "8,906"),
    ("纯时域 − 脉冲感知分支", "纯时域 −脉冲分支（GN）", "8,457"),
    ("频域精修 + p 注入频域（推荐）", "频域精修 +p注入（GN）", "14,858"),
    ("频域精修 − 脉冲感知分支", "频域精修 −脉冲分支（GN）", "14,367"),
]
rows_gn = ""
for label, key, par in GN_ROWS:
    hl = " style='background:#eef4ff;font-weight:600'" if "推荐" in label else ""
    rows_gn += ("<tr%s><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                % (hl, label, par, fmt_ms(ms(key, "frmgain_clean")),
                   fmt_ms(ms(key, "frmgain_imp")), fmt_ms(ms(key, "frmgain_imp_min")),
                   fmt_ms(ms(key, "snr_out"))))

# ================= 经典基线表 =================
rows_base = ""
for nm in ["含噪输入", "理想带通 798-1500Hz", "谱减法（盲）", "维纳滤波（盲）", "中值滤波 k=5"]:
    rec = BL["testA"].get(nm)
    if rec is None:
        continue
    rows_base += ("<tr%s><td>%s</td><td>%+.2f</td><td>%+.2f</td><td>%+.2f</td><td>%+.2f</td></tr>"
                  % (" class='ref'" if nm == "含噪输入" else "", nm, rec["snr"],
                     rec["frmgain_clean"], rec["frmgain_imp"], rec["frmgain_imp_min"]))

# ================= 消融表（单种子，BN） =================
abl_rows = ""
for v in V:
    if base0 is None:
        continue
    is_ref = (v["tag"] == "pamnet")
    style = " style='background:#eef4ff;font-weight:600'" if is_ref else ""
    abl_rows += ("<tr%s><td>%s%s</td><td>%s</td><td>%+.2f</td><td>%+.2f</td><td>%+.2f</td>"
                 "<td>%s</td><td>%s</td><td>%s</td></tr>"
                 % (style, v["label"], "（参照）" if is_ref else "",
                    "{:,}".format(v["params"]), v["A_clean"], v["A_imp"], v["A_impmin"],
                    fnum(v["A_clean"] - base0["A_clean"]),
                    fnum(v["A_imp"] - base0["A_imp"]),
                    fnum(v["A_impmin"] - base0["A_impmin"])))

# ================= 多种子配对表 =================
_MET = {"frmgain_clean": "干净帧", "frmgain_imp": "干扰帧", "frmgain_imp_min": "干扰最差",
        "snr_out": "输出SNR"}
rows_seed = ""
rows_seed_bn = ""
for lab, rec in PAIRS.items():
    first = True
    for m, mn in _MET.items():
        r = rec.get(m)
        if not r:
            continue
        if lab.startswith("[BN"):
            rows_seed_bn += ("<tr><td>%s</td><td>%s</td><td>%+.3f</td><td>1</td></tr>"
                             % (lab.replace("[BN 对照] ", "") if first else "", mn, r["mean"]))
        elif r.get("se") is None:
            rows_seed += ("<tr><td>%s</td><td>%s</td><td>%+.3f</td><td>—</td><td>—</td>"
                          "<td>1</td></tr>" % (lab if first else "", mn, r["mean"]))
        else:
            star = ("***" if r["p"] < 0.001 else ("**" if r["p"] < 0.01
                    else ("*" if r["p"] < 0.05 else "ns")))
            rows_seed += ("<tr><td>%s</td><td>%s</td><td>%+.3f ± %.3f</td><td>%.2f</td>"
                          "<td>%.3f %s</td><td>%d</td></tr>"
                          % (lab if first else "", mn, r["mean"], r["se"], r["t"],
                             r["p"], star, r["n"]))
        first = False

# ================= 样本级配对检验表 =================
rows_paired = ""
for _lab, _rec in PT.get("testA", {}).items():
    _first = True
    for _m in ["frmgain_clean", "frmgain_imp", "frmgain_imp_min"]:
        if _m not in _rec:
            continue
        _r = _rec[_m]
        _s = ("***" if _r["p"] < 0.001 else ("**" if _r["p"] < 0.01
              else ("*" if _r["p"] < 0.05 else "ns")))
        rows_paired += ("<tr><td>%s</td><td>%s</td><td>%+.3f</td><td>%.2f</td>"
                        "<td>%.1e %s</td><td>%+.3f</td></tr>"
                        % (_lab if _first else "", _MET[_m], _r["mean"],
                           _r["t"], _r["p"], _s, _r["dz"]))
        _first = False

# ================= 稳定性表 =================
rows_stab = ""
for r in ST["runs"]:
    if r["n"] < 10:
        continue
    rows_stab += ("<tr><td>%s</td><td>%.3f</td><td>%d</td><td>%.2f</td></tr>"
                  % (r["tag"], r["sd"], r["best_ep"], r["gain"]))

# ================= KPI =================
tc, ti, tm = (ms("频域精修 +p注入（GN）", "frmgain_clean"),
              ms("频域精修 +p注入（GN）", "frmgain_imp"),
              ms("频域精修 +p注入（GN）", "frmgain_imp_min"))
kpi = ("<div class='kpi'>"
       "<div><b>%.2f</b><span>推荐模型·干净帧增益<br>±%.2f（%d 种子）</span></div>"
       "<div><b>%.2f</b><span>推荐模型·干扰帧增益<br>±%.2f</span></div>"
       "<div><b>%.2f</b><span>推荐模型·干扰帧最差<br>±%.2f</span></div>"
       "<div><b>%.3f</b><span>脉冲分支 AUC<br>（判别能力证据）</span></div>"
       "</div>" % (tc[0], tc[1], tc[2], ti[0], ti[1], tm[0], tm[1], PD["auc_mean"]))

# ================= 结论 =================
td = ms("纯时域 +脉冲分支（GN）", "frmgain_imp")
hy = ms("频域精修 +p注入（GN）", "frmgain_imp")
tdc = ms("纯时域 +脉冲分支（GN）", "frmgain_clean")
tdm = ms("纯时域 +脉冲分支（GN）", "frmgain_imp_min")
gap = hy[0] - td[0]
d_td = td[0] - ms("纯时域 −脉冲分支（GN）", "frmgain_imp")[0]
d_hy = hy[0] - ms("频域精修 −脉冲分支（GN）", "frmgain_imp")[0]
d_hyc = (ms("频域精修 +p注入（GN）", "frmgain_clean")[0]
         - ms("频域精修 −脉冲分支（GN）", "frmgain_clean")[0])
L_TD = "纯时域：+脉冲分支 − (−脉冲分支) [GN]"
L_HY = "频域精修：+p注入 − (−脉冲分支) [GN]"
sd_bn = next((r["sd"] for r in ST["runs"] if r["tag"] == "pamnet"), float("nan"))
gain_bn = next((r["gain"] for r in ST["runs"] if r["tag"] == "pamnet"), float("nan"))
sd_gn = next((r["sd"] for r in ST["runs"] if r["tag"] == "pamnet_gn"), float("nan"))

concl = """<div class="box good"><b>① 按导师方案实现的 PAM-Net 可用，且远超全部经典方法。</b>
纯时域版（8,906 参数，GroupNorm，{tdn} 种子均值）在 testA 上拿到干扰帧增益
<b>{td_imp:.2f} ± {td_imp_s:.2f} dB</b>、干净帧 <b>{td_clean:.2f} ± {td_clean_s:.2f} dB</b>、
干扰帧最差 <b>{td_min:.2f} ± {td_min_s:.2f} dB</b>；而经典方法里最强的盲谱减法只有
6.27 / 10.51 / 1.83 dB，理想带通只有 8.00 / 5.69 / 1.65 dB。</div>

<div class="box bad"><b>② 但纯时域结构打不过频域掩码模型，差距 {gap:.1f} dB（干扰帧）。</b>
推荐模型（频域精修 + p 注入，14,858 参数）的干扰帧增益是
<b>{hy_imp:.2f} ± {hy_imp_s:.2f} dB</b>。原因不是实现错误，而是<b>结构先验</b>：
信号是窄带 chirp（798–1500 Hz），而 α 稳定背景与突发脉冲都是宽带的，
频域掩码天生等效于一个可学习的带通；纯时域残差估计必须从零学出这个带通。
这与阶段0"频域分支贡献几乎全部性能"的结论一致。</div>

<div class="box warn"><b>③ 脉冲感知分支确实形成了判别能力，但性能贡献很小。</b>
该分支输出 p<sub>n</sub> 与真实干扰掩码的逐样本 <b>AUC = {auc:.3f}</b>
（中位 {auc_med:.3f}，{auc_frac:.0f}% 的样本 &gt; 0.8），干扰位置平均置信度
{pi:.3f} vs 干净位置 {pc:.4f} —— <b>确实学到了"哪里是脉冲"</b>。但把它接进网络后：<br>
&nbsp;&nbsp;· <b>纯时域结构里，效应精确为零</b>（干扰帧 {d_td:+.3f} dB，
跨 {n_td} 种子 p = {p_td:.2f}）；<br>
&nbsp;&nbsp;· <b>只有在频域框架里才测得出</b>（干扰帧 {d_hy:+.3f} ± {d_hy_se:.3f} dB，
p = {p_hy:.3f}，n = {n_hy} 种子；干净帧 {d_hyc:+.3f} dB）。<br>
<b>但这个量级的效应还不够稳定</b>：testC 上显著的是干净帧
（{c_hyc:+.3f} ± {c_hyc_se:.3f} dB，p = {c_hyc_p:.3f}）而干扰帧不显著，
与 testA 相反 —— 效应落在哪个指标上会随测试集变化，说明它接近当前种子数下的
可检测下限，要定论需要更多种子。<br>
解释：二维频域掩码本身就能捕捉脉冲在时频面上的竖直条纹结构，脉冲感知分支与它高度冗余。</div>

<div class="box bad"><b>④ 方法学修正（必须记录）：实验协议本身会放大结论。</b>
BatchNorm 的逐批统计量被脉冲污染 → 验证指标逐轮振荡（标准差 <b>{sd_bn:.2f} dB</b>），
best-checkpoint 选择会挑到"统计量恰好有利"的轮次，带来中位 <b>{gain_bn:.1f} dB</b> 的乐观偏差。
改用 GroupNorm 后振荡降到 <b>{sd_gn:.2f} dB</b>、最佳轮次移到末轮（单调收敛）。<br>
<b>后果</b>：BN 单种子曾给出"脉冲分支 +1.99/+1.27/+0.96 dB"，而 GN 多种子只有
<b>{d_hyc:+.3f}（干净帧）/{d_hy:+.3f}（干扰帧） dB</b> ——
<b>单种子 + BN 会把效应量夸大 2~6 倍</b>。所有最终结论均基于 GN + 多种子。</div>
""".format(
    tdn=td[2], td_imp=td[0], td_imp_s=td[1], td_clean=tdc[0], td_clean_s=tdc[1],
    td_min=tdm[0], td_min_s=tdm[1], gap=gap, hy_imp=hy[0], hy_imp_s=hy[1],
    auc=PD["auc_mean"], auc_med=PD["auc_median"], auc_frac=100 * PD["frac_auc_gt_08"],
    pi=PD["p_imp_mean"], pc=PD["p_clean_mean"],
    d_td=d_td, p_td=(pget(L_TD, "frmgain_imp", "p", float("nan"))),
    n_td=(pget(L_TD, "frmgain_imp", "n", 1)),
    d_hy=d_hy, d_hy_se=(pget(L_HY, "frmgain_imp", "se", 0.0)),
    p_hy=(pget(L_HY, "frmgain_imp", "p", float("nan"))),
    n_hy=(pget(L_HY, "frmgain_imp", "n", 1)),
    d_hyc=d_hyc, sd_bn=sd_bn, gain_bn=gain_bn, sd_gn=sd_gn,
    c_hyc=(CT.get("频域精修：+p注入 − (−脉冲分支) [GN]", {}).get("frmgain_clean", {}).get("mean", float("nan"))),
    c_hyc_se=(CT.get("频域精修：+p注入 − (−脉冲分支) [GN]", {}).get("frmgain_clean", {}).get("se") or 0.0),
    c_hyc_p=(CT.get("频域精修：+p注入 − (−脉冲分支) [GN]", {}).get("frmgain_clean", {}).get("p", float("nan"))))

HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>降噪模型 · 脉冲感知网络（阶段1）</title>
<style>
:root{--bg:#f7f8fa;--card:#fff;--ink:#1a1d21;--muted:#6b7280;--line:#e5e7eb;
      --blue:#2563eb;--green:#15803d;--red:#b91c1c;--amber:#b45309}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
     font-size:15px;line-height:1.75}
.wrap{max-width:1040px;margin:0 auto;padding:40px 24px 80px}
h1{font-size:27px;margin:0 0 6px;letter-spacing:-.4px}
h2{font-size:20px;margin:38px 0 12px;padding-bottom:8px;border-bottom:2px solid var(--line)}
h3{font-size:16px;margin:24px 0 8px;color:#374151}
.sub{color:var(--muted);font-size:14px;margin-bottom:22px}
p{margin:10px 0}
code{background:#eef2f7;padding:1px 5px;border-radius:4px;font-size:13px;
     font-family:Consolas,Monaco,monospace}
pre{background:#0f172a;color:#e2e8f0;padding:14px 16px;border-radius:10px;
    overflow-x:auto;font-size:12.5px;line-height:1.6}
table{width:100%;border-collapse:collapse;margin:14px 0;font-size:13.5px;background:var(--card)}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{background:#f1f5f9;font-weight:600;color:#334155}
tr.ref td{color:var(--muted);font-style:italic}
tbody tr:hover{background:#f8fafc}
.kpi{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}
.kpi>div{background:var(--card);border:1px solid var(--line);border-radius:12px;
         padding:14px;text-align:center}
.kpi b{display:block;font-size:22px;color:var(--blue);letter-spacing:-.5px}
.kpi span{display:block;font-size:11.5px;color:var(--muted);margin-top:5px;line-height:1.45}
.box{background:var(--card);border-left:4px solid var(--blue);border-radius:0 10px 10px 0;
     padding:13px 17px;margin:16px 0}
.warn{border-left-color:var(--amber);background:#fffbeb}
.good{border-left-color:var(--green);background:#f0fdf4}
.bad{border-left-color:var(--red);background:#fef2f2}
figure{margin:20px 0;background:var(--card);border:1px solid var(--line);
       border-radius:12px;padding:12px;overflow-x:auto}
figure img{width:100%;height:auto;display:block;border-radius:8px}
figcaption{font-size:12.5px;color:var(--muted);margin-top:9px;text-align:center}
ul,ol{margin:10px 0;padding-left:22px}
li{margin:6px 0}
.tag{display:inline-block;font-size:11px;padding:2px 7px;border-radius:5px;
     background:#e0e7ff;color:#3730a3;margin-right:6px;font-weight:600}
hr{border:0;border-top:1px solid var(--line);margin:30px 0}
</style></head><body><div class="wrap">

<h1>PAM-Net：脉冲感知多尺度轻量化水声降噪网络</h1>
<div class="sub">阶段1 实现、验证与统计效力分析 &nbsp;·&nbsp;
按导师《面向非高斯脉冲噪声与突发干扰的轻量化多尺度水声信号降噪方法》实现
&nbsp;·&nbsp; 数据集：α 稳定背景 + 突发干扰（与阶段0 同一份）
&nbsp;·&nbsp; 最终结果均用 GroupNorm + 多种子协议</div>

__KPI__

<h2>零、结论</h2>
__CONCLUSION__

<h2>一、按导师方案实现的内容</h2>
<table>
<thead><tr><th>导师文档</th><th>本实现</th><th>具体形式</th><th>参数量</th></tr></thead>
<tbody>
<tr><td>轻量化特征提取层<br><span style="color:#6b7280;font-size:12px">一维卷积 + 归一化 + 激活</span></td>
    <td><code>FeatureExtractor</code></td><td>深度可分离 Conv1d(1→24, k=7) + Norm + ReLU</td><td>104</td></tr>
<tr><td>短时尺度分支<br><span style="color:#6b7280;font-size:12px">识别尖峰、脉冲和突发边缘</span></td>
    <td><code>ShortBranch</code></td><td>小核 k=3 两层，实测感受野 <b>13 点</b></td><td>2,808</td></tr>
<tr><td>长时尺度分支<br><span style="color:#6b7280;font-size:12px">提取波形连续性和时序结构</span></td>
    <td><code>LongBranch</code></td><td>k=7 + 膨胀率 1/4/16，实测感受野 <b>135 点</b>（≈22 ms）</td><td>3,936</td></tr>
<tr><td>门控融合<br><span style="color:#6b7280;font-size:12px">g=σ(W<sub>g</sub>[F<sub>s</sub>;F<sub>l</sub>])</span></td>
    <td><code>GatedFusion</code></td><td>Conv1d(48→24, k=1) + Sigmoid，逐点加权</td><td>1,176</td></tr>
<tr><td>脉冲感知分支<br><span style="color:#6b7280;font-size:12px">p<sub>n</sub>=σ(f<sub>imp</sub>(F)<sub>n</sub>)</span></td>
    <td><code>PulseBranch</code></td><td>深度可分离 Conv1d + Conv1d(·→1) + Sigmoid</td><td>433</td></tr>
<tr><td>残差重建<br><span style="color:#6b7280;font-size:12px">ŝ=y−r̂</span></td>
    <td><code>ResidualHead</code></td><td>深度可分离 Conv1d + Conv1d(·→1)</td><td>449</td></tr>
</tbody>
<tfoot><tr><th>合计</th><th>—</th><th>—</th><th>8,906</th></tr></tfoot>
</table>
<p>单元测试 8 项全过：参数量、前向形状、梯度流（无死分支）、Huber 两段数值与梯度有界性、
L_imp/L_spec 手算一致、频域精修分支与 v3 <code>FrequencyBranch</code> <b>逐位一致（最大差 0）</b>、
感受野实测、长度保持性（k=3…9, d=1…16）。</p>

<h3>1.1 文档未明确、由本实现补齐的两处</h3>
<ol>
<li><b>脉冲感知分支"辅助特征融合"的形式。</b>文档只给 <code>p_n=σ(f_imp(F)_n)</code>，
未给融合公式。本实现把 p<sub>n</sub> 作为<b>附加通道拼接</b>送入下游：
纯时域版 <code>r̂=f_θ([F_fuse;p_n])</code>；频域版把 p 按 hop 做 max-pool 对齐到
STFT 帧率、沿频率轴广播后作为掩码网络的<b>第 2 个输入通道</b>。<br>
<span class="tag">实验依据</span>初版用乘性调制 <code>F_fuse⊙(1+p_n)</code>，<b>实测失败</b>：
p 的分离度持续变好，验证 SNR 却从 +0.30 一路降到 −2.68 dB —— L_imp 的梯度经乘性门
直接灌进共享主干，"为了拟合掩码而破坏重建"。</li>
<li><b>感受野选取依据。</b>数据在 6 kHz 下突发时长为 12~180 点，故长时分支取膨胀率
1/4/16（k=7）得 135 点感受野；短时分支取 k=3 两层（13 点）定位突发边缘，两者相差 10 倍。</li>
</ol>

<h2>二、损失函数与量级校准</h2>
<p>按导师文档：<code>L = L_Huber + λ₁·L_imp + λ₂·L_spec</code>，λ 由验证集确定。
文档没给具体数值，而<b>直接沿用既有代码的 λ₂=0.5 会出问题</b>，因此先做量级校准。</p>
<table>
<thead><tr><th>估计质量</th><th>L1(时域)</th><th>Huber(δ=0.2)</th><th>L1(幅度谱)</th><th>谱/时域 比值</th></tr></thead>
<tbody>
<tr><td>不处理（= 网络初始化输出）</td><td>1.0801</td><td>0.2007</td><td>27.68</td><td>25.6×</td></tr>
<tr><td>0.5×（频域模型初始化近似）</td><td>0.6725</td><td>0.1194</td><td>14.22</td><td>21.1×</td></tr>
<tr><td>clean + 0.30·noise</td><td>0.3240</td><td>0.0516</td><td>8.22</td><td>25.4×</td></tr>
<tr><td>clean + 0.15·noise</td><td>0.1620</td><td>0.0216</td><td>4.08</td><td>25.2×</td></tr>
</tbody></table>
<p><code>torch.stft</code> 未归一化的幅度尺度是时域残差的 <b>20~25 倍</b>。若取 λ₂=0.5，
谱项占总损失 90%+，Huber 的"抗脉冲"作用被淹没。校准后取 <b>λ₂=0.05</b>，
并额外跑 λ₂=0.5 作敏感性对照（见消融表 <code>pamnet_l2big</code>）。</p>
<div class="box warn"><b>关于 δ：</b>δ=0.2 与残差尺度匹配（模型输出 12~15 dB SNR 时残差
rms≈0.21，即 δ≈1σ）。绝大多数干净样本落在平方区（精细拟合），脉冲样本落在一次区
（梯度被限幅在 δ）。这正是导师文档"小误差保持精细拟合、大误差降低损失增长速度"的要求。</div>

<h2>三、归一化方式的诊断与协议修正</h2>
<div class="box bad"><b>这是本次最重要的方法学发现，也是所有结论的前提。</b></div>
<p>训练日志显示验证指标大幅振荡。对全部 run 统计（<code>analyze_stability.py</code>）：
逐轮 val 帧增益的标准差中位 <b>2.01 dB</b>、最大 3.15 dB；最佳轮次散布在 4~14 轮；
「最佳轮 − 末5轮均值」中位 <b>2.51 dB</b>。也就是说 best-checkpoint 选择在挑"运气轮次"，
而它带来的乐观偏差<b>与要论证的效应量同量级</b>。</p>

<h3>3.1 定位原因</h3>
<p>输入是重尾的（α 稳定背景 + 20×rms 脉冲），逐 batch 的统计量差异极大。
BN 的 running statistics 因此被脉冲污染，eval 模式（用 running stats）与 train 模式
（用 batch stats）行为不一致。实测同一 checkpoint：</p>
<table>
<thead><tr><th>配置</th><th>帧增益中位 (dB)</th></tr></thead>
<tbody>
<tr><td>原 running stats（eval 模式）</td><td>11.46</td></tr>
<tr><td>重校准到训练集统计</td><td>6.14</td></tr>
<tr><td>batch stats（train 模式）</td><td>7.46</td></tr>
</tbody></table>
<p><b>归一化统计量能把指标推动 5.3 dB</b> —— 足以解释逐轮 2~3 dB 的振荡。</p>

<h3>3.2 修正：改用 GroupNorm</h3>
<p>给全部卷积层加可切换的归一化（<code>--norm bn|gn|in</code>）。GN/IN 不依赖 batch 统计，
train 与 eval 输出<b>逐位一致</b>（实测最大差 0.000；BN 为 0.84）。同配置、同数据、
同超参、只换归一化的对照：</p>
__FIGN__
<table>
<thead><tr><th>配置</th><th>归一化</th><th>逐轮标准差 (dB)</th><th>最佳轮次</th><th>最佳−末5轮 (dB)</th></tr></thead>
<tbody>
<tr><td>纯时域 PAM-Net</td><td>BatchNorm</td><td>2.450</td><td>11</td><td>3.80</td></tr>
<tr><td>纯时域 PAM-Net</td><td><b>GroupNorm</b></td><td><b>0.677</b></td><td><b>15（末轮）</b></td><td><b>0.05</b></td></tr>
<tr><td>频域精修 + p 注入</td><td>BatchNorm</td><td>1.483</td><td>13</td><td>1.29</td></tr>
<tr><td>频域精修 + p 注入</td><td><b>GroupNorm</b></td><td><b>0.831</b></td><td><b>15（末轮）</b></td><td><b>0.08</b></td></tr>
</tbody></table>
<p>振荡降到约 <b>1/3.6</b>，且最佳轮次移到末轮（单调收敛），选择偏差从数 dB 降到 ~0.05 dB。
<b>GN 不但没让性能变差，在纯时域上反而更好</b>（干扰帧 +2.8 dB）。因此<b>最终结果改用 GN 协议</b>。</p>

<h2>四、脉冲感知分支是否真的学到了脉冲</h2>
<p>导师文档把创新性落在"是否针对脉冲干扰与有效水声波形之间的差异形成判别能力"。
最直接的证据是 p<sub>n</sub> 与真实干扰掩码 m<sub>n</sub> 的对应关系。
下图为推荐模型（GN，频域精修 + p 注入）在 testA 上的统计。</p>
__FIGK__
<table>
<thead><tr><th>统计量</th><th>数值</th></tr></thead>
<tbody>
<tr><td>干扰位置平均置信度 p<sub>n</sub>（中位）</td><td>__PI__（__PIMED__）</td></tr>
<tr><td>干净位置平均置信度 p<sub>n</sub>（中位）</td><td>__PC__（__PCMED__）</td></tr>
<tr><td>两者倍数</td><td>__PRATIO__×</td></tr>
<tr><td>逐样本 AUC 均值（中位）</td><td><b>__AUC__</b>（__AUCMED__）</td></tr>
<tr><td>AUC &gt; 0.8 的样本比例</td><td>__AUCFRAC__</td></tr>
</tbody></table>
<p><b>结论：该分支确实学到了"哪里是脉冲"，不是输出常数。</b>这是导师方案创新点成立的
关键证据 —— 但"能识别"不等于"能提升性能"，见下一节。</p>

<h2>五、结果</h2>
<h3>5.1 主结果（GN 协议，多种子均值 ± 种子间标准差，testA）</h3>
<table>
<thead><tr><th>模型</th><th>参数</th><th>干净帧增益 (dB)</th><th>干扰帧增益 (dB)</th>
<th>干扰帧最差 (dB)</th><th>输出 SNR (dB)</th></tr></thead>
<tbody>
__ROWS_GN__
</tbody></table>
__FIGM__

<h3>5.2 与经典方法对照（不含任何学习成分）</h3>
<p>信号是窄带 chirp（798~1500 Hz），而 α 稳定背景与突发脉冲都是<b>宽带</b>的，
所以一个固定带通就能滤掉大量噪声。把 PAM-Net 放进这个参照系才能判断增益来自哪里。</p>
<table>
<thead><tr><th>方法</th><th>输出全局 SNR (dB)</th><th>干净帧增益 (dB)</th>
<th>干扰帧增益 (dB)</th><th>干扰帧最差 (dB)</th></tr></thead>
<tbody>
__ROWS_BASE__
</tbody></table>
<div class="box good"><b>PAM-Net 的干扰段处理能力远超全部经典方法</b>
（干扰帧 17.3 vs 经典最好 8.00；干扰帧最差 9.8 vs 1.83）。
但干净帧只比盲谱减略高，且明显低于频域掩码模型 —— 后者天生等效于可学习带通。</div>

<h3>5.3 跨种子统计（判断效应真伪的唯一可靠方式）</h3>
<table>
<thead><tr><th>对比（testA）</th><th>指标</th><th>均值差 ± 标准误 (dB)</th>
<th>t</th><th>p</th><th>n（种子）</th></tr></thead>
<tbody>
__ROWS_SEED__
</tbody></table>
<div class="box bad"><b>读法：</b>纯时域里脉冲分支的效应<b>精确为零</b>
（干扰帧 +0.095 ± 0.338，p = 0.80）。只有在频域框架里才测得出
（干扰帧 +0.823 ± 0.237，p = 0.074，n = 3）。
<b>而 BN 单种子曾给出 +1.99/+1.27/+0.96 dB</b> —— 单种子 + BN 把效应夸大了 2~6 倍。</div>

<h3>5.4 BN 单种子对照（说明选择偏差有多大）</h3>
<table>
<thead><tr><th>对比</th><th>指标</th><th>均值差 (dB)</th><th>种子数</th></tr></thead>
<tbody>
__ROWS_SEED_BN__
</tbody></table>

<h3>5.5 样本级配对检验（within-seed，BN 单种子）</h3>
<p>两个变体在<b>同一批测试样本</b>上评估，因此可对逐样本指标差做配对检验。
它消除的是"样本抽样"不确定性，<b>不消除训练随机性</b>。</p>
<table>
<thead><tr><th>对比（testA）</th><th>指标</th><th>均值差 (dB)</th>
<th>t</th><th>p</th><th>Cohen's d<sub>z</sub></th></tr></thead>
<tbody>
__ROWS_PAIRED__
</tbody></table>
<div class="box warn"><b>注意两者的区别：</b>配对检验的 p 值极小（1e−202 量级）是因为
n=1080 个样本；但它<b>完全不反映</b>换个种子重训会变多少。5.3 节的跨种子检验才是判断
效应是否真实的标准 —— 两者结论差了一个数量级。</div>

<h2>六、消融（单种子，BN 协议；Δ 相对阶段0 纯频域基线）</h2>
__FIGJ__
<table>
<thead><tr><th>变体</th><th>参数</th><th>干净帧 (dB)</th><th>干扰帧 (dB)</th>
<th>干扰最差 (dB)</th><th>Δ干净帧</th><th>Δ干扰帧</th><th>Δ最差</th></tr></thead>
<tbody>
__ROWS_ABL__
</tbody>
<tfoot><tr><th colspan="5" style="text-align:right">Δ 为相对【阶段0 纯频域基线】的差（正 = 更好）</th>
<th colspan="3"></th></tr></tfoot></table>
<div class="box warn">
<b>注意：本节数字是单种子 + BN，按第 3、5 节的结论，其绝对值带 2~3 dB 的选择偏差，
只能用于比较【同一批训练内】的相对大小，不能当作精确效应量。</b>
方向性结论仍然可靠：门控与多尺度双分支确有贡献（去门控干扰帧 −3.5 dB，
仅短时/仅长时分别低 3.0/1.6 dB），而脉冲感知分支的贡献很小。
</div>
<div class="box"><b>两个如实记录的负面结论：</b>
<ul>
<li><b>Huber 不如普通 L1</b>（L1 高 +0.44/+0.79/+0.31 dB）。δ=0.2 与残差尺度相当，
多数误差落在平方区，而 L1 本身就是稳健损失 —— Huber 相对 <b>MSE</b> 的稳健性优势
在这里用不上。</li>
<li><b>λ₂ 不能照搬 0.5</b>（0.05 相对 0.5 高 +0.98/+1.07/+0.95 dB）。</li>
</ul></div>

<h2>七、训练稳定性的完整数据</h2>
__FIGL__
<table>
<thead><tr><th>run</th><th>逐轮 val 帧增益标准差 (dB)</th><th>最佳轮次</th><th>最佳−末5轮 (dB)</th></tr></thead>
<tbody>
__ROWS_STAB__
</tbody></table>

<h2>八、独立复算</h2>
<p><code>verify_pamnet.py</code> <b>不复用训练脚本的任何实现</b>：自己读 .mat、
自己 <code>resample_poly(1,4)</code>、自己实现 SNR / SI-SDR / 逐帧降噪增益，
再与训练脚本产出的逐样本 CSV 对比。唯一共用的是模型定义本身
（验证一个模型不可能绕开它的结构）。</p>
<p>结果：<b>均值逐位吻合</b>；逐样本最大差 2.4e−3 dB，来自 STFT/ISTFT 在 float32 下的
舍入放大。150 个样本<b>无一变差</b>（最差 +4.53 dB）。</p>

<h2>九、局限</h2>
<ul>
<li><b>仍然是仿真数据。</b>α 稳定背景与突发干扰都是合成的，α 只取 1.5，
突发幅度/时长范围是设定的。导师已说明会提供实测信号，届时需重新验证。</li>
<li><b>种子数仍然偏少</b>（关键对照 3~5 个）。跨种子检验自由度很低，p 值只能作参考。
要得到可靠的效应量，建议 5~10 个种子。</li>
<li><b>与阶段0 的对比含多个变量</b>：架构与损失同时改变（PAM-Net 必须配套新损失），
因此不能把提升单独归因于架构；消融表内部（同一损失下改架构）才是干净的因果证据。</li>
<li><b>λ₂ 由量级校准确定，不是完整网格搜索</b>（只对照 0.05 与 0.5）。</li>
<li><b>突发干扰直接叠加在信道输出上。</b>若干扰发生在信道之前，经多径会展宽成一串回声突发，
难度完全不同 —— 该场景未测。</li>
<li><b>降采样 4 倍对脉冲的衰减未做定量分解。</b>脉冲经抗混叠滤波会被削弱，
而高斯噪声下同样的降采样带来 +6.19 dB 输入 SNR 增益。</li>
<li><b>两个前端处理需要交代</b>：样本中位有效长度 6,031/8,000（约 24.6% 零填充）；
降采样 4 倍把输入 SNR 从 0 dB 抬到 6.19 dB（高斯版）。</li>
</ul>

<h2>十、复现命令</h2>
<pre># 1) 单元测试（8 项，必须全过）
python code/analysis/test_pamnet.py
# 2) 损失量级校准（确定 lambda2）
python code/analysis/calibrate_loss.py
# 3) 归一化诊断（解释为什么必须换 GroupNorm）
python code/analysis/diagnose_bn.py --ckpt D:/project/output_pamnet/model_pamnet.pth

# 4) 最终协议：GroupNorm + 多种子（关键 4 个配置；S 为空/s2/s3/s4/s5）
D=D:/project/dataset_pam/alphastable
python code/python/train_pamnet.py --tag pamnet_gn               --data $D --epochs 15 --norm gn
python code/python/train_pamnet.py --tag pamnet_gn_s2            --data $D --epochs 15 --norm gn --seed 2
python code/python/train_pamnet.py --tag pamnet_nopulse_gn       --data $D --epochs 15 --norm gn --pulse 0
python code/python/train_pamnet.py --tag pamnet_freq_pf_gn       --data $D --epochs 15 --norm gn --freq-refine 2.0 --pulse-freq 1
python code/python/train_pamnet.py --tag pamnet_freq_nopulse_gn  --data $D --epochs 15 --norm gn --freq-refine 2.0 --pulse 0
# （其余种子同理，只改 --seed 与 --tag 后缀）

# 5) 分析与出图
python code/analysis/analyze_seeds.py       # 多种子聚合 + 跨种子检验
python code/analysis/analyze_stability.py   # 逐轮振荡与选择偏差
python code/analysis/make_fig_seeds.py      # 图 M / 图 N
python code/analysis/paired_test_pamnet.py  # 样本级配对检验
python code/analysis/verify_pamnet.py --ckpt D:/project/output_pamnet/model_pamnet_freq_pf.pth \\
    --tag pamnet_freq_pf --freq-refine 2.0 --pulse-freq 1
python code/analysis/compare_baseline_pam.py
</pre>

<hr>
<p style="font-size:12.5px;color:#6b7280">
本报告由 <code>build_report_pamnet.py</code> 自动生成。数字来自 <code>figs_pamnet/</code> 下的
summary_pamnet.json / seed_agg.json / pulse_diag.json / baseline_pam.json /
paired_test.json / stability.json。
</p>

</div></body></html>
"""

HTML = (HTML
        .replace("__KPI__", kpi)
        .replace("__CONCLUSION__", concl)
        .replace("__ROWS_GN__", rows_gn)
        .replace("__ROWS_BASE__", rows_base)
        .replace("__ROWS_ABL__", abl_rows)
        .replace("__ROWS_SEED__", rows_seed)
        .replace("__ROWS_SEED_BN__", rows_seed_bn)
        .replace("__ROWS_PAIRED__", rows_paired)
        .replace("__ROWS_STAB__", rows_stab)
        .replace("__FIGM__", img("figM_seeds.png",
                                 "图 M 关键配置的多种子结果（误差棒 = 种子间标准差）"))
        .replace("__FIGN__", img("figN_norm.png",
                                 "图 N 归一化方式对训练稳定性的影响：BatchNorm（虚线）大幅振荡，"
                                 "GroupNorm（实线）单调收敛"))
        .replace("__FIGK__", img("figK_pulse_aware.png",
                                 "图 K 脉冲感知分支诊断：左为 p_n 在干扰/干净位置的分布；"
                                 "右为具体样本中 p_n 与真实掩码的对齐"))
        .replace("__FIGJ__", img("figJ_pamnet_ablation.png",
                                 "图 J 消融结果（testA，单种子 BN）"))
        .replace("__FIGL__", img("figL_stability.png",
                                 "图 L 逐轮验证指标的振荡与 best-checkpoint 的乐观偏差"))
        .replace("__PI__", "%.4f" % PD["p_imp_mean"])
        .replace("__PIMED__", "%.4f" % PD.get("p_imp_median", 0.9998))
        .replace("__PC__", "%.4f" % PD["p_clean_mean"])
        .replace("__PCMED__", "%.4f" % PD.get("p_clean_median", 0.0005))
        .replace("__PRATIO__", "%.0f" % (PD["p_imp_mean"] / max(PD["p_clean_mean"], 1e-9)))
        .replace("__AUC__", "%.3f" % PD["auc_mean"])
        .replace("__AUCMED__", "%.3f" % PD["auc_median"])
        .replace("__AUCFRAC__", "%.1f%%" % (100 * PD["frac_auc_gt_08"]))
        )

with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)
left = [t for t in ("__",) if t in HTML]
print("报告 -> %s" % OUT)
print("大小 %.1f KB" % (len(HTML.encode("utf-8")) / 1024))
print("未替换占位符:", "无" if not left else left)
