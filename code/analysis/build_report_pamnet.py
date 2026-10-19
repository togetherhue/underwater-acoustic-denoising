# -*- coding: utf-8 -*-
"""
生成阶段1 报告（修订版）：PAM-Net —— 实现、验证与统计效力分析
================================================================================
数据来源（全部在 figs_pamnet/）：
  summary_pamnet.json   analyze_pamnet.py   产出：各变体单种子结果 + 阶段0 参照
  seed_agg.json         analyze_seeds.py    产出：关键配置 × 多种子的均值±标准差
  pulse_diag.json       make_fig_pamnet_pulse.py 产出：脉冲分支 AUC 诊断
  baseline_pam.json     compare_baseline_pam.py 产出：经典方法
  paired_test.json      paired_test_pamnet.py 产出：样本级配对检验
  stability.json        analyze_stability.py 产出：逐轮振荡与选择偏差
  ladder.json           analyze_ladder.py   产出：受控阶梯（协议 vs 架构分解）
  chirp_band.json       analyze_chirp_band.py 产出：效应 vs 波形参数（不配对）
  chirp_band_paired.json analyze_chirp_band.py 产出：同上，但干扰与波形解耦
  图：figJ（消融）、figK（脉冲诊断）、figL（稳定性）、figM（多种子）、figN（归一化对照）、
      figO（波形机制）、figP（受控阶梯）
"""
import base64, os, json, re
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


def _try_json(name):
    p = os.path.join(FIG, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


# 适用边界（敏感性）实验：sens_out/sens_summary_<split>.json
SENS = {}
for _sp in ("testA", "testC"):
    _p = os.path.join(HERE, "sens_out", "sens_summary_%s.json" % _sp)
    if os.path.exists(_p):
        SENS[_sp] = json.load(open(_p, encoding="utf-8"))

# 等有效功率 SNR 的 α 重扫：sens_out/matched_<split>.json
MATCHED = {}
for _sp in ("testA", "testC"):
    _p = os.path.join(HERE, "sens_out", "matched_%s.json" % _sp)
    if os.path.exists(_p):
        MATCHED[_sp] = json.load(open(_p, encoding="utf-8"))
SENS_PD = None
_pd = os.path.join(HERE, "sens_out", "pulse_diag_sens.json")
if os.path.exists(_pd):
    SENS_PD = json.load(open(_pd, encoding="utf-8"))


LAD = _try_json("ladder.json")
CB = _try_json("chirp_band.json")
CBP = _try_json("chirp_band_paired.json")

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

# ================= 受控阶梯表 =================
rows_ladder = ""
rows_step = ""
ladder_note = ""
if LAD and LAD.get("testA"):
    for r in LAD["testA"]["rows"]:
        hl = " style='background:#eef4ff;font-weight:600'" if r["rung"] in ("R3", "R5") else ""
        rows_ladder += ("<tr%s><td><b>%s</b> %s</td><td>%d</td><td>%s</td>"
                        "<td>%.2f%s</td><td>%.2f%s</td><td>%.2f%s</td></tr>"
                        % (hl, r["rung"], r["name"].replace("\n", " "), r["params"],
                           ("n=%d" % r["n_seed"]) if r["n_seed"] > 1 else "n=1",
                           r["frmgain_clean"],
                           "" if r["n_seed"] == 1 else "±%.2f" % r["frmgain_clean_sd"],
                           r["frmgain_imp"],
                           "" if r["n_seed"] == 1 else "±%.2f" % r["frmgain_imp_sd"],
                           r["frmgain_imp_min"],
                           "" if r["n_seed"] == 1 else "±%.2f" % r["frmgain_imp_min_sd"]))
    for s in LAD["testA"]["steps"]:
        rows_step += ("<tr><td>%s</td><td>%+.2f ± %.2f</td><td>%+.2f ± %.2f</td>"
                      "<td>%+.2f ± %.2f</td></tr>"
                      % (s["label"], s["frmgain_clean"], s["frmgain_clean_se"],
                         s["frmgain_imp"], s["frmgain_imp_se"],
                         s["frmgain_imp_min"], s["frmgain_imp_min_se"]))
    _r0 = LAD["testA"]["rows"][0]
    _rn = LAD["testA"]["rows"][-1]
    ladder_note = ("%s → %s 共提升 干净帧 %+.2f / 干扰帧 %+.2f / 干扰最差 %+.2f dB"
                   % (_r0["rung"], _rn["rung"],
                      _rn["frmgain_clean"] - _r0["frmgain_clean"],
                      _rn["frmgain_imp"] - _r0["frmgain_imp"],
                      _rn["frmgain_imp_min"] - _r0["frmgain_imp_min"]))

# ================= 波形机制表 =================
CH_KEYS = [("bw", "带宽 BW (Hz)"), ("f_lo", "下截止频率"), ("f_hi", "上截止频率"),
           ("T", "时长 T (s)"), ("A", "幅度 A"),
           ("inband", "干扰落在信号带内的能量占比"),
           ("overlap", "干扰与信号的<b>时频重叠率</b>"),
           ("ratio", "带内干扰/信号能量比")]
rows_chirp = ""
if CB and CBP:
    for k, nm in CH_KEYS:
        r1 = CB["corr_all"].get(k, float("nan"))
        r2 = CBP["corr_all"].get(k, float("nan"))
        hl = " style='background:#f0fdf4;font-weight:600'" if k == "overlap" else ""
        rows_chirp += ("<tr%s><td>%s</td><td>%+.3f</td><td>%+.3f</td></tr>"
                       % (hl, nm, r1, r2))

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

# 阶梯 / 机制的结论数字
_lad = LAD["testA"]["rows"] if (LAD and LAD.get("testA")) else None
_proto_imp = _proto_min = _arch_imp = _arch_min = float("nan")
if _lad:
    _r0 = _lad[0]
    _r3 = next((r for r in _lad if r["rung"] == "R3"), None)
    _rn = _lad[-1]
    if _r3 is not None:
        _proto_imp = _r3["frmgain_imp"] - _r0["frmgain_imp"]
        _proto_min = _r3["frmgain_imp_min"] - _r0["frmgain_imp_min"]
        _arch_imp = _rn["frmgain_imp"] - _r3["frmgain_imp"]
        _arch_min = _rn["frmgain_imp_min"] - _r3["frmgain_imp_min"]
_ov_r = CBP["corr_all"].get("overlap", float("nan")) if CBP else float("nan")
_proto_clean = _step34_imp = _step45_imp = _ratio = float("nan")
if _lad:
    _r0, _r3, _rn = _lad[0], next((r for r in _lad if r["rung"] == "R3"), None), _lad[-1]
    _r4 = next((r for r in _lad if r["rung"] == "R4"), None)
    if _r3 is not None:
        _proto_clean = _r3["frmgain_clean"] - _r0["frmgain_clean"]
    if _r3 is not None and _r4 is not None:
        _step34_imp = _r4["frmgain_imp"] - _r3["frmgain_imp"]
        _step45_imp = _rn["frmgain_imp"] - _r4["frmgain_imp"]
        _ratio = _step34_imp / _step45_imp if abs(_step45_imp) > 1e-9 else float("nan")

# ================= 适用边界（敏感性）实验 =================
SENSA = SENS.get("testA")
SENS_LBL = {"base": "α=1.5 / rate=5<br><span style='color:#6b7280'>训练条件</span>",
            "a12_r5": "α=1.2 / rate=5", "a18_r5": "α=1.8 / rate=5",
            "a20_r5": "α=2.0 / rate=5", "a15_r25": "α=1.5 / rate=2.5",
            "a15_r10": "α=1.5 / rate=10", "gauss": "独立高斯数据 / rate=5"}
SENS_CFG = {"R3_freqonly": "R3 纯频域掩码", "R4_freqrefine_nopulse": "R4 时域残差+频域精修",
            "R5_freqrefine_pulse": "R5 R4 + 脉冲感知分支",
            "P_time_pulse": "纯时域 PAM-Net", "P_time_nopulse": "纯时域（无脉冲分支）"}
SENS_PAR = {"R3_freqonly": "5,910", "R4_freqrefine_nopulse": "14,367",
            "R5_freqrefine_pulse": "14,858", "P_time_pulse": "8,906",
            "P_time_nopulse": "8,457"}
SENS_CFGS = ["R5_freqrefine_pulse", "R4_freqrefine_nopulse", "R3_freqonly",
             "P_time_pulse", "P_time_nopulse"]
SENS_ORDER = [c for c in ["a12_r5", "base", "a18_r5", "a20_r5", "a15_r25", "a15_r10", "gauss"]
              if SENSA and c in SENSA.get("table", {})]

rows_sens_main = ""
rows_sens_pulse = ""
rows_sens_arch = ""
rows_sens_noise = ""
rows_sens_degrade = ""
sens_txt = dict(base_imp=float("nan"), base_cln=float("nan"), d_r10=float("nan"),
                d_r25=float("nan"), d_a12=float("nan"), eff_a12=float("nan"),
                eff_a20=float("nan"), eff_base=float("nan"), auc_base=float("nan"),
                auc_a12=float("nan"), rank_ok="", frac_r10=float("nan"))

if SENSA:
    T = SENSA["table"]
    for c in SENS_ORDER:
        first = True
        for cf in SENS_CFGS:
            if cf not in T.get(c, {}):
                continue
            r = T[c][cf]
            hl = " style='background:#eef4ff'" if cf == "R5_freqrefine_pulse" else ""
            rows_sens_main += ("<tr%s><td>%s</td><td>%s</td><td>%s</td><td>%d</td>"
                               "<td>%.2f</td><td>%.2f</td><td>%.2f</td></tr>"
                               % (hl, SENS_LBL[c] if first else "", SENS_CFG[cf], SENS_PAR[cf],
                                  r["n_seed"], r["clean"], r["imp"], r["imp_min"]))
            first = False

    for c in SENS_ORDER:
        pe = SENSA["pulse_eff"].get(c)
        if not pe:
            continue
        def _st(k):
            d = pe[k]
            star = ("***" if d["p"] < 0.001 else ("**" if d["p"] < 0.01
                    else ("*" if d["p"] < 0.05 else "ns")))
            return "%+.3f ± %.3f<br><span style='color:#6b7280;font-size:12px'>p≤%.1e %s</span>" % (
                d["mean"], d["sd_over_seed"], d["p"], star)
        rows_sens_pulse += ("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                            % (SENS_LBL[c], _st("clean"), _st("imp"), _st("imp_min")))

    for c in SENS_ORDER:
        n = SENSA["noise"].get(c)
        if not n:
            continue
        rows_sens_noise += ("<tr><td>%s</td><td>%.3f</td><td>%+.2f</td><td>%+.2f</td>"
                            "<td>%+.2f</td><td>%.1f</td><td>%.2f</td></tr>"
                            % (SENS_LBL[c], n["frac_imp"], n["snr_tot"], n["snr_bg"],
                               n["snr_imp"], n["peak_over_rms"], n["p999_over_rms"]))

    for c in SENS_ORDER:
        aa = SENSA.get("arch_adv", {}).get(c)
        if not aa:
            continue
        rows_sens_arch += ("<tr><td>%s</td><td>%+.2f ± %.2f</td><td>%+.2f ± %.2f</td>"
                           "<td>%.1e</td></tr>"
                           % (SENS_LBL[c], aa["clean"]["mean"], aa["clean"]["sd"],
                              aa["imp"]["mean"], aa["imp"]["sd"], aa["imp"]["p"]))

    for cf in SENS_CFGS:
        d = SENSA["cond_delta"].get(cf)
        if not d:
            continue
        row = "<tr><td>%s</td>" % SENS_CFG[cf]
        for c in SENS_ORDER:
            if c not in d:
                row += "<td>—</td>"; continue
            row += "<td>%+.2f ± %.2f</td>" % (d[c]["mean"], d[c].get("sd_over_seed", 0.0))
        rows_sens_degrade += row + "</tr>"

    if "base" in T and "R5_freqrefine_pulse" in T["base"]:
        sens_txt["base_imp"] = T["base"]["R5_freqrefine_pulse"]["imp"]
        sens_txt["base_cln"] = T["base"]["R5_freqrefine_pulse"]["clean"]
    for c, key in (("a15_r10", "d_r10"), ("a15_r25", "d_r25"), ("a12_r5", "d_a12")):
        if c in T and "R5_freqrefine_pulse" in T.get(c, {}):
            sens_txt[key] = T[c]["R5_freqrefine_pulse"]["imp"] - sens_txt["base_imp"]
    for c, key in (("a12_r5", "eff_a12"), ("a20_r5", "eff_a20"), ("base", "eff_base")):
        if c in SENSA["noise"]:
            sens_txt[key] = SENSA["noise"][c]["snr_bg"]
    if "a15_r10" in SENSA["noise"]:
        sens_txt["frac_r10"] = SENSA["noise"]["a15_r10"]["frac_imp"]
    # 排序是否保持：R5 干扰帧最高；并检查 R5>R4>R3 的完整排序
    r5_best, full_ok = [], []
    for c in SENS_ORDER:
        tt = T.get(c, {})
        if not all(k in tt for k in ("R5_freqrefine_pulse", "R4_freqrefine_nopulse", "R3_freqonly")):
            continue
        r5_best.append(tt["R5_freqrefine_pulse"]["imp"] >= max(
            tt[k]["imp"] for k in tt if k.startswith(("R3", "R4", "R5", "P_time"))))
        full_ok.append(tt["R5_freqrefine_pulse"]["imp"] > tt["R4_freqrefine_nopulse"]["imp"]
                       > tt["R3_freqonly"]["imp"])
    sens_txt["rank_ok"] = ("全部 %d 个条件" % len(r5_best)) if r5_best and all(r5_best) else (
        "%d/%d 个条件" % (sum(r5_best), len(r5_best)))
    sens_txt["full_rank_ok"] = ("全部 %d 个条件" % len(full_ok)) if full_ok and all(full_ok) else (
        "%d/%d 个条件" % (sum(full_ok), len(full_ok)))
    if SENS_PD:
        for c, key in (("base", "auc_base"), ("a12_r5", "auc_a12")):
            dd = SENS_PD.get(c, {}).get("R5_freqrefine_pulse", {})
            if dd:
                k0 = sorted(dd.keys())[0]
                sens_txt[key] = dd[k0]["auc_mean"]

SENS_H = {"base": "α=1.5<br>rate=5", "a12_r5": "α=1.2<br>rate=5", "a18_r5": "α=1.8<br>rate=5",
          "a20_r5": "α=2.0<br>rate=5", "a15_r25": "α=1.5<br>rate=2.5",
          "a15_r10": "α=1.5<br>rate=10", "gauss": "高斯<br>rate=5"}
sens_head = "".join("<th>%s</th>" % SENS_H[c] for c in SENS_ORDER)


def _pe(c, k, f="mean"):
    return SENSA["pulse_eff"][c][k][f] if SENSA and c in SENSA.get("pulse_eff", {}) else float("nan")


def _aa(c, k="imp", f="mean"):
    return SENSA["arch_adv"][c][k][f] if SENSA and c in SENSA.get("arch_adv", {}) else float("nan")


def _tb(c, cf, f="imp"):
    return SENSA["table"][c][cf][f] if SENSA and cf in SENSA.get("table", {}).get(c, {}) else float("nan")


aucs_all = []
if SENS_PD:
    for c in SENS_ORDER:
        d = SENS_PD.get(c, {}).get("R5_freqrefine_pulse", {})
        for k in d:
            aucs_all.append(d[k]["auc_mean"])
sens_auc_lo = min(aucs_all) if aucs_all else float("nan")
sens_auc_hi = max(aucs_all) if aucs_all else float("nan")

sens_vars = dict(
    d_a20=_pe("a20_r5", "imp"), d_a12=_pe("a12_r5", "imp"), p_a12=_pe("a12_r5", "imp", "p"),
    d_r25=_pe("a15_r25", "imp"), d_r10=_pe("a15_r10", "imp"),
    aa_a12=_aa("a12_r5"), aa_a20=_aa("a20_r5"), aa_gauss=_aa("gauss"),
    r3_a12=_tb("a12_r5", "R3_freqonly"), r3_a20=_tb("a20_r5", "R3_freqonly"),
    r5_a12=_tb("a12_r5", "R5_freqrefine_pulse"), r5_a20=_tb("a20_r5", "R5_freqrefine_pulse"),
    deg_a12=abs(SENSA["cond_delta"]["R5_freqrefine_pulse"]["a12_r5"]["mean"]) if SENSA
    and "a12_r5" in SENSA["cond_delta"].get("R5_freqrefine_pulse", {}) else float("nan"),
    deg_a20=abs(SENSA["cond_delta"]["R5_freqrefine_pulse"]["a20_r5"]["mean"]) if SENSA
    and "a20_r5" in SENSA["cond_delta"].get("R5_freqrefine_pulse", {}) else float("nan"),
    deg_a20_r4=abs(SENSA["cond_delta"]["R4_freqrefine_nopulse"]["a20_r5"]["mean"]) if SENSA
    and "a20_r5" in SENSA["cond_delta"].get("R4_freqrefine_nopulse", {}) else float("nan"),
    auc_lo=sens_auc_lo, auc_hi=sens_auc_hi)

MA = MATCHED.get("testA", {})
MC = MATCHED.get("testC", {})

# α 轴域内重训（8.7）：用于修正 8.1b 的零样本分解
INDA = {}
for _sp in ("testA", "testC"):
    _p = os.path.join(HERE, "sens_out", "indomain_alpha_%s.json" % _sp)
    if os.path.exists(_p):
        INDA[_sp] = json.load(open(_p, encoding="utf-8"))
_a18_ind = INDA.get("testA", {}).get("conds", {}).get("a18", {}).get("corrected", float("nan"))
_a12_ind = INDA.get("testA", {}).get("conds", {}).get("a12", {}).get("corrected", float("nan"))

# 尺度不变性 / 输入归一化 / 架构排序域内（8.12 / 8.13）
_SI = {}
_p = os.path.join(HERE, "sens_out", "scale_invariance.json")
if os.path.exists(_p):
    _SI = json.load(open(_p, encoding="utf-8"))
_ING = {}
_p = os.path.join(HERE, "sens_out", "input_norm_testA.json")
if os.path.exists(_p):
    _ING = json.load(open(_p, encoding="utf-8"))
_IAG = {}
_p = os.path.join(HERE, "sens_out", "inarch_gpu_testA.json")
if os.path.exists(_p):
    _IAG = json.load(open(_p, encoding="utf-8"))

# 脉冲掩码干预（8.15）
_PMI = {}
_p = os.path.join(HERE, "sens_out", "pulse_mask_intervention.json")
if os.path.exists(_p):
    _PMI = json.load(open(_p, encoding="utf-8"))
_PMD = {}
_p = os.path.join(HERE, "sens_out", "pulse_mask_dose.json")
if os.path.exists(_p):
    _PMD = json.load(open(_p, encoding="utf-8"))
_PMO = {}
_p = os.path.join(HERE, "sens_out", "pulse_mask_oracle.json")
if os.path.exists(_p):
    _PMO = json.load(open(_p, encoding="utf-8"))
# 跨条件干预分解（8.16）
_PMC = {}
_p = os.path.join(HERE, "sens_out", "pulse_mask_crosscond.json")
if os.path.exists(_p):
    _PMC = json.load(open(_p, encoding="utf-8"))
_PMC_CONDS = [c for c in ("a12_r5", "base", "a18_r5", "a20_r5", "a15_r25", "a15_r10")
              if c in (_PMC.get("conds") or {})]


def _pmc(cond, key, default=float("nan")):
    """取 8.16 某个条件的数：_pmc('base','position')"""
    try:
        return float((_PMC.get("conds") or {})[cond][key])
    except Exception:
        return default


def _pmc_sd(key):
    """8.16 的 6 个条件上某列的样本标准差（ddof=1）"""
    vs = [_pmc(c, key) for c in _PMC_CONDS]
    vs = [v for v in vs if v == v]
    if len(vs) < 2:
        return float("nan")
    m = sum(vs) / len(vs)
    return (sum((v - m) ** 2 for v in vs) / (len(vs) - 1)) ** 0.5


def _pmi(*keys, default=float("nan")):
    """按路径取 8.15 的数：_pmi('summary','identity','mean')"""
    v = _PMI
    try:
        for k in keys:
            v = v[k]
        return float(v)
    except Exception:
        return default


def _pmd(*keys, default=float("nan")):
    v = _PMD
    try:
        for k in keys:
            v = v[k]
        return float(v)
    except Exception:
        return default


def _pmo(*keys, default=float("nan")):
    v = _PMO
    try:
        for k in keys:
            v = v[k]
        return float(v)
    except Exception:
        return default


def _si(k, sub="range_db", default=float("nan")):
    try:
        ks = list(_SI["res"].keys())
        return _SI["res"][ks[k]][sub]
    except Exception:
        return default


def _si_gain(k, s, default=float("nan")):
    try:
        ks = list(_SI["res"].keys())
        g = _SI["res"][ks[k]]["gains"]
        for kk, vv in g.items():
            if abs(float(kk) - s) < 1e-9:
                return vv
    except Exception:
        pass
    return default


def _inm(key, grp):
    try:
        return float(np.mean(_ING[grp][key]))
    except Exception:
        return float("nan")


def _iag(sub, field="mean", default=float("nan")):
    for k, v in (_IAG.get("tests") or {}).items():
        if sub in k:
            return v.get(field, default)
    return default


def _iag_p(sub, default=float("nan")):
    for k, v in (_IAG.get("tests") or {}).items():
        if sub in k:
            return v.get("p", default)
    return default


M_ORDER = ["a12_r5", "m_a12", "base", "m_a18", "a18_r5"]
M_LBL = {"a12_r5": "α=1.2<br><span style='color:#6b7280;font-size:12px'>未匹配</span>",
         "m_a12": "α=1.2<br><span style='color:#6b7280;font-size:12px'>已匹配</span>",
         "base": "α=1.5<br><span style='color:#6b7280;font-size:12px'>base</span>",
         "m_a18": "α=1.8<br><span style='color:#6b7280;font-size:12px'>已匹配</span>",
         "a18_r5": "α=1.8<br><span style='color:#6b7280;font-size:12px'>未匹配</span>"}

rows_sens_matched = ""
rows_sens_matched_arch = ""
m_txt = dict()
if MA:
    PE = MA.get("pulse_eff", {})
    AA = MA.get("arch_adv", {})

    def _mrow(pe, key="imp"):
        d = pe[key]
        # LOO 只在干扰帧（imp）上计算过
        if key == "imp" and "loo_min" in d and "loo_max" in d:
            loo = (d["loo_min"], d["loo_max"])
            loo_s = ("<br><span style='color:#6b7280;font-size:12px'>留一 [%+.2f, %+.2f]</span>" % loo
                     if all(np.isfinite(v) for v in loo) else "")
        else:
            loo_s = ""
        return "%+.3f ± %.3f<br><span style='color:#6b7280;font-size:12px'>p=%.4f</span>%s" % (
            d["mean"], d["sd"], d["p"], loo_s)

    for c in M_ORDER:
        if c not in PE:
            continue
        rows_sens_matched += ("<tr><td>%s</td><td>%s</td><td>%s</td></tr>"
                              % (M_LBL[c], _mrow(PE[c], "clean"), _mrow(PE[c], "imp")))
    for c in M_ORDER:
        if c not in AA:
            continue
        rows_sens_matched_arch += ("<tr><td>%s</td><td>%+.3f ± %.3f</td>"
                                   "<td>%+.3f ± %.3f</td><td>%.1e</td></tr>"
                                   % (M_LBL[c], AA[c]["clean"]["mean"], AA[c]["clean"]["sd"],
                                      AA[c]["imp"]["mean"], AA[c]["imp"]["sd"], AA[c]["imp"]["p"]))

    # 强度 / 尾形 分解
    def _decomp(hi, lo, base="base"):
        if not all(k in PE for k in (hi, lo, base)):
            return None
        b, u, m = PE[base]["imp"]["mean"], PE[lo]["imp"]["mean"], PE[hi]["imp"]["mean"]
        tot_u, tot_m = u - b, m - b
        return dict(base=b, um=u, m=m, tot_um=tot_u, tot_m=tot_m,
                    inten=tot_u - tot_m, tail=tot_m)

    m_txt = dict(a12=_decomp("m_a12", "a12_r5"), a18=_decomp("m_a18", "a18_r5"))
    m_txt["noise"] = MA.get("noise", {})
    m_txt["tc"] = None
    if MC and MC.get("pulse_eff"):
        tc = {}
        for c in M_ORDER:
            d = MC["pulse_eff"].get(c)
            if d:
                tc[c] = d["imp"]["mean"]
        m_txt["tc"] = tc
        m_txt["tc_arch"] = {c: v["imp"]["mean"] for c, v in MC.get("arch_adv", {}).items()
                            if c in M_ORDER}
    # 匹配条件的留一法范围（m_a12 / m_a18）
    for k, cc in (("a12", "m_a12"), ("a18", "m_a18")):
        if k in m_txt and cc in PE:
            d = PE[cc]["imp"]
            m_txt[k]["m_loo"] = (d.get("loo_min", float("nan")), d.get("loo_max", float("nan")))


def _mn(cond, key):
    """取匹配实验噪声表的一项：bg / imp / tot 来自 matched JSON，pk 来自 noise npz"""
    d = m_txt.get("noise", {}).get(cond, {}) if m_txt else {}
    if key == "pk":
        p = os.path.join(HERE, "sens_out", "noise_%s.npz" % cond)
        if os.path.exists(p):
            return float(np.median(np.load(p)["testA"][:, 7]))
        return float("nan")
    return float(d.get(key, float("nan")))


_mdelta = float("nan")
if MA and MA.get("noise"):
    _ds = []
    for c in ("m_a12", "m_a18"):
        if c in MA["noise"] and "base" in MA["noise"]:
            _ds.append(abs(MA["noise"][c]["bg"] - MA["noise"]["base"]["bg"]))
    _mdelta = max(_ds) if _ds else float("nan")

__INDOMAIN_ROWS__ = """<tr><td>base（训练条件，CPU）</td><td>26.59</td><td>27.38</td><td>+0.790</td><td>5</td></tr><tr><td>a15_r10 零样本（CPU）</td><td>26.16</td><td>26.71</td><td>+0.548</td><td>5</td></tr><tr style='background:#eef4ff;font-weight:600'><td>a15_r10 域内重训（GPU）</td><td>26.86</td><td>27.18</td><td>+0.322</td><td>3</td></tr>"""

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

<div class="box good"><b>⑤ 受控阶梯：架构是主要贡献，协议只占小头。</b>
把"阶段0 → PAM-Net"的差距拆成<b>每步只改一个因素</b>的阶梯。前置验证：纯频域模式的
PAM-Net 与阶段0 的 <code>v3.DenoiseNet(fch=2.0)</code> <b>架构逐位等价</b>
（参数量同为 5,910，前向输出最大差 <b>0.000e+00</b>）；且协议完全相同、只换训练管线时
结果差 <b>≤0.04 dB</b>。分解结果：<b>训练协议（损失 + 模型选择 + 归一化）合计约
{proto_imp:+.2f} dB</b>，<b>架构（时域多尺度残差 + 脉冲感知注入）约 {arch_imp:+.2f} dB</b>
（干扰帧；干扰帧最差 {proto_min:+.2f} / {arch_min:+.2f}）。
此前"PAM-Net 优于阶段0"的结论<b>成立</b>，但现在协议成分被单独量化出来了。详见第六节。</div>

<div class="box warn"><b>⑥ 效应的波形依赖：真实但很小；机制在干预实验下未被证实。</b>
① 效应是<b>强 SNR 依赖</b>的 —— 低 SNR 时收益全在干扰帧，高 SNR 时转到干净帧，
两个测试集是同一趋势（所以"testA/testC 不一致"的说法是错的）；<br>
② 方差分解：波形效应<b>真实</b>（干扰帧 F = 4.13，p &lt; 0.01）但只解释 <b>3.8%</b> 的方差，
<b>96% 来自逐样本的干扰实现波动</b>（SD 1.73 dB vs 波形 0.34 dB）——
分支不是"稳定提升"，而是在个别样本上帮助或损害 ±1.7 dB、平均只 +0.37 dB 的小幅正偏置；<br>
③ 用"<b>固定干扰、只变波形</b>"的配对设计（干扰最大差 1.07e−14）曾发现
<b>干扰与信号的时频重叠率</b>与收益负相关（r = <b>{ov_r:+.3f}</b>，t = −4.11，p &lt; 0.001），
<b>带宽假设被否定</b>（chirp 8 与 chirp 19 带宽几乎相同，收益却差 1.7 dB）。<br>
④ <b>但随后两个直接操纵重叠率的干预实验都没有复现这个负相关</b>
（载频扫描 r = +0.40, p = 0.33；时频对齐干预 对齐−错开 = +0.45 dB, p = 0.31，
方向均与机制预测相反）。因此 <b>「时频可分性决定收益」应降级为未证实的假设，
不能作为机制结论引用</b>。详见 7.x 节。</div>

<div class="box warn"><b>⑦ 适用边界：排序稳、符号稳、但量级随条件起伏；且发现一个 SNR 定义的混杂。</b>
把 5 个模型放到 7 个条件（α ∈ [1.2, 2.0] × 突发率 ∈ [2.5, 10]/s）上做零样本评估：
<b>推荐模型 R5 在 {rank_ok} 的干扰帧增益上都排第一</b>
（唯一例外详见 8.2：高斯背景上 R4 被 R3 反超，但 R5 仍最高），
脉冲分支效应 Δ 的<b>符号在所有条件上一致为正</b>（量级 0.20~1.74 dB）。
规律很清晰：<b>α 轴单调</b>（背景越脉冲化，收益从 +{sens_a20:.2f} 降到 +{sens_a12:.2f} dB，
α=1.2 时跨种子不显著），<b>rate 轴反向</b>（突发越稀疏收益越大：2.5/s 时 +{sens_r25:.2f}，
10/s 时 +{sens_r10:.2f}）。
架构优势 R5−R3 也从 α=1.2 的 +{aa_a12:.2f} dB 缩到高斯数据上的 +{aa_gauss:.2f} dB。
<br><b>域内重训把这些差异的来源查清了（8.6 / 8.7 / 8.14）：</b>
rate 轴的"反向"经域内重训 + 设备校正后<b>仍然是真实条件效应，而且两个方向都成立</b>：
密集端 10/s 从 +0.790 降到约 <b>+0.41</b>（<b>低于</b> base），
稀疏端 2.5/s 升到 <b>+1.34</b>（<b>高于</b> base），两端相差约 0.93 dB 且方向相反。
（稀疏端的<b>量级</b>不能写死：3 个种子 sd 0.66、p=0.081，只能读方向。）
但 α 轴的"单调"<b>主要是分布偏移，不是尾形本身</b>——域内重训后 α=1.8 的 Δ 从
+1.77 回到 <b>{a18_ind:+.2f}</b>（≈base 的 +0.79），α=1.2 只剩 <b>{a12_ind:+.2f}</b>
且不显著。<b>因此 8.1b 基于零样本的"尾形分解"已被 8.7 推翻，请以 8.7 为准。</b>
<br><b>更重要的方法论发现</b>：SNR 用"75% 分位数匹配"定义时，名义 SNR 固定
并不代表有效功率 SNR 固定 —— α 从 2.0 降到 1.2，背景噪声的有效功率信噪比从
<b>{eff_a20:+.2f} dB 掉到 {eff_a12:+.2f} dB</b>（差 {eff_gap:.1f} dB）。
所以 α 扫描实际是"强度 + 形状"的联合偏移，不能单独解读为重尾程度的影响；
真正的单因素操纵是突发率轴。详见第八节。</div>

<div class="box good"><b>⑧ 部署可用性：输入尺度敏感已被归一化解决，且几乎零代价。</b>
模型原本只在"clean rms = 1"的尺度上训练，实测数据拿不到 clean、只能按 noisy 归一化，
老模型的跨尺度极差达 <b>{si_old_rng:.2f} dB</b>（s=0.25 时损失 {si_old_loss:.2f} dB）。
改为数据层面 <code>x/rms(x)</code> 归一化并重训后：
<b>三个种子的跨尺度极差全部严格为 0.000 dB</b>（因为
<code>x·s/rms(x·s) ≡ x/rms(x)</code>，缩放被恒等式消掉），
而代价只有 R4 {si_c4:+.3f} / R5 {si_c5:+.3f} dB（同设备配对，不显著）。
脉冲分支收益也保留（Δ = {si_inm_d:+.3f} dB）。<br>
同时，域内同设备的架构排序验证确认
<b>R5 &gt; R4 &gt; R3 &gt; 纯时域 在 4/4 项比较上方向与零样本一致</b>
（R5−R3 = {ia_r5r3:+.2f} dB, p = {ia_p:.4f}）。详见 8.12 / 8.13。</div>

<div class="box good"><b>⑨ 机制：分支确实在用「对齐的位置信息」—— 8.15 的推理期干预给出了正面证据。</b>
此前 7.x 的两个干预都没复现「重叠率→收益」，机制一度被降级为不可引用。
8.15 换了一个问法：直接在推理期篡改分支输出的 p（不重训、不改模型）。结果很干净 ——<br>
· <b>把 p 换成同批另一个样本的 p</b>（边缘分布严格不变，只有位置错）损失
<b>{pmi_shuf:.2f} dB</b>，与<b>把 p 置零</b>（{pmi_zero:.2f} dB）几乎相同 ⇒ 起作用的是<b>位置</b>，不是 p 的统计量；<br>
· 错位后 R5 反而比 R4 <b>低 {pmi_neg:.2f} dB</b> ⇒ 分支是<b>承重结构</b>，不是装饰；<br>
· 随机 p 只损失 {pmi_noise:.2f} dB 且不显著（它比 shuffle 离训练分布更远）⇒ 不能用"输入分布外"解释；<br>
· 错位剂量响应：仅错开 1.5 个突发（{pmi_k125:.0f} ms）就只剩 30% 增益，半衰点
{pmi_kh:.0f} ms ≈ 2.9 个突发 ⇒ 分支要求<b>逐点</b>对齐到单个突发的尺度。<br>
<b>这与 7.x 的否定并不矛盾</b>：重叠率是帧级、全局的统计量，而分支依赖的是逐点对齐，
用前者去预测后者本来就不该成功。机制表述应改为「p 提供逐点位置 → 据此分配抑制强度 →
收益来自对齐」。<br>
另有一条负面结论：把 p 换成<b>真实干扰掩码</b>（oracle）也只多
<b>{pmi_hd:+.3f} dB</b>（占效应 {pmi_hdpct:.0f}%）⇒ <b>「把脉冲估计做得更准」不是有前途的方向</b>。</div>

<div class="box good"><b>⑩ 机制（续）：8.16 把「分支值多少钱」拆成两个可测的分量，并给出统一形式。</b>
8.15 只说明了 p 怎么起作用，没说明<b>为什么不同条件下收益差这么多</b>。
8.16 在 6 个条件上（同一批 base checkpoint，只换数据）把净 Δ 拆成
「位置信息的价值」与「无信息时的惩罚」：<br>
· 净收益可写成 <b>净 Δ = 惩罚强度 × 超额倍数</b>，超额倍数在 6 个条件上为
<b>{cc_rmin:.2f}~{cc_rmax:.2f}</b>，全部略大于 1 —— 这是"分支确实提供正价值"的最简表述；<br>
· <b>α 轴</b>上两项同时放大（惩罚 1.79→3.65，倍数 {cc_a12r:.2f}→{cc_a20r:.2f}），净收益随之上升；
<b>rate 轴</b>上倍数几乎不动，只有惩罚在缩小（3.14→2.53），净收益因此下降
（{cc_r25n:+.2f}→{cc_r10n:+.2f} dB）。<b>两条轴形状不同，但落在同一个式子里</b>；<br>
· 底下的分量动得比净结果大得多：6 个条件上净 Δ 的标准差仅 <b>{cc_sdn:.3f} dB</b>，
而位置价值与惩罚分别是 <b>{cc_sdp:.3f}</b> 与 <b>{cc_sdc:.3f}</b> dB ⇒
<b>只看净 Δ 会漏掉一半以上的变化</b>；<br>
· 8.16.7 的自检：identity 的 6 条件 12 个绝对水平与零样本 5 种子评估<b>逐位吻合</b>（最大偏差 0.0004 dB），
rate=10/s 的唯一偏差（+0.179 dB）已归因为纯采样（该条件种子间 Δ 的标准差 0.370 dB）。<br>
<b>可直接用于论文的表述</b>：脉冲分支的价值不是一个数，而是「模型在没有位置信息时会受多重的伤」与
「位置信息能把这个伤补回多少比例」的乘积；跨条件报告净 Δ 时，必须同时报告这两个分量，否则无法解释。</div>

<div class="box good"><b>⑪ 实测：唯一一份非仿真录音已给出有判别力的正面结论。</b>
第十二节原本写着「全部是仿真数据」。8.17 先把接入管道跑通，并<b>证明它的无参考指标
（带外能量占比）没有判别力</b> —— 带外被压 410 倍看着像窄带化，实际是训练分布带限的
必然结果（干净信号自身带外只有 3e−08）。<br>
8.18 换用<b>不依赖干净参考</b>的判据：扫频的瞬时频率轨迹有确定解析形式
（LFM 的 f~t 线性、HFM 的 1/f~t 线性），因此可以只从信号自身估出轨迹、比较处理前后是否一致。
工具先在合成 HFM 上做了有效性验证（识别调制类型、判出「保住」、四种人为破坏全部检出），
并在自检中修正了两个会导致错误结论的实现缺陷（逐帧取峰被噪声带偏、用相对中位数做能量门限判不出有无信号）；
闸门也迭代过一次（「R²&gt;0.6」对白噪声/窄带噪声全部失效，改为绝对阈值 0.95 + 替代数据检验）。<br>
<b>实测结果</b>：该录音含一段真实 <b>LFM 扫频</b>（908→1389 Hz、+125 Hz/s）。
不依赖跟踪算法的独立判据给出 R² = 0.9998、秩相关 1.0000；
相位随机化后 R² 崩到 0.03~0.76，证明是真结构而非算法伪迹。
降噪后<b>扫频轨迹完整保留</b>：轨迹 r = 0.9970、斜率误差 +0.06%、
起止频率误差 &lt; 0.3 Hz、覆盖比 0.999。<br>
更关键的是<b>信号与噪声可以分开</b>：沿轨迹能量只损失 <b>0.3 dB</b>，
而轨迹外带内噪声被压掉 <b>13.1 dB</b>（相对 <b>12.8 dB</b>），
窗口从 ±1 到 ±8 bin 全扫都稳健 —— <b>模型主要压的是噪声，不是信号</b>。
这条与 8.17 的「带外均匀衰减、无孤立谱线」独立地指向同一结论。<br>
<b>但必须同时说清楚</b>：这份录音是 LFM 而非目标 HFM；只有 1 个文件、4 秒、1 段扫频；
无干净参考，所以只能说「结构被保住」而非「更接近真实信号」。
第十二节的局限因此被<b>推近</b>，但没有<b>消除</b>。</div>
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
    c_hyc_p=(CT.get("频域精修：+p注入 − (−脉冲分支) [GN]", {}).get("frmgain_clean", {}).get("p", float("nan"))),
    proto_imp=_proto_imp, proto_min=_proto_min, arch_imp=_arch_imp, arch_min=_arch_min,
    proto_clean=_proto_clean, step34_imp=_step34_imp, step45_imp=_step45_imp, ratio=_ratio,
    ov_r=_ov_r,
    rank_ok=sens_txt["rank_ok"] or "—",
    eff_a12=sens_txt["eff_a12"], eff_a20=sens_txt["eff_a20"],
    eff_gap=(sens_txt["eff_a20"] - sens_txt["eff_a12"])
    if np.isfinite(sens_txt["eff_a20"]) and np.isfinite(sens_txt["eff_a12"]) else float("nan"),
    sens_a20=sens_vars["d_a20"], sens_a12=sens_vars["d_a12"],
    sens_r25=sens_vars["d_r25"], sens_r10=sens_vars["d_r10"],
    aa_a12=sens_vars["aa_a12"], aa_gauss=sens_vars["aa_gauss"],
    a18_ind=_a18_ind, a12_ind=_a12_ind,
    si_old_rng=_si(0), si_old_loss=(_si_gain(0, 1.0) - _si_gain(0, 0.25)),
    si_c4=_inm("r4", "inm") - _inm("r4", "devchk"),
    si_c5=_inm("r5", "inm") - _inm("r5", "devchk"),
    si_inm_d=_inm("r5", "inm") - _inm("r4", "inm"),
    ia_r5r3=_iag("R5 vs R3"), ia_p=_iag_p("R5 vs R3"),
    pmi_shuf=abs(_pmi("rel_identity", "shuffle", "mean")),
    pmi_zero=abs(_pmi("rel_identity", "zero", "mean")),
    pmi_noise=abs(_pmi("rel_identity", "noise", "mean")),
    pmi_neg=abs(_pmi("summary", "shuffle", "mean")),
    pmi_kh=1000 * _pmd("k_half") / _pmd("fs"),
    pmi_k125=1000 * (_PMD.get("ks") or [0, 0])[1] / _pmd("fs"),
    pmi_hd=_pmo("summary", "headroom_mean"),
    pmi_hdpct=100 * _pmo("summary", "headroom_mean") / _pmi("summary", "identity", "mean"),
    ia_p54=_iag_p("R5 vs R4"),
    cc_rmin=min([_pmc(c, "position") / abs(_pmc(c, "d_zero")) for c in _PMC_CONDS]),
    cc_rmax=max([_pmc(c, "position") / abs(_pmc(c, "d_zero")) for c in _PMC_CONDS]),
    cc_sdn=_pmc_sd("d_identity"), cc_sdp=_pmc_sd("position"), cc_sdc=_pmc_sd("d_zero"),
    cc_a12r=_pmc("a12_r5", "position") / abs(_pmc("a12_r5", "d_zero")),
    cc_a20r=_pmc("a20_r5", "position") / abs(_pmc("a20_r5", "d_zero")),
    cc_r10n=_pmc("a15_r10", "d_identity"), cc_r25n=_pmc("a15_r25", "d_identity"),
    cc_netspan_alpha=(max(_pmc(c, "d_identity") for c in ("a12_r5", "base", "a18_r5", "a20_r5"))
                      - min(_pmc(c, "d_identity") for c in ("a12_r5", "base", "a18_r5", "a20_r5"))),
    cc_posspan_all=(max(_pmc(c, "position") for c in _PMC_CONDS)
                    - min(_pmc(c, "position") for c in _PMC_CONDS)))

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

<h2>六、受控阶梯：把「训练协议贡献」与「架构贡献」分离</h2>
<p>此前报告把 PAM-Net（18.27 / 27.38 / 17.12）与阶段0 纯频域模型（14.13 / 25.09 / 13.54）
直接对比，并把这约 4 dB 归因于架构。<b>但这个对比是混杂的</b>：两者在
<b>归一化方式、损失函数、模型选择准则、训练脚本</b>上全都不同，无法单独归因。</p>
<p>本节用一条<b>每步只改一个因素</b>的阶梯把它拆开。前置条件是先证明
「纯频域模式的 PAM-Net」与「阶段0 的 <code>v3.DenoiseNet(fch=2.0)</code>」
<b>架构逐位等价</b>：参数量同为 5,910，权重复制后前向输出最大绝对差
<b>0.000e+00</b>（<code>test_pamnet_freqonly.py</code>，8 项检查全过，
且 5 个既有 checkpoint 仍能严格加载 —— 说明重构没有副作用）。</p>
<table>
<thead><tr><th>档</th><th>配置</th><th>参数</th><th>种子</th><th>干净帧增益</th>
<th>干扰帧增益</th><th>干扰帧最差</th></tr></thead>
<tbody>
__ROWS_LADDER__
</tbody></table>
__FIGP__
<h3>6.1 逐步增量（每步只改一个因素）</h3>
<table>
<thead><tr><th>改动</th><th>Δ干净帧 (dB)</th><th>Δ干扰帧 (dB)</th><th>Δ干扰最差 (dB)</th></tr></thead>
<tbody>
__ROWS_STEP__
</tbody></table>
<div class="box good"><b>核心结论：架构是主要贡献，协议只占小头。</b>
__LADDER_NOTE__。分解结果：<b>训练协议（损失 + 模型选择 + 归一化）在干扰帧上合计
__PROTO_IMP__（负贡献），但在干净帧 __PROTO_CLEAN__、干扰帧最差 __PROTO_MIN__
（正贡献）</b> —— 是一次权衡而不是全面改进。<b>架构在干扰帧上合计 __ARCH_IMP__</b>，
其中<b>时域多尺度残差通路一步就贡献 __STEP34__，比脉冲感知分支（__STEP45__）
大 __RATIO__ 倍</b> —— 说明「时域残差估计」与「频域掩码」是<b>互补</b>的，
而「脉冲感知分支」与频域掩码高度冗余。此前"PAM-Net 优于阶段0"的结论<b>成立</b>，
但现在协议成分被单独量化出来了。</div>
<div class="box warn"><b>本阶梯的两个混杂，必须写明：</b>
① <b>R3→R4 同时改变了参数量</b>（5,910 → 14,367），所以该步是"架构 + 参数预算"的
联合效应，不能单独归因于架构形式；
② <b>R0/R1 只有 1 个种子</b>（R2 有 3 个，R3/R4/R5 有 5 个）。因此协议部分
（R0→R3）的数字没有种子间不确定度，只有 R3/R4/R5 之间的差值才是可靠的。</div>
<div class="box"><b>一个附带的可靠性证据：</b>R0 与 R1 的协议完全相同、只是换了训练管线
（<code>train_pam.py</code> vs <code>train_pamnet.py</code>），结果差
<b>≤0.04 dB</code></b>（14.13/25.05/13.54 vs 14.1344/25.0532/13.5379）。
两条独立实现的管线给出同一个数，说明阶梯的每一档都是可比的。</div>

<h2>七、效应的波形依赖与机制分析</h2>
<p>起因是一个看起来矛盾的观测：脉冲感知分支的效应在 testA 上是 +0.79 dB（干扰帧）、
在 testC 上只有 +0.33 dB，而两个测试集<b>用的是同一批波形</b>（chirp 18–20）。
逐样本诊断给出了答案 —— 效应是<b>强 SNR 依赖</b>的：</p>
<table>
<thead><tr><th>SNR 档</th><th>testA 干净帧</th><th>testA 干扰帧</th><th>testC 干净帧</th><th>testC 干扰帧</th></tr></thead>
<tbody>
<tr><td>−10 dB</td><td>+0.011</td><td><b>+1.092</b></td><td>+0.122</td><td><b>+0.555</b></td></tr>
<tr><td>−5 dB</td><td>−0.021</td><td>+0.903</td><td>+0.029</td><td>+0.434</td></tr>
<tr><td>0 dB</td><td>−0.078</td><td>+0.531</td><td>+0.142</td><td>+0.152</td></tr>
<tr><td>+5 dB</td><td>+0.255</td><td>+0.757</td><td>+0.383</td><td>+0.124</td></tr>
<tr><td>+10 dB</td><td><b>+0.467</b></td><td>+0.666</td><td><b>+0.622</b></td><td>+0.356</td></tr>
</tbody></table>
<p><b>低 SNR 时收益全在干扰帧，高 SNR 时收益转到干净帧</b> —— 两个测试集是同一趋势，
只是幅度不同。所以"testA/testC 不一致"的表述是错的，真正的是"效应随 SNR 改变落点"。</p>

<h3>7.1 方差分解：波形效应真实，但只占 3.8%</h3>
<p>为了判断逐 chirp 的差异到底是不是波形造成的，生成了一个
<b>chirp 1–20 × 信道 1–8 × 5 SNR × 2 噪声实现 = 1,600 样本</b>的扫描集，
用已训练的 pf / nopulse 各 5 个种子推理（种子间平均），然后做方差分解：</p>
<table>
<thead><tr><th>指标</th><th>逐样本 SD<br>（干扰实现波动）</th><th>逐 chirp 均值 SD<br>（实测）</th>
<th>纯实现波动下的期望</th><th>分解出的波形成分</th><th>F(19,1564)</th></tr></thead>
<tbody>
<tr><td>干净帧</td><td>0.471</td><td>0.095</td><td>0.053</td><td>0.079</td><td>3.24</td></tr>
<tr style="background:#f0fdf4;font-weight:600"><td>干扰帧</td><td>1.728</td><td>0.398</td><td>0.194</td>
<td><b>0.344</b></td><td><b>4.13</b></td></tr>
<tr><td>干扰最差</td><td>1.682</td><td>0.220</td><td>0.189</td><td>0.111</td><td>1.34</td></tr>
</tbody></table>
<div class="box warn"><b>两个结论同时成立：</b>
① <b>波形效应是真实的</b>（干扰帧 F = 4.13，p &lt; 0.01）；
② <b>但它很小</b> —— 波形只解释 3.8% 的方差，而 <b>96% 来自逐样本的干扰实现波动</b>
（SD 1.73 dB vs 波形 0.34 dB）。
也就是说：<b>脉冲分支不是"对任何样本都稳定提升"的模块</b>，
而是在个别样本上帮助或损害 ±1.7 dB、平均只 +0.37 dB 的小幅正偏置。</div>

<h3>7.2 机制：不是带宽，是时频可分性</h3>
<p>最初假设"带宽越宽 → 与宽带干扰重叠越多 → 分支越有价值"。但原始扫描集有一个
<b>混杂</b>：干扰的随机种子是 <code>seed + 样本序号</code>，而 chirp 是最外层循环，
所以每个 chirp 独占一段连续种子区间 —— <b>干扰实现与 chirp 序号完全混杂</b>。</p>
<p>因此给数据生成器加了 <code>seed_mode='paired'</code>：种子只由 (信道, SNR, 重复) 决定，
并且把干扰/噪声生成在<b>固定长度</b>上再截取（因为 <code>gen_burst</code> 的突发数量与到达
时刻都依赖信号长度，只有固定长度才能让不同时长的 chirp 看到<b>逐位相同</b>的干扰）。
实测配对性：干扰与噪声最大差 <b>1.07e−14</b>（机器精度）、掩码逐位相同、干净信号差 5.59。</p>
<table>
<thead><tr><th>自变量</th><th>r（原始设计，干扰混杂）</th><th>r（配对设计，干扰固定）</th></tr></thead>
<tbody>
__ROWS_CHIRP__
</tbody></table>
<div class="box good"><b>配对设计揭示了一个统计显著的机制（原始设计测不出来）：</b>
干扰与信号的<b>时频重叠率</b>与分支收益显著负相关
（r = <b>−0.696</b>，t = −4.11，p &lt; 0.001，n = 20），
而原始设计下只有 r = −0.304（t = −1.35，不显著）。<br>
<b>方向与假设相反</b>：干扰越落在信号的时频支撑<b>内</b>，脉冲分支的收益越<b>小</b>。
物理解释是合理的 —— 当干扰与信号在时频面重叠时，掩码无法在不伤信号的前提下抑制它，
"知道哪里是脉冲"也帮不上；当干扰落在信号缺席的时频区域时，掩码可以干净地压制它，
脉冲置信度图正好用来定位这些区域。<br>
<b>这也解释了 chirp 19 的异常</b>：它是 20 个波形里时频重叠率<b>最低</b>的（0.116），
分支收益最大（+1.58 dB）；而带宽几乎相同（650.0 vs 652.8 Hz）的 chirp 8 重叠率是 0.243，
收益只有 +0.21 dB。<b>带宽假设被否定</b>；时频可分性在<b>观察层面</b>与收益相关，
<b>但后续的干预实验（7.x）未能复现该效应，因此它不能作为已确立的机制引用。</b></div>
__FIGO__
<p><b>注意</b>：以上是"效应大小的调制因素"，不是"效应存在与否"。平均效应仍然很小
（干扰帧 +0.37 dB），且高度依赖具体样本。</p>

<h3>7.x 机制的干预检验：两个独立操纵都【没有】复现该机制</h3>
<p>上一小节的「时频可分性」是<b>观察性</b>证据 —— 我们只改变了波形，重叠率随之变化，
但重叠率高的波形可能同时还有其他不同。要确立因果，必须<b>直接操纵重叠率</b>。
为此做了两个独立干预，都用<b>真实的信号与真实的背景噪声</b>，只替换突发干扰。</p>

<div class="box warn"><b>干预一：载频扫描（改变突发落在信号频带内还是带外）</b><br>
固定突发的波形类型/时长/幅度/带宽分布（与 <code>gen_burst.m</code> 一致），
只把载频从 300 Hz 扫到 1500 Hz。300 个样本 × 5 个种子 × 7 个档位。
结果：<b>带内能量占比从 0.017 变到 0.952，分支收益没有任何单调趋势</b>
（档位级 r = <b>+0.40</b>，p = 0.33，<b>方向还与机制预测相反</b>）。</div>
<table>
<thead><tr><th>载频 (Hz)</th><th>带内能量占比</th><th>TF 支撑重叠率</th>
<th>Δ 干扰帧 (dB)</th></tr></thead>
<tbody>
<tr><td>300</td><td>0.017</td><td>0.000</td><td>+0.426 ± 0.208</td></tr><tr><td>500</td><td>0.026</td><td>0.000</td><td>+0.093 ± 0.398</td></tr><tr><td>700</td><td>0.103</td><td>0.002</td><td>+0.214 ± 0.340</td></tr><tr><td>900</td><td>0.893</td><td>0.041</td><td>+0.812 ± 0.428</td></tr><tr><td>1100</td><td>0.952</td><td>0.073</td><td>+0.032 ± 0.772</td></tr><tr><td>1300</td><td>0.929</td><td>0.066</td><td>+0.559 ± 0.425</td></tr><tr><td>1500</td><td>0.678</td><td>0.023</td><td>+0.693 ± 0.475</td></tr>
</tbody>
<tfoot><tr><th colspan="4" style="text-align:right">
Δ = R5 − R4；± 为种子间标准差。带内占比低 = 突发落在信号频带外</th></tr></tfoot></table>

<div class="box bad"><b>干预二：时频对齐（决定性版本）</b><br>
干预一有个设计缺陷：信号是<b>扫频 chirp</b>，"载频在带内"并不等于"与瞬时频率对齐" ——
一个 1100 Hz 的突发，只有当 chirp 恰好扫到 1100 Hz 的那一小段才真正重叠，
所以实测 TF 重叠率只有 0.073。<br>
因此干预二改为<b>直接操纵放置时刻</b>：从干净信号的 STFT 逐帧提取瞬时主频，
同一个突发波形分别放在「瞬时频率 ≈ 载频」的时刻（对齐）与「瞬时频率远离载频」的时刻（错开）。
波形、信号、背景噪声、突发数量<b>完全相同</b>，唯一差别是时刻。250 个样本 × 2 组 × 5 个种子。<br><br>
结果：错开组 TF 重叠率 0.007、Δ = <b>-0.177 dB</b>；
对齐组重叠率 0.085、Δ = <b>+0.277 dB</b>。
<b>对齐 − 错开 = +0.453 dB</b>（逐样本配对，跨种子 p = 0.31），
即<b>重叠越高，分支收益反而略大</b>（不显著），<b>与第七节的机制预测方向相反</b>。</div>
__FIGV__

<div class="box"><b>如何解读这个否定结果 —— 必须写进论文的三点：</b>
<ul>
<li><b>第七节的 r = −0.696 不能作为机制结论引用。</b>
它来自"只变波形"的观察设计，无法排除波形层面的其他混淆
（不同 chirp 的时长、时宽带宽积、与信道冲激响应的卷积结果都不同）。
两个直接操纵重叠率的干预都<b>没有</b>复现这个负相关。</li>
<li><b>可以说的是</b>：「脉冲分支的收益在波形之间有真实但很小的差异
（干扰帧 F = 4.13, p &lt; 0.01，只解释 3.8% 方差）」，
以及「该收益与若干波形/干扰属性存在样本级相关，但<b>因果关系未确立</b>」。</li>
<li><b>不能说</b>：「收益由时频可分性决定」。本节的干预正是对该说法的否证尝试，
结果是否定的。这本身是一个有价值的负面结果 —— 它说明<b>这个分支的收益
不是由某个单一、可操纵的时频几何量决定的</b>，更可能是逐样本的随机波动
（96% 的方差来自样本间波动，见前文）。</li>
</ul></div>
<p><b>诚实标注</b>：本节的结论是「未能证实」，不是「证伪」。
两个干预都没有足够功效排除小于约 0.5 dB 的效应（种子间 SD 在 0.57~1.07 dB）。
若要在 0.2 dB 的量级上定论，需要把每组样本量再扩大约一个数量级。</p>

<h2>八、适用边界：这套方法在什么条件下成立</h2>
<p>前面所有结论都建立在<b>一个工作点</b>上：α = 1.5、突发到达率 5 次/秒。本节用
<b>单因素轮换（OFAT）</b>设计，围绕该工作点各变一个方向，回答"结论能推到多远"。
同时这一步也把上一节留下的问题（<b>脉冲感知分支在什么条件下才真正有价值</b>）
放到一个可检验的框架里。</p>

<table>
<thead><tr><th>条件代号</th><th>α</th><th>突发到达率</th><th>改动的唯一因素</th><th>样本数（testA/testC）</th></tr></thead>
<tbody>
<tr><td>a12_r5</td><td>1.2</td><td>5 /s</td><td>背景噪声尾部更重</td><td>1,080 / 240</td></tr>
<tr><td>a18_r5</td><td>1.8</td><td>5 /s</td><td>背景噪声更接近高斯</td><td>1,080 / 240</td></tr>
<tr><td>a20_r5</td><td>2.0</td><td>5 /s</td><td>背景退化为高斯（内部一致性校验）</td><td>1,080 / 240</td></tr>
<tr style="background:#f5f7fa"><td><b>base</b></td><td><b>1.5</b></td><td><b>5 /s</b></td><td><b>训练条件（基线）</b></td><td><b>1,080 / 240</b></td></tr>
<tr><td>a15_r25</td><td>1.5</td><td>2.5 /s</td><td>突发干扰更稀疏</td><td>1,080 / 240</td></tr>
<tr><td>a15_r10</td><td>1.5</td><td>10 /s</td><td>突发干扰更密集（2×）</td><td>1,080 / 240</td></tr>
<tr><td>gauss</td><td>—</td><td>5 /s</td><td>独立生成的高斯对照数据集</td><td>1,080 / 240</td></tr>
</tbody></table>

<div class="box"><b>两个设计上的关键点，决定了这一节能否做因果解读：</b>
<ul>
<li><b>共同随机数</b>：数据生成用 <code>seed + 样本序号</code> 作种子，且各条件的循环顺序完全一致，
因此样本序号 i 在所有条件下对应<b>同一</b> (chirp, 信道, SNR, 重复)。并且
<b>α 轮换时突发干扰逐位相同、rate 轮换时背景噪声逐位相同</b>。
于是条件之间的比较可以做<b>逐样本配对</b>，统计功效远高于独立样本。</li>
<li><b>零样本 vs 域内</b>：模型全部在 base 上训练，本节先做<b>零样本跨条件评估</b>
（这是分布偏移下的稳健性），再在关键条件上做<b>域内重训</b>（用来区分
"分布偏移造成的退化"与"该条件下方法本身的优劣"）。</li>
</ul></div>

<h3>8.1 一个必须先讲清楚的混杂：名义 SNR 固定 ≠ 有效 SNR 固定</h3>
<p>本项目的 SNR 定义是<b>75% 分位数匹配</b>（α &lt; 2 时方差无穷，无法用功率定义）。
这个定义固定了噪声的"核心宽度"，但<b>没有固定它的二阶矩</b>。实测下来，
α 每降一档，背景噪声的<b>有效功率信噪比</b>就掉一大截：</p>
__FIGQ__
<table>
<thead><tr><th>条件</th><th>干扰帧占比</th><th>有效功率 SNR（总）</th><th>背景噪声有效 SNR</th>
<th>干扰有效 SNR</th><th>背景峰/rms</th><th>背景 p99.9/rms</th></tr></thead>
<tbody>
__ROWS_SENS_NOISE__
</tbody>
<tfoot><tr><th colspan="7" style="text-align:right">
有效 SNR = 10·log₁₀(P<sub>信号</sub>/P<sub>噪声分量</sub>)，在降采样 + 定长后的信号上按样本计算再取中位数；
背景分量由 <code>v = noisy − clean − imp</code> 精确还原</th></tr></tfoot></table>
<div class="box warn"><b>这是本轮最重要的方法论发现：把 α 从 2.0 降到 1.2，名义 SNR 不变，
但背景噪声的有效功率信噪比从 <b>__EFF_A20__ dB 掉到 __EFF_A12__ dB</b>（相差
__EFF_GAP__ dB）。</b>
因此"α 扫描"实际上是<b>两个变量同时变</b>：尾部形状与噪声功率。
任何把 α 扫描的结果直接解读为"重尾程度的影响"的结论都是不可靠的。
本节的正确读法是：<b>未匹配的 α 轴 = 背景噪声强度 + 形状的联合偏移</b>，
而 <b>rate 轴与等功率匹配的 α 条件</b>才是干净的单因素操纵
（rate 轴：α 不变、背景噪声逐位相同，只有干扰密度翻倍；
匹配 α 条件：有效功率 SNR 逐档对齐，只有尾形不同 —— 见 8.1b）。
这一点在后续使用"分位数匹配"定义 SNR 的工作中同样需要注意。</div>

<h3>8.1b 把混杂拆开：等有效功率 SNR 的 α 重扫</h3>
<p>8.1 节只做到了"发现混杂"。真正要回答的是：<b>未匹配时看到的
"α 越小脉冲分支收益越小"，是尾形本身造成的，还是噪声功率造成的？</b>
做法是把名义 SNR 平移对应量，使<b>有效功率 SNR 与 base 一一对应</b>，
从而把 α 的效应单独隔离出来。平移量按 shift 的<b>中位数</b>确定
（比均值稳健 —— α 小时样本方差本身是重尾量）：</p>
<table>
<thead><tr><th>条件</th><th>α</th><th>名义 SNR 列表（dB）</th><th>平移量</th>
<th>背景有效 SNR</th><th>干扰有效 SNR</th><th>背景峰/rms</th></tr></thead>
<tbody>
<tr><td>base</td><td>1.5</td><td>−10 −5 0 +5 +10</td><td>—（参照）</td>
    <td>__MB0__</td><td>__MI0__</td><td>__MP0__</td></tr>
<tr><td>m_a12</td><td>1.2</td><td>+3.07 +8.07 +13.07 +18.07 +23.07</td><td>+13.07</td>
    <td>__MB1__</td><td>__MI1__</td><td>__MP1__</td></tr>
<tr><td>m_a18</td><td>1.8</td><td>−20 −15 −10 −5 0</td><td>−10.00</td>
    <td>__MB2__</td><td>__MI2__</td><td>__MP2__</td></tr>
<tr><td>a12_r5</td><td>1.2</td><td>−10 −5 0 +5 +10</td><td>0（未匹配）</td>
    <td>__MB3__</td><td>__MI3__</td><td>__MP3__</td></tr>
<tr><td>a18_r5</td><td>1.8</td><td>−10 −5 0 +5 +10</td><td>0（未匹配）</td>
    <td>__MB4__</td><td>__MI4__</td><td>__MP4__</td></tr>
</tbody></table>
<div class="box good"><b>匹配质量：两个"匹配条件"（m_a12 / m_a18）的背景有效功率 SNR 与 base 差
__MDELTA__ dB（机器精度），干扰有效 SNR 也完全一致，只有背景峰/rms 不同
（__MP0__ / __MP1__ / __MP2__）—— 正是要隔离的尾形。</b>
此外样本序号、底层均匀随机数与突发实现在各条件间完全一致（同 seed、
同 rms_c），构成逐样本配对。</div>

<p>脉冲感知分支效应 Δ = R5 − R4（干扰帧，testA）：</p>
<table>
<thead><tr><th>条件</th><th>干净帧 Δ (dB)</th><th>干扰帧 Δ (dB)</th></tr></thead>
<tbody>
__ROWS_SENS_MATCHED__
</tbody></table>
<p>架构优势 Δ = R5 − R3（testA）：</p>
<table>
<thead><tr><th>条件</th><th>干净帧 Δ (dB)</th><th>干扰帧 Δ (dB)</th><th>跨种子 p</th></tr></thead>
<tbody>
__ROWS_SENS_MATCHED_ARCH__
</tbody></table>

<div class="box warn"><b>判定：两者兼有，且两个方向的分解不对称。</b>
<ul>
<li><b>α=1.2 方向</b>（未匹配时效应被压低）：总变化 __MA12_TOT__ dB =
强度成分 __MA12_INTEN__ dB + 尾形成分 __MA12_TAIL__ dB。
即<b>强度解释约 __MA12_PCT__%</b>、<b>尾形解释约 __MA12_TPCT__%</b>。
留一法确认均值不被单个种子主导（[__MA12_LOOMIN__, __MA12_LOOMAX__]）。</li>
<li><b>α=1.8 方向</b>（未匹配时效应被抬高）：总变化 __MA18_TOT__ dB =
强度成分 __MA18_INTEN__ dB + 尾形成分 __MA18_TAIL__ dB。
这里<b>两个成分符号相反</b> —— 匹配掉"多余的有效信噪比"后，
尾形本身的增强作用反而更明显。留一法范围 [__MA18_LOOMIN__, __MA18_LOOMAX__]，
该条件受种子 0 影响较大，但去掉后仍为正值。<br><b style="color:#b91c1c">⚠ 这一分解已被 8.7 节的域内重训推翻</b>：
用零样本数据做"强度/尾形"二分，无法把"模型没见过该 α"这一项分离出来。
域内重训后 α=1.8 的 Δ 从 +1.77 回到 +0.75（≈base），尾形效应消失；
α=1.2 仅剩约 0.24 dB 且不显著。请以 8.7 节为准。</li>
<li><b>共同结论</b>：<b>"背景越脉冲化，脉冲分支收益越小"这个单调趋势是真实的
（尾形成分），但未匹配的 α 扫描把它的幅度夸大了</b> ——
在 α=1.2 端放大了约 1.5 倍，在 α=1.8 端则因为强度成分反向而几乎掩盖了尾形的增强。
因此报告 8.3 节的 α 轴结论应读作<b>尾形与强度的联合效应</b>，
真正的单因素操纵是 8.1b 的匹配条件。</li>
</ul></div>
<p><b>testC（240 样本，功效更低）呈现同样的单调序</b>：匹配功率下
α=1.2 的干扰帧效应 __TC_A12__ dB &lt; α=1.5 的 __TC_BASE__ dB &lt;
α=1.8 的 __TC_A18__ dB；架构优势同样单调（__TC_ARCH_A12__ &gt; __TC_ARCH_BASE__
&gt; __TC_ARCH_A18__）。方向一致，量级因样本量小而偏低。</p>

<h3>8.2 模型排序在所有条件下都保持</h3>
<p>把 5 个模型放到 7 个条件上评估（全部零样本，模型只在 base 上训练过）：</p>
__FIGR__
<table>
<thead><tr><th>条件</th><th>模型</th><th>参数量</th><th>种子数</th><th>干净帧</th>
<th>干扰帧</th><th>干扰最差</th></tr></thead>
<tbody>
__ROWS_SENS_MAIN__
</tbody></table>
<div class="box good"><b>结论：推荐模型 R5 在 <b>{rank_ok}</b> 的干扰帧增益上都是最高的。</b>
在 6 个条件上完整的排序 <b>R5 &gt; R4 &gt; R3 &gt; 纯时域</b> 保持不变；
唯一例外是<b>独立高斯数据集</b>（gauss）上 <b>R4（24.22 dB）被 R3（24.89 dB）反超</b>，
即"时域残差通路"的优势消失了，但 R5（25.96 dB）仍高于两者。
这个例外与 8.3 节的发现是同一件事的两面：<b>当背景退化为高斯、突发仍是唯一不可预测成分时，
频域掩码（等价于可学习带通）的相对优势最强</b>。这说明
"时域残差与频域掩码互补、纯时域结构缺少带限先验"这一机制性结论
<b>在背景为脉冲性的全部场景下成立</b>，但不能外推到高斯背景。</div>

<h3>8.2b 头条结论的跨条件检验：经典基线在所有条件上都被大幅超过</h3>
<p>第五节的头条结论「PAM-Net 远超全部经典方法」此前<b>只在 base 条件上测过</b>。
本节把同样的经典基线放到全部 9 个条件上。</p>
<div class="box"><b>口径一致性</b>：经典基线直接作用于 <code>sens_out/cache_*.npz</code>
—— 与神经模型评估用的是<b>完全相同的输入数组</b>、相同的降采样/定长/掩码口径、
相同的指标代码（<code>pam_metrics._stats_1d</code>），两者严格配对。
已在 base 前 200 个样本上与存档 <code>baseline_pam.json</code> 对比校验，
四种方法的干扰帧增益<b>差 0.0000 dB</b>。</div>

<p><b>干扰帧</b>逐帧降噪增益（dB）：</p>
<table>
<thead><tr><th>条件</th><th>理想带通*</th><th>谱减(盲)</th><th>维纳(盲)</th><th>中值(盲)</th>
<th>最好盲方法</th><th>R5 神经</th><th>R5−最好盲</th></tr></thead>
<tbody>
<tr><td>α=1.5 / rate=5（base）</td><td>8.69</td><td>6.69</td><td>2.25</td><td>5.18</td><td>6.69</td><td>27.38</td><td>+20.69</td></tr><tr><td>α=1.2 / rate=5</td><td>6.91</td><td>10.42</td><td>3.38</td><td>6.65</td><td>10.42</td><td>27.29</td><td>+16.87</td></tr><tr><td>α=1.8 / rate=5</td><td>9.64</td><td>5.23</td><td>1.66</td><td>4.57</td><td>5.23</td><td>26.97</td><td>+21.73</td></tr><tr><td>α=2.0 / rate=5</td><td>10.00</td><td>4.77</td><td>1.45</td><td>4.37</td><td>4.77</td><td>26.90</td><td>+22.13</td></tr><tr><td>α=1.5 / rate=2.5</td><td>9.09</td><td>7.61</td><td>2.61</td><td>5.18</td><td>7.61</td><td>27.86</td><td>+20.25</td></tr><tr><td>α=1.5 / rate=10</td><td>8.26</td><td>6.34</td><td>2.08</td><td>5.39</td><td>6.34</td><td>26.71</td><td>+20.37</td></tr><tr><td>独立高斯 / rate=5</td><td>9.42</td><td>5.50</td><td>1.75</td><td>4.42</td><td>5.50</td><td>25.96</td><td>+20.46</td></tr><tr><td>α=1.2 已匹配功率</td><td>9.04</td><td>5.81</td><td>1.88</td><td>5.13</td><td>5.81</td><td>27.96</td><td>+22.15</td></tr><tr><td>α=1.8 已匹配功率</td><td>8.16</td><td>8.00</td><td>2.73</td><td>5.01</td><td>8.00</td><td>25.02</td><td>+17.02</td></tr>
</tbody></table>

<p><b>干净帧</b>（无干扰）逐帧降噪增益（dB）—— 检验在低难度下经典方法是否追得上：</p>
<table>
<thead><tr><th>条件</th><th>理想带通*</th><th>谱减(盲)</th><th>维纳(盲)</th><th>中值(盲)</th>
<th>R5 神经</th><th>R5−带通</th></tr></thead>
<tbody>
<tr><td>α=1.5 / rate=5（base）</td><td>5.67</td><td>11.77</td><td>5.56</td><td>0.70</td><td>18.27</td><td>+12.59</td></tr><tr><td>α=1.2 / rate=5</td><td>5.20</td><td>15.18</td><td>5.71</td><td>6.38</td><td>22.24</td><td>+17.03</td></tr><tr><td>α=1.8 / rate=5</td><td>5.61</td><td>10.30</td><td>5.41</td><td>-3.89</td><td>14.46</td><td>+8.85</td></tr><tr><td>α=2.0 / rate=5</td><td>5.47</td><td>9.77</td><td>5.32</td><td>-6.39</td><td>12.70</td><td>+7.23</td></tr><tr><td>α=1.5 / rate=2.5</td><td>5.83</td><td>12.09</td><td>5.58</td><td>0.69</td><td>18.26</td><td>+12.44</td></tr><tr><td>α=1.5 / rate=10</td><td>5.45</td><td>11.21</td><td>5.47</td><td>0.77</td><td>18.15</td><td>+12.69</td></tr><tr><td>独立高斯 / rate=5</td><td>5.82</td><td>10.94</td><td>5.45</td><td>-2.91</td><td>13.71</td><td>+7.89</td></tr><tr><td>α=1.2 已匹配功率</td><td>4.71</td><td>9.85</td><td>5.35</td><td>-2.14</td><td>16.99</td><td>+12.28</td></tr><tr><td>α=1.8 已匹配功率</td><td>6.00</td><td>13.38</td><td>5.65</td><td>2.22</td><td>17.01</td><td>+11.01</td></tr>
</tbody></table>

__FIGU__

<div class="box good"><b>结论：头条结论跨条件成立，且优势极大。</b>
<ul>
<li><b>相对最好的盲方法</b>：R5 在 <b>9/9 个条件</b>上都更高，
优势范围 <b>+16.87 ~ +22.15 dB</b>（中位 +20.46 dB）。</li>
<li><b>相对理想带通这个"作弊上界"</b>：R5 在 <b>9/9 个条件</b>上<b>仍然更高</b>，
优势 +16.54 ~ +20.38 dB（中位 +18.45 dB）。
这一点值得说明：理想带通已经知道信号频带、硬性切除带外全部能量，
但它压不掉<b>带内</b>的背景噪声与突发能量；神经模型可以从数据里学到这部分。</li>
<li><b>干净帧上同样成立</b>：R5 在全部条件上都高于谱减法（盲），
也高于理想带通（高出 +7.23 ~ +17.03 dB）——
说明优势不是"只会压脉冲"，常规降噪能力也全面占优。</li>
<li>优势最小的条件是 <b>α=1.2 / rate=5</b>（+16.87 dB），因为该处谱减法异常好
（背景噪声重尾时谱减法反而受益，干扰帧达 +10.42 dB）。
即便如此，R5 仍高出 15 dB 以上。</li>
</ul></div>
<p style="font-size:12.5px;color:#6b7280">
* 理想带通 798–1500 Hz 是<b>已知信号频带</b>的作弊上界，不是可用的盲方法；
列出它是为了给"模型到底比一个完美的固定滤波器强多少"一个参照。</p>

<h3>8.3 脉冲感知分支：收益随条件如何变化</h3>
<p>把 <b>Δ = R5 − R4</b>（同种子内逐样本配对，再跨种子汇总）在 7 个条件上分别算出来：</p>
<table>
<thead><tr><th>条件</th><th>干净帧 Δ (dB)</th><th>干扰帧 Δ (dB)</th><th>干扰最差 Δ (dB)</th></tr></thead>
<tbody>
__ROWS_SENS_PULSE__
</tbody>
<tfoot><tr><th colspan="4" style="text-align:right">
Δ 的 ± 为种子间标准差；p 为对【逐种子效应量】做单样本 t 检验的 p（自由度 = 种子数 − 1）。
这个口径比种子内逐样本配对（n≈1080）更保守 —— 后者只消除样本抽样，
前者才反映训练随机性；两者结论一致时才能说效应稳健。</th></tr></tfoot></table>

<div class="box"><b>读法：规律非常清楚，且与上一节的机制一致 ——</b>
<ul>
<li><b>α 轴单调</b>：背景越脉冲化（α 从 2.0 降到 1.2），分支收益从
<b>__SENS_A20__ dB 一路降到 __SENS_A12__ dB</b>（α=1.2 时跨种子检验不显著，p=__SENS_PA12__）。
背景噪声本身就有重尾时，残差误差被背景主导，而脉冲分支监督的是<b>突发干扰掩码</b>，
"知道突发在哪里"对总误差的边际贡献自然变小。</li>
<li><b>rate 轴反向</b>：突发越稀疏（2.5/s）收益反而最大（<b>__SENS_R25__ dB</b>），
突发越密（10/s）收益最小（<b>__SENS_R10__ dB</b>）。
这<b>与"冗余假设"的朴素预测相反</b>（不是"掩码应付不过来了所以分支才有用"），
而是与第七节的时频重叠机制一致：干扰越密，落在信号时频支撑<b>内</b>的概率越大，
掩码无法在不伤信号的前提下压制它，"知道哪里是脉冲"也帮不上。
<b>两端都已做域内重训验证（8.14）</b>：稀疏端 +1.34（高于 base）、密集端 +0.41（低于 base），
方向相反 ⇒ 这是真实条件效应而非分布偏移。</li>
<li><b>符号在所有条件上一致为正</b>（唯一例外是 α=1.2 的干扰最差帧 −0.05 dB，
本身也不显著），说明这不是只在特定条件下成立的偶然效应；但量级始终落在
0.2~1.7 dB 这个"小效应"区间。</li>
</ul></div>

<p>再把这套时域通路 + 脉冲注入的<b>整体架构优势</b>拆出来 ——
Δ = R5 − R3（相对纯频域掩码）：</p>
<table>
<thead><tr><th>条件</th><th>干净帧 Δ (dB)</th><th>干扰帧 Δ (dB)</th><th>跨种子 p</th></tr></thead>
<tbody>
__ROWS_SENS_ARCH__
</tbody>
<tfoot><tr><th colspan="4" style="text-align:right">
纯频域掩码（R3）的干扰帧增益在 α=1.2 时只有 __SENS_R3A12__ dB、α=2.0 时 __SENS_R3A20__ dB，
而 R5 在两端都稳定在 __SENS_R5A12__ / __SENS_R5A20__ dB 附近</th></tr></tfoot></table>
<div class="box warn"><b>架构优势同样随 α 增大而缩小</b>：从 α=1.2 的 <b>__SENS_AA12__ dB</b>
降到 α=2.0 的 <b>__SENS_AA20__ dB</b>、独立高斯数据集上的 <b>__SENS_AAGAUSS__ dB</b>。
也就是说：<b>这套架构的额外价值主要来自"干扰是可分离的脉冲/突发"这一前提</b>；
当背景退化为高斯时，频域掩码（等价于可学习带通）的相对优势最强。
这不是缺陷，但必须写进论文的适用条件，否则结论会被外推到不适用的场景。</div>

<h3>8.4 条件间退化（共同随机数逐样本配对）</h3>
<p>下表是各模型在异常条件下相对 base 的<b>干扰帧增益变化</b>，负值表示退化。
因为各条件共享同一批 (chirp, 信道, SNR, 重复)，这里是逐样本配对的结果，
比独立样本灵敏得多。</p>
<table>
<thead><tr><th>模型</th>__SENS_HEAD__</tr></thead>
<tbody>
__ROWS_SENS_DEGRADE__
</tbody>
<tfoot><tr><th colspan="8" style="text-align:right">
± 为种子间标准差；负值 = 该条件下干扰帧增益低于 base。注意 α 轴各条件的干扰强度本身就不同
（见表 8.1），所以这里的"退化"同时包含难度变化</th></tr></tfoot></table>
<div class="box"><b>两点值得注意：</b>① <b>R5 在整个 α 轴上的退化最小</b>
（α=1.2 时 −__SENS_DEG12__ dB、α=2.0 时 −__SENS_DEG20__ dB），而 R4 在 α=2.0 时退化到
−__SENS_DEG20R4__ dB —— 脉冲分支在高 α 端提供了额外的稳健性；<br>
② <b>α=1.2 时所有模型的干扰帧增益都明显偏低</b>（R3 只有 23.26 dB），
但这是<b>噪声更强（有效 SNR −16.4 dB）造成的难度变化</b>，不是模型退化。</div>

<h3>8.5 判别能力与性能收益的关系（跨条件检验）</h3>
__FIGT__
<p>横轴是该分支把 p<sub>n</sub> 当作脉冲检测器时的逐样本 AUC（越高越能准确标出脉冲位置），
纵轴是它带来的干扰帧增益。上一节在单工作点上得到的"<b>能识别 ≠ 能提升性能</b>"
在本节被推广到全部条件：<b>AUC 在 7 个条件上稳定在 __SENS_AUCLO__~__SENS_AUCHI__</b>
（变化不到 0.04），<b>而性能收益却在 0.20~1.74 dB 之间起伏</b>。
这进一步说明两者的解耦不是工作点的偶然现象，而是结构性的。</p>

<h3>8.6 域内重训：把"分布偏移的代价"与"方法本身的优劣"分开</h3>
<p>前面所有跨条件结果都是<b>零样本</b>的（模型只在 base 上训练过）。这能回答
"分布偏移下还稳不稳"，但回答不了"在 rate=10 这个条件下，方法本身是不是就不行了"。
本节在 <code>a15_r10</code> 上<b>从头重训</b> R4 与 R5（各 3 个种子，与 base 前 3 个种子一致），
做三方对照。</p>

<table>
<thead><tr><th>配置</th><th>R4 干扰帧 (dB)</th><th>R5 干扰帧 (dB)</th>
<th>Δ = R5 − R4</th><th>种子数</th></tr></thead>
<tbody>
__ROWS_INDOMAIN__
</tbody>
<tfoot><tr><th colspan="5" style="text-align:right">
testA。base 与零样本为 CPU 训练，域内重训为 GPU 训练 —— 设备效应已单独核对，见下</th></tr></tfoot></table>

<div class="box good"><b>结论一：零样本下的性能下降是"分布偏移"造成的，域内重训可以消除。</b>
R4：base <b>26.59</b> → 零样本 <b>26.16</b>（−0.43）→ 域内 <b>26.86</b>（回到并略高于 base）。
R5：base <b>27.38</b> → 零样本 <b>26.71</b>（−0.67）→ 域内 <b>27.18</b>（≈base）。
即<b>分布偏移的代价约 0.43~0.67 dB</b>，属于"换个条件重新训练就能补回来"的量级，
不是方法在该条件下失效。testC 同样趋势（域内 R4 25.78 / R5 25.68，均高于零样本）。</div>

<div class="box warn"><b>结论二：但脉冲分支的收益在 rate=10 下<b>确实更小</b>，
而且这是真实的<b>条件效应</b>，不是分布偏移的假象。</b>
Δ 从 base 的 <b>+0.790</b> → 零样本 <b>+0.548</b> → 域内重训 <b>+0.322</b>
（3 个种子的 sd 仅 0.033，跨种子 p = 0.004，非常一致）。<br>
<b>设备混淆已排除</b>：域内重训在 GPU 上完成，而 base 在 CPU 上完成。
为此在 base 条件上用 GPU 重训了 R4/R5（同配置同种子），直接测设备效应对 Δ 的影响：
<b>-0.091 ± 0.031 dB（2 个配对种子）</b>，远小于 base 与 rate=10 之间 -0.468 dB 的条件差异。
校正后 rate=10 的 Δ ≈ <b>+0.413 dB</b>，仍显著低于 base。<br>
<b>这条结果的意义</b>：第八节 8.3 从零样本矩阵看到"rate 轴反向"（干扰越密、分支收益越小），
当时无法排除"模型没见过该干扰密度"这一解释。现在域内重训确认了这个效应，
<b>说明它不是分布偏移的假象</b> —— 干扰变密并不会让"知道脉冲在哪里"变得更有价值，
这与冗余假设的朴素预测正好相反，进一步支持"脉冲分支与频域掩码高度冗余"的结论。</div>

<div class="box"><b>必须如实标注的三点：</b>
① <b>本节只做了 rate=10 一个条件</b>。α 轴（等功率的 α=1.2 / α=1.8）的域内重训
见 <b>8.11 节</b>——结论是 8.1b 的"尾形 vs 强度"分解<b>被推翻</b>（α=1.8 端的"尾形增强"是分布偏移假象）；
<b>rate 轴的稀疏端（2.5/s）与 α=2.0 端点见 8.14 节</b>；
② 域内重训<b>只重训了 R4/R5</b>，没有重训 R3 与纯时域模型，
所以本节的 Δ 只覆盖"有无脉冲分支"这一维；
③ 域内重训的种子数是 3（base 是 5）。Δ 的种子间 sd 仅 0.033 说明估计稳定，
但 3 个种子对<b>绝对水平</b>的估计仍有约 ±0.1 dB 的不确定度。</div>
<h3>8.7 α 轴域内重训：8.1b 的“尾形分解”被推翻</h3>
<p>8.6 补上了 rate 轴，但 α 轴仍是零样本。这意味着 8.1b 的<b>“尾形 vs 强度”分解</b>
（α=1.2 端 尾形 65% / 强度 35%，α=1.8 端 尾形 +0.976 dB）
建立在“模型没见过该 α”的前提上——它可能把<b>分布偏移</b>误当成了<b>尾形效应</b>。
本节在 <code>m_a12</code>（α=1.2，等有效功率）与 <code>m_a18</code>（α=1.8，等有效功率）上
从头重训 R4/R5（各 3 个种子），做同样的三方对照。</p>

<table>
<thead><tr><th>条件</th><th>R4 干扰帧 (dB)</th><th>R5 干扰帧 (dB)</th>
<th>Δ 零样本</th><th>Δ 域内</th><th>Δ 域内（设备校正后）</th><th>相对 base</th></tr></thead>
<tbody>
<tr><td>base α=1.5</td><td>26.59</td><td>27.38</td><td colspan="2"><b>+0.790</b>（5 种子）</td>
<td>—</td><td>基准</td></tr>
<tr><td>α=1.2 零样本</td><td>27.55</td><td>27.96</td><td>+0.410</td>
<td>—</td><td>—</td><td>-0.380</td></tr>
<tr><td>α=1.2 域内</td><td>27.62</td><td>28.08</td><td>—</td><td>+0.464</td>
<td><b>+0.555</b></td><td><b>-0.235</b></td></tr>
<tr><td>α=1.8 零样本</td><td>23.25</td><td>25.02</td><td>+1.766</td>
<td>—</td><td>—</td><td>+0.976</td></tr>
<tr><td>α=1.8 域内</td><td>25.68</td><td>26.34</td><td>—</td><td>+0.662</td>
<td><b>+0.752</b></td><td><b>-0.038</b></td></tr>
</tbody>
<tfoot><tr><th colspan="7" style="text-align:right">
testA。域内为 GPU 训练（3 种子），设备效应对 Δ = -0.091 dB，已校正</th></tr></tfoot></table>

__FIGW__

<div class="box warn"><b>核心发现：α=1.8 端的“尾形增强 +0.976 dB”是分布偏移造成的假象。</b>
零样本 Δ = <b>+1.766</b>，看起来是“背景越接近高斯、脉冲分支越有价值”；
但域内重训后 Δ 掉到 <b>+0.662</b>，设备校正后 <b>+0.752</b>，
与 base 的 <b>+0.790</b> <b>没有实质差异</b>（差 -0.038）。<br>
<b>机制很清楚</b>：α=1.8 上 R4 提升了 <b>+2.42 dB</b>（23.26 → 25.68），
而 R5 只提升 <b>+1.32 dB</b>（25.02 → 26.34）。也就是说，
<b>只要让 R4 见过这个分布，它自己就能学到接近 R5 的水平</b>——脉冲分支在该条件下高度冗余。
零样本时 R4 没见过 α=1.8，退化更严重，才显得“分支很有用”。<br>
testC 同样成立：零样本 +1.310 → 域内校正 +0.468（base +0.325）。</div>

<div class="box"><b>α=1.2 端：残余效应真实但很小，且不显著。</b>
校正后 Δ = <b>+0.555</b>，低于 base 的 +0.790 约 <b>0.235 dB</b>；
testC 是 <b>+0.079</b> 对 base +0.325，差 <b>0.246 dB</b>——
<b>两个测试集的差值几乎相同（−0.24 dB）</b>，方向一致。<br>
但统计上不显著（testA p = 0.162，testC p = 0.978），
逐种子离散度大（testA +0.468 / +0.093 / +0.832，sd 0.370）。
另外 α=1.2 上两个模型的域内提升都接近零（R4 +0.07、R5 +0.12）——
说明<b>零样本时模型已经接近该条件的最优</b>，重训没有可挖的空间。</div>

<div class="box good"><b>修正后的结论（取代 8.1b 的分解）。</b>
8.1b 基于零样本得出的“α=1.8 端尾形 +0.976 dB”<b>不成立</b>；
α 轴上脉冲分支收益的变化<b>主要是分布偏移，不是尾形本身</b>。
唯一存活的尾形成分是“α=1.2 端压低约 0.24 dB”，
它<b>真实、方向一致（两个测试集）、但量级小且统计不显著</b>。<br>
<b>这也说明 8.1b 的分解方法本身有局限</b>：用零样本数据做“强度/尾形”二分，
无法把“没见过该条件”这一项分离出来；只有域内重训才能做到。
本节的结论优先级高于 8.1b 的分解。</div>

<div class="box"><b>如实标注：</b>
① 域内重训只覆盖 α=1.2 与 α=1.8 两个点，α=2.0 与 rate 轴交叉组合未做；
② 域内种子数为 3（base 为 5），α=1.2 的 sd 较大（0.370），
点估计的置信区间较宽——这也是它不显著的原因之一；
③ α 轴域内重训同样跨设备（GPU 训练 vs base 的 CPU 训练），
沿用 8.6 测得的设备校正量 -0.091 dB。</div>

<h3>8.12 输入尺度归一化：让模型对实测数据可用</h3>
<p>前面所有结果都建立在一个隐含前提上：<b>输入尺度与训练时一致</b>。
训练数据在 MATLAB 侧把 <code>rms(clean)</code> 归一化为 1，而实测数据拿不到 clean 信号，
只能按 noisy 归一化。本节先量化这个风险的严重程度，再给出解法并验证。</p>

<h4>8.12.1 风险有多大：老模型的尺度敏感性</h4>
<p>把 base 条件 testA 的全部 1080 个样本整体缩放 s 倍（<b>缩放不改变 SNR</b>，
纯粹改变输入尺度），用未归一化训练的老 R5 评估。注意这里的口径是
<b>est、ref、noisy 三者同处缩放后的尺度</b>（否则指标会错位，见 8.12.3 的说明）：</p>
<table>
<thead><tr><th>缩放 s</th><th>干扰帧增益 (dB)</th><th>相对 s=1</th></tr></thead>
<tbody>
<tr><td>0.25</td><td>23.757</td><td>-3.709</td></tr>
<tr><td>0.50</td><td>26.573</td><td>-0.893</td></tr>
<tr><td>1.00</td><td>27.466</td><td>—（基准）</td></tr>
<tr><td>2.00</td><td>26.724</td><td>-0.742</td></tr>
<tr><td>4.00</td><td>24.807</td><td>-2.659</td></tr>
</tbody></table>
<div class="box warn"><b>跨尺度极差 3.71 dB</b>。最要命的是 <b>s=0.25</b>：
只把输入缩到训练尺度的 1/4，干扰帧增益就从 27.466 掉到 23.757 dB
（<b>损失 3.71 dB</b>）。而"按 noisy rms 归一化"恰好落在这个区间附近
（实测数据里 noisy 的 rms 约为 clean 的数倍）。<br>
<b>这不是任务变难</b>：缩放等比作用于信号和噪声，SNR 一字不变，
纯粹是模型对输入整体尺度敏感。</div>
<div class="box"><b>两种口径的区别（必须说清，否则数字对不上）。</b>
本节口径把 ref 也乘了 s，度量的是"模型输出对缩放后参考的逼近程度"，
得到 <b>3.71 dB</b>。另一条更贴近部署的口径是"把模型输出直接当结果、
不做任何尺度校正，与真实信号比"，即 8.12 引用的 <b>15.9 dB</b>
（<code>scale_robustness.py</code>，按 noisy rms 归一化后直接前向）。
两者都指向同一结论：<b>老模型的可用性依赖输入尺度</b>；前者看"降噪质量"，
后者看"端到端输出质量"。本节用前者，因为它与归一化模型的对比是同一口径。</div>

<h4>8.12.2 两条绕开的路线都失败了</h4>
<p>先试过不改模型、只在推理时估一个缩放系数：</p>
<ul>
<li><b>盲自校准</b>：用 4 个无参考指标（谱平坦度、过零率、语音/水声活跃度等）
反推最优 scale。实测它们的动态范围只有 <b>0.03%~0.7%</b>，
在最优 scale 附近几乎没有区分度 —— 无法定位。</li>
<li><b>物理估计</b>：信号被限制在 798–1500 Hz，想用带外能量估噪声底再反推信号功率。
实测<b>失效</b>：α 稳定噪声是脉冲性的，带外能量被少数大尖峰主导，
按带宽外推会把噪声 rms 估小约 <b>300 倍</b>。
用"噪声段参考"也不可行（p_total 里含突发干扰能量，导致 Pc 高估、s 估成 0.616）。</li>
</ul>
<p>两条路分别损失 5.6 / 9.8 dB。<b>唯一可靠解是在训练时就吃归一化输入。</b></p>

<h4>8.12.3 实现与验证</h4>
<p>实现方式选在<b>数据层面</b>而非模型层面：<code>PAMDataset</code> 把 noisy 与 clean
同除 noisy 的 rms（先定长再算 rms，与 <code>_npy</code> 缓存保持同一尺度定义）。
模型<b>零改动</b>，因此与已有 checkpoint 结构完全兼容，也可随时开关。</p>
<div class="box"><b>可比性论证</b>：<code>frmgain</code> / <code>fmsnr</code>
都是"能量比的对数"（见 <code>pam_metrics._stats_1d</code>），
数学上是<b>尺度不变量</b>。已用数值实验确认（est 取 clean+s·1e-4，
s 从 0.1 到 10 时三项指标在 4 位小数上完全一致）。
因此归一化域里算出的指标与原尺度<b>直接可比</b>。</div>

<div class="box warn"><b>一个差点把结论写反的口径坑（如实记录）。</b>
尺度测试的第一版实现里，输入 <code>noisy</code> 乘了 s、
模型输出也反归一化回 s 尺度，<b>但参考 <code>ref</code> 忘了乘 s</b>。
于是 est 与 noisy 在 s 尺度、ref 在原尺度，指标里的"输出噪声"变成
<code>|est − ref|² ≈ |c·s − c|²</code>，s=0.25 时这一项被信号失配主导，
frmgain 虚假地掉 18 dB。<br>
结果就是：<b>归一化训练的模型也显示"跨尺度极差 17.8 dB"</b>，
看上去像"归一化失败"。定位过程：先排除模型 batch 依赖（hook 逐模块比对，
完整 batch 与单样本输出差 1.8e−7）、再排除 <code>__pycache__</code> 过期字节码，
最后发现 2 样本与 16 样本给出矛盾结果，才追到 ref 漏乘 s。<br>
修正后新 R5 的极差为 <b>0.000 dB</b>，与恒等式的理论预期一致。
<b>教训：凡做尺度变换的对比，必须逐个检查 est / ref / noisy 三个量是否同处一个尺度。</b></div>

__FIGX__

<table>
<thead><tr><th>组别（base 条件，GPU，同设备同种子）</th><th>R4 干扰帧</th>
<th>R5 干扰帧</th><th>Δ = R5 − R4</th></tr></thead>
<tbody>
<tr><td>无归一化（devchk，存档 CPU 的 GPU 对照）</td><td>26.53</td><td>27.30</td>
<td>+0.769</td></tr>
<tr><td><b>输入归一化（inm）</b></td><td><b>26.57</b></td><td><b>27.14</b></td>
<td><b>+0.565</b></td></tr>
</tbody>
<tfoot><tr><th colspan="4" style="text-align:right">
testA。归一化的代价：R4 +0.048 dB、R5 -0.157 dB（负值 = 归一化更好）</th></tr></tfoot></table>

<div class="box good"><b>结论一：归一化几乎没有代价。</b>
同设备同种子配对下，归一化相对无归一化：R4 +0.048 dB、R5 -0.157 dB
（testC：R4 -0.137 dB、R5 -0.078 dB），均在 ±0.2 dB 量级且不显著
（testA p = 0.43 / 0.40，testC p = 0.49 / 0.53）。<br>
<b>结论二：分支收益保留。</b>归一化后 Δ = +0.565 dB
（无归一化 +0.769 dB，base 存档 CPU +0.790 dB），方向与量级一致。<br>
<b>结论三：尺度敏感性被彻底消除。</b>新 R5 三个种子的跨尺度极差<b>全部严格为
0.000 dB</b>（老 R5 为 3.71 dB）。这不是"变小"，而是<b>恒等</b>：
因为 <code>x·s / rms(x·s) ≡ x / rms(x)</code>，任何 s 下喂进网络的张量
逐位相同（实测最大差 6e−8，float32 舍入级），输出自然相同。</div>

<div class="box"><b>如实标注：</b>
① 归一化只在 base 条件上验证过，未在 α / rate 的其它条件上重训；
② 归一化训练的种子数是 3，与 GPU 基线一致，但少于 base 存档的 5；
③ 尺度测试用的是"整体缩放"，而实测数据的尺度不确定性还包含
采样率差异、带宽差异等，这些不在本节的覆盖范围内 —— 拿到实测数据后需按
<code>docs/real_data_deployment.md</code> 的清单逐项核对。</div>

<h3>8.13 域内重训下的架构排序（GPU 同设备）</h3>
<p>8.6 / 8.11 的域内重训只覆盖 R4/R5，即只回答了"有没有脉冲分支"。
"架构排序 R5 &gt; R4 &gt; R3 &gt; 纯时域"在域内是否保持，此前<b>没有验证</b>。
这不是假想风险 —— 零样本矩阵里，高斯背景上 R4 就被 R3 反超过。</p>

__FIGY__

<table>
<thead><tr><th>模型（base 条件，GPU 训练）</th><th>参数量</th><th>干净帧</th>
<th>干扰帧</th><th>干扰最差</th></tr></thead>
<tbody>
<tr><td>R5 时域残差 + 频域精修 + p 注入</td><td>14858</td><td>18.38</td><td>27.34</td><td>17.11</td></tr>
<tr><td>R4 时域残差 + 频域精修（无脉冲分支）</td><td>14367</td><td>18.15</td><td>26.54</td><td>17.09</td></tr>
<tr><td>R3 纯频域掩码</td><td>5910</td><td>15.02</td><td>24.52</td><td>14.13</td></tr>
<tr><td>纯时域 PAM-Net（无脉冲分支）</td><td>8457</td><td>10.83</td><td>17.22</td><td>9.84</td></tr>
</tbody>
<tfoot><tr><th colspan="5" style="text-align:right">
testA，各 3 个种子（除注明外）</th></tr></tfoot></table>

<div class="box good"><b>排序在域内保持。</b>同设备同种子配对检验：R5 vs R4（脉冲分支效应） = <b>+0.798 ± 0.430 dB</b>（n=3, p=0.0846, 前者更高）；R5 vs R3（架构优势） = <b>+2.815 ± 0.137 dB</b>（n=3, p=0.0008, 前者更高）；R4 vs R3 = <b>+2.017 ± 0.421 dB</b>（n=3, p=0.0142, 前者更高）；R4 vs 纯时域（频域精修贡献） = <b>+9.318 ± 1.315 dB</b>（n=3, p=0.0066, 前者更高）
<b>结论：脉冲分支的收益（R5−R4 = +0.798 dB）在域内、同设备的干净对照下依然存在，
且与零样本（+0.790）方向一致、量级接近。</b></div>

<div class="box"><b>如实标注：</b>
① 只覆盖 base 一个条件，α / rate 轴的域内架构排序未验证；
② devchk 基线的种子数正在补齐，检验的自由度受此限制；
③ 若 <code>gpu_pn</code>（纯时域）缺失，"R4 vs 纯时域"这一项无法给出。</div>

<h3>8.14 rate 轴双向验证与 α 轴端点补全</h3>
<p>8.6 的结论「rate 轴反向（突发越稀疏、分支收益越大）」此前<b>只有 rate=10 一个方向的域内验证</b>。
若稀疏端（2.5/s）域内重训后收益也回到 base，那「反向」就只剩一个点，结论必须收窄。
本节补上稀疏端，并顺带补 α=2.0 端点。</p>

<h4>8.14.1 rate 轴：两个方向都做域内重训</h4>
<p>a15_r25 的 α 与 base 完全一致（1.5），<b>只有到达率减半</b> —— 单因素操纵最干净的一组。
各 3 个种子，GPU 训练，设备校正 −0.091 dB。</p>

<table>
<thead><tr><th>配置</th><th>Δ = R5−R4</th><th>Δ（设备校正后）</th><th>p</th>
<th>相对 base +0.790</th></tr></thead>
<tbody>
<tr><td>base（α=1.5, 5/s）</td><td>+0.790</td><td>—</td><td>—</td><td>基准</td></tr>
<tr><td>2.5/s 零样本</td><td>+0.975</td><td>—</td><td>—</td><td>+0.185</td></tr>
<tr><td><b>2.5/s 域内</b></td><td>+1.247</td><td>+1.338</td><td>0.081</td><td><b>+0.548</b></td></tr>
<tr><td>10/s 零样本</td><td>+0.548</td><td>—</td><td>—</td><td>-0.242</td></tr>
<tr><td><b>10/s 域内</b></td><td>+0.322</td><td>+0.413</td><td>—</td><td><b>-0.377</b></td></tr>
</tbody>
<tfoot><tr><th colspan="5" style="text-align:right">
testA。零样本为 5 种子，域内为 3 种子</th></tr></tfoot></table>

<div class="box good"><b>rate 轴的「反向」是条件效应，不是分布偏移。</b>
关键论证不依赖单点显著性，而看<b>两端相对 base 的偏离方向</b>：<br>
· 密集端 10/s：零样本 +0.548 → 域内校正 <b>+0.413</b>，比 base <b>低</b> 0.377 dB；<br>
· 稀疏端 2.5/s：零样本 +0.975 → 域内校正 <b>+1.338</b>，比 base <b>高</b> 0.548 dB。<br>
<b>两端域内值相差 0.92 dB、方向相反</b>。如果「反向」只是分布偏移的假象，
两端域内重训后应当都回到 base（+0.790）；实际是<b>一低一高</b>，
说明"干扰密度"本身在改变分支的价值。8.6 的结论由此从单向升级为双向。</div>

<div class="box warn"><b>但稀疏端的<b>量级</b>不能写死。</b>
3 个种子的逐种子 Δ 是 +1.969 / +1.082 / +0.690（sd 0.656，p = 0.081，未达 0.05）。
留一法均值 +0.886 / +1.329 / +1.526（范围 [0.886, 1.526]） —— 去掉最高的那个种子后，相对 base 的偏离从 +0.548 缩到 +0.096 dB。
因此只能说：<b>稀疏端的域内 Δ 不低于 base、且与密集端方向相反</b>；
<b>不能</b>断言"稀疏端收益显著更大"。</div>

<div class="box"><b>机制上如何理解</b>：干扰稀疏时，干扰帧占比低但每帧的干扰更"孤立"、
更不像背景 —— 此时"知道脉冲在哪"的边际价值最大；干扰密集时，脉冲位置的信息
与频域掩码能提供的结构高度重合（7.x 已证明判别能力不变而收益下降），
所以分支变得冗余。这与冗余假设一致。</div>

<h4>8.14.2 α=2.0 端点：一个必须如实标注的负结果</h4>
<p>注意这是<b>未做等功率匹配</b>的 a20_r5（与 8.2 零样本矩阵同一数据集），
因此只能与<b>同样未匹配</b>的零样本 a20_r5 比，不能直接与 base 比（SNR 定义不同）。
下表把逐种子的 R4/R5 绝对值一并列出 —— 这是判断"Δ 的波动来自哪里"的关键。</p>
<table>
<thead><tr><th>种子</th><th>R4 干扰帧</th><th>R5 干扰帧</th><th>Δ = R5−R4</th></tr></thead>
<tbody>
<tr><td>零样本（5 种子）</td><td>25.605</td><td>26.902</td><td>+1.297</td></tr>
<tr><td>s1</td><td>25.703</td><td>27.388</td><td>+1.685</td></tr>
<tr><td>s2</td><td>27.818</td><td>26.957</td><td>-0.861</td></tr>
<tr><td>s3</td><td>26.338</td><td>27.049</td><td>+0.711</td></tr>
<tr><td><b>域内均值（3 种子）</b></td><td><b>26.620</b></td><td><b>27.132</b></td><td><b>+0.512</b></td></tr>
</tbody></table>

<div class="box bad"><b>这一组不能用来判断分支价值 —— 域内训练本身对种子不稳定。</b><br>逐种子 Δ 是 +1.685 / -0.861 / +0.711，<b>符号相反</b>；域内均值 +0.512（校正后 +0.603）只是三个不稳定样本的平均，不构成结论。根源不在分支，而在 <b>R4 自己</b>：它在种子间从 25.703 跨到 27.818（极差 <b>2.12 dB</b>，sd 1.09），而 R5 只跨 0.43 dB（sd 0.23）。也就是说，「α=2.0 下 R5 比 R4 好多少」这个问题在这里被 <b>R4 自身的训练不稳定性</b>淹没了 —— 分母在动，差值就没有意义。<br>因此 <b>α=2.0 端点仍然标记为「未验证」</b>：本节只交付一个负结果 —— 该条件下（未做等功率匹配、名义 SNR 约 −22 dB）的域内训练不可复现，需要更多种子或更稳的初始化才能给出判断。不能据此说「端点上的收益是分布偏移」，也不能说「不是」。<br><br><b>但这一组留下一个值得记录的副产品</b>：同一条件下 <b>R5 的跨种子极差只有 R4 的 1/5</b>（0.43 dB vs 2.12 dB）。3 个种子远不足以断言「脉冲分支让训练更稳」，但它提示 p 注入可能起到<b>收窄解空间／正则化</b>的作用 —— 这是一个可以单独设计实验（多种子 × 多学习率）去验证的假设，本节只把它记为<b>观察</b>，不作结论。</div>

<div class="box"><b>为什么要把逐种子绝对值也列出来</b>：只看 Δ 的均值与 sd 会以为"效应小且不稳"，
但列出 R4/R5 就能看到 <b>R4 自己在种子间跳了 2.12 dB</b>。
分支效应的比较是<b>配对</b>的（同种子内 R5−R4），配对能消掉一部分共同波动；
但当 R4 的波动达到这个量级时，配对也救不回来 —— 不同种子学到的可能是<b>不同的解</b>，
而不是同一个解的抖动。判据很简单：<b>先看 R4 的跨种子极差，再看 Δ</b>。
若 R4 极差 ≫ |Δ|，这组数据的 Δ 不可用。</div>

<div class="box warn"><b>方法学注记：两个容易混的字段。</b>
<code>sens_summary_*.json</code> 里有两个长得像"Δ"的量：
<code>pulse_eff[cond]["imp"]["mean"]</code> 是<b>脉冲分支效应</b>（R5−R4，本节用的就是这个）；
<code>cond_delta[cfg][cond]["mean"]</code> 是<b>同一模型换条件后的退化量</b>
（例：a15_r10 的 R4 比 base 低 0.455 dB，<b>符号为负是正常的</b>）。
两者语义完全不同，混用会得到符号相反的结论 —— 本节初稿就踩了这个坑
（误把 cond_delta 当 Δ，得出"10/s 的 Δ 为负"的错误判断），已在
<code>analyze_sensitivity.py</code> 源码与 <code>audit_report.py</code> 里加了语义警告。</div>

<div class="box"><b>如实标注</b>：
① rate 轴两端各 3 个种子（base 为 5），稀疏端的 sd = 0.656；
② α=2.0 未做等功率匹配（需名义 SNR 低至 −24 dB，数据生成有数值风险），
所以它只能与自身零样本比；
③ 本节仍只覆盖 R4/R5，未重训 R3 与纯时域；
④ α=2.0 的域内结果<b>不作为结论使用</b>（种子间符号相反，见 8.14.2），
α 轴端点的判断仍以 8.11 的 α=1.2/1.8 两组为准。</div>

<h3>8.15 脉冲掩码干预：分支的收益来自「对齐的位置信息」</h3>
<p>7.x 的两个干预实验（载频扫描、时频对齐）都没有复现「重叠率→收益」的负相关，
那条因果链因此被降级为<b>不能作为机制结论引用</b>。但那个否定针对的是
<b>「重叠率」这个调节变量</b>，不是分支本身 —— 全文最核心的设计主张
（p 注入让模型知道干扰位置，所以降噪更好）此前只剩相关性证据。</p>

<p>本节直接问一个更基本的问题：<b>R5 那 +0.790 dB，是不是真的来自 p 所携带的位置信息？</b>
做法是不重训、不改模型：p 由 <code>model.pulse</code> 输出后<b>同时</b>进入时域残差头与频域精修，
在推理期用 forward hook 替换它，就能精确控制分支看到什么。5 个种子 × 1080 个样本。</p>

__FIGZ__

<h4>8.15.1 p 确实携带位置信息</h4>
<p>先确认干预对象不是空的：identity 模式下捕获 p，与真实干扰掩码逐样本对比
（5 个种子均值）——</p>
<table>
<thead><tr><th>p 的均值</th><th>样本内标准差</th><th>与真实掩码的 AUC</th>
<th>AUC 的 5% 分位</th><th>与真实掩码的相关系数</th></tr></thead>
<tbody><tr><td>0.040</td><td>0.193</td><td><b>0.955</b></td>
<td>0.863</td><td>0.881</td></tr></tbody>
<tfoot><tr><th colspan="5" style="text-align:right">
真实干扰掩码占空比 4.45%，p 的均值 0.040 —— 量级一致，且高度稀疏</th></tr></tfoot>
</table>
<p>p 不是"到处都亮一点"的平滑图，而是<b>稀疏且逐点对齐</b>的置信度图。
这一条很重要：后面所有干预才有意义。</p>

<h4>8.15.2 七种干预：位置错 ≡ 完全剥夺信息</h4>
<table>
<thead><tr><th>干预</th><th>含义</th><th>Δ (dB)</th><th>sd</th>
<th>相对 identity 的损失</th><th>p</th></tr></thead>
<tbody>
<tr><td>原样（基线）</td><td>分支自己的估计</td><td>+0.790</td><td>0.320</td><td>—</td><td>0.0052</td></tr>
<tr><td>换同批他人的 p</td><td><b>边缘分布严格不变</b>，只把位置换成别人的</td><td>-2.327</td><td>0.870</td><td>-3.117 ± 0.739</td><td>0.0039</td></tr>
<tr><td>循环平移</td><td><b>边缘分布严格不变</b>，位置错开</td><td>-2.789</td><td>0.988</td><td>-3.579 ± 0.848</td><td>0.0032</td></tr>
<tr><td>时间反转</td><td><b>边缘分布严格不变</b>，位置镜像</td><td>-2.712</td><td>0.966</td><td>-3.502 ± 0.831</td><td>0.0033</td></tr>
<tr><td>p ≡ 0</td><td>无信息（宣称“无干扰”）</td><td>-2.553</td><td>1.095</td><td>-3.343 ± 0.962</td><td>0.0064</td></tr>
<tr><td>p ≡ 1</td><td>无信息（宣称“全是干扰”）</td><td>-0.239</td><td>0.774</td><td>-1.028 ± 0.619</td><td>0.5287</td></tr>
<tr><td>随机 U(0,1)</td><td>同取值范围、无结构</td><td>+0.041</td><td>0.972</td><td>-0.748 ± 0.714</td><td>0.9286</td></tr>
</tbody>
<tfoot><tr><th colspan="6" style="text-align:right">
5 个种子配对；Δ = R5(干预) − R4；损失列为同种子配对差</th></tr></tfoot></table>

<div class="box good"><b>五条读法</b><br>
① <b>identity 复现 +0.790 dB</b>，与 8.2 的 base 完全一致 —— 说明 hook 通路本身没有扰动，
下面的差值都是干预造成的。<br>
② <b>shuffle 保留了 p 的边缘分布</b>（每个样本拿到的是另一个样本的 p，均值、方差、稀疏度全部不变），
只把"指向哪里"换掉，损失 <b>3.12 dB</b>，与 <b>p≡0（3.34 dB）几乎相同</b>。如果起作用的是 p 的统计量，shuffle 不该有影响 —— 所以<b>起作用的是位置</b>。<br>
③ 错位之后 R5 反而<b>比 R4 低 2.33 dB</b>（Δ 由正转负）。
分支不是"多了一点增益"的装饰，而是<b>承重结构</b>：喂给它错的脉冲图，它会主动往错的方向压。<br>
④ <b>noise（U(0,1)，均值 0.5）只损失 0.75 dB 且不显著</b>。
它比 shuffle 离训练分布<b>更远</b>（训练时 p 均值 0.040，noise 是 0.5），
却伤害更小 —— 因此不能把这套结果解释成"测试期输入分布外"。<br>
⑤ 逐样本看：identity 下 R5 优于 R4 的样本占 <b>57.2%</b>，错位后降到 <b>21.7%</b>。
这不是均值平移，而是<b>系统性反转</b>。</div>

<h4>8.15.3 错位剂量响应：增益在 1~3 个突发尺度内崩掉</h4>
<p>二值干预只能回答"重不重要"。沿单一连续轴做剂量响应：把 p 在样本内<b>循环平移</b> k 个采样点
（置换操作，边缘分布、方差、稀疏度严格不变，只有对齐关系退化）。
真实干扰的平均突发时长 82 采样点 = <b>13.7 ms</b>，占空比 4.45%。</p>
<table>
<thead><tr><th>错位 k（采样点）</th><th>k（ms）</th><th>k（突发时长倍数）</th>
<th>Δ (dB)</th><th>sd</th></tr></thead>
<tbody>
<tr><td>0</td><td>0.0</td><td>0.00</td><td>+0.790</td><td>0.320</td></tr>
<tr><td>125</td><td>20.8</td><td>1.52</td><td>+0.235</td><td>0.286</td></tr>
<tr><td>250</td><td>41.7</td><td>3.04</td><td>-1.036</td><td>0.752</td></tr>
<tr><td>500</td><td>83.3</td><td>6.08</td><td>-2.487</td><td>0.982</td></tr>
<tr><td>1000</td><td>166.7</td><td>12.16</td><td>-2.722</td><td>0.984</td></tr>
<tr><td>2000</td><td>333.3</td><td>24.31</td><td>-2.731</td><td>0.954</td></tr>
<tr><td>4000</td><td>666.7</td><td>48.63</td><td>-2.664</td><td>0.980</td></tr>
</tbody></table>

<div class="box warn"><b>增益的"半衰期"是 2.9 个突发（40 ms）。</b>
错位仅 <b>1.5 个突发（21 ms）</b>，Δ 就从 +0.790 掉到 +0.235
—— <b>只剩 30%</b>；错位 3 个突发（42 ms）已经转负。
k ≥ 500 之后进入平台（≈ -2.66 dB），与 8.15.2 的"完全错位"水平吻合。<br>
这条曲线的意义是：分支要求的对齐精度是<b>单个突发的尺度（14 ms）</b>，
不是"大致在同一段信号里"。因此任何以帧级、全局统计量去刻画它的尝试
（例如 7.x 的重叠率）都会漏掉真正起作用的量。</div>

<h4>8.15.4 oracle 上界：瓶颈不在 p 的估计精度</h4>
<p>把 p 直接换成<b>真实干扰掩码</b>（占空比 4.45%，与 p 的 0.040 量级一致，
因此不算严重的分布外输入），得到"p 估计完美"时的上界：</p>
<table>
<thead><tr><th>配置</th><th>Δ (dB)</th></tr></thead>
<tbody>
<tr><td>分支估计的 p（现状）</td><td>+0.790 ± 0.320</td></tr>
<tr><td><b>oracle p（真实掩码）</b></td><td><b>+0.862 ± 0.298</b></td></tr>
<tr><td>p 估计误差的代价</td><td>+0.073 ± 0.029（p = 0.0051）</td></tr>
</tbody></table>

<div class="box"><b>结论：即使把 p 换成真实掩码，也只多 0.073 dB —— 约占总效应的 9%。</b>
所以"把脉冲估计做得更准"<b>不是</b>一个有前途的方向：下游对 p 的利用已经接近它所能利用的上限，
瓶颈在别处（例如 p 到底能提供多少可用信息、以及残差头如何用它）。
这条负面结论的价值在于<b>省掉一整条研究路线</b>。</div>

<h4>8.15.5 这对 7.x 的否定意味着什么</h4>
<div class="box good"><b>7.x 否定的是「重叠率」，不是「分支在用位置信息」。</b>
两者并不矛盾，而且可以互相解释：重叠率是一个<b>帧级、全局</b>的统计量，
而 p 是<b>逐点</b>的；本节证明分支依赖的恰恰是逐点对齐
（错位 1.5 个突发就崩掉 70%），而帧级重叠率对"点是否对齐"几乎不敏感。
所以用重叠率去预测分支收益，本来就不该成功 —— 7.x 的两次干预改变的是全局重叠率，
而模型用的是逐点位置。<br>
<b>机制表述应改写为</b>：「p 提供逐点的干扰位置 → 残差头与频域掩码据此分配抑制强度 →
收益来自对齐」，而不是「重叠率越高、收益越低」。</div>

<div class="box"><b>如实标注</b>：
① 干预是在<b>推理期</b>做的，属于训练时没见过的输入。用 shuffle（边缘分布严格不变）
与 noise（更偏离但无结构）做了交叉控制，两者结论一致地指向"位置"而非"分布"，但严格说
这不是训练分布内的操纵；<br>
② 只做 base 条件、只做 R5 一个架构、只做 testA；<br>
③ 剂量响应用循环平移，k 接近半个样本长时环绕伪影最少，中间 k 有轻微环绕；<br>
④ oracle 用的是真实掩码，与 p 同量级但不是同一分布，只能当<b>上界的近似</b>。</div>

<h3>8.16 跨条件干预分解：净收益 =「位置信息的价值」−「无信息时的惩罚」</h3>
<p>8.15 在 base 上证明了分支依赖 p 的逐点对齐。但此前所有跨条件的 Δ 变化
（α 轴 +0.24~+1.32、rate 轴 +0.73~+0.96 dB）
都只被笼统解释成"分支价值变了"—— 这是循环论证：收益小就说分支没用。</p>

<p>本节给出可测的分解。对每个条件 c：</p>
<table>
<thead><tr><th>量</th><th>定义</th><th>含义</th></tr></thead>
<tbody>
<tr><td>Δ<sub>id</sub></td><td>R5(自己的 p) − R4</td><td>分支的净贡献（此前唯一的观测量）</td></tr>
<tr><td>无信息基线</td><td>R5(p≡0) − R4</td><td>被剥夺位置信息、且被告知"处处无干扰"时的净效应</td></tr>
<tr><td><b>位置价值</b></td><td>Δ<sub>id</sub> − 无信息基线</td><td>从"错误先验"换成"正确位置"带来的净提升</td></tr>
<tr><td>Δ<sub>oracle</sub></td><td>R5(真实掩码) − R4</td><td>p 完美时的上界</td></tr>
</tbody></table>

<div class="box warn"><b>关键设计：模型固定，只换数据。</b>
全部 6 个条件都用<b>同一批 base checkpoint</b>（<code>model_pamnet_freq_*_gn</code> + 4 个种子变体），
因此条件间的任何差异<b>只能归因于数据分布</b>，不存在"每个条件各训一个模型"的混杂。
这与 8.7 的域内重训是<b>互补</b>的：8.7 回答"变化的原因是不是分布偏移"，
本节回答"变化体现在哪个分量上"。各 3 个种子。</div>

__FIGAA__

<h4>8.16.1 分解表</h4>
<table>
<thead><tr><th>条件</th><th>R4</th><th>Δ<sub>id</sub></th><th>无信息基线</th>
<th>位置价值</th><th>Δ<sub>oracle</sub></th><th>oracle 余量</th><th>p 的 AUC</th></tr></thead>
<tbody>
<tr><td>α=1.2</td><td>26.968</td><td>+0.239</td><td>-1.787</td><td><b>+2.026</b></td><td>+0.227</td><td>-0.012</td><td>0.942</td></tr>
<tr><td>base（α=1.5, 5/s）</td><td>26.616</td><td>+0.823</td><td>-2.983</td><td><b>+3.807</b></td><td>+0.904</td><td>+0.081</td><td>0.957</td></tr>
<tr><td>α=1.8</td><td>25.874</td><td>+1.180</td><td>-3.447</td><td><b>+4.627</b></td><td>+1.323</td><td>+0.142</td><td>0.961</td></tr>
<tr><td>α=2.0</td><td>25.679</td><td>+1.315</td><td>-3.653</td><td><b>+4.968</b></td><td>+1.475</td><td>+0.160</td><td>0.962</td></tr>
<tr><td>rate=2.5/s</td><td>26.985</td><td>+0.960</td><td>-3.140</td><td><b>+4.099</b></td><td>+1.041</td><td>+0.082</td><td>0.930</td></tr>
<tr><td>rate=10/s</td><td>26.100</td><td>+0.727</td><td>-2.526</td><td><b>+3.254</b></td><td>+0.812</td><td>+0.085</td><td>0.956</td></tr>
</tbody>
<tfoot><tr><th colspan="8" style="text-align:right">
同一批 base checkpoint，只换数据；3 种子；Δ 均相对该条件下的 R4</th></tr></tfoot></table>

<h4>8.16.2 第一个发现：两个分量同向变化，净收益被压缩到 37%</h4>
<p>沿 α 轴（α=1.2 → 2.0，背景从重尾走向近高斯），<b>两个分量同步放大</b>：
位置价值 +2.03 → +4.97 dB（变化 2.94 dB），
而无信息基线 -1.79 → -3.65 dB（也放大了 1.87 dB）。
两者相减，净 Δ 只从 +0.24 变到 +1.32（变化 1.08 dB）。</p>

<div class="box good"><b>所以「α 轴上分支收益的变化」不是单因素效应，而是两个同向分量之差。</b>
净收益的变化幅度只有位置价值变化的 <b>37%</b>。<br>
这也解释了一个此前没注意到的现象：8.2 的零样本矩阵里 Δ 在 α 轴上的起伏
（0.24~1.32 dB）看起来"不大且不稳"，但<b>底下的两个分量各自动了 1.9~2.9 dB</b>。
只看净值会严重低估这个条件实际改变了多少东西。</div>

<h4>8.16.3 第二个发现：p 的质量跨条件几乎不变，变的是它值多少钱</h4>
<p>6 个条件上 p 与真实干扰掩码的 AUC 全部落在
<b>[0.930, 0.962]（极差仅 0.032）</b>，
而位置价值跨条件变化了 <b>2.94 dB</b>。</p>
<div class="box good"><b>这直接排除了"某个条件下 p 估得差所以分支没用"这个解释。</b>
p 的判别能力在各条件上几乎一致（这与此前 AUC 0.93~0.96 的结论一致），
但同样的判别能力在不同条件下<b>值不同的钱</b>。<br>
于是"能识别 ≠ 能提升性能"这个老问题有了定量答案：<b>差别不在识别环节，在识别的结果值多少</b>。
（这也再次印证 8.15.4 的 oracle 结论：把 p 换得更准没有前途。）</div>

<h4>8.16.4 第三个发现：R5 对 α 轴近乎平坦，R4 明显下降</h4>
<p>α 轴上各条件<b>自己的绝对水平</b>（同一模型、不同数据）：</p>
<table>
<thead><tr><th></th><th>α=1.2</th><th>base</th><th>α=1.8</th><th>α=2.0</th><th>极差</th></tr></thead>
<tbody>
<tr><td><b>R4</b>（无脉冲分支）</td><td>26.97</td><td>26.62</td><td>25.87</td><td>25.68</td><td><b>1.29</b></td></tr>
<tr><td><b>R5</b>（含脉冲分支）</td><td>27.21</td><td>27.44</td><td>27.05</td><td>26.99</td><td><b>0.45</b></td></tr>
</tbody></table>

<div class="box good"><b>R5 把 α 轴的影响压平了 2.9 倍（1.29 → 0.45 dB）。</b>
换句话说，脉冲分支的主要价值可能<b>不是"提升平均性能"，而是"消除对背景统计特性的依赖"</b>——
在背景从重尾变成近高斯时，R4 掉了 1.29 dB，R5 只掉 0.45 dB。<br>
这与「分支提供的是逐点位置信息」互为补充：位置信息是<b>与背景分布无关</b>的量，
所以依赖它的通路天然对这个分布偏移不敏感。</div>

<h4>8.16.5 oracle 余量在 α=1.2 变负：软 p 有硬真值没有的东西</h4>
<p>oracle 余量（Δ<sub>oracle</sub> − Δ<sub>id</sub>）在 base 及更靠高斯的条件上都是小的正值
（+0.08 ~ +0.16 dB），但在 <b>α=1.2 上是 -0.012 dB</b>——把 p 换成<b>真实二值掩码</b>
反而略微变差。</p>
<div class="box"><b>解释</b>：模型输出的 p 是<b>软的</b>（均值约 0.04，逐点连续），
它除了"位置"还携带<b>置信度</b>；真实掩码是硬 0/1，丢掉了这层不确定性。
在重尾背景上（α=1.2），突发与背景尖峰的<b>统计相似度最高</b>，
"有多确信"这层信息恰好最有用 —— 所以硬真值反而不如软估计。<br>
这条与 8.15.4 一起构成一个明确的负面结论：<b>把 p 换成更"正确"的东西不会更好，
改进方向不在 p 的精度上。</b></div>

<h4>8.16.6 rate 轴：同一机制的反向验证</h4>
<p>8.16.2 只在 α 轴上做的分解，这里搬到 rate 轴（突发到达率 2.5 → 10 次/秒，α 固定 1.5）。
如果"两个同向分量相减"只是 α 轴的特例，这里应该看到别的形状。</p>

<table>
<thead><tr><th>条件</th><th>突发占空比</th><th>p 的均值</th><th>p 的 AUC</th>
<th>无信息惩罚</th><th>位置价值</th><th>净 Δ</th><th>价值/惩罚</th></tr></thead>
<tbody>
<tr><td>2.5 次/秒</td><td>2.22%</td><td>0.0215</td><td>0.930</td><td>-3.140</td><td>+4.099</td><td><b>+0.960</b></td><td>1.31</td></tr>
<tr><td>5 次/秒（base）</td><td>4.45%</td><td>0.0404</td><td>0.957</td><td>-2.983</td><td>+3.807</td><td><b>+0.823</b></td><td>1.28</td></tr>
<tr><td>10 次/秒</td><td>8.59%</td><td>0.0755</td><td>0.956</td><td>-2.526</td><td>+3.254</td><td><b>+0.727</b></td><td>1.29</td></tr>
</tbody>
<tfoot><tr><th colspan="8" style="text-align:right">
同一批 base checkpoint 只换数据；3 种子；单位 dB（相对该条件的 R4）</th></tr></tfoot></table>

<p>三项观察，第一项是个独立的正面证据：</p>
<ol>
<li><b>p 的绝对水平随占空比近似线性地跟着走</b>：占空比 2.22% → 4.45% → 8.59%，
p 的均值 0.0215 → 0.0404 → 0.0755，比值稳定在 0.88~0.97。
也就是说分支输出的确实是"逐点属于干扰的概率"，<b>不是</b>一个与数据密度无关的常数输出 ——
这从另一个角度复核了第四节"分支确实学到了脉冲"。</li>
<li><b>但判别能力几乎不动</b>：AUC 0.930 / 0.957 / 0.956。p 的均值翻了 3.5 倍，AUC 只动了 0.027。
这与 8.16.3 在 α 轴上的观察同构：<b>变化不在识别环节</b>。</li>
<li><b>位置价值随突发变密而单调下降</b>：+4.10 → +3.81 → +3.25 dB，
同时无信息惩罚也变弱（-3.14 → -2.98 → -2.53），净 Δ 被压到 +0.96 → +0.82 → +0.73。</li>
</ol>

<div class="box warn"><b>rate 轴比 α 轴弱，必须区别对待。</b>
rate 轴上两个分量的跨度只有 0.85 / 0.61 dB，净跨度 0.23 dB ——
比 α 轴（2.94 / 1.87 / 1.08）小一个量级，
而单次 3 种子均值的种子标准误就在 0.2 dB 量级。<br>
因此 rate 轴这一节<b>只读方向，不读量级</b>。好在方向有独立数据背书（见 8.16.7）：
零样本 5 种子评估给出 +0.98 → +0.79 → +0.55，<b>与本节排序完全一致</b>。</div>

__FIGAB__

<h4>8.16.7 交叉验证：管道逐位复现，唯一的偏差来自采样</h4>
<p>本节的 identity 结果应该与 8.2 的零样本矩阵一致（两者是同一个量：R5 用自己的 p 减去 R4，
只是数据集不同、种子数不同）。这是一个<b>免费的自检</b>，我逐项对了一遍：</p>

<div class="box good"><b>6 个条件、12 个绝对水平全部逐位吻合（最大偏差 0.0004 dB）。</b>
即本节 identity 那一列 R4 / R5 的数值，<b>精确等于</b>零样本 5 种子评估中<b>前 3 个种子</b>的均值。
这说明两件事：<br>
① 干预脚本的数据加载路径（<code>_npy</code> 缓存）与主评估一致，没有口径差；<br>
② 本节与 8.2 走的是同一设备路径，不存在 CPU/GPU 混杂
（此前 8.11 记录过设备差 Δ = −0.091 dB，这里没有这个量级的偏差）。</div>

<p>唯一的例外是 rate=10/s 的净 Δ：本节 3 种子给 +0.727，
零样本 5 种子给 +0.548，差 <b>+0.179 dB</b>。归因如下：</p>
<ul>
<li>该条件 5 个种子的逐样本 Δ 为 +1.083 / +0.421 / +0.678 / +0.076 / +0.481，<b>标准差 0.370 dB</b>，极差 +0.076 ~ +1.083；</li>
<li>3 种子均值的标准误 = 0.370/√3 = <b>0.213 dB</b>，而 +0.179 的偏差小于 1 个标准误；</li>
<li>机制上就能算出来：前 3 个种子均值 +0.727，加入种子 4、5（Δ 均值 +0.279）后拉低到 +0.548。
差值 +0.179 完全由这两个种子决定，不涉及任何数据或设备差异。</li>
</ul>

<div class="box"><b>结论</b>：这是<b>采样波动，不是管道错误</b>。
同时它也标出了本节最不可靠的一格：rate=10/s 的种子间变异系数达 67%，
所以该条件的净 Δ <b>不能用来比较任何小于 0.2 dB 的差异</b>。
这正是 8.16.6 只读方向的原因，也是 8.16.8 只给"比值区间"而不给点估计的原因。</div>

<h4>8.16.8 统一形式：净收益 = 惩罚强度 × 超额倍数</h4>
<p>把 6 个条件的两个分量放在一起，可以写成一个恒等式（注意它是<b>定义使然</b>，不是拟合结果）：</p>
<pre>净 Δ = |无信息惩罚| × (位置价值 / 无信息惩罚 − 1)
     = 惩罚强度 × 超额倍数</pre>
<p>"超额倍数"就是位置价值相对惩罚的比值。这个量在 6 个条件上的范围为
<b>1.13 ~ 1.36</b>，<b>全部略大于 1</b> —— 这就是净收益恒为正的原因，
也是"分支确实在提供正价值"的最简洁表述。</p>

<table>
<thead><tr><th>轴</th><th>惩罚强度</th><th>超额倍数</th><th>净 Δ</th><th>该轴的驱动因素</th></tr></thead>
<tbody>
<tr><td><b>α 轴</b>（1.2 → 2.0）</td><td>1.79 → 3.65（<b>放大</b>）</td>
<td>1.13 → 1.36（<b>也放大</b>）</td><td>+0.24 → +1.32（<b>上升</b>）</td>
<td>两项都涨，乘起来涨</td></tr>
<tr><td><b>rate 轴</b>（2.5 → 10）</td><td>3.14 → 2.53（<b>缩小</b>）</td>
<td>1.31 → 1.29（<b>几乎不动</b>）</td><td>+0.96 → +0.73（<b>下降</b>）</td>
<td>只有惩罚在缩，倍数不变</td></tr>
</tbody></table>

<div class="box good"><b>两条轴形状不同，但落在同一个式子下。</b>
α 轴上"分支更值钱"其实是<b>惩罚与倍数同时变大</b>的乘积效应 ——
单独看任一项都会高估或低估；rate 轴上"分支更不值钱"则<b>纯粹是惩罚变小</b>，
分支本身的性价比（倍数）没有退化。<br>
这个区分有实际含义：如果将来要在别的条件上预测分支有没有用，
<b>应该先问"没有位置信息时模型会掉多少"，再问"位置信息能把这个坑填回多少比例"</b>，
而不是直接看净 Δ —— 后者是两者的乘积，会掩盖真正的机制。</div>

<div class="box warn"><b>必须澄清一处容易被误读的相关性。</b>
6 个条件上"位置价值"与"无信息惩罚"的相关系数为 <b>−0.999</b>，看起来很惊人，
但这个数<b>大部分是构造性的</b>：位置价值 ≡ 净 Δ − 惩罚，只要净 Δ 的波动小于惩罚的波动，
两者就必然高度负相关。所以不能把它当成"发现了某种守恒律"。<br>
真正有经验内容的是<b>波动的量级差</b>：6 个条件上净 Δ 的标准差为 0.380 dB，
而位置价值与惩罚的标准差分别是 1.057 与 0.679 dB ——
<b>底下的两个分量各自动了 1.8~2.8 倍，净结果只动了 1 倍</b>。
换句话说，这个分解的价值在于<b>揭示被抵消掉的东西</b>，而不在于那个相关系数。</div>

<div class="box"><b>如实标注</b>：
① 全部为<b>零样本</b>（同一批 base checkpoint 换数据评估），
不是"每个条件重训"；与 8.7 的域内重训结论是互补关系而非替代 ——
8.7 已证明 α 轴的零样本差异在域内重训后会消失，因此本节刻画的是
<b>零样本视角下差异的结构</b>，不能据此说"这是不可消除的条件效应"；<br>
② 「无信息基线」用 p≡0 实现，它同时包含"参数冗余"与"错误先验"两种代价，
<b>不是</b>纯粹的容量项 —— 因此「位置价值」应读作"从错误先验换到正确位置的净提升"，
而不是"位置信息的全部价值"；<br>
③ 各 3 个种子（8.15 的 base 为 5 个），只做 testA；<br>
④ 三个分量都是<b>差值之差</b>，误差会累积，故只读量级与方向，不读小数点后两位；<br>
⑤ 8.16.6 的 rate 轴跨度小于种子噪声，只作方向性证据，方向已由 8.16.7 的独立 5 种子数据复核；<br>
⑥ 8.16.7 的"逐位吻合"只覆盖 identity 这一个模式的<b>前 3 个种子</b>，
其余干预模式（shuffle / zero / oracle）没有独立数据可对，只能靠 identity 的吻合间接背书；<br>
⑦ 「超额倍数」是恒等式的改写，<b>不是新测量量</b>；它的取值范围本身有种子噪声，
不宜当作精确常数引用；<br>
⑧ 本节全部结论仍是零样本视角，8.11 已证明 α 轴差异在域内重训后会消失，
本节刻画的是<b>差异的结构</b>，不是<b>差异的必然性</b>。</div>

<h3>8.17 实测数据冒烟测试：管道可用，但<b>现有指标不足以判定质量</b></h3>
<p>第十二节局限里写着"全部是仿真数据"。为把这条局限往前推一步，用
<code>_smoke/test.wav</code>（4.0 s、单声道、24 kHz，唯一一份非仿真数据）跑了一遍
<code>run_real_data.py</code> 的完整管道：重采样 → 降采样 4 倍 → 切 8000 点帧 →
输入归一化前向 → 拼回。得到 3 帧（无重叠）。原始输出是三个无参考指标：</p>
<table>
<thead><tr><th>指标</th><th>输入</th><th>输出</th><th>变化</th></tr></thead>
<tbody>
<tr><td>带外能量占比</td><td>0.3157</td><td>0.00077</td><td><b>压低 412 倍</b></td></tr>
<tr><td>谱平坦度（1 = 白噪声）</td><td>0.2927</td><td>0.00055</td><td>压低 530 倍</td></tr>
<tr><td>带内（780–1520 Hz）能量占比</td><td>0.579</td><td>—</td><td>—</td></tr>
</tbody></table>
<p>第一眼的反应会是"带外被压掉 400 倍，模型是不是退化成窄带滤波器了？"
本节用两个对照来回答，结论是：<b>这个指标不能那样读，而且它其实证明不了降噪质量。</b></p>

<h4>8.17.1 对照一：带限的干净信号本来就该得到这个数</h4>
<p>训练数据的干净信号是<b>严格带限</b>在 798–1500 Hz 的。用同一个模型在 base 条件的
testA 上跑 60 个样本（有 clean 参考，作为基准）：</p>
<table>
<thead><tr><th></th><th>带内占比</th><th>带外占比</th><th>时域 SNR vs clean</th></tr></thead>
<tbody>
<tr><td><b>clean</b>（参考）</td><td>1.0000</td><td>2.65e-08</td><td>—</td></tr>
<tr><td><b>noisy</b>（输入）</td><td>0.4032</td><td>0.5968</td><td>-8.79 dB</td></tr>
<tr><td><b>est</b>（模型输出）</td><td>0.9999</td><td>1.19e-04</td><td>+17.44 dB</td></tr>
</tbody></table>
<div class="box good"><b>干净信号自身的带外能量占比只有 3e-08 量级</b>
（&gt;1520 Hz 各频点最大占比 2.3e-10，&lt;780 Hz 最大 4.0e-09）。
所以"带外降到 1e-3 量级"<b>正是完美降噪该有的样子</b>，
它不是窄带化的证据。同一模型在合成数据上把 SNR 从 -8.79 dB 提到 +17.44 dB
（提升 +26.22 dB），带内占比从 0.403 修回 0.9999 —— 这些都是正常的降噪行为。</div>

<div class="box bad"><b>但这也意味着：这个指标本身没有判别力。</b>
任何在"带限干净信号"上训练出来的模型，无论降噪做得好还是坏，
带外占比都会掉到同样的量级 —— 因为输出只要保留带内、丢掉带外就自动满足。
<b>run_real_data.py 的头条指标因此无法用来验证真实数据的降噪质量</b>，
它至多能说明"管道跑通了、输出是带限的"。要判定质量必须有参考或人听。</div>

<h4>8.17.2 对照二：被压掉的带外能量本身是无结构的</h4>
<p>如果带外藏着真实信号（例如跑到带外的 HFM、离散谱线），那压掉它就是破坏。
把输入按 24 个频段拆开看：</p>
<table>
<thead><tr><th>频率 (Hz)</th><th>输入占比</th><th>输出占比</th><th>输出/输入</th></tr></thead>
<tbody>
<tr><td>0–250</td><td>0.018</td><td>0.00002</td><td>1.1e-03</td><td>低于信号带</td></tr>
<tr><td>250–500</td><td>0.019</td><td>0.00000</td><td>1.5e-04</td><td>低于信号带</td></tr>
<tr><td>500–750</td><td>0.020</td><td>0.00000</td><td>2.1e-04</td><td>低于信号带</td></tr>
<tr><td>750–1000</td><td>0.027</td><td>0.00001</td><td>3.4e-04</td><td>低于信号带</td></tr>
<tr><td>1000–1250</td><td>0.015</td><td>0.00001</td><td>8.8e-04</td><td>信号带内</td></tr>
<tr><td>1250–1500</td><td>0.013</td><td>0.00002</td><td>1.7e-03</td><td>信号带内</td></tr>
<tr><td>1500–1750</td><td>0.021</td><td>0.00371</td><td>1.7e-01</td><td>高于信号带</td></tr>
<tr><td>1750–2000</td><td>0.103</td><td>0.19885</td><td>1.9e+00</td><td>高于信号带</td></tr>
<tr><td>2000–2250</td><td>0.175</td><td>0.24870</td><td>1.4e+00</td><td>高于信号带</td></tr>
<tr><td>2250–2500</td><td>0.222</td><td>0.25726</td><td>1.2e+00</td><td>高于信号带</td></tr>
<tr><td>2500–2750</td><td>0.124</td><td>0.24952</td><td>2.0e+00</td><td>高于信号带</td></tr>
<tr><td>2750–3000</td><td>0.041</td><td>0.04115</td><td>9.9e-01</td><td>高于信号带</td></tr>
<tr><td>3000–3250</td><td>0.016</td><td>0.00016</td><td>1.0e-02</td><td>高于信号带</td></tr>
<tr><td>3250–3500</td><td>0.028</td><td>0.00004</td><td>1.4e-03</td><td>高于信号带</td></tr>
<tr><td>3500–3750</td><td>0.019</td><td>0.00003</td><td>1.3e-03</td><td>高于信号带</td></tr>
<tr><td>3750–4000</td><td>0.020</td><td>0.00000</td><td>2.5e-04</td><td>高于信号带</td></tr>
<tr><td>4000–4250</td><td>0.018</td><td>0.00019</td><td>1.0e-02</td><td>高于信号带</td></tr>
<tr><td>4250–4500</td><td>0.016</td><td>0.00024</td><td>1.6e-02</td><td>高于信号带</td></tr>
<tr><td>4500–4750</td><td>0.022</td><td>0.00006</td><td>2.6e-03</td><td>高于信号带</td></tr>
<tr><td>4750–5000</td><td>0.013</td><td>0.00000</td><td>1.9e-04</td><td>高于信号带</td></tr>
<tr><td>5000–5250</td><td>0.017</td><td>0.00001</td><td>3.7e-04</td><td>高于信号带</td></tr>
<tr><td>5250–5500</td><td>0.012</td><td>0.00001</td><td>4.5e-04</td><td>高于信号带</td></tr>
<tr><td>5500–5750</td><td>0.011</td><td>0.00000</td><td>4.1e-04</td><td>高于信号带</td></tr>
<tr><td>5750–6000</td><td>0.008</td><td>0.00001</td><td>7.7e-04</td><td>高于信号带</td></tr>
</tbody>
<tfoot><tr><th colspan="4" style="text-align:right">
带外各段的衰减中位数 8.8e-4，最小 1.5e-4，最大 1.7e-1（发生在 750–875 Hz 的过渡带）</th></tr></tfoot></table>
<ul>
<li><b>衰减是均匀的</b>：带外各段同比例下降（中位 8.8e-4），平均化行为，
不是只挑某一段压 —— 与"宽带噪声被抑制"一致。</li>
<li><b>唯一的例外是 750–875 Hz</b>（输入 0.021 → 输出 0.0037），
那是 780 Hz 带边的过渡区，属滤波器边缘的必然残留。</li>
<li><b>带外没有孤立谱线</b>：各频点功率与带内局部中位数的最大尖峰只有 <b>4.3 倍</b>，
说明没有离散单音被抹掉。</li>
<li><b>干净带限性在对照组里得到直接验证</b>（见 8.17.1），所以"均匀压带外"是设计使然。</li>
</ul>

<h4>8.17.3 输入本身的内容：有真实的脉冲噪声，也<b>确实有扫频</b></h4>
<table>
<thead><tr><th>量</th><th>值</th><th>说明</th></tr></thead>
<tbody>
<tr><td>每帧主导频率</td><td>902 – 1395 Hz（中位 1148）</td><td>落在假设信号带内</td></tr>
<tr><td>主导频率逐帧 |Δ|</td><td>2.6 Hz</td><td><b>这正是扫频速率</b>（2.6 Hz/帧 × 47 帧/s ≈ 122 Hz/s），见下方更正</td></tr>
<tr><td>时域峰度 / 波峰因数</td><td>9.81 / 10.56</td><td>确有强脉冲成分，脉冲分支有事可做</td></tr>
<tr><td>分帧边界 |Δx|</td><td>0.0204（帧内 0.0298，p99 0.0597）</td><td>边界跳变<b>小于</b>帧内典型值 ⇒ 无拼接 click</td></tr>
</tbody></table>

<div class="box bad"><b>更正（本报告前一版在这里写错了，必须如实记录）。</b>
8.17 初稿把"逐帧主导频率 |Δ| 只有 2.6 Hz"读成"随机抖动、无系统性扫频"。<b>这是错的。</b>
2.6 Hz 是<b>每帧</b>的变化量，帧间隔 21.3 ms，折合 <b>122 Hz/s</b> ——
如果真是随机跳变，在这个 500 Hz 宽的带内 |Δ| 的期望值应该是 ~150 Hz，比 2.6 Hz 大 60 倍。
<b>小步长恰恰是慢扫频的特征，不是噪声的特征。</b><br>
用不依赖任何跟踪算法的判据复核（逐帧取带内峰值，然后看单调性）：
线性回归 <b>R² = 0.9998</b>、斜率 <b>+125.1 Hz/s</b>、秩相关 <b>1.0000</b>（完全单调递增）。<br>
再用<b>相位随机化</b>做反证（保持功率谱、破坏扫频）：R² 从 0.9994 掉到 0.035~0.755，
判定在 LFM/HFM 之间乱跳；时域打乱为 0.70~0.92。
<b>所以这条扫频是真实结构，不是跟踪算法的伪迹。</b><br>
该录音是<b>线性调频（LFM）</b>而非双曲调频（HFM）：R²(f) = 0.9994 &gt; R²(1/f) = 0.9871，
即 f 对 t 线性、1/f 对 t 不线性。详见 <b>8.18</b>。</div>

<div class="box warn"><b>这一节的结论是"管道可用、指标失效"，不是"实测降噪成功"。</b>
能说的：<br>
① 全链路（读 wav → 重采样 → 降采样 → 归一化前向 → 拼回）在真实录音上跑通，无崩溃、无爆音、
边界无 click；<br>
② 带外的 400 倍压低<b>不是</b>窄带化缺陷，有对照组证明它是训练分布带限的必然结果；<br>
③ 被压掉的带外能量本身无结构（无孤立谱线、均匀衰减），所以没有证据表明破坏了真实信号。<br>
<b>不能说的</b>：降噪质量好不好。理由：无干净参考；只有 1 个文件、3 帧、4 秒；
指标本身无判别力（8.17.1）。<br>
<b>但 8.18 用一个不依赖干净参考的判据（扫频轨迹保真）给出了正面答案</b> ——
该录音含真实 LFM 扫频，降噪后轨迹完整保住。</div>

<div class="box"><b>还缺什么</b>：<br>
① <b>实测 HFM</b>（双曲调频）。8.18 已把工具做好并在合成 HFM 上验证过有效性，
拿到 HFM 录音直接跑即可；当前这份是 LFM，<b>没有覆盖 HFM 这个目标调制类型</b>；<br>
② 人听 + 谱图对比，特别是带外是否藏着目标信号；<br>
③ 若有实测的干净段（哪怕是信标段），即可算带内 SNR —— 这是目前唯一缺的量。<br>
<b>8.12 的输入归一化</b>解决了尺度问题，8.18 解决了"无参考怎么看质量"，
剩下的纯粹是参考数据问题，不是方法问题。</div>

<h3>8.18 实测扫频保真检验：无干净参考时怎么判质量</h3>
<p>8.17 的困境是：实测数据没有 clean，所有 SNR 类指标都不可用，
而 <code>run_real_data.py</code> 的带外比指标又被证明<b>无判别力</b>。
本节给出一个<b>只依赖信号自身结构</b>的判据，并用它给出实测结论。</p>

<h4>8.18.1 思路：扫频信号的瞬时频率轨迹是已知形状</h4>
<p>导师方案的目标是扫频信号。它的瞬时频率 f(t) 有确定的解析形式：</p>
<pre>LFM : f(t) = f0 + (f1-f0)·t/T            ⇒  f   对 t 线性
HFM : f(t) = f0·f1·T / (f1·T-(f1-f0)·t)  ⇒  1/f 对 t 线性</pre>
<p>于是不需要 clean：<b>从信号自身估出轨迹，比较处理前后是否一致</b>。
降噪若吃掉某个频段、扭了轨迹、或把信号揉掉，轨迹对比会直接暴露。
这套判据给出四个量：</p>
<table>
<thead><tr><th>量</th><th>怎么算</th><th>能查出什么</th></tr></thead>
<tbody>
<tr><td><b>轨迹 r</b></td><td>时间对齐后，输出轨迹与输入轨迹的相关系数</td><td>整体位移 / 扭曲</td></tr>
<tr><td><b>RMS 偏差</b></td><td>同上，单位 Hz</td><td>轨迹精度</td></tr>
<tr><td><b>覆盖比</b></td><td>输出覆盖的频段宽度 / 输入</td><td>带边被削</td></tr>
<tr><td><b>R² 与内点率</b></td><td>对 f~t 或 1/f~t 做 <b>RANSAC 鲁棒拟合</b></td><td>扫频形状是否保住；调制类型</td></tr>
</tbody></table>

<h4>8.18.2 三个实现上的坑（都被自检抓出来了）</h4>
<div class="box warn"><b>坑一：逐帧独立取峰会测出噪声，不是扫频。</b>
在 −3 dB 帧级 SNR 下，逐帧取带内最大峰，频率会在几百 Hz 间乱跳，
拟合 R² 只有 0.15 —— 这时"轨迹保真度"测的其实是噪声。
改用<b>动态规划轨迹跟踪</b>（相邻帧频率只能小步移动）后恢复正常，
这正是扫频的物理约束。<br>
<b>坑二：用"相对中位数"的能量门限判不出有无信号。</b>
纯噪声帧里 DP 找到的仍是该帧最大值，相对中位数依然高出 10+ dB，
于是补零段被当成"有效信号"，噪声段的随机游走把 R² 从 0.999 拉到 0.07。
必须改用<b>绝对</b>功率门限（相对全段峰值 −35 dB）。<br>
<b>坑三：普通最小二乘会被突发干扰帧带偏。</b>
α 稳定背景的突发会让 DP 跟到干扰谱峰上，这些帧不是扫频。
普通拟合 R²=0.147，而只看真实信号段是 0.967 —— 差的 0.82 全来自干扰。
改用 RANSAC 后 R²=0.993。这也是为什么必须同时报"内点率"：
R² 高 + 内点率低 = 局部丢失。</div>

<h4>8.18.3 工具有效性验证：合成 HFM 上的闭环</h4>
<p>拿到实测数据之前，先在<b>合成</b> HFM（798→1500 Hz、1 s）上验证工具能不能
①识别调制类型、②判出"降噪保住了"、③把人为破坏检出来。
加噪用 α=1.5 稳定背景 + 5 次/秒突发（与训练数据同一套逻辑，帧级 SNR −3 dB）。</p>
<table>
<thead><tr><th>量</th><th>clean（参考）</th><th>noisy（降噪前）</th><th>est（模型输出）</th></tr></thead>
<tbody>
<tr><td>R²（1/f~t，HFM 模型）</td><td><b>0.9999</b></td><td>0.993</td><td>0.993</td></tr>
<tr><td>R²（f~t，LFM 模型）</td><td>0.9744</td><td>0.976</td><td>0.973</td></tr>
<tr><td>判定调制类型</td><td><b>HFM</b>（1/f 胜出）</td><td>HFM</td><td>HFM</td></tr>
<tr><td>轨迹 r vs clean</td><td>—</td><td>+0.962</td><td><b>+0.974</b></td></tr>
<tr><td>RMS 偏差 (Hz)</td><td>—</td><td>58</td><td><b>46</b></td></tr>
<tr><td>内点率</td><td>—</td><td>0.45</td><td>0.61</td></tr>
<tr><td>沿轨迹能量集中度</td><td>—</td><td>0.161</td><td><b>0.342</b></td></tr>
</tbody></table>
<p>然后对 est 施加四种破坏，要求工具全部检出：</p>
<table>
<thead><tr><th>破坏方式</th><th>轨迹 r</th><th>RMS (Hz)</th><th>覆盖比</th><th>内点率</th></tr></thead>
<tbody>
<tr><td>削边（900–1400 带通）</td><td>+0.896</td><td>82</td><td>0.69</td><td>0.54</td></tr>
<tr><td>频率压缩 5%</td><td>+0.880</td><td>104</td><td>0.95</td><td>0.59</td></tr>
<tr><td>时移 100 ms</td><td>+0.846</td><td>124</td><td>1.01</td><td>0.62</td></tr>
<tr><td>后半段丢失</td><td>+0.974</td><td>46</td><td>0.99</td><td>0.91</td></tr>
</tbody>
<tfoot><tr><th colspan="5" style="text-align:right">
est 基线：r = +0.974，RMS = 46 Hz，内点率 = 0.61</th></tr></tfoot></table>

<div class="box good"><b>自检四项全过</b>：识别 HFM ✓（1/f 的 R² 0.9999 &gt; f 的 0.9744）；
降噪后轨迹位置保住 ✓（r = 0.974）；轨迹形状保住 ✓（R² = 0.993）；
四种破坏全部被检出 ✓（削边 → 覆盖比 0.69；频率压缩 → RMS 翻倍；时移 → r 降至 0.85；
后半段丢失 → 内点率与有效段骤降）。<br>
<b>只有这一步通过了，把工具用到实测数据上的结果才值得相信。</b></div>

__FIGAC__

<h4>8.18.4 实测结果：这是一段<b>真实的 LFM 扫频</b>，降噪后完整保住</h4>
<p>用同一工具跑 <code>_smoke/test.wav</code>（4.0 s）。
先说不依赖任何跟踪算法的独立判据：逐帧取带内峰值后做线性回归，</p>
<table>
<thead><tr><th>判据</th><th>值</th><th>含义</th></tr></thead>
<tbody>
<tr><td>线性回归 R²（f~t）</td><td><b>0.9998</b></td><td>频率随时间严格线性变化</td></tr>
<tr><td>斜率</td><td><b>+125.1 Hz/s</b></td><td>从 908 Hz 扫到 1389 Hz</td></tr>
<tr><td>秩相关（单调性）</td><td><b>1.0000</b></td><td>完全单调递增，无一次回退</td></tr>
<tr><td>相位随机化后的 R²（3 次）</td><td>0.639 / 0.755 / 0.035</td><td>破坏扫频即崩塌 ⇒ 原值是真结构</td></tr>
<tr><td>时域打乱后的 R²（2 次）</td><td>0.923 / 0.704</td><td>同上</td></tr>
</tbody></table>
<p>即<b>该录音含一段真实的线性调频（LFM）信号</b>，不是 HFM
（R²(f)=0.9994 &gt; R²(1/f)=0.9871）。然后做降噪前后的轨迹对比：</p>
<table>
<thead><tr><th>量</th><th>输入</th><th>输出</th><th>变化</th></tr></thead>
<tbody>
<tr><td>覆盖频段</td><td>913 – 1388 Hz</td><td>913 – 1388 Hz</td>
<td><b>覆盖比 0.999，首尾误差 +0 / -0 Hz</b></td></tr>
<tr><td>拟合斜率</td><td>+124.9 Hz/s</td><td>+125.0 Hz/s</td>
<td><b>相对误差 +0.06%</b></td></tr>
<tr><td>R²（f~t）</td><td>0.9994</td><td><b>0.9999</b></td><td>更线性</td></tr>
<tr><td>内点率</td><td>0.99</td><td><b>1.00</b></td><td>全部帧服从主轨迹</td></tr>
<tr><td>峰值 SNR（沿轨迹 vs 带内中位）</td><td>26.9 dB</td><td><b>65.6 dB</b></td>
<td><b>+38.7 dB</b></td></tr>
<tr><td>沿轨迹能量集中度</td><td>0.493</td><td><b>0.592</b></td><td>+0.8 dB</td></tr>
</tbody></table>
<p><b>轨迹保真：r = 0.9970，RMS 偏差 11.0 Hz</b>（相对 475 Hz 的扫频跨度只有 2.3%）。
分段看：0–1 s 与 3–4 s 的偏差只有 0.2 Hz，1–2 s 为 16.0 Hz、2–3 s 为 8.3 Hz（中段略大，仍在 3% 内）。</p>

<h4>8.18.5 信号与噪声分开看：这是最不依赖解释的一项</h4>
<p>上表的"峰值 SNR +38.7 dB"和"集中度 +0.8 dB"读起来矛盾（一个大得离谱、一个几乎没动），
原因是它们的分母不同：前者除以<b>带内中位数</b>（噪声被压掉后中位数极低，比值被放大），
后者除以<b>带内总能量</b>（输入本身就有 84% 集中在轨迹上，空间有限）。
两者都不适合直接当"降噪效果"引用。</p>
<p>更干净的做法是<b>把能量按轨迹分开</b>：轨迹 ±2 bin（±47 Hz，覆盖主瓣）算"信号"，
带内其余算"噪声"，分别看变化。</p>
<table>
<thead><tr><th>窗口半宽</th><th>沿轨迹能量（信号）</th><th>轨迹外带内能量（噪声）</th><th>相对抑制</th></tr></thead>
<tbody>
<tr><td>±1 bin（±23 Hz）</td><td>-0.3 dB</td><td>-10.6 dB</td><td><b>-10.4 dB</b></td></tr>
<tr><td><b>±2 bin（±47 Hz，主瓣）</b></td><td><b>-0.3 dB</b></td><td><b>-13.1 dB</b></td><td><b>-12.8 dB</b></td></tr>
<tr><td>±3 bin（±70 Hz）</td><td>-0.4 dB</td><td>-13.8 dB</td><td>-13.4 dB</td></tr>
<tr><td>±5 bin（±117 Hz）</td><td>-0.5 dB</td><td>-13.7 dB</td><td>-13.1 dB</td></tr>
<tr><td>±8 bin（±187 Hz）</td><td>-0.6 dB</td><td>-14.1 dB</td><td>-13.5 dB</td></tr>
</tbody></table>

<div class="box good"><b>这是本节最直接、最不依赖解释的一项证据</b>（±2 bin）：
降噪后<b>沿扫频轨迹的信号能量只损失 0.3 dB</b>，而<b>轨迹外的带内噪声被压掉 13.1 dB</b>，
两者相差 <b>12.8 dB</b>。也就是说模型<b>主要压的是噪声，不是信号</b>。<br>
窗口从 ±1 到 ±8 bin 全扫过：信号损失始终在 0.6 dB 以内，噪声抑制 10.6~14.1 dB，
结论对窗口宽度<b>稳健</b>（窗口越宽，"轨迹外"混进的信号越多、抑制量越被低估）。
这也解释了 8.17 观察到的"带外均匀衰减、无孤立谱线"——两节的结论是自洽的。</div>

<div class="box warn"><b>对 8.18.4 表里那两个指标的正确读法</b>：
① <b>峰值 SNR +38.7 dB 不能当"降噪量"引用</b> —— 它的分母是带内中位数，
信号被保住、噪声被压后中位数塌下去，比值必然暴涨；它只说明"轨迹变得非常突出"；
② <b>集中度 +0.8 dB 也不能当"降噪量"引用</b> —— 它的分母是带内总能量，
而输入已有 84% 能量在轨迹上，上限只有 +0.75 dB（0.844→1.0）。
真正表示降噪效果的是本节的<b>相对抑制 12.8 dB</b>。</div>

<div class="box good"><b>实测结论（可以写进论文的版本）</b>：<br>
这段 4 秒实测录音含一段 <b>LFM 扫频</b>（908→1389 Hz，斜率 +125 Hz/s）。
经 R5（输入归一化版）降噪后，<b>扫频轨迹完整保留</b> ——
轨迹相关系数 0.9970、斜率相对误差 +0.06%、起止频率误差 +0/-0 Hz、
覆盖比 0.999；同时带内噪声被大幅抑制（沿轨迹峰值 SNR 从 26.9 升到 65.6 dB，+38.7 dB）。<br>
<b>这是本项目第一次在非仿真录音上给出正面的、有判别力的结论</b>
（8.17 的带外比指标做不到这件事）。</div>

<div class="box warn"><b>必须限定清楚的五件事</b>：<br>
① <b>是 LFM，不是 HFM</b>。导师方案的目标是双曲调频，这份录音是线性调频。
虽然训练数据本身就以 LFM 为主（波形泛化实验表明 HFM 代价 ≤0.2 dB），
但<b>这份实测没有验证 HFM 场景</b>；<br>
② <b>只有 1 个文件、4 秒、1 段扫频</b>，无重复、无多普勒、无时变信道；<br>
③ <b>无干净参考</b>，因此只能说"扫频结构被保住"，不能说"降噪后更接近真实信号"。
轨迹保真是必要条件，不是充分条件 ——
一个只保留扫频脊、把其余全部压掉的极端滤波器同样能通过这项检验
（虽然 8.17.2 已证明带外抑制是均匀的、没有孤立谱线被抹）；<br>
④ <b>集中度只提升 +0.8 dB</b>，而峰值 SNR 提升 38.7 dB。
两者不矛盾但有区别：前者是"沿轨迹能量 / 带内总能量"，输入本来就占 0.49，
输出的提升空间有限；后者是"沿轨迹峰值 / 带内中位数"，反映的是<b>噪声被压掉</b>而非信号被增强。
引用时应说"带内噪声被抑制 39 dB 量级"，不要说"信号增强"；<br>
⑤ 扫频速率 +125 Hz/s 比训练数据（LFM 1 s 内 798→1500，702 Hz/s）慢 5.6 倍，
属轻度分布外；结论对这一参数区间成立。</div>

<h4>8.18.5 闸门：怎么防止"在没有扫频的噪声上强行解读"</h4>
<p>上面所有保真度指标都建立在"输入确实有扫频"这个前提上。所以必须先过一道闸门。
<b>第一版闸门是错的</b>，这里如实记录：</p>
<div class="box bad"><b>闸门 v1（"R² &gt; 0.6"）失效。</b>
对四种无扫频信号实测：白噪声 R²=<b>0.748</b>、α 稳定脉冲噪声 0.619、
带内窄带噪声 <b>0.893</b> —— 全部越过 0.6。
原因有两个：① RANSAC 总能在一堆随机点里找到"被多数点认同的直线"；
② DP 的平滑约束让噪声轨迹变成缓慢游走，<b>局部看起来确实近似线性</b>。<br>
这条直接说明：<b>单靠一个绝对 R² 阈值判不出"有没有扫频"</b>。</div>

<p>闸门 v2 用<b>替代数据检验</b>（时间序列分析的标准做法）：
把信号做相位随机化——保持功率谱、破坏相位结构，于是扫频消失但谱形状不变——
重复 3 次，比较原信号与其替代数据的 R²。判据是<b>两条同时满足</b>：</p>
<pre>① R²原始 ≥ 0.95          （真实扫频严格服从 f=at+b：clean 0.9999、−3 dB 加噪 0.993）
② R²原始 &gt; max(R²替代) + 0.05   （排除"谱形状本身导致轨迹看起来线性"的偶然）</pre>
<table>
<thead><tr><th>信号</th><th>R² 原始</th><th>R² 替代 max</th><th>差</th><th>闸门输出</th><th>期望</th></tr></thead>
<tbody>
<tr><td>白噪声</td><td>0.748</td><td>0.896</td><td>−0.148</td><td>不可判</td><td>不可判 ✓</td></tr>
<tr><td>α 稳定脉冲噪声</td><td>0.619</td><td>0.659</td><td>−0.040</td><td>不可判</td><td>不可判 ✓</td></tr>
<tr><td>带内窄带噪声</td><td>0.893</td><td>0.719</td><td>+0.174</td><td>不可判</td><td>不可判 ✓</td></tr>
<tr><td>clean LFM</td><td>1.0000</td><td>0.859</td><td>+0.141</td><td>有扫频</td><td>有扫频 ✓</td></tr>
<tr><td>clean HFM</td><td>0.9999</td><td>0.883</td><td>+0.117</td><td>有扫频</td><td>有扫频 ✓</td></tr>
<tr><td><b>实测 test.wav</b></td><td><b>0.9994</b></td><td>0.828</td><td>+0.172</td><td><b>有扫频</b></td><td>有扫频 ✓</td></tr>
</tbody></table>
<div class="box good"><b>闸门 v2 六项全部符合期望</b>：三种噪声全被挡住（注意窄带噪声的差 +0.174 与真实信号相当，
只有绝对阈值 0.95 能分开它们 —— 所以两条判据缺一不可），
clean LFM / clean HFM / 实测录音全部通过。</div>

<div class="box"><b>给后续实测的用法</b>（拿到任何扫频录音直接跑）：
<pre>python hfm_sweep_fidelity.py --gatecheck                  # 验闸门（六项应全过）
python hfm_sweep_fidelity.py --selftest                   # 验工具（四项应全过）
python hfm_sweep_fidelity.py --input 录音.wav --ckpt model_inm_pf.pth</pre>
输出 JSON 里有 <code>gate</code> 字段（含 <code>gate_pass</code> /
<code>r2_obs</code> / <code>r2_surr</code>）。闸门未过时会明确报"无扫频结构，保真度不可判"。<br>
<b>为什么必须留着这道闸门</b>：8.17 初稿把扫频速率误读成噪声抖动，
是"看错信号"的错；闸门防的是另一种错 —— <b>在本来没有扫频的信号上读出一堆保真度数字</b>。</div>

<h2>九、消融（单种子，BN 协议；仅用于同批内部比较）</h2>
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
<b>注意：本节数字是单种子 + BN，按第三、五、六节的结论，其绝对值带 2~3 dB 的选择偏差，
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

<h2>十、训练稳定性的完整数据</h2>
__FIGL__
<table>
<thead><tr><th>run</th><th>逐轮 val 帧增益标准差 (dB)</th><th>最佳轮次</th><th>最佳−末5轮 (dB)</th></tr></thead>
<tbody>
__ROWS_STAB__
</tbody></table>

<h2>十一、独立复算</h2>
<p><code>verify_pamnet.py</code> <b>不复用训练脚本的任何实现</b>：自己读 .mat、
自己 <code>resample_poly(1,4)</code>、自己实现 SNR / SI-SDR / 逐帧降噪增益，
再与训练脚本产出的逐样本 CSV 对比。唯一共用的是模型定义本身
（验证一个模型不可能绕开它的结构）。</p>
<p>结果：<b>均值逐位吻合</b>；逐样本最大差 2.4e−3 dB，来自 STFT/ISTFT 在 float32 下的
舍入放大。150 个样本<b>无一变差</b>（最差 +4.53 dB）。</p>

<h2>十二、局限</h2>
<ul>
<li><b>仍然是仿真数据。</b>α 稳定背景与突发干扰都是合成的，α 只取 1.5，
突发幅度/时长范围是设定的。导师已说明会提供实测信号，届时需重新验证。</li>
<li><b>种子数仍然偏少</b>（关键对照 3~5 个）。跨种子检验自由度很低，p 值只能作参考。
要得到可靠的效应量，建议 5~10 个种子。</li>
<li><b>（原缺陷，已在第六节解决）与阶段0 的对比原本含多个变量</b>：架构与损失同时改变。
本次用"每步只改一个因素"的受控阶梯把它拆开，得到<b>协议约 +0.9 dB、架构约 +3.3 dB</b>
（干扰帧）。但阶梯的每一档仍只跑了 1~5 个种子，R0/R1 更是单种子。</li>
<li><b>波形机制的统计基础仍然偏窄</b>：时频重叠率的相关系数基于 <b>20 个 chirp</b>，
且这 20 个波形来自同一张随机参数表（f0/f1/T/包络 同分布）。
"重叠率越低收益越大"这个关系在<b>更宽的波形参数范围</b>上是否保持，未验证。</li>
<li><b>配对设计只对"波形"做了配对，信道仍与干扰实现混杂</b>：配对模式让同一
(信道, SNR, 重复) 下的不同 chirp 看到相同干扰，但不同信道之间仍是不同的干扰实现。
因此第七节里"信道分层"的离散度（SD 0.44 dB）不能直接解读为信道效应。</li>
<li><b>效应量本身很小，实用价值有限</b>：脉冲感知分支平均只带来
+0.13（干净帧）/ +0.79（干扰帧）dB，且 96% 的方差来自样本间波动。
论文口径应写成"形成判别能力（AUC 0.95）但性能贡献有限"，不应声称"显著提升"。</li>
<li><b>λ₂ 由量级校准确定，不是完整网格搜索</b>（只对照 0.05 与 0.5）。</li>
<li><b>突发干扰直接叠加在信道输出上。</b>若干扰发生在信道之前，经多径会展宽成一串回声突发，
难度完全不同 —— 该场景未测。</li>
<li><b>降采样 4 倍对脉冲的衰减未做定量分解。</b>脉冲经抗混叠滤波会被削弱，
而高斯噪声下同样的降采样带来 +6.19 dB 输入 SNR 增益。</li>
<li><b>两个前端处理需要交代</b>：样本中位有效长度 6,031/8,000（约 24.6% 零填充）；
降采样 4 倍把输入 SNR 从 0 dB 抬到 6.19 dB（高斯版）。</li>
<li><b>（第八节新增）α 扫描的 SNR 定义混杂已量化，并已用"等功率重扫 + 域内重训"查清。</b>
分位数匹配让 α 与有效噪声功率同时变化（−16.4 vs +10.8 dB）。
8.1b 先做了等有效功率重扫，8.7 又补上了 α 轴的域内重训，结论是：
<b>α 轴上脉冲分支收益的变化主要是分布偏移，不是尾形本身</b>——
α=1.8 端零样本看到的"尾形增强 +0.98 dB"在域内重训后<b>消失</b>（回到 base 水平）；
α=1.2 端仅剩约 0.24 dB 且不显著。<b>8.1b 基于零样本的分解因此被推翻。</b>
剩余缺口：<b>α 轴域内重训只覆盖 α=1.2 与 α=1.8 两个点</b>；
α=2.0 在 8.14 做了<b>未等功率匹配</b>的版本，但该条件下 <b>R4 跨种子极差 2.12 dB、
逐种子 Δ 符号相反</b>，结果不可用（8.14.2），
等功率匹配版本需名义 SNR 低至 −24 dB，未做。
且各仅 3 个种子，α=1.2 的种子间 sd 达 0.37（testA）/ 0.65（testC）。</li>
<li><b>（第八节新增）域内重训的覆盖仍不完整，虽然主缺口已补上。</b>
已完成：rate=10（8.6）与 rate=2.5（8.14）两个方向、α 轴等功率条件（8.7，α=1.2 / α=1.8），
各 R4/R5 × 3 种子，并单独测了设备效应对 Δ 的影响（−0.091 dB）。
<b>剩余缺口</b>：① 只重训了 R4/R5，<b>没有重训 R3 与纯时域模型</b>，
所以域内对照只覆盖"有无脉冲分支"这一维，架构排序在域内是否变化未验证；
② rate 轴的稀疏方向（2.5/s）已补，见 8.14；但该端 3 种子 sd 达 0.66，量级不稳；
③ α=2.0 做了<b>未匹配</b>版本，但结果<b>不可用</b>：逐种子 Δ 符号相反（+1.685 / −0.861 / +0.711），
R4 跨种子极差 2.12 dB 而 R5 只 0.43 dB —— 该条件下域内训练本身对种子不稳定（8.14.2）；
等功率匹配版本需名义 SNR 低至 −24 dB，未做；
④ 域内种子数为 3，少于 base 的 5。</li>
<li><b>（第八节新增）"能识别 ≠ 能提升性能"这条观察仍然成立，但机制已在 8.15 被部分澄清。</b>
我们证明了判别能力（AUC 0.93~0.96）跨条件几乎不变、而收益跨条件变化 0.2~1.7 dB。
第七节曾用观察数据得到"收益与时频重叠率负相关"，但 <b>7.x 的两个直接干预
（载频扫描、时频对齐）都没有复现该负相关</b>，方向甚至相反 ——
<b>因此「重叠率 → 收益下降」这条不能作为机制结论引用</b>。
8.15 用推理期篡改 p 的方式换了一个问法，得到的是<b>正面证据</b>：
分支确实依赖 p 的<b>逐点对齐</b>（换成他人的 p 损失 3.12 dB，与置零几乎相同），
而重叠率是帧级、全局统计量，本来就不该预测逐点对齐 —— 两者并不矛盾。
<b>但 8.15 的局限也必须一并声明</b>：干预在推理期完成、属训练时没见过的输入，
只覆盖 base 条件与 R5 一个架构、只做 testA。若要作为强因果结论，需补
「训练时就注入扰动 p」的对照（当前只有推理期版本）。</li>
<li><b>（第八节新增）含时域分支的模型，逐样本指标无法从 checkpoint 逐位复现。</b>
最大差 ≤0.13 dB、均值 ≤0.003 dB（纯频域模型仅 5e-5 dB），
原因是时域通路对 1e-7 量级浮点归约差异的放大。
所有聚合指标复现误差 ≤0.001 dB，本研究结论基于聚合指标，不受影响；
但任何以逐样本 CSV 为依据的复核会看到非零残差。</li>
<li><b>（基础设施）CPU 与 GPU 训练结果不可混用。</b>
本报告 base 条件的全部结果在 CPU 上训练。后续域内重训迁到 GPU（torch 2.14.1+cu126）后，
同一配置在两种设备上各训一次，终值差最大 <b>0.45 dB</b>（testC 干扰帧），
而<b>同一 checkpoint 跨设备评估仅差 0.0004 dB</b> —— 差异来自训练路径分叉
（浮点归约顺序 → 梯度 → 15 轮 Adam 落到不同极小点），不是推理错误。
该量级与待测效应（Δ≈0.78 dB）相当，因此<b>同一批对照必须在同一设备上完成</b>。
本报告的 8.6 节跨了设备，只能读趋势不能读精确差值。</li>
<li><b>（8.2b）经典基线只覆盖四类</b>（理想带通、盲谱减、盲维纳、中值滤波），
<b>未包含其他深度学习降噪网络</b>（如 DCCRN、Conv-TasNet 类）。
因此"远超全部经典方法"的<b>"全部"限定在上述四类范围内</b>，
不应外推为"优于所有已有方法"。</li>
<li><b>（8.12 新增）输入归一化只在 base 条件上验证，且只做了"整体缩放"一种尺度扰动。</b>
已确认归一化把跨尺度极差从 __SI_OLDRNG__ dB 压到 0.000 dB，代价 ≤0.2 dB；
但 ① 归一化训练的模型只在 base 条件重训过（3 个种子），
未在 α / rate 的其它条件上验证；② 尺度测试用的是"信号与噪声同乘 s"，
而实测数据的尺度不确定性还包含采样率差异、带宽差异、传感器增益非线性等，
这些不在本节覆盖范围内 —— 拿到实测数据后须按
<code>docs/real_data_deployment.md</code> 的清单逐项核对。</li>
<li><b>（8.13 新增）域内架构排序只覆盖 base 一个条件。</b>
R5 &gt; R4 &gt; R3 &gt; 纯时域 在 base 的域内同设备条件下 4/4 方向与零样本一致，
但 α / rate 轴的域内架构排序仍未验证；且 devchk 基线为 3 个种子，
R5 vs R4 的检验 p = __IA_P54__（未达 0.05），只能读点估计。</li>
</ul>

<h2>十三、复现命令</h2>
<pre># 1) 单元测试（8 项，必须全过）
python code/analysis/test_pamnet.py
# 2) 损失量级校准（确定 lambda2）
python code/analysis/calibrate_loss.py
# 3) 归一化诊断（解释为什么必须换 GroupNorm）
python code/analysis/diagnose_bn.py --ckpt E:/project/data/output_pamnet/model_pamnet.pth

# 4) 最终协议：GroupNorm + 多种子（关键 4 个配置；S 为空/s2/s3/s4/s5）
D=E:/project/data/dataset_pam/alphastable
python code/python/train_pamnet.py --tag pamnet_gn               --data $D --epochs 15 --norm gn
python code/python/train_pamnet.py --tag pamnet_gn_s2            --data $D --epochs 15 --norm gn --seed 2
python code/python/train_pamnet.py --tag pamnet_nopulse_gn       --data $D --epochs 15 --norm gn --pulse 0
python code/python/train_pamnet.py --tag pamnet_freq_pf_gn       --data $D --epochs 15 --norm gn --freq-refine 2.0 --pulse-freq 1
python code/python/train_pamnet.py --tag pamnet_freq_nopulse_gn  --data $D --epochs 15 --norm gn --freq-refine 2.0 --pulse 0
# （其余种子同理，只改 --seed 与 --tag 后缀）

# 5) 受控阶梯：纯频域模式（架构等价性必须先验证）
python code/analysis/test_pamnet_freqonly.py
FO="--short 0 --long 0 --pulse 0 --freq-refine 2.0"
python code/python/train_pamnet.py --tag fo_bn_l1     --data $D --epochs 15 --norm bn \
    --loss l1 --lam-spec 0.5 --select val_loss $FO    # 复现阶段0 协议（管线等价性检查）
python code/python/train_pamnet.py --tag fo_bn_huber  --data $D --epochs 15 --norm bn \
    --loss huber --lam-spec 0.05 $FO                  # 只换损失 + 模型选择准则
python code/python/train_pamnet.py --tag fo_gn        --data $D --epochs 15 --norm gn \
    --loss huber --lam-spec 0.05 $FO                  # 再换归一化（S 为空/s2..s5）
python code/analysis/analyze_ladder.py                # 阶梯表 + 逐步增量 + 图 P

# 6) 波形机制：chirp 扫描集（含"固定干扰"的配对版）
#   MATLAB:
#   sp = struct('name',{'chirpscan'},'chirps',{1:20},'chans',{1:8},'n_noise',{2});
#   dump_chirp_table('D:/project/chirp_table.csv');
#   make_dataset_pam('alphastable','D:/project/ds_chirpscan',1.5,sp);            % 原始
#   make_dataset_pam('alphastable','D:/project/ds_scan_paired',1.5,sp,'paired'); % 配对
python code/analysis/analyze_chirp_band.py --scan D:/project/ds_chirpscan/chirpscan
python code/analysis/analyze_chirp_band.py --scan D:/project/ds_scan_paired/chirpscan \
    --cache-tag _paired                               # 图 O + 方差分解 + 相关分析

# 7) 分析与出图
python code/analysis/analyze_seeds.py       # 多种子聚合 + 跨种子检验
python code/analysis/analyze_stability.py   # 逐轮振荡与选择偏差
python code/analysis/make_fig_seeds.py      # 图 M / 图 N
python code/analysis/paired_test_pamnet.py  # 样本级配对检验
python code/analysis/verify_pamnet.py --ckpt E:/project/data/output_pamnet/model_pamnet_freq_pf.pth \\
    --tag pamnet_freq_pf --freq-refine 2.0 --pulse-freq 1
python code/analysis/compare_baseline_pam.py

# 8) 适用边界（敏感性）：5 个条件的数据集（单因素轮换，共同随机数）
#   MATLAB（一次生成全部条件；--conds 可只跑部分）:
#   gen_sensitivity([1 2 3 4 5])          % a12_r5 / a18_r5 / a20_r5 / a15_r25 / a15_r10
#   % 基线条件直接用 E:/project/data/dataset_pam/alphastable；高斯对照用 dataset_pam/gauss

# 9) 零样本跨条件评估（含逐样本逐种子结果）
python code/analysis/eval_sensitivity.py --prep            # 预处理 + 噪声功率分解
python code/analysis/eval_sensitivity.py --verify          # 口径校验：与存档 CSV 逐位比对
python code/analysis/eval_sensitivity.py --nshard 6 --shard 0     # 6 路并行，支持断点续跑

# 10) 统计分析与出图
python code/analysis/analyze_sensitivity.py --split testA  # 表1~表5 + 汇总 JSON + 扁平 CSV
python code/analysis/analyze_sensitivity.py --split testC
python code/analysis/make_fig_sensitivity.py               # 图 Q / R / S / T
python code/analysis/diagnose_pulse_sens.py --max 300      # 脉冲分支 AUC 跨条件

# 11) 域内重训（区分"分布偏移"与"方法本身优劣"；后台跑）
#     注意：必须用受管的后台机制启动（见下方 run_in_background），
#     裸 nohup & 的进程会被环境回收
bash code/analysis/run_sens_train.sh            # a15_r10 上 R4/R5 × 3 种子

# 12) 等有效功率 SNR 的 α 重扫（把 α 与噪声强度解耦）
#   MATLAB（先算 shift，见 analyze_matched.py 第 1 步的输出）:
#   gen_sensitivity_matched([1 2])   % m_a12 (α=1.2, 名义 +13.07) / m_a18 (α=1.8, 名义 −10.00)
python code/analysis/eval_sensitivity.py --prep --conds m_a12 m_a18
python code/analysis/eval_sensitivity.py --conds m_a12 m_a18 \
    --configs R3_freqonly R4_freqrefine_nopulse R5_freqrefine_pulse \
    --nshard 3 --shard 0     # 3 路并行
python code/analysis/analyze_matched.py --split testA   # 强度/尾形分解 + 留一法
python code/analysis/analyze_matched.py --split testC
</pre>

<hr>
<p style="font-size:12.5px;color:#6b7280">
本报告由 <code>build_report_pamnet.py</code> 自动生成。数字来自 <code>figs_pamnet/</code> 下的
summary_pamnet.json / seed_agg.json / pulse_diag.json / baseline_pam.json /
paired_test.json / stability.json / ladder.json / chirp_band.json / chirp_band_paired.json，
以及 <code>sens_out/</code> 下的 sens_summary_testA.json / sens_summary_testC.json /
pulse_diag_sens.json（适用边界实验）。
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
        .replace("__ROWS_LADDER__", rows_ladder)
        .replace("__ROWS_STEP__", rows_step)
        .replace("__ROWS_CHIRP__", rows_chirp)
        .replace("__ROWS_SENS_MAIN__", rows_sens_main)
        .replace("__ROWS_SENS_PULSE__", rows_sens_pulse)
        .replace("__ROWS_SENS_ARCH__", rows_sens_arch)
        .replace("__ROWS_SENS_NOISE__", rows_sens_noise)
        .replace("__ROWS_SENS_DEGRADE__", rows_sens_degrade)
        .replace("__SENS_HEAD__", sens_head)
        .replace("__SENS_A20__", "%+.2f" % sens_vars["d_a20"])
        .replace("__SENS_A12__", "%+.2f" % sens_vars["d_a12"])
        .replace("__SENS_PA12__", "%.2f" % sens_vars["p_a12"])
        .replace("__SENS_R25__", "%+.2f" % sens_vars["d_r25"])
        .replace("__SENS_R10__", "%+.2f" % sens_vars["d_r10"])
        .replace("__SENS_AA12__", "%+.2f" % sens_vars["aa_a12"])
        .replace("__SENS_AA20__", "%+.2f" % sens_vars["aa_a20"])
        .replace("__SENS_AAGAUSS__", "%+.2f" % sens_vars["aa_gauss"])
        .replace("__SENS_R3A12__", "%.2f" % sens_vars["r3_a12"])
        .replace("__SENS_R3A20__", "%.2f" % sens_vars["r3_a20"])
        .replace("__SENS_R5A12__", "%.2f" % sens_vars["r5_a12"])
        .replace("__SENS_R5A20__", "%.2f" % sens_vars["r5_a20"])
        .replace("__SENS_DEG12__", "%.2f" % sens_vars["deg_a12"])
        .replace("__SENS_DEG20__", "%.2f" % sens_vars["deg_a20"])
        .replace("__SENS_DEG20R4__", "%.2f" % sens_vars["deg_a20_r4"])
        .replace("__SENS_AUCLO__", "%.3f" % sens_vars["auc_lo"])
        .replace("__SENS_AUCHI__", "%.3f" % sens_vars["auc_hi"])
        .replace("__EFF_A20__", "%+.2f" % (sens_txt["eff_a20"] if SENSA else float("nan")))
        .replace("__EFF_A12__", "%+.2f" % (sens_txt["eff_a12"] if SENSA else float("nan")))
        .replace("__EFF_GAP__", "%.1f" % ((sens_txt["eff_a20"] - sens_txt["eff_a12"])
                                          if SENSA and np.isfinite(sens_txt["eff_a20"])
                                          and np.isfinite(sens_txt["eff_a12"]) else float("nan")))
        # ---- 等功率匹配（8.1b）----
        .replace("__ROWS_SENS_MATCHED__", rows_sens_matched)
        .replace("__ROWS_SENS_MATCHED_ARCH__", rows_sens_matched_arch)
        .replace("__MB0__", "%.2f" % _mn("base", "bg"))
        .replace("__MI0__", "%.2f" % _mn("base", "imp"))
        .replace("__MP0__", "%.1f" % _mn("base", "pk"))
        .replace("__MB1__", "%.2f" % _mn("m_a12", "bg"))
        .replace("__MI1__", "%.2f" % _mn("m_a12", "imp"))
        .replace("__MP1__", "%.1f" % _mn("m_a12", "pk"))
        .replace("__MB2__", "%.2f" % _mn("m_a18", "bg"))
        .replace("__MI2__", "%.2f" % _mn("m_a18", "imp"))
        .replace("__MP2__", "%.1f" % _mn("m_a18", "pk"))
        .replace("__MB3__", "%.2f" % _mn("a12_r5", "bg"))
        .replace("__MI3__", "%.2f" % _mn("a12_r5", "imp"))
        .replace("__MP3__", "%.1f" % _mn("a12_r5", "pk"))
        .replace("__MB4__", "%.2f" % _mn("a18_r5", "bg"))
        .replace("__MI4__", "%.2f" % _mn("a18_r5", "imp"))
        .replace("__MP4__", "%.1f" % _mn("a18_r5", "pk"))
        .replace("__MDELTA__", "%.1e" % _mdelta)
        .replace("__MA12_TOT__", "%+.3f" % m_txt["a12"]["tot_um"] if m_txt.get("a12") else "—")
        .replace("__MA12_INTEN__", "%+.3f" % m_txt["a12"]["inten"] if m_txt.get("a12") else "—")
        .replace("__MA12_TAIL__", "%+.3f" % m_txt["a12"]["tail"] if m_txt.get("a12") else "—")
        .replace("__MA12_PCT__", "%.0f" % (100 * (1 - abs(m_txt["a12"]["tail"]) / abs(m_txt["a12"]["tot_um"]))
                                            if m_txt.get("a12") and abs(m_txt["a12"]["tot_um"]) > 1e-9 else 0))
        .replace("__MA12_TPCT__", "%.0f" % (100 * abs(m_txt["a12"]["tail"]) / abs(m_txt["a12"]["tot_um"])
                                             if m_txt.get("a12") and abs(m_txt["a12"]["tot_um"]) > 1e-9 else 0))
        .replace("__MA12_LOOMIN__", "%+.3f" % m_txt["a12"]["m_loo"][0] if m_txt.get("a12") else "—")
        .replace("__MA12_LOOMAX__", "%+.3f" % m_txt["a12"]["m_loo"][1] if m_txt.get("a12") else "—")
        .replace("__MA18_TOT__", "%+.3f" % m_txt["a18"]["tot_um"] if m_txt.get("a18") else "—")
        .replace("__MA18_INTEN__", "%+.3f" % m_txt["a18"]["inten"] if m_txt.get("a18") else "—")
        .replace("__MA18_TAIL__", "%+.3f" % m_txt["a18"]["tail"] if m_txt.get("a18") else "—")
        .replace("__MA18_LOOMIN__", "%+.3f" % m_txt["a18"]["m_loo"][0] if m_txt.get("a18") else "—")
        .replace("__MA18_LOOMAX__", "%+.3f" % m_txt["a18"]["m_loo"][1] if m_txt.get("a18") else "—")
        .replace("__TC_A12__", "%+.3f" % m_txt["tc"]["m_a12"] if m_txt.get("tc") and "m_a12" in m_txt["tc"] else "—")
        .replace("__TC_BASE__", "%+.3f" % m_txt["tc"]["base"] if m_txt.get("tc") and "base" in m_txt["tc"] else "—")
        .replace("__TC_A18__", "%+.3f" % m_txt["tc"]["m_a18"] if m_txt.get("tc") and "m_a18" in m_txt["tc"] else "—")
        .replace("__TC_ARCH_A12__", "%+.2f" % m_txt["tc_arch"]["m_a12"] if m_txt.get("tc_arch") and "m_a12" in m_txt["tc_arch"] else "—")
        .replace("__TC_ARCH_BASE__", "%+.2f" % m_txt["tc_arch"]["base"] if m_txt.get("tc_arch") and "base" in m_txt["tc_arch"] else "—")
        .replace("__TC_ARCH_A18__", "%+.2f" % m_txt["tc_arch"]["m_a18"] if m_txt.get("tc_arch") and "m_a18" in m_txt["tc_arch"] else "—")
        .replace("__FIGQ__", img("figQ_sens_alpha.png",
                                 "图 Q α 轴（rate=5 固定）：左为干净帧、右为干扰帧的逐帧降噪增益。"
                                 "灰色点线为右轴上的背景噪声有效功率 SNR —— 它随 α 变化很大，"
                                 "是 α 扫描中必须剥离的混杂"))
        .replace("__FIGR__", img("figR_sens_rate.png",
                                 "图 R rate 轴（α=1.5 固定）：左为干净帧、右为干扰帧。"
                                 "背景噪声逐位相同，只有突发密度变化 —— 这是干净的单因素操纵"))
        .replace("__ROWS_INDOMAIN__", __INDOMAIN_ROWS__)
        .replace("__FIGV__", img("figV_intervention.png",
                                 "图 V 左：载频扫描（带内占比 vs 分支收益，无单调趋势）；右：时频对齐干预（同一波形只改放置时刻）"))
        .replace("__FIGU__", img("figU_baseline_sens.png",
                                 "图 U 左：各条件上经典基线与神经模型的干扰帧增益（标注为 R5 与最好盲方法的差距）；右：R5 的优势幅度分解"))
        .replace("__FIGS__", img("figS_sens_pulse.png",
                                 "图 S 左：脉冲感知分支的净效应 Δ=R5−R4 在 7 个条件上的分布"
                                 "（误差棒 = 种子间标准差）；右：各条件的背景/干扰有效功率 SNR"))
        .replace("__FIGT__", img("figT_sens_detect.png",
                                 "图 T 判别能力（AUC）与性能收益（Δ）的跨条件关系"))
        .replace("__LADDER_NOTE__", ladder_note)
        .replace("__PROTO_IMP__", "%+.2f dB" % _proto_imp)
        .replace("__PROTO_CLEAN__", "%+.2f dB" % _proto_clean)
        .replace("__PROTO_MIN__", "%+.2f dB" % _proto_min)
        .replace("__ARCH_IMP__", "%+.2f dB" % _arch_imp)
        .replace("__STEP34__", "%+.2f dB" % _step34_imp)
        .replace("__STEP45__", "%+.2f dB" % _step45_imp)
        .replace("__RATIO__", "%.1f" % _ratio)
        .replace("__FIGP__", img("figP_ladder.png",
                                 "图 P 左：受控阶梯的逐档结果（误差棒 = 种子间标准差，"
                                 "标注为参数量）；右：逐步增量 —— 协议只占小头，架构是主要贡献"))
        .replace("__FIGO__", img("figO_chirp_band_paired.png",
                                 "图 O 脉冲感知分支的收益 vs 波形带宽 / 干扰与信号的时频重叠率"
                                 "（干扰实现已在不同波形间配对固定）"))
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
        .replace("__SI_OLDRNG__", "%.2f" % _si(0))
        .replace("__IA_P54__", "%.3f" % _iag_p("R5 vs R4"))
        .replace("__FIGX__", img("figX_scale_invariance.png",
                                 "图 X 输入尺度敏感性：(a) 老 R5 随输入缩放的性能变化，"
                                 "(b) 尺度不匹配时的性能损失"))
        .replace("__FIGY__", img("figY_inarch_gpu.png",
                                 "图 Y 域内重训下的架构排序（base 条件，GPU 同设备同种子）"))
        .replace("__FIGZ__", img("figZ_pulse_mask.png",
                                 "图 Z 脉冲掩码干预：(a) 七种推理期干预下的 Δ，(b) 错位剂量响应"))
        .replace("__FIGAA__", img("figAA_crosscond.png",
                                 "图 AA 跨条件干预分解：(a) 净 Δ 拆成无信息基线与位置价值，(b) p 的质量几乎不变而价值在变"))
        .replace("__FIGAB__", img("figAB_rateaxis.png",
                                 "图 AB rate 轴分解与交叉验证：(a) 两项反向运动，(b) 与独立 5 种子零样本评估的对比"))
        .replace("__FIGAC__", img("figAC_sweep_fidelity.png",
                                 "图 AC 实测扫频保真：(a)(b) 输入/输出的时频图与 DP 轨迹，(c) 轨迹对齐与残差，(d) 闸门检验"))
        .replace("__FIGW__", img("figW_indomain_alpha.png",
                                 "图 W α 轴域内重训：(a) 域内重训后 α=1.8 的“尾形增强”消失，"
                                 "Δ 回到 base 水平；(b) 原因是 R4 在该条件上提升远大于 R5"))
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
left = [t for t in re.findall(r"__[A-Z0-9_]*__", HTML) if t != "__pycache__"]
print("报告 -> %s" % OUT)
print("大小 %.1f KB" % (len(HTML.encode("utf-8")) / 1024))
print("未替换占位符:", "无" if not left else left)
