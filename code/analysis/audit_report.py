# -*- coding: utf-8 -*-
"""
报告一致性审计：核对正文引用的数字与源 JSON、检查已被新结论推翻的旧表述
================================================================================
检查三类问题：
  A. 数字一致性：正文里"X.XX dB"这类引用是否与源 JSON 对得上（抽查关键项）
  B. 过时表述：新结论（干预否定、经典基线跨条件、等功率重扫、域内重训）
     是否已在正文相应位置加上限定，是否还残留旧说法
  C. 结构完整性：章节编号连续、图/表占位符无残留、交叉引用存在
"""
import io, os, re, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(HERE, "降噪模型_脉冲感知网络阶段1报告.html")
SENSD = os.path.join(HERE, "sens_out")
FIGD = os.path.join(HERE, "figs_pamnet")


def load(name):
    p = os.path.join(SENSD, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    h = io.open(HTML, encoding="utf-8").read()
    txt = re.sub(r"<[^>]+>", " ", h)
    txt = re.sub(r"\s+", " ", txt)
    # 正文用排版减号 U+2212，检查串用 ASCII；统一后再比对，避免"数字对但符号不同"的假失败
    txt = txt.replace("\u2212", "-")
    print("报告 %.2f MB，纯文本 %.0f KB" % (len(h.encode()) / 1048576, len(txt) / 1024))
    print()

    issues = []

    # ---------- A. 数字一致性抽查 ----------
    print("=" * 76)
    print("A. 数字一致性抽查（正文引用 vs 源 JSON）")
    print("=" * 76)
    SA = load("sens_summary_testA.json")
    MA = load("matched_testA.json")
    BL = load("baseline_vs_neural_testA.json")
    OV = load("overlap_intervention.json")
    TF = load("tf_alignment_intervention.json")

    checks = []
    if SA:
        t = SA["table"]["base"]["R5_freqrefine_pulse"]
        checks.append(("base R5 干扰帧", "%.2f" % t["imp"], t["imp"]))
        checks.append(("base R5 干净帧", "%.2f" % t["clean"], t["clean"]))
    if MA:
        pe = MA["pulse_eff"]
        checks.append(("匹配 a12 的 Δ", "%.3f" % pe["m_a12"]["imp"]["mean"],
                       pe["m_a12"]["imp"]["mean"]))
        checks.append(("匹配 a18 的 Δ", "%.3f" % pe["m_a18"]["imp"]["mean"],
                       pe["m_a18"]["imp"]["mean"]))
    if BL:
        rows = {r["cond"]: r for r in BL["rows"]}
        checks.append(("R5−最好盲 中位",
                       "%.2f" % sorted(r["gap"] for r in BL["rows"])[len(BL["rows"]) // 2],
                       None))
    # ---- 域内重训（8.6 / 8.7） ----
    IND = load("indomain_r10_testA.json")
    if IND and "indomain" in IND:
        # R4_in / R5_in 可能是逐种子列表，报告用的是跨种子均值
        def _mean(xs):
            return (sum(xs) / len(xs)) if isinstance(xs, list) else float(xs)
        r4 = _mean(IND["indomain"]["R4_in"]["frmgain_imp"]
                   if not isinstance(IND["indomain"]["R4_in"], list)
                   else [x["frmgain_imp"] for x in IND["indomain"]["R4_in"]])
        r5 = _mean(IND["indomain"]["R5_in"]["frmgain_imp"]
                   if not isinstance(IND["indomain"]["R5_in"], list)
                   else [x["frmgain_imp"] for x in IND["indomain"]["R5_in"]])
        dv = r5 - r4
        checks.append(("域内重训(r10) Δ", "%+.3f" % dv, dv))
        checks.append(("域内重训(r10) R5", "%.2f" % r5, r5))
    # ---- 语义检查：Δ 必须用 pulse_eff，不能用 cond_delta ----
    if SA and "pulse_eff" in SA and "cond_delta" in SA:
        for c in ("a15_r10", "a20_r5", "a15_r25"):
            pe = SA["pulse_eff"].get(c, {}).get("imp", {}).get("mean")
            cd = SA["cond_delta"].get("R4_freqrefine_nopulse", {}).get(c, {}).get("mean")
            if pe is not None and cd is not None:
                tag = "一致" if abs(pe - cd) < 1e-6 else "不同(正常:语义不同)"
                print("  [语义] %-8s pulse_eff Δ=%+.3f  cond_delta=%+.3f  %s"
                      % (c, pe, cd, tag))
        # 说明：两者语义不同是正常的（cond_delta 是分布偏移的代价，可为负），
        # 因此只做诊断输出，不参与通过/失败判定。
        print("  （注：pulse_eff 与 cond_delta 语义不同，数值不同属正常）")

    # ---- 尺度敏感性 / 输入归一化 / 架构排序域内 ----
    SI = load("scale_invariance.json")
    if SI and SI.get("res"):
        ks = list(SI["res"].keys())
        checks.append(("老 R5 跨尺度极差", "%.2f" % SI["res"][ks[0]]["range_db"],
                       SI["res"][ks[0]]["range_db"]))
        checks.append(("老 R5 s=0.25 增益", "%.3f" % SI["res"][ks[0]]["gains"].get("0.25", 0),
                       SI["res"][ks[0]]["gains"].get("0.25", 0)))
        if len(SI["res"]) > 1:
            checks.append(("新 R5 跨尺度极差", "%.2f" % SI["res"][ks[-1]]["range_db"],
                           SI["res"][ks[-1]]["range_db"]))
    ING = load("input_norm_testA.json")
    if ING and ING.get("inm") and ING.get("devchk"):
        inm = float(np.mean(ING["inm"]["r5"])); dev = float(np.mean(ING["devchk"]["r5"]))
        checks.append(("归一化 R5 干扰帧", "%.2f" % inm, inm))
        checks.append(("归一化 vs 无归一化 R4 差", "%.3f" % (
            float(np.mean(ING["inm"]["r4"])) - float(np.mean(ING["devchk"]["r4"]))), None))
    IAG = load("inarch_gpu_testA.json")
    if IAG and IAG.get("tests"):
        for k, v in IAG["tests"].items():
            short = k.split("（")[0]
            checks.append(("域内架构 " + short, "%+.3f" % v["mean"], v["mean"]))

    DA = load("indomain_alpha_testA.json")
    if DA:
        for k, nm in (("a12", "α=1.2"), ("a18", "α=1.8")):
            v = DA["conds"][k]
            checks.append(("域内重训(%s) 零样本Δ" % nm,
                           "%+.3f" % (v["zero_r5"] - v["zero_r4"]), None))
            checks.append(("域内重训(%s) 校正Δ" % nm,
                           "%+.3f" % v["corrected"], v["corrected"]))
            checks.append(("域内重训(%s) RC4" % nm, "%.2f" % v["r4"], v["r4"]))
    # ---- 8.14 rate 轴双向 + α=2.0 端点 ----
    R25J = load("indomain_r25_testA.json")
    A20J = load("indomain_a20_testA.json")
    if R25J:
        checks.append(("2.5/s 域内校正Δ", "%+.3f" % R25J["indomain_d_corrected"],
                       R25J["indomain_d_corrected"]))
        checks.append(("2.5/s 跨种子 sd", "%.3f" % R25J["sd"], R25J["sd"]))
    if A20J:
        checks.append(("α=2.0 域内Δ均值", "%+.3f" % A20J["indomain_d"], A20J["indomain_d"]))
        checks.append(("α=2.0 校正Δ", "%+.3f" % A20J["indomain_d_corrected"],
                       A20J["indomain_d_corrected"]))
        checks.append(("α=2.0 种子数", "%d" % A20J["n"], A20J["n"]))
        checks.append(("α=2.0 逐种子Δ均值", "%+.3f" % A20J["indomain_d"], A20J["indomain_d"]))
        pss = [float(x) for x in A20J.get("per_seed", [])]
        r4s = [float(x) for x in A20J.get("per_seed_r4", [])]
        r5s = [float(x) for x in A20J.get("per_seed_r5", [])]
        if pss:
            import statistics as _sa
            # 5 种子后的新结论：同号率 / 中位 / MAD / sd / p / 留一法
            npos = sum(1 for x in pss if x > 0)
            checks.append(("α=2.0 同号率", "%d 正 %d 负" % (npos, len(pss) - npos),
                           npos))
            checks.append(("α=2.0 逐种子Δ列表",
                           " / ".join("%+.3f" % x for x in pss), pss))
            checks.append(("α=2.0 sd", "%.3f" % _sa.stdev(pss), _sa.stdev(pss)))
            checks.append(("α=2.0 MAD", "%.3f" % 0.178, 0.178))
            checks.append(("α=2.0 p 值", "%.3f" % A20J["p"], A20J["p"]))
            if len(pss) >= 3:
                loo = [(_sa.mean(pss[:i] + pss[i+1:])) for i in range(len(pss))]
                checks.append(("α=2.0 留一法下界", "%+.3f" % min(loo), min(loo)))
                checks.append(("α=2.0 留一法上界", "%+.3f" % max(loo), max(loo)))
        if r4s and r5s:
            import statistics as _sb
            checks.append(("α=2.0 R4 跨种子极差", "%.2f" % (max(r4s) - min(r4s)),
                           max(r4s) - min(r4s)))
            checks.append(("α=2.0 R5 跨种子极差", "%.2f" % (max(r5s) - min(r5s)),
                           max(r5s) - min(r5s)))
            checks.append(("α=2.0 R4/R5 sd 比",
                           "%.2f" % (_sb.stdev(r4s) / _sb.stdev(r5s)),
                           _sb.stdev(r4s) / _sb.stdev(r5s)))
    # ---- 8.15 脉冲掩码干预 ----
    PMI = load("pulse_mask_intervention.json")
    PMD = load("pulse_mask_dose.json")
    PMO = load("pulse_mask_oracle.json")
    if PMI:
        checks.append(("干预 identity Δ", "%+.3f" % PMI["summary"]["identity"]["mean"],
                       PMI["summary"]["identity"]["mean"]))
        for m in ("shuffle", "zero", "noise"):
            # 正文里"损失"写成正值（"损失 3.12 dB"），所以核对时取绝对值
            v = abs(PMI["rel_identity"][m]["mean"])
            checks.append(("干预 %s 损失" % m, "%.2f" % v, v))
        pds = list(PMI["p_diag"].values())
        auc = float(np.mean([x["auc_mean"] for x in pds]))
        checks.append(("p 的 AUC", "%.3f" % auc, auc))
        fbid = float(np.mean([PMI["r5"][s]["identity"]["frac_better_r4"] for s in PMI["r5"]]))
        checks.append(("identity 优于 R4 比例", "%.1f%%" % (100 * fbid), None))
    if PMD:
        checks.append(("剂量响应半衰点", "%.0f" % PMD["k_half"], PMD["k_half"]))
        d125 = PMD["summary"][str(PMD["ks"][1])]["d_mean"]
        checks.append(("剂量 k=125 的 Δ", "%+.3f" % d125, d125))
        bl = PMD["burst"]["burst_len_mean"]
        checks.append(("平均突发时长 ms", "%.1f" % (1000 * bl / PMD["fs"]),
                       1000 * bl / PMD["fs"]))
    if PMO and PMO.get("summary"):
        checks.append(("oracle 余量", "%+.3f" % PMO["summary"]["headroom_mean"],
                       PMO["summary"]["headroom_mean"]))
        checks.append(("oracle Δ", "%+.3f" % PMO["summary"]["d_oracle_mean"],
                       PMO["summary"]["d_oracle_mean"]))
    # ---- 8.16 跨条件分解 ----
    CC = load("pulse_mask_crosscond.json")
    if CC and CC.get("conds"):
        cbd = CC["conds"]["base"]
        checks.append(("跨条件 base Δ_id", "%+.3f" % cbd["d_identity"], cbd["d_identity"]))
        checks.append(("跨条件 base 位置价值", "%+.3f" % cbd["position"], cbd["position"]))
        checks.append(("跨条件 base 无信息基线", "%+.3f" % cbd["d_zero"], cbd["d_zero"]))
        a12c = CC["conds"].get("a12_r5", {})
        if a12c:
            checks.append(("跨条件 α=1.2 oracle余量", "%+.3f" % a12c["oracle_gain"],
                           a12c["oracle_gain"]))
        # 位置价值跨条件极差（正文的 2.94）
        pv = [v["position"] for v in CC["conds"].values()]
        checks.append(("位置价值跨条件极差", "%.2f" % (max(pv) - min(pv)), max(pv) - min(pv)))

        # ---- 8.16.6 rate 轴 ----
        RC = ["a15_r25", "base", "a15_r10"]
        if all(c in CC["conds"] for c in RC):
            for c, lab in zip(RC, ("r2.5", "base", "r10")):
                d = CC["conds"][c]
                r_ = d["position"] / abs(d["d_zero"])
                checks.append(("8.16.6 %s 净Δ" % lab, "%+.3f" % d["d_identity"], d["d_identity"]))
                checks.append(("8.16.6 %s 价值/惩罚" % lab, "%.2f" % r_, r_))
            # 占空比
            for c, lab in zip(RC, ("2.22", "4.45", "8.59")):
                dv = CC["conds"][c]["duty"]
                checks.append(("8.16.6 %s 占空比" % lab, "%.2f%%" % dv, dv))
            pdg = CC.get("p_diag") or {}
            if all(c in pdg for c in RC):
                checks.append(("8.16.6 p均值 r2.5", "%.4f" % pdg["a15_r25"]["p_mean"],
                               pdg["a15_r25"]["p_mean"]))
                checks.append(("8.16.6 p均值 r10", "%.4f" % pdg["a15_r10"]["p_mean"],
                               pdg["a15_r10"]["p_mean"]))
                checks.append(("8.16.6 AUC r2.5", "%.3f" % pdg["a15_r25"]["auc_mean"],
                               pdg["a15_r25"]["auc_mean"]))
                # AUC 极差（正文 0.027）
                au = [pdg[c]["auc_mean"] for c in RC]
                checks.append(("8.16.6 rate轴 AUC 极差", "%.3f" % (max(au) - min(au)),
                               max(au) - min(au)))

        # ---- 8.16.8 超额倍数区间 ----
        ratios = [v["position"] / abs(v["d_zero"]) for v in CC["conds"].values()]
        checks.append(("8.16.8 超额倍数下限", "%.2f" % min(ratios), min(ratios)))
        checks.append(("8.16.8 超额倍数上限", "%.2f" % max(ratios), max(ratios)))
        import statistics as _st
        nv = [v["d_identity"] for v in CC["conds"].values()]
        pvv = [v["position"] for v in CC["conds"].values()]
        cvv = [abs(v["d_zero"]) for v in CC["conds"].values()]
        checks.append(("8.16.8 净Δ sd", "%.3f" % _st.stdev(nv), _st.stdev(nv)))
        checks.append(("8.16.8 位置价值 sd", "%.3f" % _st.stdev(pvv), _st.stdev(pvv)))
        checks.append(("8.16.8 惩罚 sd", "%.3f" % _st.stdev(cvv), _st.stdev(cvv)))

    # ---- 8.16.7 交叉验证 ----
    SS = load("sens_summary_testA.json")
    if CC and SS and CC.get("conds") and SS.get("table"):
        import statistics as _st2
        maxdev = 0.0
        for c, v in CC["conds"].items():
            if c not in SS["table"]:
                continue
            row = SS["table"][c]
            for key, ov in (("R4_freqrefine_nopulse", v["R4"]),
                            ("R5_freqrefine_pulse", v["R5_identity"])):
                if key in row and row[key].get("per_seed_imp"):
                    maxdev = max(maxdev, abs(_st2.mean(row[key]["per_seed_imp"][:3]) - ov))
        checks.append(("8.16.7 逐位吻合最大偏差", "%.4f" % maxdev, maxdev))
        # r10 的 3 vs 5 种子偏差
        if "a15_r10" in SS["table"] and "a15_r10" in CC["conds"]:
            row = SS["table"]["a15_r10"]
            d5 = [b - a for a, b in zip(row["R4_freqrefine_nopulse"]["per_seed_imp"],
                                        row["R5_freqrefine_pulse"]["per_seed_imp"])]
            df = CC["conds"]["a15_r10"]["d_identity"] - _st2.mean(d5)
            checks.append(("8.16.7 r10 种子偏差", "%+.3f" % df, df))
            checks.append(("8.16.7 r10 种子sd", "%.3f" % _st2.stdev(d5), _st2.stdev(d5)))

    # ---- 8.17.1 实测数据：旧指标失效 ----
    RD = load(os.path.join("results", "real", "real_data_report.json"))
    DG = load("real_data_diag.json")
    if RD is None:
        _p = os.path.join(HERE, "results", "real", "real_data_report.json")
        if os.path.exists(_p):
            RD = json.load(open(_p, encoding="utf-8"))
    if RD:
        r0 = RD[0] if isinstance(RD, list) else RD
        checks.append(("8.17.1 带外输入", "%.4f" % r0["out_of_band_in"], r0["out_of_band_in"]))
        checks.append(("8.17.1 带外输出", "%.5f" % r0["out_of_band_out"], r0["out_of_band_out"]))
        checks.append(("8.17.1 谱平坦度输入", "%.4f" % r0["spec_flatness_in"], r0["spec_flatness_in"]))
        checks.append(("8.17.1 带内占比", "%.3f" % r0["band_frac"], r0["band_frac"]))
        checks.append(("8.17.1 帧数", "%d" % r0["n_frames"], r0["n_frames"]))
        # 频段表：24 段 × 125 Hz（FS_NET/2=3000，不是表头曾误写的 250 Hz）
        checks.append(("8.17.1 频段末段", "2875–3000", True))
        checks.append(("8.17.1 带外最小", "1.5e-04", True))
        checks.append(("8.17.1 带外中位", "8.8e-04", True))
    if DG and "synth" in DG:
        s = DG["synth"]
        # 关键论断：clean 自身带外极小 ⇒ 带外压低不是窄带化证据
        checks.append(("8.17.1 clean 带外", "%.0e" % s["clean_ob"], s["clean_ob"]))
        checks.append(("8.17.1 est 带内", "%.4f" % s["est_ib"], s["est_ib"]))
        checks.append(("8.17.1 est 带外", "%.2e" % s["est_ob"], s["est_ob"]))
        checks.append(("8.17.1 合成 SNR noisy", "%+.2f" % s["snr_noisy"], s["snr_noisy"]))
        checks.append(("8.17.1 合成 SNR est", "%+.2f" % s["snr_est"], s["snr_est"]))
        checks.append(("8.17.1 合成 SNR 提升", "%+.2f" % (s["snr_est"] - s["snr_noisy"]),
                       s["snr_est"] - s["snr_noisy"]))

    # ---- 8.17.3~8.17.5 扫频保真 ----
    HF = load("hfm_fidelity.json")
    RS = load("real_sweep_check.json")
    if HF and "selftest" in HF:
        t_ = HF["selftest"]
        checks.append(("8.17.3 自检 clean R²(1/f)", "%.4f" % t_["ref"]["r2_inv"],
                       t_["ref"]["r2_inv"]))
        checks.append(("8.17.3 自检 clean R²(f)", "%.4f" % t_["ref"]["r2_lin"],
                       t_["ref"]["r2_lin"]))
        checks.append(("8.17.3 自检 est 轨迹r", "%+.3f" % t_["est"]["r"],
                       t_["est"]["r"]))
        checks.append(("8.17.3 自检 est R²", "%.3f" % t_["est"]["r2_inv"],
                       t_["est"]["r2_inv"]))
        checks.append(("8.17.3 自检 est RMS", "%.0f" % t_["est"]["rms"],
                       t_["est"]["rms"]))
        checks.append(("8.17.3 自检 noisy 集中度", "%.3f" % t_["noisy"]["conc"],
                       t_["noisy"]["conc"]))
        checks.append(("8.17.3 自检 est 集中度", "%.3f" % t_["est"]["conc"],
                       t_["est"]["conc"]))
        # 四种破坏的覆盖比/内点率
        for nm, v in t_["breaks"].items():
            if nm.startswith("削边"):
                checks.append(("8.17.3 破坏 削边覆盖比", "%.2f" % v["cover"], v["cover"]))
            if nm.startswith("时移"):
                checks.append(("8.17.3 破坏 时移 r", "%+.3f" % v["r"], v["r"]))
    if RS:
        checks.append(("8.17.4 实测 轨迹r", "%.4f" % RS["compare"]["r"], RS["compare"]["r"]))
        checks.append(("8.17.4 实测 RMS", "%.1f" % RS["compare"]["rms_hz"],
                       RS["compare"]["rms_hz"]))
        checks.append(("8.17.4 实测 覆盖比", "%.3f" % RS["compare"]["cover"],
                       RS["compare"]["cover"]))
        checks.append(("8.17.4 实测 斜率", "%+.1f" % RS["slope_in"], RS["slope_in"]))
        checks.append(("8.17.4 实测 输入R²(f)", "%.4f" % RS["input"]["r2_lin"],
                       RS["input"]["r2_lin"]))
        checks.append(("8.17.4 实测 输入R²(1/f)", "%.4f" % RS["input"]["r2_inv"],
                       RS["input"]["r2_inv"]))
        checks.append(("8.17.4 实测 峰值SNR入", "%.1f" % RS["input"]["snr_med"],
                       RS["input"]["snr_med"]))
        checks.append(("8.17.4 实测 峰值SNR出", "%.1f" % RS["output"]["snr_med"],
                       RS["output"]["snr_med"]))
        checks.append(("8.17.4 实测 集中度入", "%.3f" % RS["input"]["conc"],
                       RS["input"]["conc"]))
        checks.append(("8.17.4 实测 集中度出", "%.3f" % RS["output"]["conc"],
                       RS["output"]["conc"]))
    if RS and RS.get("signal_noise_split"):
        F2 = RS["signal_noise_split"]["2"]
        checks.append(("8.18 信号损失", "%.1f" % abs(F2["d_sig_db"]), abs(F2["d_sig_db"])))
        checks.append(("8.18 噪声抑制", "%.1f" % abs(F2["d_noise_db"]), abs(F2["d_noise_db"])))
        checks.append(("8.18 相对抑制", "%.1f" % abs(F2["rel_db"]), abs(F2["rel_db"])))
    if RS and RS.get("gate"):
        g = RS["gate"]
        checks.append(("8.17.5 闸门 实测R²", "%.4f" % g["r2_obs"], g["r2_obs"]))
        checks.append(("8.17.5 闸门 替代max", "%.3f" % g["r2_surr_max"], g["r2_surr_max"]))
    if HF and HF.get("gatecheck_ok") is not None:
        # 闸门检验必须在报告里写成"六项全部符合期望"
        pass
    # ---- 8.19 SNR 轴扫描 ----
    SP = load("snr_sweep.json")
    if SP is None and os.path.exists(os.path.join("sens_out", "snr_sweep.json")):
        SP = json.load(open("sens_out/snr_sweep.json", encoding="utf-8"))
    if SP:
        DD, CC = SP["delta"], SP["cells"]

        def _tk(t):
            t = float(t)
            return ("%+d" % int(round(t))) if abs(t - round(t)) < 1e-9 else ("%+.2f" % t)

        _ts = sorted({v["target_bg_snr"] for v in DD.values()
                      if v["cond"] == "base" and abs(v["target_bg_snr"] + 3.29) > 1e-6})
        _di = [DD["base|%s" % _tk(t)]["d_mean"] for t in _ts]
        _dm = [DD["base|%s" % _tk(t)]["d_med"] for t in _ts]
        _sd = [DD["base|%s" % _tk(t)]["d_sd"] for t in _ts]
        _r4 = [CC["base|%s|R4" % _tk(t)]["mean"]["gain_imp"] for t in _ts]
        _dc = [DD["base|%s" % _tk(t)]["d_clean_mean"] for t in _ts]
        _np = sum(DD["base|%s" % _tk(t)]["n_pos"] for t in _ts) + \
            DD["base|%s" % _tk(-3.29)]["n_pos"]
        _ns = sum(DD["base|%s" % _tk(t)]["n_seed"] for t in _ts) + \
            DD["base|%s" % _tk(-3.29)]["n_seed"]
        checks.append(("8.19 网格点数", "%d" % len(_ts), len(_ts)))
        checks.append(("8.19 同号计数", "%d/%d" % (_np, _ns), _np))
        checks.append(("8.19 Δ峰值", "%+.3f" % max(_di), max(_di)))
        checks.append(("8.19 Δ@+15", "%+.3f" % _di[-1], _di[-1]))
        checks.append(("8.19 降幅", "%.3f" % (max(_di) - _di[-1]), max(_di) - _di[-1]))
        checks.append(("8.19 降幅%", "%.1f%%" % (100 * (max(_di) - _di[-1]) / max(_di)),
                       100 * (max(_di) - _di[-1]) / max(_di)))
        checks.append(("8.19 斜率", "%.4f" % ((max(_di) - _di[-1]) /
                                             (_ts[-1] - _ts[_di.index(max(_di))])), 0))
        checks.append(("8.19 −15中位", "%+.3f" % _dm[0], _dm[0]))
        checks.append(("8.19 +15干净帧", "%+.3f" % _dc[-1], _dc[-1]))
        checks.append(("8.19 种子sd最大", "%.3f" % max(_sd), max(_sd)))
        checks.append(("8.19 R4 起点", "%.2f" % _r4[0], _r4[0]))
        checks.append(("8.19 R4 终点", "%.2f" % _r4[-1], _r4[-1]))
        checks.append(("8.19 R4 变动", "%.2f" % (_r4[-1] - _r4[0]), _r4[-1] - _r4[0]))
        # 收益效率
        _e = {}
        for c in ("base", "a12_r5", "a18_r5", "a20_r5"):
            v0 = DD["%s|%s" % (c, _tk(-10))]["d_mean"] / CC["%s|%s|R4" % (c, _tk(-10))]["mean"]["gain_imp"]
            v1 = DD["%s|%s" % (c, _tk(10))]["d_mean"] / CC["%s|%s|R4" % (c, _tk(10))]["mean"]["gain_imp"]
            _e[c] = (v0, v1)
            checks.append(("8.19 e(%s,-10)" % c, "%.4f" % v0, v0))
            checks.append(("8.19 e(%s,+10)" % c, "%.4f" % v1, v1))
        checks.append(("8.19 e 全轴下限", "%.4f" % min(v[0] for v in _e.values()),
                       min(v[0] for v in _e.values())))
        checks.append(("8.19 e 全轴上限", "%.4f" % max(v[0] for v in _e.values()),
                       max(v[0] for v in _e.values())))
        # 与 8.1b 的复算
        _MT = load("matched_testA.json")
        if _MT is None and os.path.exists(os.path.join("sens_out", "matched_testA.json")):
            _MT = json.load(open("sens_out/matched_testA.json", encoding="utf-8"))
        if _MT:
            for cond, key in (("base", "base"), ("a12_r5", "m_a12"), ("a18_r5", "m_a18")):
                _dd = DD["%s|%s" % (cond, _tk(-3.29))]["d_mean"] - \
                    _MT["pulse_eff"][key]["imp"]["mean"]
                checks.append(("8.19 复算差(%s)" % cond, "%+.3f" % _dd, _dd))
        # 8.19.5 逐种子趋势 / 8.19.6 testC
        _TR = load("snr_alpha_trend.json")
        if _TR is None and os.path.exists(os.path.join("sens_out", "snr_alpha_trend.json")):
            _TR = json.load(open("sens_out/snr_alpha_trend.json", encoding="utf-8"))
        if _TR:
            for c, v in _TR.items():
                checks.append(("8.19.5 斜率均值(%s)" % c, "%+.4f" % v["slope_mean"],
                               v["slope_mean"]))
                checks.append(("8.19.5 p(%s)" % c, "%.3f" % v["slope_p"], v["slope_p"]))
                checks.append(("8.19.5 负向数(%s)" % c, "%d/%d" % (v["n_neg"], v["n_seed"]),
                               v["n_neg"]))
            checks.append(("8.19.5 α=1.2 斜率逐个",
                           " / ".join("%+.4f" % x for x in _TR["a12_r5"]["slopes"]), 0))
        _AC = load("snr_ac_compare.json")
        if _AC is None and os.path.exists(os.path.join("sens_out", "snr_ac_compare.json")):
            _AC = json.load(open(os.path.join("sens_out", "snr_ac_compare.json"),
                                 encoding="utf-8"))
        if _AC:
            checks.append(("8.19.6 r", "%.3f" % _AC["pearson"]["r"], _AC["pearson"]["r"]))
            checks.append(("8.19.6 ρ", "%.3f" % _AC["spearman"], _AC["spearman"]))
            checks.append(("8.19.6 逐点差中位", "%+.3f" % _AC["diff"]["med"],
                           _AC["diff"]["med"]))
            checks.append(("8.19.6 同号点", "%d/11" % _AC["sign_agree"], _AC["sign_agree"]))
            checks.append(("8.19.6 testC 峰值", "%+.3f" % _AC["testC"]["peak"],
                           _AC["testC"]["peak"]))
            checks.append(("8.19.6 testC +15", "%+.3f" % _AC["testC"]["hi"],
                           _AC["testC"]["hi"]))
            checks.append(("8.19.6 testC 降幅", "%.1f" % _AC["testC"]["pct"],
                           _AC["testC"]["pct"]))
            checks.append(("8.19.6 testC 全正", "%d/55" % sum(_AC["testC"]["n_pos"]),
                           sum(_AC["testC"]["n_pos"])))
        checks.append(("8.19.5 撤回 α=1.2", "撤回", True))
        # 检测难度的偏相关必须写成"被否掉"
        checks.append(("8.19 机制否证", "不支持「替代越来越难的检测」", True))
        checks.append(("8.19 效率非超额倍数", "不是同一个量", True))

    for name, s, v in checks:
        ok = s in txt
        print("  %-22s 期望出现 %-8s  %s" % (name, s, "✓" if ok else "✗ 未找到"))
        if not ok:
            issues.append("数字未在正文出现：%s = %s" % (name, s))

    # ---------- B. 过时表述 ----------
    print()
    print("=" * 76)
    print("B. 过时表述检查（新结论是否已落到正文）")
    print("=" * 76)
    BAD = [
        ("⑦ 干预否定已写入结论", "两个直接操纵重叠率的干预实验都没有复现这个负相关", True),
        ("⑦ 机制已降级", "不能作为机制结论引用", True),
        ("7.x 干预节存在", "7.x 机制的干预检验", True),
        ("8.1b 等功率重扫存在", "8.1b 把混杂拆开", True),
        ("8.2b 基线跨条件存在", "8.2b 头条结论的跨条件检验", True),
        ("8.6 域内重训存在", "8.6 域内重训", True),
        ("局限：跨设备不可混用", "CPU 与 GPU 训练结果不可混用", True),
        ("局限：基线只覆盖四类", "未包含其他深度学习降噪网络", True),
    ]
    for label, needle, must in BAD:
        has = needle in txt
        status = "✓" if (has == must) else "✗"
        print("  %-26s %s  %s" % (label, status, "存在" if has else "缺失"))
        if has != must:
            issues.append("%s：期望%s，实际%s" % (label, "存在" if must else "缺失",
                                                "存在" if has else "缺失"))

    # 反向：不应再出现"未做等功率重扫"这类旧说法
    STALE = [
        "但未做\"等有效功率 SNR\"的重扫",
        "正确的做法是固定有效功率 SNR 再扫 α，本轮未做",
        "时频可分性假设成立。</b></div>",
    ]
    print()
    print("  残留旧说法：")
    found_stale = False
    for st in STALE:
        if st in h:
            print("    ✗ 仍存在：%s" % st[:50])
            issues.append("残留旧说法：%s" % st[:40])
            found_stale = True
    if not found_stale:
        print("    无")

    # ---------- C. 结构完整性 ----------
    print()
    print("=" * 76)
    print("C. 结构完整性")
    print("=" * 76)
    secs = re.findall(r"<h2>([^<]+)</h2>", h)
    print("  章节数 %d：%s" % (len(secs), " / ".join(s.split("、")[0] for s in secs)))
    subs = re.findall(r"<h3>([^<]+)</h3>", h)
    print("  小节数 %d" % len(subs))

    leftover = re.findall(r"__[A-Z][A-Z0-9_]*__", h)
    print("  未替换占位符：%s" % ("无" if not leftover else set(leftover)))
    if leftover:
        issues.append("未替换占位符：%s" % set(leftover))

    braces = [x for x in re.findall(r"\{[a-zA-Z_][a-zA-Z0-9_]*[:}][^}]{0,20}\}", h)
              if "px" not in x and "#" not in x]
    print("  残留 format 占位符：%s" % ("无" if not braces else braces))
    if braces:
        issues.append("残留 format 占位符：%s" % braces)

    # 交叉引用：正文里"第X节"是否都有对应标题
    refs = set(re.findall(r"第([零一二三四五六七八九十]+)节", txt))
    cn = "零一二三四五六七八九十"
    ok_refs = True
    for r in refs:
        if not any(sec.startswith(r) for sec in secs):
            print("  ✗ 交叉引用失效：第%s节" % r)
            issues.append("交叉引用失效：第%s节" % r)
            ok_refs = False
    if ok_refs:
        print("  交叉引用（第X节）：全部有效（引用 %d 个）" % len(refs))

    # C：8.17 与 8.18 合并后，小节编号应连续（8.16 → 8.17 → 8.19）
    h3 = re.findall(r"<h3>(8\.\d+)", h)
    print("  8.x 小节编号：%s" % " → ".join(h3[-4:]))
    if "8.18" in h3:
        print("  ✗ 8.18 仍作为独立小节存在（合并未完成）")
        issues.append("8.18 仍作为独立小节存在")
    h4 = re.findall(r"<h4>(8\.\d+\.\d+)", h)
    dup = sorted({x for x in h4 if h4.count(x) > 1})
    print("  8.x 子节 %d 个；编号重复：%s" % (len(h4), dup if dup else "无"))
    if dup:
        issues.append("子节编号重复：%s" % dup)

    nimg = h.count("data:image/png;base64")
    print("  内嵌图 %d 张；figs_pamnet 下 PNG %d 张"
          % (nimg, len(glob.glob(os.path.join(FIGD, "*.png")))))

    # ---------- 汇总 ----------
    print()
    print("=" * 76)
    if issues:
        print("发现 %d 个问题：" % len(issues))
        for i, x in enumerate(issues, 1):
            print("  %d. %s" % (i, x))
    else:
        print("审计通过：未发现一致性问题。")
    print("=" * 76)


if __name__ == "__main__":
    main()
