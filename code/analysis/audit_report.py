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
        pss = [float(x) for x in A20J.get("per_seed", [])]
        if pss and min(pss) < 0 < max(pss) and A20J.get("per_seed_r4"):
            # 报告在"符号相反"分支里会打印 R4 跨种子极差，必须与 JSON 一致
            rr = max(A20J["per_seed_r4"]) - min(A20J["per_seed_r4"])
            checks.append(("α=2.0 R4 跨种子极差", "%.2f" % rr, rr))
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
