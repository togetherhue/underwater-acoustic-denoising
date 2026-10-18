# -*- coding: utf-8 -*-
"""生成阶段0 报告：脉冲噪声下的现有模型 + 时域分支复检"""
import base64, os, json, re
import numpy as np

FIG = r"E:/workbuddy/store/2026-10-12-22-50-08/figs_pam"
OUT = r"E:/workbuddy/store/2026-10-12-22-50-08/降噪模型_脉冲噪声阶段0报告.html"

S = json.load(open(os.path.join(FIG, "summary_pam.json"), encoding="utf-8"))
TESTS = ["testA", "testC"]


def img(name, cap):
    p = os.path.join(FIG, name)
    if not os.path.exists(p):
        return "<figure><figcaption>（%s：图尚未生成）</figcaption></figure>" % cap
    with open(p, "rb") as f:
        b = base64.b64encode(f.read()).decode()
    return ('<figure><img src="data:image/png;base64,%s" alt="%s">'
            "<figcaption>%s</figcaption></figure>") % (b, cap, cap)


ORDER = ["pam_freq", "pam_full", "pam_freq_g"]
SHORT = {"pam_freq": "α稳定+脉冲 · 纯频域", "pam_full": "α稳定+脉冲 · 时域+频域",
         "pam_freq_g": "高斯+脉冲 · 纯频域"}

rows = ""
for tag in ORDER:
    if tag not in S:
        continue
    d = S[tag]
    for ts in TESTS:
        m = d["d"][ts]
        rows += ("<tr><td>%s</td><td>%s</td><td>%s</td><td>%d</td>"
                 "<td>%+.2f</td><td>%+.2f</td><td>%.2f</td><td>%.2f</td>"
                 "<td>%.2f</td><td>%.2f</td></tr>"
                 % (SHORT[tag], ts, "{:,}".format(d["params"]), int(m["n"]),
                    m["snr_in"], m["snr_out"] - m["snr_in"], m["frmgain_med"],
                    m["frmgain_clean"], m["frmgain_imp"], m["frmgain_imp_min"]))

a = S["pam_freq"]["d"]["testA"]
b = S["pam_freq_g"]["d"]["testA"]
c = S["pam_full"]["d"]["testA"]
wt = S["pam_full"]["w_time"]

HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>降噪模型 · 脉冲噪声阶段0</title>
<style>
:root{--bg:#f7f8fa;--card:#fff;--ink:#1a1d21;--muted:#6b7280;--line:#e5e7eb;
      --blue:#2563eb;--green:#15803d;--red:#b91c1c;--amber:#b45309}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
     font-size:15px;line-height:1.75}
.wrap{max-width:1020px;margin:0 auto;padding:40px 24px 80px}
h1{font-size:27px;margin:0 0 6px;letter-spacing:-.4px}
h2{font-size:20px;margin:38px 0 12px;padding-bottom:8px;border-bottom:2px solid var(--line)}
h3{font-size:16px;margin:24px 0 8px;color:#374151}
.sub{color:var(--muted);font-size:13.5px;margin-bottom:26px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
      padding:20px 24px;margin:18px 0}
.kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:20px 0}
.kpi div{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.kpi b{display:block;font-size:24px;letter-spacing:-.5px;margin-bottom:2px}
.kpi span{color:var(--muted);font-size:12.5px}
table{width:100%;border-collapse:collapse;margin:14px 0;font-size:13.5px}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--line)}
th{background:#f3f4f6;font-weight:600;color:#374151}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
tr:last-child td{border-bottom:none}
figure{margin:22px 0;text-align:center}
figure img{max-width:100%;border:1px solid var(--line);border-radius:8px;background:#fff}
figcaption{color:var(--muted);font-size:12.5px;margin-top:8px}
code{background:#f3f4f6;padding:1.5px 6px;border-radius:4px;font-size:12.5px;
     font-family:Consolas,monospace}
pre{background:#1e293b;color:#e2e8f0;padding:14px 16px;border-radius:8px;
    overflow-x:auto;font-size:12.5px;line-height:1.6}
ul,ol{padding-left:22px}li{margin:5px 0}
.note{border-left:3px solid var(--blue);background:#eff6ff;padding:12px 16px;
      border-radius:0 8px 8px 0;margin:16px 0;font-size:14px}
.warn{border-left:3px solid var(--amber);background:#fffbeb;padding:12px 16px;
      border-radius:0 8px 8px 0;margin:16px 0;font-size:14px}
.ok{border-left:3px solid var(--green);background:#f0fdf4;padding:12px 16px;
    border-radius:0 8px 8px 0;margin:16px 0;font-size:14px}
</style></head><body><div class="wrap">

<h1>阶段0：脉冲噪声下的现有模型 + 时域分支复检</h1>
<div class="sub">轻量化水声信号降噪模型 · 按导师方案改造 · 第一阶段 · 生成于 __DATE__</div>

<div class="card">
<h3 style="margin-top:0">一句话结论</h3>
<p>导师指出「实测信号的脉冲噪声比较大」，要求把噪声模型改成 <b>y = s + v + i</b>
（v 为 α 稳定分布背景噪声，i 为突发干扰）。本阶段<b>不改网络、不改损失</b>，
只换数据与指标，先拿到干净基线。</p>
<p><b>三个结论：</b></p>
<ol>
<li><b>现有模型在脉冲噪声下仍然有效。</b>5,910 参数纯频域模型在 testA 上把
输入 −4.61 dB 提升到 <b>+12.51 dB</b>；逐帧看，干净帧降噪 <b>14.13 dB</b>、
含突发干扰的帧降噪 <b>25.09 dB</b>。干扰段反而压得更狠（脉冲幅度大，dB 收益自然高）。</li>
<li><b>α 稳定背景并没有让降噪"更难"。</b>把背景从高斯换成 α 稳定（突发干扰完全相同），
干净帧增益 <b>__DA__ dB</b>、干扰帧增益 <b>__DB__ dB</b> —— 几乎不变。
输出 SNR 差的 4.08 dB <b>全部来自输入变差</b>，不是模型变弱。</li>
<li><b>时域分支依然没用。</b>高斯噪声下它的融合权重衰减到 0.006；换到脉冲噪声后
放大到 <b>__WT__</b>（约 7 倍），但仍然很小，而且<b>加上它反而全面变差</b>
（干扰帧增益 −1.20 dB）。</li>
</ol>
</div>

<div class="kpi">
  <div><b>__A_OUT__ dB</b><span>testA 输出 SNR（输入 −4.61）</span></div>
  <div><b>__A_CLEAN__ dB</b><span>干净帧降噪增益</span></div>
  <div><b>__A_IMP__ dB</b><span>干扰帧降噪增益</span></div>
  <div><b>__WT__</b><span>时域分支最终权重（高斯下 0.006）</span></div>
</div>

<h2>1. 噪声模型改造</h2>
<div class="card">
<h3>1.1 信号模型</h3>
<p>按老师方案：<code>y_n = s_n + v_n + i_n</code>。s 是 chirp 过 RAM 多径信道后的接收波形，
v 是背景噪声，i 是脉冲/突发干扰。</p>

<h3>1.2 α 稳定分布背景噪声</h3>
<p>用 Chambers-Mallows-Stuck 方法实现 <code>stblrnd.m</code>。α=1.5、β=0（对称）。</p>
<table>
<thead><tr><th>验证项</th><th>结果</th></tr></thead>
<tbody>
<tr><td>α=2 是否退化为高斯</td><td>样本 std 与理论 2γ² 吻合，峰度 <b>3.00</b>（理论 3）</td></tr>
<tr><td>用特征函数反推 α</td><td>E[cos(tX)]=exp(−γ^α|t|^α)，拟合 α 误差 <b>≤0.06%</b></td></tr>
<tr><td>尾部指数</td><td>log-log 生存函数斜率与 −α 吻合（α=1.2 时误差 0.02%）</td></tr>
<tr><td>偏斜参数 β</td><td>β 从 −1 到 +1，分位数对称变化，β=0 时严格对称</td></tr>
<tr><td>脉冲性</td><td>α=1.2 时 0.65% 的样本超过 20×q75；α=2 时为 <b>0</b></td></tr>
</tbody></table>

<h3>1.3 突发干扰</h3>
<p>泊松到达 + 随机幅度 + 随机持续时间，四种波形覆盖老师提到的「尖峰、脉冲和突发边缘」：
尖峰 impulse / 衰减振荡 decay / 限带噪声 burst / 矩形 rect。同时输出干扰位置掩码
<code>imp_mask</code>（供后续 L_imp 辅助损失使用）。</p>
<table>
<thead><tr><th>验证项</th><th>实测</th><th>理论/设定</th></tr></thead>
<tbody>
<tr><td>到达次数（2 s 内）</td><td>15.86 ± 4.31</td><td>16.0 ± 4.00（泊松）</td></tr>
<tr><td>占空比</td><td>9.14%</td><td>9.62%</td></tr>
<tr><td>波形类型分布</td><td>24.8 / 30.2 / 24.5 / 20.5 %</td><td>25 / 30 / 25 / 20 %</td></tr>
<tr><td>掩码一致性</td><td colspan="2">mask=0 处干扰严格为 0、mask=1 处非 0，全部通过</td></tr>
</tbody></table>

<h3>1.4 噪声强度怎么定义（α 稳定分布的关键问题）</h3>
<div class="warn">
<b>α &lt; 2 时方差无穷，无法用功率定义信噪比。</b>本实验采用
<b>75% 分位数匹配</b>：令噪声的 q75(|v|) 等于同功率高斯噪声的 q75（= 0.6745σ）。
这样「核心宽度」一致，α 越小尾部越重。这是配对比较的前提。
</div>

<h3>1.5 配对设计</h3>
<p>背景噪声与突发干扰都<b>按样本序号定种子</b>生成，因此
「α稳定」与「高斯」两套数据的<b>突发干扰实现完全相同</b>（实测最大差 0），
只有背景噪声的分布不同 —— 构成严格配对对照。干净信号与既有 v3 数据集
<b>逐位一致</b>（最大差 0），保证与历史结果可比。</p>
</div>

<h2>2. 指标问题：为什么不能只看全局 SNR</h2>
<div class="card">
<p>沿用 <code>10log10(||s||²/||e||²)</code> 这种全局 SNR，结果会被少数几个极大脉冲完全主导。
自检实验（合成数据，已知真值）说明问题：</p>
<table>
<thead><tr><th>情形</th><th>全局 SNR</th><th>帧 SNR 中位</th><th>干扰帧增益</th><th>干扰帧最差</th></tr></thead>
<tbody>
<tr><td>好模型（残差 0.30×）</td><td>9.02</td><td>9.11</td><td>10.46</td><td>10.46</td></tr>
<tr><td><b>漏掉 1 个脉冲</b></td><td><b>10.16</b></td><td>15.13</td><td>16.48</td><td><b>−0.03</b></td></tr>
</tbody></table>
<div class="warn">
<b>全局 SNR 把「漏掉脉冲」判成了更好</b>（10.16 &gt; 9.02）。只有
<b>干扰帧增益的最差值</b>能精确抓住漏掉的脉冲（−0.03 vs 10.46）。
</div>
<p>因此本实验采用的主指标是<b>逐帧降噪增益</b>：</p>
<pre>frmgain_f = 10·log10( ||n_f − s_f||² / ||ŝ_f − s_f||² )</pre>
<p>即「该帧的输入噪声能量 / 输出噪声能量」。相比帧 SNR，它不受该帧信号强弱影响
（分子分母用同一帧），在信号包络起伏大的 chirp 上更稳定。再按「该帧是否含干扰」
分解成 <code>frmgain_clean</code>（干净段）与 <code>frmgain_imp</code>（脉冲段）——
这正是 PAM-Net 要解决的问题。</p>
</div>

<h2>3. 结果</h2>
__FIG__
<h3>3.1 主表</h3>
<table>
<thead><tr><th>模型</th><th>测试集</th><th>参数量</th><th>n</th>
<th>输入SNR</th><th>输出提升</th><th>帧增益中位</th><th>干净帧增益</th>
<th>干扰帧增益</th><th>干扰帧最差</th></tr></thead>
<tbody>
__ROWS__
</tbody></table>
<p>注意：α 稳定与高斯两套数据的<b>输入 SNR 不同</b>（−4.61 vs −0.64 dB），
所以输出 SNR 不可直接比；但<b>帧增益是相对量，可以直接比</b>。</p>

<h3>3.2 背景分布的影响（α稳定 vs 高斯，突发干扰完全相同）</h3>
<table>
<thead><tr><th>指标</th><th>α稳定背景</th><th>高斯背景</th><th>差值</th></tr></thead>
<tbody>
<tr><td>干净帧增益</td><td>__A_CLEAN__</td><td>__G_CLEAN__</td><td><b>__DA__</b></td></tr>
<tr><td>干扰帧增益</td><td>__A_IMP__</td><td>__G_IMP__</td><td>__DB__</td></tr>
<tr><td>干扰帧最差</td><td>__A_IMPMIN__</td><td>__G_IMPMIN__</td><td>__D_MIN__</td></tr>
<tr><td>全局输出 SNR</td><td>__A_OUT__</td><td>__G_OUT__</td><td>__D_OUT__</td></tr>
</tbody></table>
<div class="ok">
<b>关键读法：</b>帧增益（相对量）几乎不变（__DA__ / __DB__），
而全局输出 SNR 差 __D_OUT__ dB —— 差异<b>全部来自输入变差</b>
（α稳定背景的 rms 是信号的 2.2~2.6 倍，高斯只有 0.67 倍，因为重尾贡献了大量能量）。
<b>结论：α 稳定背景并没有让降噪本身变难，只是输入噪声能量更大。</b>
</div>

<h3>3.3 ★ 时域分支复检</h3>
<div class="card">
<p>高斯噪声下的既有结论：完整模型中时域分支的融合权重 <code>w_time</code> 从 0.297
一路衰减到 <b>0.006</b>，模型自己把它弃用了；删掉后只损失 0.20 dB。</p>
<p><b>当时的猜想</b>是：时域分支没用，是因为高斯噪声没有「局部异常」这种需要细粒度
时域识别的东西；换到脉冲噪声后它可能就有用了。所以这次专门复检。</p>
<table>
<thead><tr><th>噪声类型</th><th>最终 w_time</th><th>帧增益中位</th><th>干净帧增益</th><th>干扰帧增益</th></tr></thead>
<tbody>
<tr><td>高斯（既有，25 轮）</td><td>0.006</td><td colspan="3">—— 时域分支被弃用</td></tr>
<tr><td><b>α稳定+脉冲（本次，15 轮）</b></td><td><b>__WT__</b></td><td>14.55</td><td>13.59</td><td>23.89</td></tr>
<tr><td>对照：同数据纯频域（5,910 参数）</td><td>—</td><td><b>15.06</b></td><td><b>14.13</b></td><td><b>25.09</b></td></tr>
</tbody></table>
<div class="warn">
<b>结论：猜想不成立。</b>时域分支的权重确实放大了约 7 倍（0.006 → __WT__），
说明脉冲噪声让它「稍微有点用」；但它仍然只占 4% 权重，而且
<b>加上时域分支后所有指标全面变差</b>：帧增益中位 −0.51 dB、
干净帧 −0.55 dB、<b>干扰帧 −1.20 dB</b>、干扰帧最差 −0.77 dB。<br>
换句话说：<b>在脉冲噪声下，5,910 参数的纯频域模型依然胜过 57,657 参数的完整模型。</b>
</div>
<p>这对老师方案的直接含义：<b>PAM-Net 里的多尺度时域分支不能照搬现有时域分支的实现方式。</b>
现有时域分支是一个「全局残差估计器」，它对脉冲这种局部异常没有专门的归纳偏置。
老师方案里短时分支要「识别尖峰、脉冲和突发边缘」，这需要
<b>更小的感受野 + 逐点输出</b>（对应 <code>p_n</code> 脉冲感知分支），
而不是现在这种「整段残差」的形式。这是阶段1 要验证的核心假设。</p>
</div>

<h3>3.4 三种指标口径的差异</h3>
<table>
<thead><tr><th>口径</th><th>testA 数值</th><th>说明</th></tr></thead>
<tbody>
<tr><td>全局 SNR 提升</td><td>+20.48</td><td>被少数脉冲主导，偏大</td></tr>
<tr><td>帧 SNR 中位提升</td><td>+11.89</td><td>典型帧的绝对 SNR 改善</td></tr>
<tr><td>帧增益中位</td><td>+15.06</td><td>典型帧的噪声削减量（推荐主指标）</td></tr>
<tr><td>干净帧增益 / 干扰帧增益</td><td>+14.13 / +25.09</td><td>分解后看脉冲段处理能力</td></tr>
</tbody></table>
<p>同一组结果，四种口径从 11.89 到 20.48 dB，相差 8.6 dB。<b>报告里必须写清用的是哪一种。</b>
建议主口径用「帧增益」，并同时给出干净帧/干扰帧的分解。</p>
</div>

<h2>4. 对阶段1（实现 PAM-Net）的指导</h2>
<div class="card">
<ol>
<li><b>基线已确立。</b>纯频域 5,910 参数模型在脉冲噪声下：干净帧 14.13 dB、干扰帧 25.09 dB、
干扰帧最差 13.54 dB。PAM-Net 必须在这三个数上都有提升才算有效。</li>
<li><b>「干扰帧最差」是最该优化的指标。</b>它反映漏掉的脉冲——一个漏掉的脉冲会毁掉
下游处理，而中位数看不出来。</li>
<li><b>时域分支要重做，不是简单加回来。</b>老师方案要的是「短时分支识别尖峰/脉冲边缘」，
现有实现是整段残差估计，形式不对。建议直接按方案实现
<code>p_n = σ(f_imp(F)_n)</code> 的逐点脉冲置信度，而不是复用旧的时域分支。</li>
<li><b>损失必须换。</b>现在用的 L1 + 谱 L1 在 α 稳定噪声下训练不稳定
（最后 3 轮验证损失从 0.41 涨到 0.63，验证 SNR 从 18.35 掉到 14.71）；
高斯对照则平滑收敛。老师要求的 Huber 损失正是为解决这个问题。</li>
<li><b>噪声参数可以扫。</b>α 稳定分布的 α 值、突发干扰的到达率/幅度都做成了参数，
阶段1 可以做敏感性分析。</li>
</ol>
</div>

<h2>5. 局限</h2>
<ul>
<li><b>仍是仿真。</b>α 稳定分布是背景噪声的模型，突发干扰是合成的，
不等同于实测脉冲噪声。老师提到正在整理实测信号，届时需重新验证。</li>
<li><b>α 值只取了一个（1.5）。</b>真实海洋噪声的 α 通常在 1.5~1.9，未做扫描。</li>
<li><b>突发干扰的幅度/时长范围是设定的</b>（4~25×rms、2~30 ms），
没有实测统计支撑，可能偏离真实情况。</li>
<li><b>脉冲经多径信道后的形态未单独分析。</b>本实验把突发干扰直接叠加在
信道输出上；如果干扰发生在信道<b>之前</b>，经多径会展宽成一串回声突发，
难度完全不同。这个场景未测。</li>
<li><b>降采样 4 倍的影响未重新评估。</b>高斯噪声下这一步带来 +6.19 dB 输入 SNR；
脉冲噪声不会被抽取平均掉，实际增益会小得多（本次输入 SNR 只有 −4.61 dB 即为佐证），
但未做定量分解。</li>
<li><b>每个配置只跑一个随机种子。</b></li>
</ul>

<h2>6. 复现</h2>
<pre># 1. 生成数据集（α稳定 + 突发干扰，含高斯对照）约 7 分钟，5.7 GB
matlab -batch "make_dataset_pam('alphastable','E:/project/data/dataset_pam/alphastable',1.5); \
               make_dataset_pam('gauss','E:/project/data/dataset_pam/gauss',1.5)"

# 2. 验证噪声模型与数据集
matlab -batch "test_stblrnd; test_gen_burst"
python verify_pam_dataset.py E:/project/data/dataset_pam/alphastable E:/project/data/dataset_pam/gauss

# 3. 阶段0 三组实验（串行，约 45 分钟）
D=E:/project/data/dataset_pam
python code/python/train_pam.py --tag pam_freq   --data $D/alphastable --epochs 15 --time 0 --fch 2.0
python code/python/train_pam.py --tag pam_full   --data $D/alphastable --epochs 15 --time 1 --freq 1 --fch 1.0
python code/python/train_pam.py --tag pam_freq_g --data $D/gauss      --epochs 15 --time 0 --fch 2.0

# 4. 汇总 + 出图 + 独立复算
python code/analysis/analyze_pam.py
python code/analysis/verify_pam_result.py</pre>

</div></body></html>
"""

REP = {
    "__DATE__": "2026-10-14",
    "__FIG__": img("figI_pam_stage0.png",
                   "图 I 左：testA 上干净帧与干扰帧的降噪增益；"
                   "右：同一结果在三种指标口径下的差异（全局 SNR 明显偏大）"),
    "__ROWS__": rows,
    "__WT__": "%.4f" % wt,
    "__A_OUT__": "%+.2f" % a["snr_out"],
    "__A_CLEAN__": "%.2f" % a["frmgain_clean"],
    "__A_IMP__": "%.2f" % a["frmgain_imp"],
    "__A_IMPMIN__": "%.2f" % a["frmgain_imp_min"],
    "__G_CLEAN__": "%.2f" % b["frmgain_clean"],
    "__G_IMP__": "%.2f" % b["frmgain_imp"],
    "__G_IMPMIN__": "%.2f" % b["frmgain_imp_min"],
    "__G_OUT__": "%+.2f" % b["snr_out"],
    "__DA__": "%+.2f dB" % (a["frmgain_clean"] - b["frmgain_clean"]),
    "__DB__": "%+.2f dB" % (a["frmgain_imp"] - b["frmgain_imp"]),
    "__D_MIN__": "%+.2f dB" % (a["frmgain_imp_min"] - b["frmgain_imp_min"]),
    "__D_OUT__": "%+.2f dB" % (a["snr_out"] - b["snr_out"]),
}
for k, v in REP.items():
    HTML = HTML.replace(k, v)
left = sorted(set(re.findall(r"__[A-Z_]+__", HTML)))
if left:
    print("警告：未替换占位符", left)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)
print("报告 ->", OUT)
print("  w_time: %.4f（高斯下 0.006）" % wt)
print("  背景影响: 干净帧 %+.2f  干扰帧 %+.2f  输出SNR %+.2f"
      % (a["frmgain_clean"] - b["frmgain_clean"], a["frmgain_imp"] - b["frmgain_imp"],
         a["snr_out"] - b["snr_out"]))
print("  时域分支: 干扰帧 %+.2f dB" % (c["frmgain_imp"] - a["frmgain_imp"]))
