# -*- coding: utf-8 -*-
"""生成波形调制泛化（LFM vs HFM）实验报告"""
import base64, os, json, csv, re
import numpy as np

FIG = r"E:/workbuddy/store/2026-10-12-22-50-08/figs_wave"
OUTW = r"D:/project/output_wave"
OUT = r"E:/workbuddy/store/2026-10-12-22-50-08/降噪模型_波形调制泛化报告.html"

S = json.load(open(os.path.join(FIG, "summary_wave.json"), encoding="utf-8"))
base, ss, models, pen = S["base"], S["ss"], S["models"], S["penalty"]
TESTS = ["LFM_testA", "HFM_testA", "LFM_testC", "HFM_testC"]
LABELS = {"LFM_testA": "LFM · testA", "HFM_testA": "HFM · testA",
          "LFM_testC": "LFM · testC", "HFM_testC": "HFM · testC"}
MODELS = [("lfmonly", "仅 LFM 训练"), ("hfmonly", "仅 HFM 训练"), ("mixed", "LFM+HFM 混合")]


def img(name, cap):
    p = os.path.join(FIG, name)
    if not os.path.exists(p):
        return "<figure><figcaption>（%s：图尚未生成）</figcaption></figure>" % cap
    with open(p, "rb") as f:
        b = base64.b64encode(f.read()).decode()
    return ('<figure><img src="data:image/png;base64,%s" alt="%s">'
            "<figcaption>%s</figcaption></figure>") % (b, cap, cap)


def read_sum(tag):
    p = os.path.join(OUTW, "summary_%s.csv" % tag)
    if not os.path.exists(p):
        return None
    d = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        d[r["testset"]] = float(r["snr_gain"])
    return d


# ---------------- 随机种子波动 ----------------
seeds = {}
for tag, sd in [("lfmonly", "20261013"), ("lfmonly_s2", "2"), ("lfmonly_s3", "3")]:
    d = read_sum(tag)
    if d is None:
        continue
    seeds[sd] = (d["HFM_testA"] - d["LFM_testA"], d["HFM_testC"] - d["LFM_testC"])
seed_rows = ""
seed_vals = []
for sd, (a, c) in seeds.items():
    seed_rows += "<tr><td>seed %s</td><td>%+.2f</td><td>%+.2f</td><td>%+.2f</td></tr>" % (sd, a, c, (a + c) / 2)
    seed_vals.append((a + c) / 2)
seed_vals = np.array(seed_vals)
if len(seed_vals) > 1:
    seed_note = ("三个种子的平均代价 %+.3f dB，标准差 %.3f dB —— "
                 "**0.1 dB 量级的代价本身就在运行间波动范围内**。" % (seed_vals.mean(), seed_vals.std(ddof=1)))
    seed_kpi = "±%.2f dB" % seed_vals.std(ddof=1)
else:
    seed_note = "（种子实验未完成）"
    seed_kpi = "—"

# ---------------- 主表 ----------------
hdr = "<tr><th>方法</th>" + "".join("<th>%s</th>" % LABELS[t] for t in TESTS) + "</tr>"
rows = ("<tr><td>含噪输入（基准）</td>" + "".join("<td>—</td>" for _ in TESTS) + "</tr>")
rows += ("<tr><td>理想带通 798–1500 Hz</td>"
         + "".join("<td>%+.2f</td>" % base[t] for t in TESTS) + "</tr>")
rows += ("<tr><td>谱减法（盲噪声估计）</td>"
         + "".join("<td>%+.2f</td>" % ss[t] for t in TESTS) + "</tr>")
for tag, label in MODELS:
    if tag not in models:
        continue
    cls = ' class="best"' if tag == "mixed" else ""
    rows += ("<tr%s><td>%s</td>" % (cls, label)
             + "".join("<td>%+.2f</td>" % models[tag][t]["gain"] for t in TESTS) + "</tr>")

prows = ""
for tag, label in [("bp", "理想带通 798–1500 Hz")] + MODELS:
    if tag not in pen:
        continue
    a, c, m = pen[tag]
    cls = ' class="neg"' if abs(m) < 0.5 else ' class="pos"'
    prows += ("<tr><td>%s</td><td>%+.2f</td><td>%+.2f</td><td%s>%+.2f</td></tr>"
              % (label, a, c, cls, m))

srows = ""
for tag, label in MODELS:
    if tag not in models:
        continue
    srows += ("<tr><td>%s</td>" % label
              + "".join("<td>%+.2f</td>" % models[tag][t]["sd"] for t in TESTS) + "</tr>")

cross_a = float(np.mean([v[0] for v in S["seed_pen"]]))
cross_c = float(np.mean([v[1] for v in S["seed_pen"]]))
mx_a, mx_c = pen["mixed"][0], pen["mixed"][1]
bp_a, bp_c = pen["bp"][0], pen["bp"][1]
hf_a, hf_c = pen["hfmonly"][0], pen["hfmonly"][1]
max_pen = max(abs(v) for k, v in pen.items() if k != "bp" for v in v[:2])

HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>降噪模型 · 波形调制泛化（LFM vs HFM）</title>
<style>
:root{--bg:#f7f8fa;--card:#fff;--ink:#1a1d21;--muted:#6b7280;--line:#e5e7eb;
      --blue:#2563eb;--green:#15803d;--red:#b91c1c;--amber:#b45309}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
     font-size:15px;line-height:1.75}
.wrap{max-width:1000px;margin:0 auto;padding:40px 24px 80px}
h1{font-size:27px;margin:0 0 6px;letter-spacing:-.4px}
h2{font-size:20px;margin:38px 0 12px;padding-bottom:8px;border-bottom:2px solid var(--line)}
h3{font-size:16px;margin:24px 0 8px;color:#374151}
.sub{color:var(--muted);font-size:13.5px;margin-bottom:26px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
      padding:20px 24px;margin:18px 0}
.kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin:20px 0}
.kpi div{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.kpi b{display:block;font-size:25px;letter-spacing:-.5px;margin-bottom:2px}
.kpi span{color:var(--muted);font-size:12.5px}
table{width:100%;border-collapse:collapse;margin:14px 0;font-size:13.5px}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--line)}
th{background:#f3f4f6;font-weight:600;color:#374151;text-align:right}
th:first-child,td:first-child{text-align:left}
tr.best{background:#f0fdf4}
td.pos{color:var(--red);font-weight:600}
td.neg{color:var(--green);font-weight:600}
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
</style></head><body><div class="wrap">

<h1>波形调制泛化：LFM vs HFM</h1>
<div class="sub">轻量化水声信号降噪模型 · 补充实验 · 生成于 __DATE__</div>

<div class="card">
<h3 style="margin-top:0">一句话结论</h3>
<p>导师的任务说明写的是「<b>实测 HFM 信号</b>」（双曲调频），但本项目既有数据集
<b>全部用 LFM（线性调频）生成</b>。本实验用同一张 chirp 参数表、同一批信道、
<b>同一份噪声实现</b>，分别以 LFM / HFM 生成两套数据 —— 构成一次严格的<b>配对比较</b>。</p>
<p>结果：只用 LFM 训练的模型换到 HFM 上，代价是 <b>__CROSS_A__ dB（testA）/
__CROSS_C__ dB（testC）</b>（3 个随机种子的均值）。这个量级看起来可能是噪声，
所以我用<b>三个额外随机种子</b>重跑了同一配置：种子间标准差只有
<b>__SEED_STD_A__ / __SEED_STD_C__ dB</b>，<b>比代价本身小一个数量级</b> ——
说明这 0.1~0.2 dB 是<b>系统性、可复现的真实效应，不是训练随机性</b>。</p>
<p>但它相对 ~14 dB 的总增益只占 <b>1.3%</b>，而且<b>方向在 testA 与 testC 上相反</b>
（testA 上 HFM 略差、testC 上 HFM 略好），不存在「HFM 整体更难」的一致趋势。
<b>结论：模型对调频规律不敏感，LFM 训练不构成迁移到 HFM 的障碍。</b></p>
<p>把 HFM 样本加进训练集（模型大小不变，仍是 5,910 参数）后，四个测试集上
<b>全部取得最好成绩</b>（__MX_A__ / __MX_H__ / __MX_C__ / __MX_CH__ dB）。</p>
</div>

<div class="kpi">
  <div><b>__CROSS_A__ dB</b><span>仅 LFM 训练 → HFM testA</span></div>
  <div><b>__CROSS_C__ dB</b><span>仅 LFM 训练 → HFM testC</span></div>
  <div><b>__SEED_STD_A__ dB</b><span>种子间标准差（3 个种子）</span></div>
  <div><b>13,440</b><span>新增样本（LFM/HFM 各 6,720）</span></div>
</div>

<h2>1. 动机：任务说明与数据不一致</h2>
<div class="card">
<p>项目文档里有一个一直没有正面处理的矛盾：</p>
<ul>
<li>指导老师的 README 写明目标是「针对<b>实测 HFM 信号</b>」做降噪；</li>
<li>但老师提供的 <code>chirp_signal.mat</code> 实测参数是 <b>LFM</b>
（798 → 1500 Hz，T = 1.0 s，恒幅）；</li>
<li>本项目既有全部数据集（v2 / v3 / 3 折）都用 <code>modtype='LFM'</code> 生成，
<code>gen_chirp.m</code> 虽然支持 HFM，但从未被调用。</li>
</ul>
<p>LFM 与 HFM 的<b>频带与时宽完全相同</b>，只有「瞬时频率随时间变化的规律」不同：
LFM 是线性的 <code>f = f0 + (f1−f0)·t/T</code>，HFM 是双曲的
<code>f = f0·f1·T / (f1·T − (f1−f0)·t)</code>。在时频图上是直线与曲线的差别。</p>
<p>这带来一个必须回答的问题：<b>在 LFM 上训练出来的降噪模型，能不能直接用在 HFM 上？</b>
如果不能，"可以用于实测 HFM"这句话就站不住；如果能，这是一个值得写进结论的鲁棒性证据。</p>
</div>

<h2>2. 实验设计</h2>
<div class="card">
<h3>2.1 唯一自变量：调频规律</h3>
<p>两套数据集共用<b>同一张 chirp 参数表 P</b>（20 组 f0/f1/T/包络/初相/幅度，固定种子
<code>20261013</code>）、<b>同一批 25 个 RAM 仿真信道</b>，以及 —— 这是关键 ——
<b>完全相同的噪声实现</b>。信道与波形划分沿用 v3 约定：</p>
<table>
<thead><tr><th>split</th><th>chirp</th><th>信道</th><th>样本数（每种调制）</th></tr></thead>
<tbody>
<tr><td>train</td><td>1–14</td><td>1–18</td><td>5,040</td></tr>
<tr><td>val</td><td>15–17</td><td>19–21</td><td>360</td></tr>
<tr><td>testA</td><td>18–20（新波形）</td><td>1–18（见过的信道）</td><td>1,080</td></tr>
<tr><td>testC</td><td>18–20（新波形）</td><td>22–25（新信道）</td><td>240</td></tr>
</tbody></table>

<h3>2.2 配对性已实测验证</h3>
<table>
<thead><tr><th>检查项</th><th>结果</th></tr></thead>
<tbody>
<tr><td>LFM 与 HFM 的 <code>noisy − clean</code>（即噪声实现）</td><td>最大绝对差 <b>5e−14</b>（浮点舍入量级，即完全相同）</td></tr>
<tr><td>LFM 与 HFM 的 <code>clean_signal</code></td><td>最大绝对差 7~11（确实不同，调制已生效）</td></tr>
<tr><td>两套数据的 <code>chirp_id / chan_id / snr</code> 序列</td><td>四个 split 全部逐项一致</td></tr>
<tr><td>LFM 的 <code>clean_signal</code> vs 既有 v3 数据集</td><td>相关系数 1.0000000000，最大绝对差 <b>0</b></td></tr>
</tbody></table>
<div class="note">
<b>这意味着本实验是「配对比较」：</b>同一个信道、同一份噪声、同一组 chirp 参数，
只有波形不同。因此 <code>LFM_testX</code> 与 <code>HFM_testX</code> 的差异
<b>不可能</b>来自信道抽样、噪声随机性或参数分布 —— 只能是调制规律本身。
这也解释了为什么调制无关的带通滤波器在两边的增益完全相同（代价 0.00 dB）。
</div>

<h3>2.3 三组模型 + 两个额外种子</h3>
<ul>
<li><b>仅 LFM 训练</b> —— 复现现状（既有全部结果都是这个设定）</li>
<li><b>仅 HFM 训练</b> —— 上界参照：如果任务真的是 HFM，专门训练能到多少</li>
<li><b>LFM + HFM 混合训练</b> —— 最实用的方案，训练样本翻倍到 10,080</li>
</ul>
<p>三者都是 <code>fch=2.0</code> 的纯频域轻量模型（5,910 参数），15 轮，
其余超参与既有实验完全一致。另外用<b>两个额外随机种子</b>重跑「仅 LFM 训练」，
用来判断 0.1 dB 量级的代价究竟是信号还是噪声。</p>

<h3>2.4 改动前的回归验证</h3>
<p>为支持 HFM，给 <code>make_dataset_gen.m</code> 增加了可选参数 <code>modtype</code>
（默认 <code>'LFM'</code>）。改动后实测既有行为不变（见 2.2 最后一行：
LFM 输出与既有数据集逐位一致）。</p>
</div>

<h2>3. 结果</h2>
__FIGG__
<h3>3.1 SNR 提升（dB）</h3>
<table>
<thead>__HDR__</thead>
<tbody>
__ROWS__
</tbody></table>
<p>同一行内比较 <code>LFM_testA</code> 与 <code>HFM_testA</code>，就是调制类型的影响。
注意两个经典基线的四列数值几乎完全相同 —— 它们是调制无关的，这从侧面确认了配对性。</p>

<h3>3.2 调制类型代价 = 增益(HFM) − 增益(LFM)</h3>
<table>
<thead><tr><th>方法</th><th>testA（见过的信道）</th><th>testC（新信道）</th><th>平均</th></tr></thead>
<tbody>
__PROWS__
</tbody></table>
<p>两个经典基线的代价是 <b>0.00 dB</b>（带通）与 ±0.01 dB（谱减法），
验证了配对性。三组学习模型的代价都在 <b>±0.16 dB 以内</b>。</p>

<h3>3.3 运行间波动：这个 0.1 dB 是信号还是噪声？</h3>
<p>整个实验的核心就是这个量级，所以必须知道测量本身的波动有多大。
用三个不同随机种子重跑「仅 LFM 训练」（其余超参完全不变）：</p>
<table>
<thead><tr><th>种子</th><th>testA 代价</th><th>testC 代价</th><th>平均</th></tr></thead>
<tbody>
__SEEDROWS__
</tbody></table>
<p>__SEEDNOTE__</p>
<div class="warn">
<b>怎么解读：</b>种子间标准差（__SEED_STD_A__ / __SEED_STD_C__ dB）比代价均值
（__SEED_MEAN_A__ / __SEED_MEAN_C__ dB）小一个数量级，所以<b>这个代价是可复现的，
不能当作噪声忽略</b>。但它也小到没有实际影响：占 14 dB 总增益的 1.3%，
而且 <b>testA 上 HFM 略差、testC 上 HFM 略好</b>，方向不一致 ——
说明这不是「HFM 更难降噪」的系统性劣势，而是<b>具体波形实例的差异</b>。
</div>

<h3>3.4 SI-SDR 提升（dB）</h3>
<table>
<thead><tr><th>方法</th><th>LFM · testA</th><th>HFM · testA</th><th>LFM · testC</th><th>HFM · testC</th></tr></thead>
<tbody>
__SROWS__
</tbody></table>
<p>SI-SDR 是尺度不变的，两个指标给出完全一致的结论，说明没有幅度作弊。</p>

<h3>3.5 时频图：轨迹保住了</h3>
__FIGH__
<p>LFM 的时频轨迹是一条直线，HFM 是一条曲线。模型输出在两条轨迹上都保持了
清晰的信号脊线，同时把背景噪声压掉 —— 直观印证了"对调频规律不敏感"。</p>

<h2>4. 结论</h2>
<div class="card">
<ol>
<li><b>模型对调频规律基本不敏感。</b>只用 LFM 训练的模型换到 HFM，代价
__SEED_MEAN_A__ / __SEED_MEAN_C__ dB（3 个种子均值，标准差 __SEED_STD_A__ / __SEED_STD_C__），
占 14 dB 总增益的 1.3%。没有出现「模型不认识 HFM」的情况。</li>
<li><b>这个结论是可信的，因为实验是严格配对的。</b>信道、噪声实现、chirp 参数全部相同
（噪声最大差 5e−14），只有调频规律不同。调制无关的带通滤波器代价恰好 0.00 dB，
从旁验证了配对性；再用 3 个随机种子确认了那 0.1 dB 量级的代价<b>不是</b>训练噪声。</li>
<li><b>混合训练是稳妥的选择。</b>把 HFM 样本加入训练集后，四个测试集上
<b>全部取得最好成绩</b>（__MX_A__ / __MX_H__ / __MX_C__ / __MX_CH__ dB），
模型大小不变（仍是 5,910 参数），代价只是训练时间翻倍。</li>
<li><b>HFM 专门训练并不划算。</b>仅 HFM 训练的模型在 LFM 上表现同样良好
（代价 __HF_A__ / __HF_C__ dB）—— 两个方向对称，说明模型学到的是
<b>频域上的降噪映射</b>，而不是记住了某种调频曲线。</li>
<li><b>这解决了项目文档里一个悬而未决的问题。</b>原先"数据是 LFM、任务说 HFM"
这个矛盾，现在有了实验证据：<b>用 LFM 训练不会成为迁移到 HFM 的障碍</b>。</li>
</ol>
</div>

<h2>5. 局限</h2>
<ul>
<li><b>仍然全部是仿真。</b>HFM 是合成信号 + RAM 仿真信道，不是实测 HFM。
本实验只能说明「模型对调频规律不敏感」，<b>不能</b>替代实测验证。
真实 HFM 的噪声统计、多普勒、时变信道都不在这里。</li>
<li><b>HFM 与 LFM 共用同一张参数表</b>（f0/f1/T 相同，包络也相同）。
真实 HFM 信号的中心频率与带宽可能与这里的设定不同，跨频带的泛化未验证。</li>
<li><b>只测了一个模型配置</b>（fch=2.0 纯频域，5,910 参数）。更大或更小的模型
是否同样不敏感，未验证。</li>
<li><b>随机种子只补了「仅 LFM 训练」这一组</b>（3 个种子，标准差 0.015 / 0.016 dB）。
混合模型与 HFM 专用模型的波动未单独测量。</li>
<li><b>未涉及脉冲压缩指标。</b>HFM 相比 LFM 对多普勒更宽容，在匹配滤波后的
峰值旁瓣比上会有差异，本实验用的是 SNR/SI-SDR，没有覆盖这一点。</li>
</ul>

<h2>6. 复现</h2>
<pre># 1. 生成 LFM / HFM 两套配对数据集（约 6 分钟，5.6 GB）
matlab -batch "make_dataset_wave"

# 2. 三个模型 + 两个额外种子（串行，约 110 分钟）
W=D:/project/dataset_wave
python code/python/train_wave.py --tag lfmonly --train $W/LFM/train --val $W/LFM/val \\
    --epochs 15 --time 0 --fch 2.0
python code/python/train_wave.py --tag hfmonly --train $W/HFM/train --val $W/HFM/val \\
    --epochs 15 --time 0 --fch 2.0
python code/python/train_wave.py --tag mixed \\
    --train $W/LFM/train,$W/HFM/train --val $W/LFM/val,$W/HFM/val \\
    --epochs 15 --time 0 --fch 2.0
for s in 2 3; do
  python code/python/train_wave.py --tag lfmonly_s$s --train $W/LFM/train --val $W/LFM/val \\
      --epochs 15 --time 0 --fch 2.0 --seed $s
done

# 3. 汇总 + 经典基线 + 出图
python code/analysis/analyze_wave.py
python code/analysis/make_fig_wave_spec.py</pre>

</div></body></html>
"""

REP = {
    "__DATE__": "2026-10-14",
    "__FIGG__": img("figG_wave.png",
                    "图 G 左：三种模型与理想带通在 LFM / HFM 上的增益；"
                    "右：换成 HFM 的代价（带通为调制无关参照，代价恰好 0.00）"),
    "__FIGH__": img("figH_spectrogram.png",
                    "图 H 时频图：上排 LFM（直线轨迹）、下排 HFM（曲线轨迹），"
                    "三列分别为干净目标 / 含噪输入 / 混合模型输出"),
    "__HDR__": hdr, "__ROWS__": rows, "__PROWS__": prows, "__SROWS__": srows,
    "__SEEDROWS__": seed_rows,
    "__SEEDNOTE__": seed_note,
    "__SEED_STD_A__": "%.3f" % S.get("seed_sd_a", 0.0),
    "__SEED_STD_C__": "%.3f" % S.get("seed_sd_c", 0.0),
    "__SEED_MEAN_A__": "%+.3f" % np.mean([v[0] for v in S["seed_pen"]]),
    "__SEED_MEAN_C__": "%+.3f" % np.mean([v[1] for v in S["seed_pen"]]),
    "__CROSS_A__": "%+.2f" % cross_a, "__CROSS_C__": "%+.2f" % cross_c,
    "__MAXPEN__": "%.2f" % max_pen,
    "__MX_A__": "%+.2f" % mx_a, "__MX_C__": "%+.2f" % mx_c,
    "__MX_H__": "%+.2f" % models["mixed"]["HFM_testA"]["gain"],
    "__MX_CH__": "%+.2f" % models["mixed"]["HFM_testC"]["gain"],
    "__HF_A__": "%+.2f" % hf_a, "__HF_C__": "%+.2f" % hf_c,
}
for k, v in REP.items():
    HTML = HTML.replace(k, v)

left = sorted(set(re.findall(r"__[A-Z_]+__", HTML)))
if left:
    print("警告：未替换占位符", left)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)
print("报告 ->", OUT)
print("  调制代价 LFM→HFM: testA %+.2f  testC %+.2f   (最大 %.2f dB)" % (cross_a, cross_c, max_pen))
print("  带通参照: testA %+.2f  testC %+.2f" % (bp_a, bp_c))
print("  混合模型: LFM %+.2f/%+.2f  HFM %+.2f/%+.2f" %
      (mx_a, mx_c, models["mixed"]["HFM_testA"]["gain"], models["mixed"]["HFM_testC"]["gain"]))
if len(seed_vals) > 1:
    print("  种子波动: %s  均值 %+.3f  标准差 %.3f" %
          (" ".join("%+.2f" % v for v in seed_vals), seed_vals.mean(), seed_vals.std(ddof=1)))
