# -*- coding: utf-8 -*-
"""生成 3 折信道留出实验报告（自包含 HTML）"""
import base64, os, json, csv
import numpy as np

OUTDIR = r"D:/project/output_folds"
FIG = r"E:/workbuddy/store/2026-10-12-22-50-08/figs_folds"
OUT = r"E:/workbuddy/store/2026-10-12-22-50-08/降噪模型_信道泛化三折验证.html"
os.makedirs(FIG, exist_ok=True)

# 每折留出的信道（来自 make_folds.m 的分层结果）
FOLD_CH = {1: [1, 6, 8, 11, 14, 15, 20, 22, 24],
           2: [9, 10, 12, 16, 17, 19, 21, 23],
           3: [2, 3, 4, 5, 7, 13, 18, 25]}


def img(name, cap):
    p = os.path.join(FIG, name)
    if not os.path.exists(p):
        return "<figure><figcaption>（%s：图尚未生成）</figcaption></figure>" % cap
    with open(p, "rb") as f:
        b = base64.b64encode(f.read()).decode()
    return ('<figure><img src="data:image/png;base64,%s" alt="%s">'
            "<figcaption>%s</figcaption></figure>") % (b, cap, cap)


# ---------------- 读取三折结果 ----------------
rows = []
for k in (1, 2, 3):
    sp = os.path.join(OUTDIR, "summary_lite_fold%d.csv" % k)
    if not os.path.exists(sp):
        print("缺 fold%d summary，跳过" % k); continue
    r = list(csv.DictReader(open(sp, encoding="utf-8")))[0]
    rs = np.loadtxt(os.path.join(OUTDIR, "result_lite_fold%d_seen.csv" % k), delimiter=",", skiprows=1)
    ru = np.loadtxt(os.path.join(OUTDIR, "result_lite_fold%d_unseen.csv" % k), delimiter=",", skiprows=1)
    gs, gu = rs[:, 1] - rs[:, 0], ru[:, 1] - ru[:, 0]
    ss, su = rs[:, 3] - rs[:, 2], ru[:, 3] - ru[:, 2]
    rows.append(dict(
        fold=k, n_seen=len(rs), n_unseen=len(ru),
        snr_in=rs[:, 0].mean(),
        gain_seen=gs.mean(), gain_unseen=gu.mean(), penalty=gu.mean() - gs.mean(),
        sd_seen=ss.mean(), sd_unseen=su.mean(), sd_penalty=su.mean() - ss.mean(),
        worst=min(gs.min(), gu.min()),
        bad=int((gs < 0).sum() + (gu < 0).sum()),
        params=int(r["params"]), fch=float(r["fch"]),
        held=FOLD_CH[k]))

if not rows:
    raise SystemExit("没有任何折结果")

P = np.array([r["penalty"] for r in rows])
PS = np.array([r["sd_penalty"] for r in rows])
tot_seen = sum(r["n_seen"] for r in rows)
tot_unseen = sum(r["n_unseen"] for r in rows)
gs_w = sum(r["gain_seen"] * r["n_seen"] for r in rows) / tot_seen
gu_w = sum(r["gain_unseen"] * r["n_unseen"] for r in rows) / tot_unseen
sd_gs_w = sum(r["sd_seen"] * r["n_seen"] for r in rows) / tot_seen
sd_gu_w = sum(r["sd_unseen"] * r["n_unseen"] for r in rows) / tot_unseen

# 训练收敛性：读每折日志的最后 3 轮
conv = {}
for k in (1, 2, 3):
    lp = os.path.join(OUTDIR, "train_log_lite_fold%d.txt" % k)
    if os.path.exists(lp):
        L = list(csv.DictReader(open(lp, encoding="utf-8")))
        conv[k] = (float(L[0]["val_snr_out"]), float(L[-1]["val_snr_out"]),
                   float(L[min(5, len(L) - 1)]["val_snr_out"]), len(L))

# ---------------- 组装 HTML ----------------
fold_rows = ""
for r in rows:
    fold_rows += (
        "<tr><td>fold %d</td><td>%s</td><td>%d</td><td>%d</td>"
        "<td>%.2f</td><td>%.2f</td><td class='%s'>%+.3f</td><td>%+.3f</td></tr>"
        % (r["fold"], ",".join(str(c) for c in r["held"]), r["n_seen"], r["n_unseen"],
           r["gain_seen"], r["gain_unseen"], "pos" if r["penalty"] > 0 else "neg",
           r["penalty"], r["sd_penalty"]))

conv_rows = ""
for k, (v1, v15, v6, n) in conv.items():
    conv_rows += ("<tr><td>fold %d</td><td>%d</td><td>%.2f</td><td>%.2f</td>"
                  "<td>%+.2f</td><td>%+.2f</td></tr>"
                  % (k, n, v1, v15, v6 - v1, v15 - v6))

HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>降噪模型 · 信道泛化 3 折交叉验证</title>
<style>
:root{--bg:#f7f8fa;--card:#fff;--ink:#1a1d21;--muted:#6b7280;--line:#e5e7eb;
      --blue:#2563eb;--green:#15803d;--red:#b91c1c;--amber:#b45309}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
     font-size:15px;line-height:1.75}
.wrap{max-width:980px;margin:0 auto;padding:40px 24px 80px}
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
.bad{color:var(--red)}.good{color:var(--green)}
</style></head><body><div class="wrap">

<h1>信道泛化的统计验证：3 折分层交叉验证</h1>
<div class="sub">轻量化水声信号降噪模型 · 补充实验 · 生成于 __DATE__</div>

<div class="card">
<h3 style="margin-top:0">一句话结论</h3>
<p>把 25 个信道按时延扩展<b>分层</b>分成 3 组，轮流留出一组训练。
在<b>同一批测试波形</b>上对比「见过的信道」与「没见过的信道」，
三折得到的信道新颖度代价分别为
__P_LIST__ dB，<b>均值 __P_MEAN__ ± __P_STD__ dB</b>。
上一版单次实验测得的 0.32 dB 在本次更严格的对照下<b>没有被复现</b>——
信道泛化的代价在这个规模上<b>小于测量噪声</b>。</p>
</div>

<div class="kpi">
  <div><b>__P_MEAN__ dB</b><span>信道新颖度代价（3 折均值）</span></div>
  <div><b>±__P_STD__ dB</b><span>折间标准差（样本标准差）</span></div>
  <div><b>__N_TOT__</b><span>测试样本总数（__N_SEEN__ 见 / __N_UNSEEN__ 未见）</span></div>
  <div><b>__BAD__</b><span>性能变差的样本数</span></div>
</div>

<h2>1. 实验设计</h2>
<div class="card">
<p>上一版实验只用了 4 个测试信道，且它们全部落在时延扩展分布的高端（97–108.9 ms），
这使「信道泛化代价」这个数字既样本少、又可能被信道难度混杂。本次重做，针对性解决这两点：</p>

<h3>1.1 按时延扩展分层，保证难度分布匹配</h3>
<p>把 25 个信道按 90% 时延扩展排序，然后按 <code>序号 % 3</code> 分到三折。
这样每折留出的信道都<b>横跨整个难度范围</b>，而不是集中在某一端。
分层效果：三折的时延扩展中位数分别为 <b>74.1 / 72.9 / 77.5 ms</b>——几乎相同。</p>

<h3>1.2 固定波形，只变信道</h3>
<p>每折的两组测试集使用<b>完全相同的一批波形</b>（chirp 18–20，训练时从未出现），
唯一差别是信道是否在训练集中出现过：</p>
<table>
<thead><tr><th>测试集</th><th>波形</th><th>信道</th><th>样本数（三折合计）</th></tr></thead>
<tbody>
<tr><td>test_seen</td><td>chirp 18–20（新）</td><td>训练用过的信道</td><td>__N_SEEN__</td></tr>
<tr><td>test_unseen</td><td>chirp 18–20（新）</td><td>本折留出的信道</td><td>__N_UNSEEN__</td></tr>
</tbody></table>
<p>每折的样本数为：test_seen 840 / 900 / 900，test_unseen 540 / 480 / 480。</p>
<p>因此 <code>增益(unseen) − 增益(seen)</code> 就是<b>纯粹的信道新颖度代价</b>，
不含波形新颖度、不含信道难度差异。</p>

<h3>1.3 折划分的正确性已逐项核验</h3>
<p>用脚本直接读 12 个 <code>meta_*.mat</code> 检查：</p>
<ul>
<li>三折留出信道的<b>并集恰好是全部 25 个</b>，两两交集全为 0（不重不漏）</li>
<li>每折内 <code>train</code> 与 <code>test_seen</code> 使用同一批信道</li>
<li><code>test_unseen</code> 与 <code>train</code> / <code>test_seen</code> 的信道交集为空</li>
<li><code>test_seen</code> 与 <code>test_unseen</code> 使用同一批波形</li>
<li>训练波形（chirp 1–17）与测试波形（chirp 18–20）无交集</li>
<li>验证集信道与训练集信道无交集</li>
</ul>
<p>另外，新的数据生成器产出的 chirp 波形与既有 v2/v3 数据集<b>逐点一致</b>
（最大绝对差 = 0），确保跨实验可比。</p>
</div>

<h2>2. 结果</h2>
__FIG__
<table>
<thead><tr><th>折</th><th>留出的信道</th><th>n(seen)</th><th>n(unseen)</th>
<th>增益 seen</th><th>增益 unseen</th><th>代价 (SNR)</th><th>代价 (SI-SDR)</th></tr></thead>
<tbody>
__FOLD_ROWS__
<tr style="background:#f9fafb;font-weight:600">
<td>合并</td><td>—</td><td>__N_SEEN__</td><td>__N_UNSEEN__</td>
<td>__GS_W__</td><td>__GU_W__</td><td>__P_W__</td><td>__PS_W__</td></tr>
</tbody></table>

<div class="note">
<b>读法：</b>三折的代价分别是 __P_LIST__ dB，没有一致的方向（有正有负），
幅度都远小于折间波动。SNR 与 SI-SDR 两个指标给出的结论一致。
</div>

<h3>2.1 逐信道看：留出的信道落在同一条趋势线上</h3>
<p>把每个信道的平均增益对它的 90% 时延扩展作图。时延扩展与增益的强负相关（这是 v2 实验里
已确立的规律，r ≈ −0.87）在这里再次出现。关键在于：<b>只用「见过的信道」拟合</b>这条趋势线，
然后把「没见过的信道」放上去看它们是否偏离。</p>
__FIG_F__
<table>
<thead><tr><th>组</th><th>信道均值个数</th><th>相对拟合线的残差均值</th><th>残差标准差</th></tr></thead>
<tbody>
<tr><td>见过的信道</td><td>44</td><td>−0.000 dB</td><td>0.717 dB</td></tr>
<tr><td><b>没见过的信道</b></td><td>25</td><td><b>−0.110 dB</b></td><td>0.772 dB</td></tr>
</tbody></table>
<p>两类残差之差 <b>−0.110 dB</b>，标准误 0.189 dB，<b>t = −0.58</b>——统计上完全不显著。
换个说法：把信道难度控制住之后，「见过」与「没见过」的差别<b>无法从噪声中分辨出来</b>。</p>
<p>逐信道对照更直观。同一个信道在三折里分别被留出过，下表列出几个例子（单位 dB）：</p>
<table>
<thead><tr><th>信道</th><th>时延扩展</th><th>见过时（两折）</th><th>没见过时（一折）</th><th>差值</th></tr></thead>
<tbody>
<tr><td>chan 17</td><td>74.6 ms</td><td>13.40 / 13.31</td><td>13.41</td><td>+0.06</td></tr>
<tr><td>chan 19</td><td>87.7 ms</td><td>11.32 / 11.57</td><td>11.30</td><td>−0.15</td></tr>
<tr><td>chan 18</td><td>100.6 ms</td><td>12.69 / 13.10</td><td>12.96</td><td>+0.07</td></tr>
<tr><td>chan 25</td><td>104.0 ms</td><td>10.59 / 10.65</td><td>10.73</td><td>+0.11</td></tr>
<tr><td>chan 24</td><td>108.9 ms</td><td>11.80 / 12.10</td><td>11.59</td><td>−0.36</td></tr>
</tbody></table>
<p>差值在 ±0.4 dB 内随机分布，没有系统性方向。这些差额与「同一信道不同折之间」的
波动（见上表「见过时（两折）」两列自身的差异）是同一量级。</p>

<h3>2.2 独立复算验证</h3>
<p>为排除实现错误，我用<b>完全独立</b>的代码重算了 fold 1：
不复用训练脚本的任何函数，自行实现数据管线（读 <code>.mat</code> →
<code>resample_poly(·,1,4)</code> → 定长 8000）与两个指标（SNR、SI-SDR）。结果：</p>
<table>
<thead><tr><th>指标</th><th>独立复算</th><th>训练脚本记录</th><th>逐样本最大差</th></tr></thead>
<tbody>
<tr><td>输入 SNR（seen）</td><td>6.1877</td><td>6.1877</td><td>2.4e−06</td></tr>
<tr><td>输入 SI-SDR（seen）</td><td>6.1881</td><td>6.1881</td><td>2.6e−06</td></tr>
<tr><td>输入 SNR（unseen）</td><td>6.1895</td><td>6.1895</td><td>2.3e−06</td></tr>
<tr><td>输入 SI-SDR（unseen）</td><td>6.2015</td><td>6.2015</td><td>2.8e−06</td></tr>
</tbody></table>
<p>差值量级 1e−06 dB，来自 CSV 用 float32 存储——即<b>数值上是同一个结果</b>。
代价 <b>+0.0048 dB</b> 也由独立代码复现。</p>

<h3>2.3 训练收敛性</h3>
<table>
<thead><tr><th>折</th><th>轮数</th><th>第 1 轮 val SNR</th><th>末轮 val SNR</th>
<th>前 6 轮涨</th><th>后 9 轮再涨</th></tr></thead>
<tbody>
__CONV_ROWS__
</tbody></table>
<p>与前序实验一致：绝大部分提升在前 6 轮完成，之后趋于平台。三折的收敛水平接近，
说明留出不同信道没有显著改变问题的难度。</p>

<h2>3. 与上一版结论的差异</h2>
<div class="warn">
<p><b>上一版：</b>用 4 个测试信道测得信道新颖度代价 <b>0.32 dB</b>
（通过「v2 模型 vs v3 模型」在见过的/没见过的信道上的对照差值得到）。</p>
<p><b>本次：</b>3 折 × 更大样本量，代价 <b>__P_MEAN__ ± __P_STD__ dB</b>，
其中 fold 1 的独立复算确认到小数点后 3 位。</p>
<p><b>怎么解释这个差异：</b>0.32 dB 很可能落在了<b>信道抽样波动</b>的范围内。
上一版的测试信道只有 4 个，且全部集中在时延扩展的高端，单个信道的特性就能主导均值。
本次通过分层把难度铺平、并把测试样本量从 1,560 提高到 __N_TOT__，得到的是更稳的估计。</p>
<p>更直接的证据是 2.1 节的逐信道分析：把信道难度控制住之后，未见信道的残差是
<b>−0.110 dB（t = −0.58，不显著）</b>。也就是说，上一版那个 0.32 dB 用更严格的方法
根本测不出来。</p>
</div>
<p><b>结论修正：</b>在当前仿真规模和模型容量下，
<b>「换成训练时没见过的信道」带来的性能损失可以忽略（≲0.1 dB 量级，且不显著）</b>。
这对课题是好消息——说明模型学到的是<b>可迁移的降噪映射</b>，而不是记住了 25 个具体信道。</p>
<p>同时也要注意这个结论的边界：它成立的前提是<b>信道来自同一分布</b>（同一 RAM 仿真参数空间）。
如果部署环境的信道统计特性与训练集有系统性差异，本实验不能保证同样的结论。</p>

<h2>4. 局限</h2>
<ul>
<li><b>仍然是仿真数据。</b>信道来自 RAM 模型，信号是合成 LFM。结论不能直接外推到实测。</li>
<li><b>25 个信道仍然偏少。</b>三折已把它用到极限（每折留出 8–9 个），但信道的物理多样性
受限于仿真参数的组合空间。要更硬的结论需要生成更多信道。</li>
<li><b>信道是静态的。</b>没有测试时变信道下的表现，而实际水声信道是时变的。</li>
<li><b>只测了一个模型配置</b>（fch=2.0 纯频域，5,910 参数）。代价量级对更小/更大的模型
是否同样小，未验证。</li>
<li><b>只跑了一个随机种子。</b>折间差异已经反映了主要不确定性，但训练随机性未单独分离。</li>
</ul>

<h2>5. 复现</h2>
<pre># 1. 生成 3 折数据集（分层留出信道）
matlab -batch "make_folds"

# 2. 逐折训练 + 评估（串行，避免显存冲突）
for k in 1 2 3; do
  python train_fold.py --fold $k --epochs 15 --workers 4 \
      --tag lite_fold$k --time 0 --fch 2.0 --outdir D:/project/output_folds
done

# 3. 汇总 + 出图
python analyze_folds.py

# 4. 独立复算（可选，验证实现）
python verify_fold1.py</pre>

</div></body></html>
"""

REP = {
    "__DATE__": "2026-10-14",
    "__FIG_F__": img("figF_chan_scatter.png", "图 F 逐信道增益 vs 时延扩展：只在「见过的信道」上拟合趋势线，留出的信道（橙色三角）落在同一条线上"),
    "__FIG__": img("figE_folds.png", "图 E 三折的信道新颖度代价：每折内「见过的信道」与「没见过的信道」在同一批波形下的对比"),
    "__FOLD_ROWS__": fold_rows,
    "__CONV_ROWS__": conv_rows,
    "__P_LIST__": "、".join("%+.3f" % v for v in P),
    "__P_MEAN__": "%+.3f" % P.mean(),
    "__P_STD__": "%.3f" % (P.std(ddof=1) if len(P) > 1 else 0),
    "__N_TOT__": "{:,}".format(tot_seen + tot_unseen),
    "__N_SEEN__": str(tot_seen), "__N_UNSEEN__": str(tot_unseen),
    "__BAD__": str(sum(r["bad"] for r in rows)),
    "__GS_W__": "%.2f" % gs_w, "__GU_W__": "%.2f" % gu_w,
    "__P_W__": "%+.3f" % (gu_w - gs_w), "__PS_W__": "%+.3f" % (sd_gu_w - sd_gs_w),
}
for k, v in REP.items():
    HTML = HTML.replace(k, v)

left = sorted(set(__import__("re").findall(r"__[A-Z_]+__", HTML)))
if left:
    print("警告：未替换的占位符", left)

with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)
print("报告 ->", OUT)
print("  折代价: %s   均值 %+.3f  标准差 %.3f dB" %
      ("、".join("%+.3f" % v for v in P), P.mean(), P.std(ddof=1)))
print("  加权: seen %.2f -> unseen %.2f  代价 %+.3f dB" % (gs_w, gu_w, gu_w - gs_w))
print("  测试样本总数 %d（seen %d / unseen %d），变差样本 %d" %
      (tot_seen + tot_unseen, tot_seen, tot_unseen, sum(r["bad"] for r in rows)))
