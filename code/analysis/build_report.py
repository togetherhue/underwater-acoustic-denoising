# -*- coding: utf-8 -*-
"""生成训练结果分析报告（自包含 HTML，图片 base64 内嵌）"""
import base64, os

FIG = r"E:/workbuddy/store/2026-10-12-22-50-08/figs"
OUT = r"E:/workbuddy/store/2026-10-12-22-50-08/降噪模型v2_训练结果分析.html"

def img(name, cap):
    with open(os.path.join(FIG, name), "rb") as f:
        b = base64.b64encode(f.read()).decode()
    return ('<figure><img src="data:image/png;base64,%s" alt="%s">'
            '<figcaption>%s</figcaption></figure>') % (b, cap, cap)

HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>水声信号降噪模型 v2 — 训练结果分析</title>
<style>
*{box-sizing:border-box}
body{margin:0;padding:44px 26px 90px;background:#f5f6f8;color:#1f2328;
  font:15px/1.75 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
.wrap{max-width:1000px;margin:0 auto}
h1{font-size:27px;margin:0 0 6px;letter-spacing:-.3px}
.sub{color:#6b7280;font-size:13.5px;margin-bottom:30px}
h2{font-size:19.5px;margin:44px 0 14px;padding-left:11px;border-left:4px solid #2f6fd0}
h3{font-size:15.5px;margin:26px 0 9px;color:#333}
p{margin:11px 0}
.card{background:#fff;border:1px solid #e3e6ea;border-radius:11px;padding:20px 24px;
  box-shadow:0 1px 3px rgba(0,0,0,.045)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:12px;margin:18px 0}
.kpi{background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:15px 17px}
.kpi .k{font-size:11.5px;color:#6b7280;letter-spacing:.4px}
.kpi .v{font-size:23px;font-weight:650;margin-top:5px;color:#1f2328;font-variant-numeric:tabular-nums}
.kpi .d{font-size:12px;color:#8a919b;margin-top:2px}
.kpi.hi .v{color:#1a7f45}
table{width:100%;border-collapse:collapse;margin:15px 0;font-size:13.5px;
  background:#fff;border:1px solid #e3e6ea;border-radius:10px;overflow:hidden}
th,td{padding:9px 13px;text-align:right;border-bottom:1px solid #eef0f3}
th{background:#f0f2f5;font-weight:600;font-size:12.5px;color:#4b5563;text-align:right}
th:first-child,td:first-child{text-align:left}
tbody tr:last-child td{border-bottom:none}
tr.best{background:#eef8f1;font-weight:600}
tr.best td{color:#12633a}
tr.base{background:#fafbfc;color:#6b7280}
.pos{color:#1a7f45;font-weight:600}
figure{margin:20px 0;background:#fff;border:1px solid #e3e6ea;border-radius:11px;
  padding:13px;box-shadow:0 1px 3px rgba(0,0,0,.045)}
figure img{width:100%;display:block;border-radius:6px}
figcaption{font-size:12.5px;color:#6b7280;margin-top:10px;padding:0 5px;line-height:1.6}
.warn{background:#fff8e8;border:1px solid #f0d9a8;border-left:4px solid #d99a1b;
  border-radius:8px;padding:15px 19px;margin:18px 0}
.warn b{color:#8a5a00}
.ok{background:#eef8f1;border:1px solid #bfe3cd;border-left:4px solid #1a7f45;
  border-radius:8px;padding:15px 19px;margin:18px 0}
.ok b{color:#12633a}
.bad{background:#fdf0ef;border:1px solid #f2c8c4;border-left:4px solid #c0392b;
  border-radius:8px;padding:15px 19px;margin:18px 0}
.bad b{color:#9c2b1f}
code{background:#eef1f4;padding:1.5px 6px;border-radius:4px;font-size:12.5px;
  font-family:"Cascadia Code",Consolas,monospace}
ul,ol{margin:11px 0;padding-left:23px}
li{margin:6px 0}
.tag{display:inline-block;font-size:11px;padding:2px 8px;border-radius:20px;
  background:#e8eefb;color:#2f6fd0;margin-right:6px;vertical-align:1px}
.tag.p0{background:#fde8e6;color:#b3392a}
.tag.p1{background:#fef3e0;color:#a4710d}
.tag.p2{background:#e9eef2;color:#5b6876}
.bar{height:9px;background:#e9ecef;border-radius:5px;overflow:hidden;margin-top:4px}
.bar i{display:block;height:100%;background:#2f6fd0}
.note{font-size:12.5px;color:#6b7280;margin-top:9px}
hr{border:none;border-top:1px solid #e3e6ea;margin:36px 0}
</style></head><body><div class="wrap">

<h1>水声信号降噪模型 v2 — 训练结果分析</h1>
<div class="sub">数据：25 个 RAM 仿真信道 × 20 组 LFM chirp × 5 档 SNR × 4 次噪声 = 10,000 样本 ·
模型 57,657 参数 · RTX 4050 · 50 轮 66.6 分钟 · 2026-10-13</div>

<h2>一句话结论</h2>
<div class="ok">
<b>训练成功，结论成立。</b>测试集 1500 个样本 <b>无一变差</b>，输出 SNR 由 6.19 dB 提升到
<b>19.85 dB（+13.66 dB）</b>；比最强的经典方法（谱减法 +7.70 dB）还高 <b>6.0 dB</b>，
比"已知信号频带的理想带通"（+6.44 dB）高 <b>7.3 dB</b>。<br>
更关键的是：测试用的 3 组 chirp 波形<b>从未出现在训练集里</b>，模型仍能处理 ——
这证明它学到的是"降噪"，不是旧版那种"背下唯一波形"。
</div>

<h2>一、结果总览</h2>
<div class="grid">
<div class="kpi"><div class="k">测试样本</div><div class="v">1,500</div><div class="d">train 7000 / val 1500</div></div>
<div class="kpi hi"><div class="k">输出 SNR</div><div class="v">19.85</div><div class="d">dB（输入 6.19）</div></div>
<div class="kpi hi"><div class="k">SNR 提升</div><div class="v">+13.66</div><div class="d">dB</div></div>
<div class="kpi hi"><div class="k">SI-SDR 提升</div><div class="v">+13.62</div><div class="d">dB（尺度不敏感）</div></div>
<div class="kpi"><div class="k">变差样本</div><div class="v">0</div><div class="d">1500 个全部改善</div></div>
<div class="kpi"><div class="k">最差样本</div><div class="v">+2.99</div><div class="d">dB（最低输出 10.06 dB）</div></div>
<div class="kpi"><div class="k">参数量</div><div class="v">57,657</div><div class="d">原版 99,005</div></div>
<div class="kpi"><div class="k">训练耗时</div><div class="v">66.6</div><div class="d">分钟（GPU）</div></div>
</div>

<p>SNR 与 SI-SDR 的提升几乎完全一致（13.66 vs 13.62 dB），说明输出<b>没有幅度偏差</b>，
不是靠"把信号整体放大"骗指标 —— 这是 SI-SDR 尺度不变性给出的保证。</p>

<h2>二、这个数字到底好不好？—— 与经典方法对比</h2>
<p>只跟"不处理"比是没有说服力的。我在<b>同一批 1500 个样本、同一套指标定义</b>下跑了四种经典方法：</p>
<table>
<thead><tr><th>方法</th><th>输出 SNR (dB)</th><th>提升 (dB)</th><th>相对本模型</th></tr></thead>
<tbody>
<tr class="base"><td>含噪输入（不处理）</td><td>6.19</td><td>—</td><td>−13.71</td></tr>
<tr><td>带通 700–1600 Hz（放宽）</td><td>11.54</td><td class="pos">+5.35</td><td>−8.36</td></tr>
<tr><td>维纳滤波（盲噪声估计）</td><td>11.07</td><td class="pos">+4.88</td><td>−8.83</td></tr>
<tr><td>带通 798–1500 Hz（<b>理想</b>，已知信号频带）</td><td>12.63</td><td class="pos">+6.44</td><td>−7.27</td></tr>
<tr><td>谱减法（盲噪声估计）</td><td>13.89</td><td class="pos">+7.70</td><td>−6.01</td></tr>
<tr class="best"><td>本模型 v2</td><td>19.90</td><td class="pos">+13.71</td><td>—</td></tr>
</tbody></table>
<p class="note">本模型行为我用独立脚本（float64 全精度重算）复现，得 19.90 / +13.71 dB，
与训练脚本自报的 19.85 / +13.66 dB 相差 0.05 dB —— 结果可复现。
其余方法在同一脚本内计算，因此可直接比较。</p>

<div class="ok">
<b>两个关键对比结论：</b>
<ul>
<li><b>比最强经典方法（谱减法）高 6.0 dB。</b>谱减法已经是用了"1500–3000 Hz 只有噪声"这一先验的
盲噪声估计方案，属于工程上很强的对手。</li>
<li><b>比"理想带通"高 7.3 dB。</b>理想带通是<b>作弊式</b>的 —— 它直接用了信号频带 798–1500 Hz
这个我们本来不知道的先验。真实场景下只能放宽到 700–1600 Hz（+5.35 dB），
那时模型领先 <b>8.4 dB</b>。</li>
</ul>
</div>

<p><b>管线可信度的一个交叉验证：</b>理想带通的理论增益 = 10·log₁₀(3000 / 702) = <b>6.31 dB</b>
（白噪声功率均匀分布在 0–3000 Hz，信号只占 702 Hz）。实测 <b>6.44 dB</b>，吻合到 0.13 dB。
说明整套测量管线（降采样 → 指标计算）没有系统性偏差。</p>

__FIG4__

<h2>三、逐样本表现</h2>
<table>
<thead><tr><th>分位</th><th>输入 SNR</th><th>输出 SNR</th><th>提升</th></tr></thead>
<tbody>
<tr><td>最差（0%）</td><td>−4.05</td><td>10.06</td><td class="pos">+2.99</td></tr>
<tr><td>1%</td><td>−3.93</td><td>10.98</td><td class="pos">+4.56</td></tr>
<tr><td>5%</td><td>−3.86</td><td>11.81</td><td class="pos">+6.80</td></tr>
<tr><td>25%</td><td>1.14</td><td>15.67</td><td class="pos">+11.68</td></tr>
<tr><td>50%（中位）</td><td>6.20</td><td>19.73</td><td class="pos">+14.29</td></tr>
<tr><td>75%</td><td>11.24</td><td>23.27</td><td class="pos">+16.14</td></tr>
<tr><td>95%</td><td>16.24</td><td>28.64</td><td class="pos">+18.55</td></tr>
<tr><td>最好（100%）</td><td>16.40</td><td>34.46</td><td class="pos">+22.57</td></tr>
</tbody></table>
<p><b>没有失败样本。</b>最坏的 1% 样本也至少有 +4.5 dB 的提升，最低输出 SNR 是 10.06 dB。
分布是单峰的、连续的，没有"部分样本崩掉"的迹象。</p>

__FIG5__

<p>散点图里能看到 <b>5 个垂直簇</b>（对应 5 档真实 SNR）。每个簇都很紧，说明模型对同一
SNR 下的不同噪声实现表现稳定。绿色（模型）在所有信噪比下都远高于橙色（理想带通），
且越往左下（信噪比越低）差距越大。</p>

<h2>四、增益由什么决定</h2>
<p>增益在样本间的标准差是 <b>3.53 dB</b>。因为实验是 3 chirp × 25 信道 × 5 SNR × 4 噪声的
<b>平衡设计</b>，可以直接做方差分解：</p>
<table>
<thead><tr><th>因素</th><th>解释方差</th><th>极差 (dB)</th><th>说明</th></tr></thead>
<tbody>
<tr><td>SNR 档位</td><td><b>55.0%</b><div class="bar"><i style="width:55%"></i></div></td><td>7.48</td><td>主导因素</td></tr>
<tr><td>chirp 波形</td><td>19.5%<div class="bar"><i style="width:19.5%"></i></div></td><td>3.37</td><td>测试集的 3 组波形难度不同</td></tr>
<tr><td>信道</td><td>16.5%<div class="bar"><i style="width:16.5%"></i></div></td><td>5.43</td><td>真实但无法用一阶物理量解释</td></tr>
<tr><td>残差（交互+噪声）</td><td>9.0%<div class="bar"><i style="width:9%"></i></div></td><td>—</td><td>—</td></tr>
</tbody></table>

<h3>4.1 低信噪比时提升更大 —— 典型的降噪器行为</h3>
<table>
<thead><tr><th>真实 SNR 档</th><th>输入 SNR</th><th>输出 SNR</th><th>提升</th></tr></thead>
<tbody>
<tr><td>−10 dB</td><td>−3.81</td><td>13.23</td><td class="pos">+17.04</td></tr>
<tr><td>−5 dB</td><td>1.19</td><td>16.77</td><td class="pos">+15.58</td></tr>
<tr><td>0 dB</td><td>6.19</td><td>20.19</td><td class="pos">+13.99</td></tr>
<tr><td>+5 dB</td><td>11.19</td><td>23.34</td><td class="pos">+12.15</td></tr>
<tr><td>+10 dB</td><td>16.19</td><td>25.75</td><td class="pos">+9.56</td></tr>
</tbody></table>
<p>注意输出 SNR 随输入单调上升（13.2 → 25.8 dB），说明模型<b>没有出现"过度抑制"</b>
—— 有些降噪器在高信噪比下会把信号一起削掉，这里没有。这也是低 SNR 时增益最大（+17.04 dB）
的原因：噪声越多，可去掉的越多。</p>

__FIG7__

<h3>4.2 信道间的 5.4 dB 差异由「多径时延扩展」解释（r = −0.862）</h3>
<p>信道平均增益从 +10.83 到 +16.26 dB，极差 5.43 dB，远超统计噪声
（每个信道 60 个样本，单样本残差标准差 1.06 dB → 均值标准误 0.137 dB，3×SEM = 0.41 dB，
极差是它的 13 倍）。所以信道差异是真实且可重复的效应。</p>
<p>找出物理原因，测量三个常见的信道特征：</p>
<table>
<thead><tr><th>信道特征</th><th>范围</th><th>与增益的相关系数</th></tr></thead>
<tbody>
<tr class="best"><td><b>90% 时延扩展</b></td><td>5.8 ~ 108.9 ms</td><td><b>−0.862</b></td></tr>
<tr><td>带内 5% 分位凹陷深度</td><td>−29.9 ~ −7.1 dB</td><td>+0.297</td></tr>
<tr><td>带内频率响应起伏（标准差）</td><td>3.6 ~ 10.1 dB</td><td>−0.396</td></tr>
</tbody></table>
<p>具体看两端：时延扩展<b>最小</b>的信道（chan_id 11，5.8 ms，接近单径）增益 <b>+14.62 dB</b>；
时延扩展<b>最大</b>的两个信道（chan_id 25 / 24，104.0 / 108.9 ms，强多径）增益只有
<b>+10.83 / +12.33 dB</b>。差距 3.8 dB，且单调。</p>
<p><b>物理含义：</b>多径越强，接收波形在时间上被展得越开、能量越分散，与噪声在时频面上的
重叠越多，可分离性越差。这符合直觉，也是对模型行为的一个合理解释。</p>
<div class="warn">
<b>一处更正：</b>本文档第一版曾给出"三个信道特征都与增益不相关（|r| &lt; 0.07）"的结论，
<b>那是错的</b>。原因是信道文件与 <code>chan_id</code> 的对应关系搞错了 —— MATLAB 的
<code>dir</code> 按字典序返回（<code>channel_H_1, channel_H_10, channel_H_11, ...</code>），
<code>chan_id</code> 是<b>字典序下标</b>，不是文件名里的数字。<br>
已用 <code>clean_signal</code> 逐点反查证实正确映射：<b>25/25 全部匹配，相关系数 = 1.000000</b>。
本节数字为修正后的结果。
</div>

__FIG6__

<h2>五、训练过程诊断</h2>
__FIG1__
<div class="warn">
<b>训练曲线暴露了两个信息：</b>
<ul>
<li><b>50 轮是浪费的。</b>验证集 SNR 从第 1 轮的 16.24 涨到第 6 轮的 19.32（+3.08 dB），
已完成总提升 3.23 dB 的 <b>95%</b>；之后 44 轮只再涨 <b>0.15 dB</b>。
<b>跑 15 轮就能拿到几乎一样的结果，时间省 2/3。</b></li>
<li><b>模型已到容量上限。</b>验证损失从 0.2080（第 5 轮）到 0.1746（第 48 轮），
只改善 16%。要再提升必须改结构或损失函数，<b>加轮次没有用</b>。</li>
</ul>
</div>
<p><b>好消息：完全没有过拟合。</b>验证损失（0.1746）一直<b>低于</b>训练损失（0.1915）。
这不是异常 —— 训练时 Dropout(0.1) 生效、验证时不生效，所以训练损失天然偏高。
真正的含义是：模型容量还有富余，可以<b>放心加大模型</b>。</p>

<h2>六、从信号本身看模型在做什么</h2>
__FIG2__
<p>红色是含噪输入 —— 目标信号完全埋在噪声里；绿色是模型输出；蓝色是干净目标。
模型输出的<b>多径包络结构被完整保留</b>（那些起伏的"节"是信道的多径干涉，不是噪声），
说明它在做<b>降噪</b>，没有顺手把信道畸变也抹掉。这一点很重要：
如果目标定义错了（像旧版那样把"源 chirp"当目标），模型会把多径结构也一起抹平，
那就不叫降噪了，叫解卷积。</p>

__FIG3__
<p>频谱上看得更清楚：信号频带 798–1500 Hz 内，输出（绿）与干净目标（蓝）几乎重合；
带外噪声被压低约 <b>70 dB</b>（红 10⁻⁴ → 绿 10⁻¹¹）。</p>
<p class="note">顺带一个发现：干净目标在 60 Hz、400 Hz 附近有微小能量（图中蓝色的小凸起），
模型也把它们复现出来了（绿色同样有凸起）。这些是样本末尾零填充造成的截断谱泄漏，
不是真实信号。模型忠实复现了它们 —— 属于"把该保留的都保留了"，
但也提示填充方式值得改进（见第七节）。</p>

<h2>七、必须指出的局限</h2>
<div class="bad">
<b>① 信道没有留出 —— 这是目前最大的缺口。</b><br>
测试集用的是 chirp 18/19/20，这 3 组波形确实没在训练里出现过（真正的波形泛化测试）。
但测试集的 <b>25 个信道全部出现在训练集里</b>。所以现在只证明了
"对<b>新波形</b>泛化"，没有证明"对<b>新信道</b>泛化"。<br>
<b>修法：</b>改 <code>make_dataset.m</code> 的划分逻辑，把信道也分组（例如 20 个训 / 5 个测），
<b>不需要重跑 RAM</b>（信道数据已在硬盘上），只需重新生成数据集（2.7 分钟）+ 重新训练（约 15 轮即可）。
</div>
<div class="warn">
<b>② 降采样 4 倍会滤掉带外噪声，输入 SNR 被抬高了 6 dB。</b><br>
数据集里 <code>true_snr</code> 平均是 0 dB，但模型实际看到的输入 SNR 是 6.19 dB。
原因是白噪声只有 1/4 功率落在 0–3000 Hz 内，而信号（798–1500 Hz）完全保留。
这是<b>合理的前端设计</b>（信号只占 798–1500 Hz，Nyquist 3000 Hz 足够），
但必须在报告里说明：<b>部署到实测数据时也必须先做同样的 4 倍降采样</b>，
否则输入分布不一致，模型会失效。
</div>
<div class="warn">
<b>③ 24.6% 的样本内容是零填充。</b><br>
样本有效长度中位 <b>6031 / 8000</b> 点（最短 5718）。因为 chirp 时长 0.6–1.0 s 变化，
而 <code>seg_len</code> 固定 8000。零填充本身不破坏指标（参考段和输出段在该区域都是 0），
但它造成了截断谱泄漏，而且浪费了 1/4 的计算量。
<b>建议</b>：按 chirp 实际时长裁剪，或把不同时长的样本分桶。
</div>
<div class="warn">
<b>④ 只在仿真数据上验证。</b><br>
老师 README 写的是"针对<b>实测 HFM 信号</b>"（双曲调频），但本项目的数据是
<b>LFM 线性调频 + RAM 仿真信道</b>。两者在匹配滤波、多普勒容限上差别很大。
如果论文要声称"可用于实测"，需要说明数据是仿真的，或补充实测验证。
</div>

<h2>八、下一步优先级</h2>
<table>
<thead><tr><th>优先级</th><th>事项</th><th>成本</th><th>为什么重要</th></tr></thead>
<tbody>
<tr><td><span class="tag p0">P0</span></td><td><b>留出信道做泛化测试</b></td><td>2.7 min 生成 + 约 20 min 训练</td>
<td>不补这一条，"模型能泛化"这句话就站不住。目前只验证了波形泛化<br>
<b>（已实施：2×2 因子实验，见配套文档）</b></td></tr>
<tr><td><span class="tag p0">P0</span></td><td>报告里改成真实 SNR 分层叙述</td><td>0（数据已有）</td>
<td>现在只报平均值，掩盖了 −10 dB 时 +17 dB 的优势</td></tr>
<tr><td><span class="tag p1">P1</span></td><td><b>消融实验</b>：只用时域分支 / 只用频域分支 / 两者</td><td>约 40 min</td>
<td>证明两个分支都有贡献，是论文里最有说服力的一张表</td></tr>
<tr><td><span class="tag p1">P1</span></td><td>加大模型 / 换损失（加 SI-SDR 损失项、多分辨率 STFT 损失）</td><td>约 30 min</td>
<td>已验证无过拟合，容量有富余，是最直接的提分手段</td></tr>
<tr><td><span class="tag p1">P1</span></td><td>查清"信道间 5.4 dB 差异"的来源</td><td>不确定</td>
<td>目前三个一阶特征都无关，可能藏着一个有价值的发现</td></tr>
<tr><td><span class="tag p2">P2</span></td><td>去掉零填充（按 chirp 时长裁剪/分桶）</td><td>约 15 min</td>
<td>省 1/4 算力，消除截断谱泄漏</td></tr>
<tr><td><span class="tag p2">P2</span></td><td>减少训练轮数到 15</td><td>省 2/3 时间</td>
<td>已验证 95% 的提升在前 6 轮完成</td></tr>
</tbody></table>

<h2>九、报告写作口径建议</h2>
<div class="card">
<p><b>不要这样写：</b>"降噪模型 SNR 提升 13.66 dB。"</p>
<p><b>建议这样写：</b></p>
<p style="border-left:3px solid #2f6fd0;padding-left:14px;color:#333;margin:12px 0">
"在 25 个 RAM 仿真水声信道、20 组线性调频信号（其中测试集 3 组波形未参与训练）、
5 档信噪比（−10 ~ +10 dB）共 10,000 个样本上训练了 57,657 参数的轻量化降噪网络。
在 1,500 个测试样本上，输出信噪比由 6.19 dB 提升至 19.85 dB（+13.66 dB），
尺度不变信噪比 SI-SDR 提升 13.62 dB，全部样本均获得改善。
作为对照，已知信号频带的理想带通滤波器提升 6.44 dB，采用盲噪声估计的谱减法提升 7.70 dB，
本模型分别高出 7.3 dB 和 6.0 dB。"
</p>
<p class="note">这样写的好处：① 交代了数据来源和规模，可复现；② 用了 SI-SDR 交叉验证，
说明没有幅度作弊；③ 有经典方法对照，"提升 13.66 dB"才有意义；
④ 明确说了测试波形未参与训练，堵住"背波形"的质疑。</p>
</div>

<h2>十、怎么复现</h2>
<div class="card">
<p>训练脚本：<code>code/python/大创_v2.py</code></p>
<p>输出：<code>D:/project/denoise_model_v2.pth</code>（模型）、
<code>D:/project/output_v2/train_log.txt</code>（逐轮指标）、
<code>D:/project/output_v2/test_results.csv</code>（1500 行逐样本结果）</p>
<p>本次分析用的脚本（在 <code>E:/workbuddy/store/2026-10-12-22-50-08/</code>）：</p>
<ul>
<li><code>analyze_result.py</code> — 按 chirp / 信道 / SNR 分组统计</li>
<li><code>compare_baseline.py</code> — 与带通、维纳、谱减法对比</li>
<li><code>channel_gain.py</code> — 信道物理特性与增益的关系</li>
<li><code>make_figs.py</code> — 生成本报告全部图表</li>
</ul>
<p class="note">运行环境：<code>D:/python/python-3.11.4-embed-amd64/project1/Scripts/python.exe</code>
（torch 2.8.0+cu128）</p>
</div>

<hr>
<p class="note">报告生成时间 2026-10-13 · 所有数字均在真实项目数据上实测，可复现</p>
</div></body></html>
"""

HTML = (HTML.replace("__FIG1__", img("fig1_training.png", "图 1 训练曲线：损失在前 6 轮收敛，之后 44 轮几乎无改善（右轴绿色虚线为验证集 SNR 提升）"))
            .replace("__FIG2__", img("fig2_waveform.png", "图 2 时域波形对比：含噪输入完全淹没在噪声中，模型输出与干净目标几乎重合，且多径包络结构被完整保留"))
            .replace("__FIG3__", img("fig3_spectrum.png", "图 3 功率谱：信号带内输出与干净目标重合，带外噪声被压低约 70 dB"))
            .replace("__FIG4__", img("fig4_methods.png", "图 4 模型与四种经典方法的对比（同一批 1500 样本，指标定义一致）"))
            .replace("__FIG5__", img("fig5_scatter.png", "图 5 逐样本输入-输出 SNR：5 个垂直簇对应 5 档信噪比，模型在所有档位都远高于理想带通"))
            .replace("__FIG6__", img("fig6_channel.png", "图 6 信道物理特性与增益的关系：90% 时延扩展与增益强负相关（r = −0.862），带内凹陷与带内起伏只有弱相关"))
            .replace("__FIG7__", img("fig7_bysnr.png", "图 7 分档对比：低信噪比时模型相对经典方法的优势最大")))

with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)
print("报告已生成:", OUT, " 大小 %.0f KB" % (os.path.getsize(OUT)/1024))
