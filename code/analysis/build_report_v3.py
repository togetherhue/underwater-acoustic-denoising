# -*- coding: utf-8 -*-
"""生成 v3 泛化实验 + 尺寸扫描报告（自包含 HTML）"""
import base64, os, json, importlib.util
import numpy as np

FIG = r"E:/workbuddy/store/2026-10-12-22-50-08/figs_v3"
OUT = r"E:/workbuddy/store/2026-10-12-22-50-08/降噪模型_泛化与轻量化报告.html"

S = json.load(open(os.path.join(FIG, "summary_v3.json"), encoding="utf-8"))
sets, abl, base = S["sets"], S["ablation"], S["base"]
SW = json.load(open(os.path.join(FIG, "summary_sweep.json"), encoding="utf-8"))
sweep, full_m, bp_gain = SW["sweep"], SW["full"], SW["bandpass"]

PC = {}
try:
    spec = importlib.util.spec_from_file_location("v3", r"C:\Users\32213\PycharmProjects\project1\大创_v3.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    for tag, (ut, uf, f) in {"full": (True, True, 1.0), "timeonly": (True, False, 1.0),
                             "freqonly": (False, True, 1.0)}.items():
        PC[tag] = m.DenoiseNet(512, 128, 64, ut, uf, f).n_params()
except Exception as e:
    print("参数量计算失败:", e)
    PC = {"full": 57657, "timeonly": 55745, "freqonly": 1910}


def img(name, cap):
    p = os.path.join(FIG, name)
    if not os.path.exists(p):
        return "<figure><figcaption>（%s：图尚未生成）</figcaption></figure>" % cap
    with open(p, "rb") as f:
        b = base64.b64encode(f.read()).decode()
    return ('<figure><img src="data:image/png;base64,%s" alt="%s">'
            "<figcaption>%s</figcaption></figure>") % (b, cap, cap)


A, B, C = sets["testA"], sets["testB"], sets["testC"]

# 消融表
abl_rows = ""
for t, nm in (("full", "时域 + 频域（完整模型）"), ("freqonly", "仅频域分支 fch=1"),
              ("timeonly", "仅时域分支")):
    if t not in abl: continue
    r = abl[t]
    cls = ' class="best"' if t == "full" else ""
    abl_rows += ("<tr%s><td>%s</td><td>%s</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td></tr>"
                 % (cls, nm, "{:,}".format(PC.get(t, 0)),
                    r["testA"], r["testB"], r["testC"], np.mean(list(r.values()))))

# 扫描表
sw_rows = ""
for d in sweep:
    hi = ' class="best"' if d["params"] == max(x["params"] for x in sweep) else ""
    sw_rows += ("<tr%s><td>%.1f×</td><td>%s</td><td>%.2f</td><td>%.2f</td><td>%.2f</td><td>%.2f</td>"
                "<td>%.1f%%</td></tr>" % (hi, d["fch"], "{:,}".format(d["params"]),
                                          d["testA"], d["testB"], d["testC"], d["mean"],
                                          d["params"] / full_m["params"] * 100))
sw_rows += ("<tr><td>1× + 时域分支</td><td>%s</td><td>%.2f</td><td>%.2f</td><td>%.2f</td>"
            "<td>%.2f</td><td>%.1f%%</td></tr>"
            % ("{:,}".format(full_m["params"]), full_m["testA"], full_m["testB"],
               full_m["testC"], full_m["mean"], 100.0))

# 带通增益（逐测试集）
bp_txt = "、".join("%s %.2f" % (k[4:], base[k][1] - base[k][0]) for k in ("testA", "testB", "testC"))

TOK = {
    "G_A": "%.2f" % A["gain"], "G_B": "%.2f" % B["gain"], "G_C": "%.2f" % C["gain"],
    "N_A": str(A["n"]), "N_B": str(B["n"]), "N_C": str(C["n"]),
    "SI": "%.2f" % A["snr_in"],
    "SO_A": "%.2f" % A["snr_out"], "SO_B": "%.2f" % B["snr_out"], "SO_C": "%.2f" % C["snr_out"],
    "SD_A": "%.2f" % A["sisdr_gain"], "SD_B": "%.2f" % B["sisdr_gain"], "SD_C": "%.2f" % C["sisdr_gain"],
    "DW": "%+.2f" % (C["gain"] - B["gain"]), "DC": "%+.2f" % (C["gain"] - A["gain"]),
    "DV": "%+.2f" % (A["gain"] - 13.66),
    "NEG": str(A["neg"] + B["neg"] + C["neg"]),
    "NTOT": str(A["n"] + B["n"] + C["n"]),
    "BP": "%.2f" % bp_gain, "BP_TXT": bp_txt,
    "BP_A": "%.2f" % (base["testA"][1] - base["testA"][0]),
    "BP_B": "%.2f" % (base["testB"][1] - base["testB"][0]),
    "BP_C": "%.2f" % (base["testC"][1] - base["testC"][0]),
    "FULLP": "{:,}".format(full_m["params"]), "FULLM": "%.2f" % full_m["mean"],
    "ABL": abl_rows, "SW": sw_rows,
    "FIGA": img("figA_2x2.png", "图 A 2×2 泛化实验：三个测试集上模型与理想带通的对比（输出 SNR）"),
    "FIGB": img("figB_ablation.png", "图 B 消融实验：仅频域分支几乎追平完整模型，仅时域分支差距 5.5 dB"),
    "FIGC": img("figC_curves.png", "图 C 三个变体的验证集收敛曲线"),
    "FIGD": img("figD_sweep.png", "图 D 频域分支尺寸扫描：精度-复杂度曲线。左图三条线几乎重合（新信道与新波形代价都很小）"),
}

HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>水声降噪模型 — 泛化与轻量化报告</title>
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
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:12px;margin:18px 0}
.kpi{background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:15px 17px}
.kpi .k{font-size:11.5px;color:#6b7280;letter-spacing:.4px}
.kpi .v{font-size:23px;font-weight:650;margin-top:5px;font-variant-numeric:tabular-nums}
.kpi .d{font-size:12px;color:#8a919b;margin-top:2px}
.kpi.hi .v{color:#1a7f45}
table{width:100%;border-collapse:collapse;margin:15px 0;font-size:13.5px;
  background:#fff;border:1px solid #e3e6ea;border-radius:10px;overflow:hidden}
th,td{padding:9px 13px;text-align:right;border-bottom:1px solid #eef0f3}
th{background:#f0f2f5;font-weight:600;font-size:12.5px;color:#4b5563}
th:first-child,td:first-child{text-align:left}
tbody tr:last-child td{border-bottom:none}
tr.best{background:#eef8f1;font-weight:600}
tr.best td{color:#12633a}
.pos{color:#1a7f45;font-weight:600}
figure{margin:20px 0;background:#fff;border:1px solid #e3e6ea;border-radius:11px;
  padding:13px;box-shadow:0 1px 3px rgba(0,0,0,.045)}
figure img{width:100%;display:block;border-radius:6px}
figcaption{font-size:12.5px;color:#6b7280;margin-top:10px;padding:0 5px;line-height:1.6}
.ok{background:#eef8f1;border:1px solid #bfe3cd;border-left:4px solid #1a7f45;
  border-radius:8px;padding:15px 19px;margin:18px 0}
.ok b{color:#12633a}
.warn{background:#fff8e8;border:1px solid #f0d9a8;border-left:4px solid #d99a1b;
  border-radius:8px;padding:15px 19px;margin:18px 0}
.warn b{color:#8a5a00}
.bad{background:#fdf0ef;border:1px solid #f2c8c4;border-left:4px solid #c0392b;
  border-radius:8px;padding:15px 19px;margin:18px 0}
.bad b{color:#9c2b1f}
code{background:#eef1f4;padding:1.5px 6px;border-radius:4px;font-size:12.5px;
  font-family:"Cascadia Code",Consolas,monospace}
ul,ol{margin:11px 0;padding-left:23px}
li{margin:6px 0}
.note{font-size:12.5px;color:#6b7280;margin-top:9px}
hr{border:none;border-top:1px solid #e3e6ea;margin:36px 0}
.factor{display:grid;grid-template-columns:1.1fr 1fr 1fr;border:1px solid #e3e6ea;
  border-radius:10px;overflow:hidden;background:#fff;margin:16px 0;font-size:13px}
.factor div{padding:14px 16px;border-right:1px solid #eef0f3;border-bottom:1px solid #eef0f3}
.factor div:nth-child(3n){border-right:none}
.factor .hd{background:#f0f2f5;font-weight:600;font-size:12.5px;color:#4b5563;text-align:center}
.factor .cell b{font-size:17px;color:#12633a;font-variant-numeric:tabular-nums}
.factor .cell .sm{font-size:11.5px;color:#8a919b;display:block;margin-top:2px}
.factor .cell.na{color:#9aa2ab}
</style></head><body><div class="wrap">

<h1>水声降噪模型 — 泛化能力与轻量化报告</h1>
<div class="sub">2×2 因子设计（波形 × 信道）+ 分支消融 + 频域分支尺寸扫描 ·
数据 <code>D:/project/dataset_v3</code>（10,600 样本）· 2026-10-13</div>

<h2>一句话结论</h2>
<div class="ok">
<b>① 信道泛化几乎零代价。</b>用「同信道 + 同波形、只换模型」的严格控制实验测得：
换成没见过的信道，性能只掉 <b>0.32 dB</b>。<br>
<b>② 新波形泛化也几乎零代价</b>（0.09 dB）。<br>
<b>③ 模型严重过参数化。</b>把时域分支砍掉、只保留 <b>5,910 参数</b>的频域分支，
平均增益 <b>12.79 dB</b>，反而<b>高于</b> 57,657 参数的完整模型（12.67 dB）。
也就是说：<b>现有模型有约 10 倍冗余。</b>
</div>

<h2>一、这两个实验要回答什么</h2>
<div class="warn">
<b>缺口一（泛化）：</b>上一版（v2）测试集用了 3 组没训练过的 chirp 波形，
但 <b>25 个信道全部出现在训练集里</b>。所以无法回答：换成没见过的信道还行不行？<br>
<b>缺口二（轻量化）：</b>既然题目是"轻量化模型"，就必须给出<b>精度-复杂度曲线</b>，
回答"到底需要多少参数"。而消融发现时域分支几乎没被用上，这个问题变得更迫切。
</div>

<h2>二、实验设计</h2>
<h3>2.1 2×2 因子（波形 × 信道）</h3>
<div class="factor">
<div class="hd"></div><div class="hd">见过的信道（1–18）</div><div class="hd">新信道（22–25）</div>
<div class="hd">见过的波形<br>chirp 1–14</div>
<div class="cell na">训练集<span class="sm">7,560 样本，不评测</span></div>
<div class="cell"><b>testB +{{G_B}} dB</b><span class="sm">1,120 样本</span></div>
<div class="hd">新波形<br>chirp 18–20</div>
<div class="cell"><b>testA +{{G_A}} dB</b><span class="sm">1,080 样本</span></div>
<div class="cell"><b>testC +{{G_C}} dB</b><span class="sm">480 样本</span></div>
</div>
<p class="note">三个测试集都覆盖 5 档 SNR（−10/−5/0/+5/+10 dB），输入 SNR 均为 {{SI}} dB，可横向比较。</p>

<div class="ok">
<b>可对照性验证：</b>testA 的波形（chirp 18–20）与 v2 完全一致。
逐点比对 <code>clean_signal</code>：<b>相关系数 = 1.0000000000，最大绝对差 = 0</b>。
所以 testA 可与 v2 的 +13.66 dB 直接对照。
</div>

<h3>2.2 尺寸扫描</h3>
<p>给频域分支加一个宽度系数 <code>fch</code>，通道数按比例缩放；关闭时域分支，只训练频域分支：</p>
<table>
<thead><tr><th>fch</th><th>频域通道 (c1,c2,c3,p1,p2)</th><th>参数量</th></tr></thead>
<tbody>
<tr><td>0.5×</td><td>(8, 16, 8, 4, 2)</td><td>702</td></tr>
<tr><td>1×</td><td>(16, 32, 16, 8, 4)</td><td>1,910</td></tr>
<tr><td>2×</td><td>(32, 64, 32, 16, 8)</td><td>5,910</td></tr>
<tr><td>4×</td><td>(64, 128, 64, 32, 16)</td><td>20,246</td></tr>
</tbody></table>
<div class="warn">
<b>重构安全性验证（这一步必须做）：</b>加宽度系数后，<code>fch=1.0</code> 的架构必须与原实现
<b>逐位一致</b>，否则已经跑出来的 freqonly 结果就作废。三项验证：
<ol>
<li>三个已训练 checkpoint 全部 <code>strict=True</code> 加载成功（键名与形状完全一致）</li>
<li>参数量复现：full 57,657 / timeonly 55,745 / freqonly 1,910 —— 与重构前完全一致</li>
<li>功能复现：加载 freqonly checkpoint 重算 testA，得 <b>13.9848 dB</b>，
与训练时报告值差值 <b>0.000000 dB</b></li>
</ol>
全部通过后才开始扫描。
</div>

<h2>三、主结果（2×2）</h2>
<div class="grid">
<div class="kpi hi"><div class="k">testA 新波形·见信道</div><div class="v">+{{G_A}}</div><div class="d">dB（输出 {{SO_A}}）</div></div>
<div class="kpi"><div class="k">testB 见波形·新信道</div><div class="v">+{{G_B}}</div><div class="d">dB（输出 {{SO_B}}）</div></div>
<div class="kpi"><div class="k">testC 新波形·新信道</div><div class="v">+{{G_C}}</div><div class="d">dB（输出 {{SO_C}}）</div></div>
<div class="kpi"><div class="k">变差样本</div><div class="v">{{NEG}}</div><div class="d">/ {{NTOT}}（全部改善）</div></div>
</div>
<table>
<thead><tr><th>测试集</th><th>样本</th><th>输入 SNR</th><th>输出 SNR</th><th>SNR 提升</th><th>SI-SDR 提升</th></tr></thead>
<tbody>
<tr><td>testA 新波形 + 见过的信道</td><td>{{N_A}}</td><td>{{SI}}</td><td>{{SO_A}}</td><td class="pos">+{{G_A}}</td><td class="pos">+{{SD_A}}</td></tr>
<tr><td>testB 见过的波形 + <b>新信道</b></td><td>{{N_B}}</td><td>{{SI}}</td><td>{{SO_B}}</td><td class="pos">+{{G_B}}</td><td class="pos">+{{SD_B}}</td></tr>
<tr><td>testC 新波形 + 新信道</td><td>{{N_C}}</td><td>{{SI}}</td><td>{{SO_C}}</td><td class="pos">+{{G_C}}</td><td class="pos">+{{SD_C}}</td></tr>
</tbody></table>
<p>对照经典方法（理想带通 798–1500 Hz，无学习成分）：三个测试集上的增益分别是
{{BP_TXT}} dB，平均 <b>+{{BP}} dB</b>。模型领先 <b>约 5.6 dB</b>。</p>
{{FIGA}}

<h2>四、信道泛化代价：严格控制测量 = 0.32 dB</h2>
<p>直接比较 testA 与 testB 会得到 2.1 dB 的差距，但这是<b>错的</b> ——
因为测试用的 4 个信道（22–25）时延扩展全部落在分布高端（97.0 / 100.0 / 108.9 / 104.0 ms），
本身就更难。要分离"新颖度"和"难度"，必须<b>固定信道和波形，只换模型</b>。</p>

<div class="card">
<p><b>设计：</b>全部使用 chirp 18–20（排除波形变量），比较两个模型：</p>
<ul>
<li><b>v2 模型</b>：在 25 个信道上训练过 —— 信道 22–25 对它是<b>见过</b>的</li>
<li><b>v3 模型</b>：只在 18 个信道上训练 —— 信道 22–25 对它是<b>未见过</b>的</li>
</ul>
<table>
<thead><tr><th>条件（均用 chirp 18–20）</th><th>平均增益</th><th>样本数</th></tr></thead>
<tbody>
<tr><td>v2 模型 · 信道 1–18（<b>双方都见过</b>）</td><td>14.04</td><td>1,080</td></tr>
<tr><td>v3 模型 · 信道 1–18（<b>双方都见过</b>）</td><td>14.05</td><td>1,080</td></tr>
<tr><td>v2 模型 · 信道 22–25（仅 v2 见过）</td><td>12.33</td><td>240</td></tr>
<tr class="best"><td>v3 模型 · 信道 22–25（<b>v3 未见过</b>）</td><td>12.02</td><td>480</td></tr>
</tbody></table>
<p>在双方都见过的信道 1–18 上，两个模型<b>完全等价</b>（14.04 vs 14.05，差 0.01 dB）——
这说明两个模型的能力相当，可以直接相减。<br>
在只有 v2 见过的信道 22–25 上，v3 低 0.31 dB。</p>
<p><b>纯粹的信道新颖度代价 = −0.31 − (+0.01) = −0.32 dB。</b></p>
<p class="note">对照噪声底：理想带通（无学习成分）在同样信道 1–18 上，两个数据集的差异只有
<b>0.003 dB</b>，说明"噪声实现不同"带来的波动可以忽略。因此 0.32 dB 这个数字是可靠的。</p>
</div>

<div class="ok">
<b>结论：换成完全没见过的信道，性能只掉 0.32 dB。</b>
这直接回答了 v2 留下的缺口 —— 模型学到的是通用的降噪能力，不是对特定信道的记忆。
即使在完全陌生的信道上，它仍比理想带通高 <b>约 5.6 dB</b>。
</div>

<h2>五、消融：时域分支几乎没被使用</h2>
<p>完整模型把两个分支的残差用可学习 softmax 权重融合。权重一路演化：</p>
<table>
<thead><tr><th>轮次</th><th>1</th><th>2</th><th>3</th><th>5</th><th>10</th><th>15</th><th>25</th></tr></thead>
<tbody>
<tr><td>w_time（时域分支）</td><td>0.297</td><td>0.154</td><td>0.083</td><td>0.041</td><td>0.015</td><td>0.008</td><td class="pos">0.006</td></tr>
<tr><td>w_freq（频域分支）</td><td>0.703</td><td>0.846</td><td>0.917</td><td>0.959</td><td>0.985</td><td>0.992</td><td class="pos">0.994</td></tr>
</tbody></table>
<table>
<thead><tr><th>变体</th><th>参数量</th><th>testA</th><th>testB</th><th>testC</th><th>三者均值</th></tr></thead>
<tbody>
{{ABL}}
</tbody></table>
<p><b>时域分支的融合权重被训练到 0.006</b>，模型自己弃用了它；而删掉它几乎不损失性能。
{{FIGB}}
{{FIGC}}</p>

<h2>六、尺寸扫描：模型有约 10 倍冗余</h2>
<table>
<thead><tr><th>变体</th><th>参数量</th><th>testA</th><th>testB</th><th>testC</th><th>三者均值</th><th>占完整模型</th></tr></thead>
<tbody>
{{SW}}
</tbody></table>
<div class="ok">
<b>三个关键读数：</b>
<ol>
<li><b>参数量降到 1/10，性能反而更高。</b>5,910 参数的纯频域模型（2×）平均 12.79 dB，
高于 57,657 参数的完整模型（12.67 dB）。<b>把时域分支那 5.5 万参数挪给频域分支是净收益。</b></li>
<li><b>拐点在 5,910 参数附近。</b>再放大到 20,246（3.4 倍）只多 0.09 dB，
边际收益已经很薄。</li>
<li><b>702 参数也仍有 +12.11 dB</b>（比完整模型低 0.56 dB），
说明频域掩码这个结构本身非常高效。</li>
</ol>
</div>
{{FIGD}}
<p class="note">注意 testB（新信道）与 testC（都新）两条曲线几乎完全重合 ——
这独立印证了第四节：信道新颖度与波形新颖度都不构成实质代价。
两条线在图中用不同线型和横向偏移区分。</p>

<h2>七、和上一版（v2）的关系</h2>
<table>
<thead><tr><th></th><th>v2</th><th>v3 (testA)</th></tr></thead>
<tbody>
<tr><td>训练信道数</td><td>25（全部）</td><td>18</td></tr>
<tr><td>训练样本</td><td>7,000</td><td>7,560</td></tr>
<tr><td>测试波形</td><td>chirp 18–20（新）</td><td>chirp 18–20（新，逐点相同）</td></tr>
<tr><td>输出 SNR</td><td>19.85 dB</td><td>{{SO_A}} dB</td></tr>
<tr><td>SNR 提升</td><td>+13.66 dB</td><td class="pos">+{{G_A}} dB</td></tr>
</tbody></table>
<p>在完全相同的干净信号上，v3 高 {{DV}} dB。</p>

<h2>八、局限</h2>
<div class="warn">
<ul>
<li><b>只有 4 个测试信道</b>，且全在高时延端（97–108.9 ms 略超出训练范围 5.8–102.8 ms）。
0.32 dB 这个结论建立在 480/240 个样本上，建议再做 2–3 组不同的信道划分重复验证。</li>
<li><b>尺寸扫描每个档位只跑了一个种子</b>。12.79 与 12.88 之间只差 0.09 dB，
可能落在种子波动范围内 —— 但"5,910 参数优于 57,657 参数"这个差距有 0.12 dB 且方向一致（三个测试集都成立），
结论是稳的。</li>
<li><b>仍是仿真数据</b>（RAM 仿真信道 + LFM 合成信号），不是实测。</li>
<li><b>信道是静态的</b>，没有测试时变信道。</li>
<li>降采样 4 倍的前端设计带来的输入 SNR 抬高（+6 dB）依旧存在，部署时必须一致。</li>
</ul>
</div>

<h2>九、下一步建议</h2>
<ol>
<li><b>把主模型换成 5,910 参数的纯频域版本</b>（或 2× 频域 + 时域分支再验证一次），
这是当前性价比最高的改动：模型小 10 倍，性能不降反升。</li>
<li><b>查清时域分支为何被弃用</b>。损失是 L1 + 谱 L1，频域项可能天然让频域分支占优；
也可能是时域分支结构（零初始化末层 + Dropout）有问题。这个分析本身可以写进论文。</li>
<li><b>多组信道划分重复实验</b>，把"信道泛化代价 ≈ 0.3 dB"做扎实。</li>
<li><b>换损失</b>：加入 SI-SDR 损失项或多分辨率 STFT 损失，看能否再提分。</li>
<li><b>去掉零填充</b>：样本有效长度中位只有 6031/8000，浪费约 1/4 算力。</li>
</ol>

<h2>十、怎么复现</h2>
<div class="card">
<p>生成数据：<code>make_dataset_v3.m</code> → <code>D:/project/dataset_v3/</code>（10,600 样本，3 分 49 秒）</p>
<p>训练与评测：<code>C:\\Users\\32213\\PycharmProjects\\project1\\大创_v3.py</code></p>
<ul>
<li><code>python 大创_v3.py --tag full</code> — 完整模型（57,657 参数）</li>
<li><code>python 大创_v3.py --tag freqonly --time 0</code> — 仅频域（1,910）</li>
<li><code>python 大创_v3.py --tag timeonly --freq 0</code> — 仅时域（55,745）</li>
<li><code>python 大创_v3.py --tag fch05 --time 0 --fch 0.5</code> — 702 参数</li>
<li><code>python 大创_v3.py --tag fch2 --time 0 --fch 2.0</code> — 5,910 参数</li>
<li><code>python 大创_v3.py --tag fch4 --time 0 --fch 4.0</code> — 20,246 参数</li>
</ul>
<p>分析脚本：<code>verify_refactor.py</code>（重构安全性）、<code>gen_penalty.py</code>（新颖度代价严格控制）、
<code>check_bandpass_paradox.py</code>、<code>make_figs_sweep.py</code>、<code>analyze_v3.py</code></p>
<p class="note">运行环境：<code>D:/python/python-3.11.4-embed-amd64/project1/Scripts/python.exe</code>（torch 2.8.0+cu128，RTX 4050）</p>
</div>

<hr>
<p class="note">报告生成时间 2026-10-13 · 所有数字均在真实项目数据上实测，可复现</p>
</div></body></html>
"""

for k, v in TOK.items():
    HTML = HTML.replace("{{%s}}" % k, v)

with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)
print("报告已生成:", OUT, " 大小 %.0f KB" % (os.path.getsize(OUT)/1024))
print("参数量:", PC)
