# 轻量化水声信号降噪模型

本科大学生创新创业训练计划（大创）项目。针对水声接收信号做**降噪**，构建了一个轻量化网络，
并在 RAM 仿真的多径信道上系统评估了它的**降噪性能、泛化能力与参数效率**。

> **任务定义（重要）**
> 干净目标 = `conv(源chirp, 信道冲激响应)`，即**无噪的接收波形**；输入 = 干净目标 + 高斯噪声。
> 这是"降噪"，不是"解卷积" —— 信道的多径畸变属于要保留的物理特征。
> 早期版本把"源 chirp"当目标，等于要求模型在不知道信道的情况下反卷积，是病态问题，
> 模型只能靠背波形作弊（见 `docs/02_dachuang_code_diagnosis.html`）。

---

## 一、核心结果

### 1. 主结果（v2，25 个信道全部参与训练）

| 指标 | 输入 | 输出 | 提升 |
|---|---|---|---|
| SNR | 6.19 dB | **19.85 dB** | **+13.66 dB** |
| SI-SDR | 6.19 dB | **19.81 dB** | **+13.62 dB** |

测试集 1,500 个样本**全部改善，无一变差**（最差样本 +2.99 dB）。
参数量 57,657，训练 50 轮 / 66.6 分钟（RTX 4050）。

**与经典方法对比**（同一批样本、同一指标定义）：

| 方法 | 输出 SNR | 提升 |
|---|---|---|
| 含噪输入 | 6.19 | — |
| 带通 700–1600 Hz | 11.54 | +5.35 |
| 维纳滤波（盲噪声估计） | 11.07 | +4.88 |
| 带通 798–1500 Hz（**理想**，已知信号频带） | 12.63 | +6.44 |
| 谱减法（盲噪声估计） | 13.89 | +7.70 |
| **本模型** | **19.90** | **+13.71** |

比最强经典方法高 **6.0 dB**，比"作弊式"的理想带通高 **7.3 dB**。

### 2. 泛化能力（2×2 因子实验：波形 × 信道）

把 25 个信道与 20 组 chirp 同时分组，构造四个格子（详见 `docs/06_generalization_and_lightweight.html`）：

| | 见过的信道 (1–18) | 新信道 (22–25) |
|---|---|---|
| **见过的波形** (chirp 1–14) | 训练集 | testB **+11.93 dB** |
| **新波形** (chirp 18–20) | testA **+14.05 dB** | testC **+12.02 dB** |

- **新波形代价 ≈ 0.09 dB**（可忽略）
- **新信道代价：单次实验测 0.32 dB，但 3 折交叉验证测得 −0.05 ± 0.21 dB（不显著）** —— 见 §3

**0.32 dB 是怎么来的（已被 §3 修正）**：直接比 testA 与 testB 会得到 2.1 dB 差距，但那是错的 ——
测试用的 4 个信道时延扩展全落在分布高端（97.0/100.0/108.9/104.0 ms），本身就更难。
正确做法是**固定信道与波形、只换模型**（全部用 chirp 18–20）：

| 条件 | 平均增益 |
|---|---|
| v2 模型 · 信道 1–18（双方都见过） | 14.04 dB |
| v3 模型 · 信道 1–18（双方都见过） | 14.05 dB ← 两模型等价 |
| v2 模型 · 信道 22–25（仅 v2 见过） | 12.33 dB |
| v3 模型 · 信道 22–25（v3 未见过） | 12.02 dB |

→ 按此方法算得代价 **−0.32 dB**。**但这个数字只基于 4 个测试信道，落在信道抽样波动范围内**，
更严格的 3 折验证（§3）无法复现它。

### 3. 信道泛化的统计验证：3 折分层交叉验证

上一节只用了 4 个测试信道，统计强度不足。本节把 25 个信道**按 90% 时延扩展分层**
分成 3 组（三折的时延中位数分别为 74.1 / 72.9 / 77.5 ms，难度分布天然匹配），
轮流留出一组训练，然后在**同一批测试波形**（chirp 18–20）上对比：

| 折 | 留出的信道 | 增益（见过的信道） | 增益（没见过的信道） | 代价 |
|---|---|---|---|---|
| 1 | 9 个（1,6,8,11,14,15,20,22,24） | 13.22 | 13.22 | **+0.005 dB** |
| 2 | 8 个（9,10,12,16,17,19,21,23） | 13.48 | 13.20 | **−0.284 dB** |
| 3 | 8 个（2,3,4,5,7,13,18,25） | 13.29 | 13.42 | **+0.128 dB** |
| 合并 | 25 个（不重不漏） | 13.33 | 13.28 | **−0.055 dB** |

**信道新颖度代价 = −0.05 ± 0.21 dB**（3 折均值 ± 标准差），4,140 个测试样本**无一变差**。

**逐信道验证**（更有说服力）：只用「见过的信道」拟合
`增益 = 15.94 − 0.0378 × 时延扩展`（r = −0.87），再把「没见过的信道」放上去看偏离：

| 组 | 信道均值个数 | 相对拟合线的残差 |
|---|---|---|
| 见过的信道 | 44 | −0.000 dB |
| 没见过的信道 | 25 | **−0.110 dB**（标准误 0.189，t = −0.58，**不显著**） |

→ **结论：控制住信道难度后，「见过」与「没见过」的差别无法从噪声中分辨。**
模型学到的是可迁移的降噪映射，而不是记住 25 个具体信道。

> 独立复算：全部指标用另写的代码（自行实现降采样管线与 SNR/SI-SDR）重算过，
> 逐样本最大差 3e−06 dB（float32 存储误差），结论一致。

### 4. 轻量化：现有模型有约 10 倍冗余

给频域分支加宽度系数 `fch`，关闭时域分支，只训练频域分支：

| 变体 | 参数量 | testA | testB | testC | 三者均值 |
|---|---|---|---|---|---|
| 频域 0.5× | **702** | 13.58 | 11.39 | 11.37 | 12.11 |
| 频域 1× | 1,910 | 13.98 | 11.71 | 11.72 | 12.47 |
| **频域 2×** | **5,910** | 14.21 | 12.07 | 12.10 | **12.79** |
| 频域 4× | 20,246 | 14.38 | 12.11 | 12.14 | **12.88** |
| 频域 1× + 时域分支 | 57,657 | 14.05 | 11.93 | 12.02 | 12.67 |

**5,910 参数的纯频域模型（12.79）高于 57,657 参数的完整模型（12.67）。**
把时域分支那 5.5 万参数挪给频域分支是净收益。拐点在 5,910 附近。

**消融**：完整模型中时域分支的融合权重 `w_time` 从 0.297 一路衰减到 **0.006**，
模型自己弃用了它；删掉后仅损失 0.20 dB。

### 5. 物理洞察

在 25 个信道上，**90% 时延扩展与降噪增益强负相关，r = −0.862**：
时延扩展 5.8 ms 的信道增益 +14.62 dB，104–109 ms 的信道只有 +10.83 ~ +12.33 dB。
即**多径越强，降噪越难**（信号能量在时频面上越分散，与噪声重叠越多）。

### 6. 波形调制泛化：LFM vs HFM

指导老师的 README 写明目标是「**实测 HFM 信号**」（双曲调频），但既有全部数据集都用
**LFM**（线性调频）生成 —— 这是一个一直没正面处理的矛盾。本实验用**同一张 chirp 参数表、
同一批信道、同一份噪声实现**分别以 LFM / HFM 生成两套数据，构成**严格配对比较**：

| 方法 | LFM · testA | HFM · testA | LFM · testC | HFM · testC |
|---|---|---|---|---|
| 理想带通 798–1500 Hz | +6.44 | +6.44 | +6.43 | +6.43 |
| 谱减法（盲噪声估计） | +7.74 | +7.75 | +7.53 | +7.53 |
| 仅 LFM 训练 | +13.89 | **+13.73** | +11.81 | **+11.90** |
| 仅 HFM 训练 | +13.79 | +13.79 | +11.69 | +11.84 |
| **LFM + HFM 混合** | **+14.09** | **+14.01** | **+12.03** | **+12.14** |

**调制类型代价（HFM − LFM）**：

| 方法 | testA | testC |
|---|---|---|
| 理想带通（调制无关参照） | −0.00 | −0.00 |
| 仅 LFM 训练（3 个种子均值 ± 标准差） | **−0.178 ± 0.015** | **+0.080 ± 0.016** |
| 仅 HFM 训练 | −0.01 | +0.15 |
| LFM + HFM 混合 | −0.09 | +0.11 |

**配对性已实测验证**：LFM 与 HFM 的 `noisy − clean`（噪声实现）最大绝对差 **5e−14**
（浮点舍入量级），而 `clean_signal` 差 7~11 —— 只有调频规律不同。

**结论**：模型对调频规律基本不敏感。代价 0.1~0.2 dB 占 ~14 dB 总增益的 **1.3%**，
且方向在 testA / testC 上相反（不存在「HFM 整体更难」的趋势）。
用 3 个随机种子确认了它不是训练噪声（种子间标准差只有 0.015 dB）。
**用 LFM 训练不构成迁移到 HFM 的障碍。**
混合训练是稳妥选择：模型大小不变（仍 5,910 参数），四个测试集上全部取得最好成绩。

### 7. 脉冲噪声（导师方案）· 阶段0：现有模型的基线

导师指出「实测信号的脉冲噪声比较大」，要求把噪声模型改成
**`y = s + v + i`**（v 为 α 稳定分布背景噪声，i 为突发干扰），并给出了 PAM-Net 方案。
阶段0 **不改网络、不改损失**，只换数据与指标，先拿到干净基线。

**噪声模型**：`stblrnd.m`（Chambers-Mallows-Stuck 法生成 α 稳定分布，α=1.5、β=0，
经特征函数反推验证误差 ≤0.06%）+ `gen_burst.m`（泊松到达、随机幅度/时长，
四种波形覆盖"尖峰/脉冲/突发边缘"，同时输出干扰位置掩码）。
**噪声强度用 75% 分位数匹配**（α<2 时方差无穷，无法用功率定义 SNR）。
两套数据（α稳定 / 高斯）的**突发干扰实现完全相同**，构成配对对照。

| 模型 | 参数 | testA 输入 | testA 输出 | 帧增益中位 | 干净帧增益 | 干扰帧增益 | 干扰帧最差 |
|---|---|---|---|---|---|---|---|
| **α稳定+脉冲 · 纯频域** | **5,910** | −4.61 | **+12.51** | **15.06** | **14.13** | **25.09** | **13.54** |
| α稳定+脉冲 · 时域+频域 | 57,657 | −4.61 | +11.79 | 14.55 | 13.59 | 23.89 | 12.77 |
| 高斯+脉冲 · 纯频域 | 5,910 | −0.64 | +16.60 | 14.81 | 14.09 | 25.66 | 14.43 |

**三个结论**：

1. **现有模型在脉冲噪声下仍然有效**。干扰帧增益（25.09 dB）**高于**干净帧（14.13 dB）——
   脉冲幅度大（20×rms），压掉它们的 dB 收益自然高。
2. **α 稳定背景没有让降噪变难**。帧增益几乎不变（干净帧 +0.04 dB、干扰帧 −0.57 dB）；
   输出 SNR 差的 −4.08 dB **全部来自输入变差**（α稳定背景 rms 是信号的 2.2~2.6 倍，
   高斯只有 0.67 倍，重尾贡献了大量能量）。
3. **时域分支依然没用（原猜想被否定）**。高斯下 `w_time` 衰减到 0.006，
   脉冲噪声下放大到 **0.0410**（约 7 倍），但仍只占 4% 权重，且
   **加上它所有指标全面变差**（干扰帧 −1.20 dB）。
   **5,910 参数纯频域模型依然胜过 57,657 参数完整模型。**

> **对阶段1 的指导**：PAM-Net 的时域分支**不能照搬现有实现**。现有的是"整段残差估计器"，
> 对局部脉冲没有归纳偏置；方案要的是「短时分支识别尖峰/脉冲边缘」+ 逐点 `p_n`
> 脉冲置信度，形式不同。另外 α 稳定噪声下现有 L1+谱L1 损失**训练不稳定**
> （末 3 轮验证损失 0.41→0.63、验证 SNR 18.35→14.71），老师要求的 Huber 损失正是对策。

**指标问题**：全局 SNR 在脉冲噪声下会判错——自检显示"漏掉 1 个脉冲"的全局 SNR
（10.16）反而**高于**"好模型"（9.02）。因此主指标改用
**逐帧降噪增益 `frmgain` = 10log10(该帧输入噪声/输出噪声)**，并分解为
干净帧 / 干扰帧 / 干扰帧最差。同一组结果三种口径从 11.89 到 20.48 dB，相差 8.6 dB。

### 8. 脉冲噪声（导师方案）· 阶段1：PAM-Net 实现与验证

严格按导师文档第三节实现 **PAM-Net（Pulse-Aware Multi-scale Lightweight Network）**，
纯时域版 **8,906 参数**。模块逐一对应：

| 导师文档 | 本实现 | 具体形式 | 参数量 |
|---|---|---|---|
| 轻量化特征提取层（一维卷积+归一化+激活） | `FeatureExtractor` | 深度可分离 Conv1d(1→24,k=7)+Norm+ReLU | 104 |
| 短时尺度分支（识别尖峰/脉冲/突发边缘） | `ShortBranch` | k=3 两层，实测感受野 **13 点** | 2,808 |
| 长时尺度分支（波形连续性与时序结构） | `LongBranch` | k=7 + 膨胀率 1/4/16，感受野 **135 点** | 3,936 |
| 门控融合 `g=σ(W_g[F_s;F_l])` | `GatedFusion` | Conv1d(48→24,1)+Sigmoid | 1,176 |
| 脉冲感知分支 `p_n=σ(f_imp(F)_n)` | `PulseBranch` | 深度可分离 Conv1d + Conv1d(·→1) + Sigmoid | 433 |
| 残差重建 `ŝ=y−r̂` | `ResidualHead` | 深度可分离 Conv1d + Conv1d(·→1) | 449 |

**损失**：`L = L_Huber + λ₁·L_imp + λ₂·L_spec`，δ=0.2、λ₁=1.0、**λ₂=0.05**（由量级校准确定，
见下）。单元测试 8 项全过（含频域精修分支与 v3 `FrequencyBranch` **逐位一致**）。

#### 8.1 ⚠️ 方法学修正：实验协议本身会放大结论

训练日志显示验证指标大幅振荡。对全部 run 统计：逐轮 val 帧增益的标准差中位
**2.01 dB**、最大 3.15 dB；最佳轮次散布在 4~14 轮；「最佳轮 − 末5轮均值」中位 **2.51 dB**。
即 best-checkpoint 选择在挑"运气轮次"，其乐观偏差**与要论证的效应量同量级**。

**原因**：输入是重尾的（α 稳定背景 + 20×rms 脉冲），逐 batch 统计量差异极大，
BN 的 running statistics 被脉冲污染。同一 checkpoint 用不同统计量评估：

| 配置 | 帧增益中位 (dB) |
|---|---|
| 原 running stats（eval 模式） | 11.46 |
| 重校准到训练集统计 | 6.14 |
| batch stats（train 模式） | 7.46 |

**归一化统计量能把指标推动 5.3 dB** —— 足以解释逐轮 2~3 dB 的振荡。

**修正**：给全部卷积层加可切换归一化（`--norm bn|gn|in`）。GN/IN 不依赖 batch 统计，
train/eval 输出**逐位一致**（实测最大差 0.000；BN 为 0.84）。同配置只换归一化：

| 配置 | 归一化 | 逐轮标准差 | 最佳轮次 | 最佳−末5轮 |
|---|---|---|---|---|
| 纯时域 PAM-Net | BatchNorm | 2.450 dB | 11 | 3.80 dB |
| 纯时域 PAM-Net | **GroupNorm** | **0.677 dB** | **15（末轮）** | **0.05 dB** |
| 频域精修 + p 注入 | BatchNorm | 1.483 dB | 13 | 1.29 dB |
| 频域精修 + p 注入 | **GroupNorm** | **0.831 dB** | **15（末轮）** | **0.08 dB** |

振荡降到约 1/3.6，最佳轮次移到末轮（单调收敛）。**GN 在纯时域上性能反而更好**
（干扰帧 +2.8 dB）。**全部最终结果改用 GN 协议。**

#### 8.2 结果（GN 协议，多种子均值 ± 种子间标准差，testA）

| 模型 | 参数 | 干净帧增益 | 干扰帧增益 | 干扰帧最差 | 输出 SNR |
|---|---|---|---|---|---|
| 经典·理想带通 798–1500 Hz | — | 5.69 | 8.00 | 1.65 | −2.09 |
| 经典·盲谱减法 | — | 10.51 | 6.27 | 1.83 | −0.25 |
| 经典·维纳（盲） | — | 5.47 | 2.21 | 0.39 | −3.64 |
| 经典·中值 k=5 | — | −1.50 | 4.58 | −2.48 | −3.26 |
| 【阶段0】纯频域掩码（BN） | 5,910 | 14.13 | 25.09 | 13.54 | +12.51 |
| **纯时域 PAM-Net** | **8,906** | **10.85 ± 0.49** | **17.29 ± 0.69** | **9.80 ± 0.39** | — |
| 纯时域 − 脉冲分支 | 8,457 | 10.84 ± 1.04 | 17.19 ± 0.84 | 9.81 ± 0.52 | — |
| **频域精修 + p 注入（推荐）** | **14,858** | **18.27 ± 0.29** | **27.38 ± 0.16** | **17.12 ± 0.19** | — |
| 频域精修 − 脉冲分支 | 14,367 | 18.14 ± 0.22 | 26.59 ± 0.26 | 17.10 ± 0.23 | — |

（纯时域 n=3 种子；频域两行 n=5 种子）

#### 8.3 四条结论（全部有跨种子显著性检验支撑）

1. **PAM-Net 远超全部经典方法。** 纯时域版干扰帧增益 17.29 dB vs 经典最好 8.00；
   干扰帧最差 9.80 vs 1.83。
2. **但纯时域结构打不过频域掩码模型，干扰帧差 10.1 dB。** 原因是**结构先验**：
   信号是窄带 chirp（798–1500 Hz），而 α 稳定背景与突发脉冲都是宽带的，
   频域掩码天生等效于可学习的带通；纯时域残差估计必须从零学出这个带通。
3. **脉冲感知分支确实形成了判别能力，但性能贡献很小。**
   p_n 与真实干扰掩码的逐样本 **AUC = 0.953**（中位 0.976，97% 的样本 > 0.8），
   干扰位置平均置信度 0.828 vs 干净位置 0.0027 —— **确实学到了"哪里是脉冲"**。
   但接进网络后：
   - **纯时域结构里效应精确为零**（干扰帧 +0.095 ± 0.338 dB，跨 3 种子 p = 0.80）；
   - **只有在频域框架里才测得出**（干扰帧 **+0.790 ± 0.143 dB，p = 0.0052，n = 5 种子**；
     干净帧 +0.127 ± 0.082 dB，ns）；
   - **但这个量级的效应还不稳定**：testC 上显著的是干净帧（+0.260 ± 0.084，p = 0.036）
     而干扰帧不显著，与 testA 相反 —— 效应落在哪个指标上随测试集变化，
     说明它接近当前种子数下的可检测下限。
   - 解释：二维频域掩码本身就能捕捉脉冲在时频面上的竖直条纹结构，
     脉冲感知分支与它高度冗余。
4. **单种子 + BatchNorm 会把效应量夸大 2~6 倍。** BN 单种子曾给出
   "脉冲分支 +1.99/+1.27/+0.96 dB"，而 GN 5 种子只有 +0.127/+0.790 dB。

#### 8.4 两个如实记录的负面结论

- **Huber 不如普通 L1**（L1 高 +0.44/+0.79/+0.31 dB，样本级配对 p<1e−12）。
  δ=0.2 与残差尺度（rms≈0.21）相当，多数误差落在平方区，而 L1 本身就是稳健损失
  —— Huber 相对 **MSE** 的稳健性优势在这里用不上。
- **λ₂ 不能照搬既有代码的 0.5。** `torch.stft` 未归一化的幅度尺度是时域残差的
  20~25 倍，λ₂=0.5 会让谱项占总损失 90%+、淹没 Huber。
  λ₂=0.05 相对 0.5 高 +0.98/+1.07/+0.95 dB。

#### 8.5 两处"文档未明确、由本实现补齐"的地方（必须写进论文）

1. **脉冲感知分支"辅助特征融合"的形式**：文档只给 `p_n=σ(f_imp(F)_n)`，未给融合公式。
   初版用乘性调制 `F_fuse⊙(1+p_n)` **实测失败**——p 分离度持续变好，验证 SNR 却从
   +0.30 一路降到 −2.68 dB（L_imp 的梯度经乘性门灌进共享主干，"为拟合掩码而破坏重建"）。
   改为**拼接**后收敛正常。频域版把 p 按 hop 做 max-pool 对齐到 STFT 帧率、
   沿频率轴广播后作为掩码网络的第 2 个输入通道。
2. **感受野选取依据**：6 kHz 下突发时长 12~180 点，故长时分支取膨胀率 1/4/16（k=7）
   得 135 点感受野；短时分支 k=3 两层（13 点）定位突发边缘。

#### 8.6 独立复算

`verify_pamnet.py` 不复用训练脚本的任何实现（自己读 .mat、自己 `resample_poly(1,4)`、
自己实现三个指标），与训练脚本输出逐样本对比：**均值逐位吻合**，
最大差 2.7e−3 dB（STFT/ISTFT 在 float32 下的舍入放大）。150 样本**无一变差**。

## 二、目录结构

```
docs/                            分析报告（HTML，图片外链到 docs/figures/）
  01_code_and_data_diagnosis.html    代码与数据诊断
  02_dachuang_code_diagnosis.html    任务定义诊断（"背波形"问题的发现）
  03_data_pipeline_diagnosis.html    数据生成链路诊断（信道频带错位）
  04_training_result_v2.html         训练结果分析（含经典方法对比、方差分解）
  05_generalization_experiment.html  2×2 泛化实验
  06_generalization_and_lightweight.html  泛化 + 轻量化
  07_channel_generalization_3fold.html    信道泛化的 3 折交叉验证
  08_waveform_modulation_generalization.html  波形调制泛化 LFM vs HFM
  09_pulse_noise_stage0.html              脉冲噪声阶段0
  10_pamnet_stage1.html                   PAM-Net 阶段1（最新；含归一化诊断与多种子统计）
  figures/v2/  figures/v3/  figures/folds/  figures/wave/  figures/pam/
  figures/pamnet/（图 J 消融 / K 脉冲诊断 / L 稳定性 / M 多种子 / N 归一化对照）

code/
  matlab/          数据生成与验证脚本
    make_dataset.m        生成 train/val/test（v2，按 chirp 分组）
    make_dataset_v3.m     生成 2×2 因子数据集（v3，按 chirp + 信道分组）
    make_dataset_gen.m    通用数据集生成器（任意 chirp/信道组合）
    make_folds.m          生成 3 折分层交叉验证数据集（按时延扩展分层）
    make_dataset_wave.m   生成 LFM / HFM 配对数据集（波形泛化实验）
    make_dataset_pam.m    生成 α稳定+突发干扰数据集（导师方案）
    stblrnd.m             α 稳定分布随机数（Chambers-Mallows-Stuck 法）
    gen_burst.m           突发脉冲干扰生成器（含干扰位置掩码）
    test_*.m              噪声模型验证脚本
    gen_chirp.m           LFM/HFM 扫频信号生成器
    gen_ch.m              信道冲激响应生成（修正版）
    fix_channel_gen.m     修正信道生成的通带错位问题
    fix_make_dataset.m    修正数据生成（chirp 重采样后再卷积）
    verify_*.m            各类验证脚本
  python/
    train_v2.py      训练脚本（v2 数据集）
    train_v3.py      训练 + 2×2 评估 + 消融 + 尺寸扫描（支持 --fch / --time / --freq）
    train_fold.py    3 折留出实验训练脚本（seen / unseen 同波形对比）
    train_wave.py    波形泛化训练脚本（支持多目录混合训练、--seed）
    train_pam.py     脉冲噪声训练脚本（带干扰掩码、稳健指标）
    pamnet.py        PAM-Net 网络与联合损失（导师方案，含全部消融开关）
    train_pamnet.py  PAM-Net 训练/消融脚本（复用 train_pam 的数据与评估口径）
  analysis/         分析与出图脚本（纯 Python，含自写 .mat 解析器）
    mat_probe.py          纯 Python MATLAB v5 .mat 解析器（无 numpy 依赖）
    compare_baseline.py   与经典降噪方法对比
    channel_gain.py       信道物理特性与增益的关系
    verify_refactor.py    重构安全性验证
    gen_penalty.py        信道新颖度代价的严格控制测量
    check_bandpass_paradox.py  带通口径排查
    analyze_folds.py      3 折结果汇总与出图
    verify_folds.py       3 折结果的独立复算（自行实现管线与指标）
    make_fig_folds2.py    逐信道增益 vs 时延扩展散点图
    analyze_wave.py       波形泛化汇总 + 经典基线 + 出图
    verify_wave.py        波形实验独立复算 + 配对性验证
    make_fig_wave_spec.py LFM/HFM 时频图对比
    pam_metrics.py        脉冲噪声下的稳健评估指标（逐帧降噪增益）
    verify_pam_metrics.py 指标行为验证（证明全局 SNR 会判错）
    verify_pam_dataset.py 数据集配对性/脉冲性验证
    analyze_pam.py        阶段0 结果汇总与出图
    verify_pam_result.py  阶段0 独立复算
    test_pamnet.py        PAM-Net 单元测试（参数量/形状/梯度/损失/感受野）
    calibrate_loss.py     损失项量级校准（确定 λ2）
    analyze_pamnet.py     PAM-Net 消融汇总与出图
    make_fig_pamnet_pulse.py  脉冲感知分支诊断（p_n vs 真实掩码，AUC）
    verify_pamnet.py      PAM-Net 独立复算
    paired_test_pamnet.py 变体间差异的配对显著性检验
    compare_baseline_pam.py   脉冲噪声下的经典基线（带通/谱减/维纳/中值）
    diagnose_bn.py        BatchNorm 统计失配诊断（解释训练振荡来源）
    analyze_stability.py  逐轮振荡与 best-checkpoint 选择偏差
    analyze_seeds.py      多种子聚合 + 跨种子显著性检验
    make_fig_seeds.py     图 M（多种子）/ 图 N（归一化对照）
    make_figs*.py / build_report*.py   出图与报告生成
  exploratory/      探索性脚本（含一条被否决的"抢救旧信道"路线）

results/
  metrics/   逐样本结果 CSV（snr_in, snr_out, sisdr_in, sisdr_out）与汇总
  logs/      逐轮训练日志
  data/      分析中间结果（信道映射、扫描汇总 JSON、3 折汇总、波形泛化汇总）

models/
  model_fch2.pth        推荐的轻量模型（5,910 参数）
  model_full.pth        完整模型（57,657 参数）
  model_lite_fold{1,2,3}.pth   3 折实验的三个模型（各 5,910 参数）
  model_{lfmonly,hfmonly,mixed}.pth        波形泛化三个模型（各 5,910 参数）
  model_lfmonly_s{2,3}.pth                 额外随机种子
  model_pam_{freq,full,freq_g}.pth         脉冲噪声阶段0 三个模型
  model_pamnet*.pth                        PAM-Net 及其 9 个消融变体（各 ~60 KB）
```

---

## 三、复现步骤

### 环境
- Python 3.11 + `torch 2.8.0+cu128` + `numpy` / `scipy` / `matplotlib`
- MATLAB R2025b（需要 Signal Processing Toolbox；数据生成还需要 RAM 声场计算程序）

### 步骤

1. **生成信道**（需要 RAM 声场计算程序，未包含在本仓库）
   ```matlab
   gen_ch(1:25, 10, 'D:/project/信道产生/data_fixed')     % 25 个信道
   ```
   自检：新信道的 99% 能量应落在归一化频率 **0.0333 ~ 0.0625**。

2. **生成数据集**
   ```matlab
   make_dataset_v3          % 2×2 因子数据集 -> D:/project/dataset_v3/
   ```

3. **训练与评估**
   ```bash
   python code/python/train_v3.py --tag full                          # 57,657 参数
   python code/python/train_v3.py --tag fch2 --time 0 --fch 2.0       # 5,910 参数（推荐）
   python code/python/train_v3.py --tag freqonly --time 0             # 1,910 参数
   python code/python/train_v3.py --tag timeonly --freq 0             # 55,745 参数
   ```

4. **3 折交叉验证（信道泛化）**
   ```matlab
   make_folds                % 按时延扩展分层，生成 3 折留出数据集 -> D:/project/dataset_folds/
   ```
   ```bash
   # 串行执行（本机 6 GB 显存跑不下并行）
   for k in 1 2 3; do
     python code/python/train_fold.py --fold $k --epochs 15 --workers 4 \
         --tag lite_fold$k --time 0 --fch 2.0 --outdir D:/project/output_folds
   done
   python code/analysis/analyze_folds.py     # 汇总 + 出图
   python code/analysis/verify_folds.py      # 独立复算（可选）
   ```

5. **波形调制泛化（LFM vs HFM）**
   ```matlab
   make_dataset_wave         % LFM / HFM 配对数据集 -> D:/project/dataset_wave/
   ```
   ```bash
   W=D:/project/dataset_wave
   python code/python/train_wave.py --tag lfmonly --train $W/LFM/train --val $W/LFM/val \
       --epochs 15 --time 0 --fch 2.0
   python code/python/train_wave.py --tag hfmonly --train $W/HFM/train --val $W/HFM/val \
       --epochs 15 --time 0 --fch 2.0
   python code/python/train_wave.py --tag mixed \
       --train $W/LFM/train,$W/HFM/train --val $W/LFM/val,$W/HFM/val \
       --epochs 15 --time 0 --fch 2.0
   for s in 2 3; do   # 额外随机种子（用于量化运行间波动）
     python code/python/train_wave.py --tag lfmonly_s$s --train $W/LFM/train \
         --val $W/LFM/val --epochs 15 --time 0 --fch 2.0 --seed $s
   done
   python code/analysis/analyze_wave.py       # 汇总 + 经典基线 + 出图
   python code/analysis/verify_wave.py        # 独立复算 + 配对性验证
   python code/analysis/make_fig_wave_spec.py # 时频图
   ```

6. **脉冲噪声（导师方案）阶段0**
   ```matlab
   make_dataset_pam('alphastable','D:/project/dataset_pam/alphastable',1.5);
   make_dataset_pam('gauss',      'D:/project/dataset_pam/gauss',1.5);
   ```
   ```bash
   D=D:/project/dataset_pam
   python code/python/train_pam.py --tag pam_freq   --data $D/alphastable --epochs 15 --time 0 --fch 2.0
   python code/python/train_pam.py --tag pam_full   --data $D/alphastable --epochs 15 --time 1 --freq 1 --fch 1.0
   python code/python/train_pam.py --tag pam_freq_g --data $D/gauss      --epochs 15 --time 0 --fch 2.0
   python code/analysis/analyze_pam.py        # 汇总 + 出图
   python code/analysis/verify_pam_result.py  # 独立复算
   ```

7. **脉冲噪声阶段1：PAM-Net**
   ```bash
   python code/analysis/test_pamnet.py         # 单元测试（8 项，必须全过）
   python code/analysis/calibrate_loss.py      # 确定 λ2
   D=D:/project/dataset_pam/alphastable
   # 消融（每项只改一个开关；全部串行）
   python code/python/train_pamnet.py --tag pamnet             --data $D --epochs 15
   python code/python/train_pamnet.py --tag pamnet_nopulse     --data $D --epochs 15 --pulse 0
   python code/python/train_pamnet.py --tag pamnet_freq        --data $D --epochs 15 --freq-refine 2.0
   python code/python/train_pamnet.py --tag pamnet_freq_nopulse --data $D --epochs 15 --freq-refine 2.0 --pulse 0
   python code/python/train_pamnet.py --tag pamnet_freq_pf     --data $D --epochs 15 --freq-refine 2.0 --pulse-freq 1
   python code/python/train_pamnet.py --tag pamnet_nogate      --data $D --epochs 15 --gate 0
   python code/python/train_pamnet.py --tag pamnet_short       --data $D --epochs 15 --long 0
   python code/python/train_pamnet.py --tag pamnet_long        --data $D --epochs 15 --short 0
   python code/python/train_pamnet.py --tag pamnet_l1          --data $D --epochs 15 --loss l1
   python code/python/train_pamnet.py --tag pamnet_l2big       --data $D --epochs 15 --lam-spec 0.5
   python code/analysis/analyze_pamnet.py
   python code/analysis/make_fig_pamnet_pulse.py \
       --ckpt D:/project/output_pamnet/model_pamnet_freq_pf.pth --freq-refine 2.0 --pulse-freq 1
   python code/analysis/paired_test_pamnet.py
   python code/analysis/verify_pamnet.py --ckpt D:/project/output_pamnet/model_pamnet_freq_pf.pth \
       --tag pamnet_freq_pf --freq-refine 2.0 --pulse-freq 1
   python code/analysis/compare_baseline_pam.py
   ```

8. **分析出图**：按 `code/analysis/` 中脚本头部的说明运行（路径为绝对路径，需按本机调整）。

---

## 四、数据说明

**本仓库不包含数据集**（原始数据约 5 GB）。数据由上面的 MATLAB 脚本生成，来源是：

```
干净信号  = 扫频信号（798 → 1500 Hz，0.6–1.0 s，fs = 9600 Hz）
            调频规律 LFM（线性）或 HFM（双曲），由 gen_chirp.m 的 modtype 控制
含噪信号  = conv(重采样到 24000 Hz 的 chirp, RAM 仿真信道) + 高斯噪声
样本长度  = 22,399 ~ 31,999 点（24000 Hz），训练时降采样 4 倍到 6000 Hz
```

波形泛化实验（`dataset_wave/`）用**同一张 chirp 参数表、同一批信道、同一份噪声实现**
分别生成 LFM 与 HFM 两套数据，构成严格配对比较。

每个 `.mat` 样本包含：`noisy_signal`、`clean_signal`、`true_snr`、`chirp_id`、`chan_id`。

**注意**：老师提供的 RAM 声场仿真代码（`Input_PE.m`、`SeaWater_attenu.m`、
`WriteenvRam.m`、`PEsim_main_parfor.m` 及 `ramgeo1.5_atten.exe`）**未包含在本仓库**，
需要向指导老师索取。

---

## 五、局限（如实记录）

1. **全部是仿真数据**：RAM 仿真信道 + 合成扫频信号，**不是实测**。
   脉冲噪声部分（第 7 节）的 α 稳定背景与突发干扰也都是合成的，
   α 只取了一个值（1.5）、突发幅度/时长范围是设定的，没有实测统计支撑。
   老师提到正在整理实测信号，届时需重新验证。
   关于"任务说 HFM、数据是 LFM"这个矛盾，第 6 节的配对实验证明
   **LFM 训练不构成迁移到 HFM 的障碍**（代价 ≤0.2 dB），但 HFM 数据仍是合成的，
   **不能替代实测验证** —— 真实 HFM 的噪声统计、多普勒、时变信道都不在这里。
2. **信道数量仍然偏少**：只有 25 个（来自同一 RAM 仿真参数空间）。
   3 折交叉验证已把它用到极限（每折留出 8–9 个），但信道的物理多样性受限于仿真参数组合。
   结论成立的前提是**部署信道与训练信道来自同一分布**，跨分布的泛化未验证。
3. **尺寸扫描与 3 折实验每个档位只跑一个随机种子**（3 折之间的差异已反映主要不确定性，
   但训练随机性未单独分离）。波形泛化实验补了 3 个种子（标准差 0.015 dB）。
4. **信道是静态的**，未测试时变信道。
5. **波形泛化实验只覆盖同一张 chirp 参数表**：LFM 与 HFM 的中心频率、带宽、时宽、包络
   全部相同，只有调频规律不同。真实 HFM 信号的参数分布可能不同，跨频带泛化未验证。
6. **PAM-Net 的对比含多个变量**：与阶段0 的对比同时改变了架构与损失函数
   （PAM-Net 必须配套新损失），因此不能把提升单独归因于架构；消融表内部
   （同一损失下改架构）才是干净的因果证据。
7. **配对显著性检验只消除样本抽样不确定性**，不消除训练随机性（未做多随机种子重复）。
   且训练过程本身不稳定：验证帧增益在 15 轮中出现大幅振荡（如 10.41→4.00→10.91→4.20），
   本实验用 best-checkpoint 选择缓解，但未系统解决。
8. **λ₂ 由量级校准确定，不是完整网格搜索**（只对照了 0.05 与 0.5 两个取值）。
9. **脉冲经多径信道后的形态未单独分析**：第 7 节把突发干扰直接叠加在信道**输出**上；
   若干扰发生在信道**之前**，经多径会展宽成一串回声突发，难度完全不同。该场景未测。
   另外降采样 4 倍对脉冲噪声的增益远小于对高斯噪声的 +6.19 dB，但未做定量分解。
10. **两个前端处理需要交代**：
   - 样本中位有效长度只有 6,031 / 8,000 点，**约 24.6% 是零填充**；
   - 降采样 4 倍会滤掉带外噪声，使输入 SNR 从数据集的 0 dB 抬到 6.19 dB。
     这是合理的前端设计（信号只占 798–1500 Hz），但**部署到实测数据时必须做同样的降采样**。

---

## 六、致谢与说明

- 信道仿真基于指导老师提供的 RAM 声场计算程序与 `PEsim_main_parfor.m` 框架。
- 本项目修正了原数据生成链路中的两个问题：**信道通带 bin 索引与频率步进 `fct` 的错误耦合**
  （导致通带从 798–1500 Hz 被搬到 237–447 Hz），以及 **chirp 未重采样就直接与信道卷积**。
  详见 `docs/03_data_pipeline_diagnosis.html`。
- 报告中所有数字均在真实项目数据上实测，可由仓库内脚本复现。
