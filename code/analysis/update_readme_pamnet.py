# -*- coding: utf-8 -*-
"""更新 _repo/README.md：加入阶段1（PAM-Net）章节 + 目录结构 + 复现步骤 + 局限"""
import io, os

p = r"E:/workbuddy/store/2026-10-12-22-50-08/_repo/README.md"
s = io.open(p, encoding="utf-8").read()

# ============ 1) 新增第 8 节 ============
anchor = "---\n\n## 二、目录结构"
sec8 = """### 8. 脉冲噪声（导师方案）· 阶段1：PAM-Net 实现与验证

严格按导师文档第三节实现 **PAM-Net（Pulse-Aware Multi-scale Lightweight Network）**，
共 **8,906 参数**（纯时域版）。模块逐一对应：

| 导师文档 | 本实现 | 具体形式 | 参数量 |
|---|---|---|---|
| 轻量化特征提取层（一维卷积+归一化+激活） | `FeatureExtractor` | 深度可分离 Conv1d(1→24,k=7)+BN+ReLU | 104 |
| 短时尺度分支（识别尖峰/脉冲/突发边缘） | `ShortBranch` | k=3 两层，实测感受野 **13 点** | 2,808 |
| 长时尺度分支（波形连续性与时序结构） | `LongBranch` | k=7 + 膨胀率 1/4/16，感受野 **135 点** | 3,936 |
| 门控融合 `g=σ(W_g[F_s;F_l])` | `GatedFusion` | Conv1d(48→24,1)+Sigmoid | 1,176 |
| 脉冲感知分支 `p_n=σ(f_imp(F)_n)` | `PulseBranch` | 深度可分离 Conv1d + Conv1d(·→1) + Sigmoid | 433 |
| 残差重建 `ŝ=y−r̂` | `ResidualHead` | 深度可分离 Conv1d + Conv1d(·→1) | 449 |

**损失**：`L = L_Huber + λ₁·L_imp + λ₂·L_spec`，δ=0.2、λ₁=1.0、**λ₂=0.05**。

**结果（testA：新波形 + 见过的信道；同一份数据集、同一套评估口径）**

| 模型 | 参数 | 干净帧增益 | 干扰帧增益 | 干扰帧最差 | 输出SNR |
|---|---|---|---|---|---|
| 经典·理想带通 798–1500 Hz | — | 5.69 | 8.00 | 1.65 | −2.09 |
| 经典·盲谱减法 | — | 10.51 | 6.27 | 1.83 | −0.25 |
| 【阶段0】纯频域掩码 | 5,910 | 14.13 | 25.09 | 13.54 | +12.51 |
| 【阶段0】时域+频域 | 57,657 | 13.59 | 23.89 | 12.77 | — |
| **PAM-Net 纯时域（按方案原样）** | **8,906** | 10.90 | 15.01 | 9.59 | +7.46 |
| PAM-Net + 后置频域精修 | 14,816 | 16.38 | 26.70 | 15.75 | +15.51 |
| **PAM-Net + p 注入频域（推荐）** | **14,858** | **18.49** | **27.79** | **16.93** | **+17.83** |

**四条结论**（均有配对显著性检验支撑，n=1080，逐样本配对）：

1. **PAM-Net 远超全部经典方法。** 干扰帧增益 15.01 dB vs 理想带通 8.00 / 盲谱减 6.27 /
   维纳 2.21 / 中值 −0.40；干扰帧最差 9.59 vs 经典最好 1.83。
2. **但纯时域结构打不过同等训练的频域掩码模型**（干净帧 −3.23、干扰帧 −10.08、
   最差 −3.95 dB，全部 p<1e−77）。原因是**结构先验**：信号是窄带 chirp（798–1500 Hz），
   而 α 稳定背景与突发脉冲都是宽带的，频域掩码天生等效于一个可学习的带通；
   纯时域残差估计必须从零学出这个带通。这与阶段0"频域分支贡献几乎全部性能"一致。
3. **脉冲感知机制确实形成了判别能力。** p_n 与真实干扰掩码的逐样本 **AUC 均值 0.943**
   （中位 0.969，94.6% 的样本 >0.8）；干扰位置平均置信度 0.806（中位 0.9998）、
   干净位置 0.0021。**不是输出常数。**
4. **但必须嵌在频域框架里用，且集成位置很关键。** 把 p 作为附加通道**喂进频域掩码**
   （而不是事后修形）后，相对同参数的"频域精修−脉冲分支"提升
   **干净帧 +1.99 / 干扰帧 +1.27 / 最差 +0.96 dB**（三项 p<1e−31，d_z=1.16/0.54/0.38）；
   而"后置频域精修"里该分支的贡献检测不到（±0.25 dB）。

**消融（同一数据、同一超参、同一口径，每项只改一个组件；Δ 相对阶段0 纯频域基线）**

| 变体 | 参数 | 干净帧 | 干扰帧 | 最差 | Δ干扰帧 |
|---|---|---|---|---|---|
| PAM-Net 完整（纯时域，参照） | 8,906 | 10.90 | 15.01 | 9.59 | −10.08 |
| 纯时域 − 脉冲感知分支 | 8,457 | 11.28 | 14.72 | 9.38 | −10.37 |
| − 门控（改简单平均） | 7,730 | 9.50 | 11.48 | 8.01 | −13.62 |
| 仅短时分支 | 3,794 | 7.50 | 11.99 | 6.83 | −13.11 |
| 仅长时分支 | 4,922 | 9.11 | 13.41 | 8.51 | −11.69 |
| 损失换 L1（去 Huber） | 8,906 | 11.34 | 15.80 | 9.89 | −9.29 |
| λ₂=0.5（谱项主导） | 8,906 | 9.92 | 13.94 | 8.63 | −11.16 |

**两个"如实记录"的负面结论**：

- **Huber 不如普通 L1。** 换成 L1 后干净帧 +0.44 / 干扰帧 +0.79 / 最差 +0.31 dB
  （p<1e−12）。原因是 δ=0.2 与残差尺度（rms≈0.21）相当，绝大多数误差落在平方区，
  而 L1 本身就是稳健损失——Huber 相对 MSE 的稳健性优势在这里用不上。
- **λ₂ 不能照搬既有代码的 0.5。** `torch.stft` 未归一化的幅度尺度是时域残差的 20~25 倍，
  λ₂=0.5 会让谱项占总损失 90%+、淹没 Huber。λ₂=0.05 相对 0.5 提升
  干净帧 +0.98 / 干扰帧 +1.07 / 最差 +0.95 dB（p<1e−76）。

**两处"文档未明确、由本实现补齐"的地方（必须写进论文）**：

1. **脉冲感知分支"辅助特征融合"的形式**：文档只给 `p_n=σ(f_imp(F)_n)`，未给融合公式。
   初版用乘性调制 `F_fuse⊙(1+p_n)` **实测失败**——p 分离度持续变好，验证 SNR 却从
   +0.30 一路降到 −2.68 dB（L_imp 的梯度经乘性门灌进共享主干，"为拟合掩码而破坏重建"）。
   改为**拼接** `r̂=f_θ([F_fuse;p_n])` 后验证帧增益单调上升。
2. **感受野选取依据**：6 kHz 下突发时长 12~180 点，故长时分支取膨胀率 1/4/16（k=7）
   得 135 点感受野；短时分支 k=3 两层（13 点）定位突发边缘，两者相差 10 倍。

**模型选择准则的修正**：不能用 val_loss 选模型（被谱项主导，且全局类指标在脉冲噪声下
会判反）。改用**验证集逐帧降噪增益中位数**，与最终评估口径一致。

**独立复算**：`verify_pamnet.py` 不复用训练脚本的任何实现（自己读 .mat、
自己 `resample_poly(1,4)`、自己实现 SNR/SI-SDR/逐帧增益），与训练脚本输出逐样本对比，
均值逐位吻合；残差 2.4e−3 dB 来自 STFT/ISTFT 在 float32 下的舍入放大。
150 个样本**无一变差**（最差 +4.53 dB）。

---

## 二、目录结构"""
assert anchor in s, "未找到锚点 1"
s = s.replace(anchor, sec8, 1)

# ============ 2) 目录结构 ============
pairs = [
    ("""  09_pulse_noise_stage0.html              脉冲噪声阶段0（最新）
  figures/v2/  figures/v3/  figures/folds/  figures/wave/  figures/pam/""",
     """  09_pulse_noise_stage0.html              脉冲噪声阶段0
  10_pamnet_stage1.html                   PAM-Net 阶段1（最新）
  figures/v2/  figures/v3/  figures/folds/  figures/wave/  figures/pam/
  figures/pamnet/"""),
    ("""    train_pam.py     脉冲噪声训练脚本（带干扰掩码、稳健指标）""",
     """    train_pam.py     脉冲噪声训练脚本（带干扰掩码、稳健指标）
    pamnet.py        PAM-Net 网络与联合损失（导师方案，含全部消融开关）
    train_pamnet.py  PAM-Net 训练/消融脚本（复用 train_pam 的数据与评估口径）"""),
    ("""    analyze_pam.py        阶段0 结果汇总与出图
    verify_pam_result.py  阶段0 独立复算""",
     """    analyze_pam.py        阶段0 结果汇总与出图
    verify_pam_result.py  阶段0 独立复算
    test_pamnet.py        PAM-Net 单元测试（参数量/形状/梯度/损失/感受野）
    calibrate_loss.py     损失项量级校准（确定 λ2）
    analyze_pamnet.py     PAM-Net 消融汇总与出图
    make_fig_pamnet_pulse.py  脉冲感知分支诊断（p_n vs 真实掩码，AUC）
    verify_pamnet.py      PAM-Net 独立复算
    paired_test_pamnet.py 变体间差异的配对显著性检验
    compare_baseline_pam.py   脉冲噪声下的经典基线（带通/谱减/维纳/中值）"""),
    ("""  model_pam_{freq,full,freq_g}.pth         脉冲噪声阶段0 三个模型""",
     """  model_pam_{freq,full,freq_g}.pth         脉冲噪声阶段0 三个模型
  model_pamnet*.pth                        PAM-Net 及其 9 个消融变体（各 ~60 KB）"""),
]
for i, (a, b) in enumerate(pairs, 1):
    assert a in s, "目录结构第 %d 处未匹配" % i
    s = s.replace(a, b)

# ============ 3) 复现步骤 ============
old = """7. **分析出图**：按 `code/analysis/` 中脚本头部的说明运行（路径为绝对路径，需按本机调整）。"""
new = """7. **脉冲噪声阶段1：PAM-Net**
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
   python code/analysis/make_fig_pamnet_pulse.py \\
       --ckpt D:/project/output_pamnet/model_pamnet_freq_pf.pth --freq-refine 2.0 --pulse-freq 1
   python code/analysis/paired_test_pamnet.py
   python code/analysis/verify_pamnet.py --ckpt D:/project/output_pamnet/model_pamnet_freq_pf.pth \\
       --tag pamnet_freq_pf --freq-refine 2.0 --pulse-freq 1
   python code/analysis/compare_baseline_pam.py
   ```

8. **分析出图**：按 `code/analysis/` 中脚本头部的说明运行（路径为绝对路径，需按本机调整）。"""
assert old in s, "未找到复现步骤锚点"
s = s.replace(old, new)

# ============ 4) 局限 ============
old2 = """6. **脉冲经多径信道后的形态未单独分析**"""
new2 = """6. **PAM-Net 的对比含多个变量**：与阶段0 的对比同时改变了架构与损失函数
   （PAM-Net 必须配套新损失），因此不能把提升单独归因于架构；消融表内部
   （同一损失下改架构）才是干净的因果证据。
7. **配对显著性检验只消除样本抽样不确定性**，不消除训练随机性（未做多随机种子重复）。
   且训练过程本身不稳定：验证帧增益在 15 轮中出现大幅振荡（如 10.41→4.00→10.91→4.20），
   本实验用 best-checkpoint 选择缓解，但未系统解决。
8. **λ₂ 由量级校准确定，不是完整网格搜索**（只对照了 0.05 与 0.5 两个取值）。
9. **脉冲经多径信道后的形态未单独分析**"""
assert old2 in s, "未找到局限锚点"
s = s.replace(old2, new2)
# 原第 6 条的后续编号顺延
s = s.replace("""   另外降采样 4 倍对脉冲噪声的增益远小于对高斯噪声的 +6.19 dB，但未做定量分解。
7. **两个前端处理需要交代**：""",
              """   另外降采样 4 倍对脉冲噪声的增益远小于对高斯噪声的 +6.19 dB，但未做定量分解。
10. **两个前端处理需要交代**：""")

io.open(p, "w", encoding="utf-8").write(s)
print("README 已更新，%.1f KB" % (len(s.encode("utf-8")) / 1024))
