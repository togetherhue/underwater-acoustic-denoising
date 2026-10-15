# -*- coding: utf-8 -*-
"""
PAM-Net：脉冲感知多尺度轻量化网络（Pulse-Aware Multi-scale Lightweight Network）
================================================================================
严格按导师《面向非高斯脉冲噪声与突发干扰的轻量化多尺度水声信号降噪方法》
的第三节实现。信号模型 y_n = s_n + v_n + i_n（v 为 α 稳定背景噪声，
i 为突发干扰）。

模块对照（导师文档 → 本文件）
--------------------------------------------------------------------------------
  「轻量化特征提取层  一维卷积 + 归一化 + 激活」
        -> FeatureExtractor    : DSConv1d(1 -> C, k=7)

  「短时尺度分支  F_s = DWConv_{k_s}(F)  识别尖峰、脉冲和突发边缘」
        -> ShortBranch         : 小核 k=3 的两层深度可分离卷积，感受野 5

  「长时尺度分支  F_l = DWConv_{k_l,d_l}(F)  提取波形连续性和时序结构」
        -> LongBranch          : k=7 + 膨胀率 1/4/16，感受野 127

  「门控融合  g = σ(W_g·[F_s;F_l])，F_fuse = g⊙F_s + (1−g)⊙F_l」
        -> GatedFusion

  「脉冲感知分支  p_n = σ(f_imp(F)_n)，预测各采样位置受脉冲干扰的程度」
        -> PulseBranch         : 输出 [B,1,L]，逐点置信度

  「残差重建  r̂_n = f_θ(y)_n,  ŝ_n = y_n − r̂_n」
        -> ResidualHead + PAMNet.forward

【实现约定（文档未明确、由本实现补齐的地方，均在报告中列出）】
1. 脉冲感知分支"辅助特征融合"的具体形式：文档只说"利用该分支辅助特征融合"。
   本实现把 p_n 作为【附加通道】拼接到融合特征后送入残差估计头，即
   r̂ = f_θ([F_fuse ; p_n])。选择拼接而非乘性调制 (1+p)⊙F_fuse 是实验结论：
   乘性形式会把 L_imp 的梯度直接灌进共享主干，导致"为了拟合掩码而破坏重建"
   （首轮实验实测：p 的分离度持续变好的同时验证 SNR 反而从 +0.30 降到 −2.68 dB）。
2. 感受野按"脉冲/突发边缘"的实际尺度选取：数据在 6 kHz 下突发时长为
   12~180 点，故长时分支感受野取 127 点（≈21 ms）覆盖绝大多数突发。
3. 归一化用 BatchNorm（与既有 v2/v3 实现一致，便于对照）。
4. 可选频域精修分支（--freq-refine）：阶段0 发现频域掩码在脉冲噪声下依然强势，
   而导师方案是纯时域残差估计。该开关用于严格检验这一张力，默认关闭。

【消融开关】use_short / use_long / use_gate / use_pulse / loss_kind
================================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================== 基础模块 ==================================
class DSConv1d(nn.Module):
    """深度可分离一维卷积：DWConv -> Pointwise -> BN -> (ReLU)

    padding = d*(k-1)//2 保证"same"长度（仅支持奇数 k）。
    """

    def __init__(self, cin, cout, k=3, d=1, act=True):
        super().__init__()
        assert k % 2 == 1, "只支持奇数卷积核"
        self.dw = nn.Conv1d(cin, cin, k, padding=d * (k - 1) // 2, dilation=d, groups=cin)
        self.pw = nn.Conv1d(cin, cout, 1)
        self.bn = nn.BatchNorm1d(cout)
        self.act = act

    def forward(self, x):
        y = self.bn(self.pw(self.dw(x)))
        return F.relu(y) if self.act else y


class DSConv2d(nn.Module):
    """深度可分离二维卷积（仅供可选的频域精修分支使用）"""

    def __init__(self, cin, cout, k=3, p=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, padding=p, groups=cin)
        self.pw = nn.Conv2d(cin, cout, 1)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))))


# ============================== PAM-Net 组件 ==============================
class FeatureExtractor(nn.Module):
    """轻量化特征提取层：一维卷积 + 归一化 + 激活"""

    def __init__(self, c):
        super().__init__()
        self.net = DSConv1d(1, c, k=7, d=1)

    def forward(self, x):
        return self.net(x)


class ShortBranch(nn.Module):
    """短时尺度分支：小核（k=3）两层，感受野 5，抓尖峰/脉冲/突发边缘"""

    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            DSConv1d(c, 2 * c, k=3, d=1),
            DSConv1d(2 * c, c, k=3, d=1))

    def forward(self, x):
        return self.net(x)


class LongBranch(nn.Module):
    """长时尺度分支：大核（k=7）+ 膨胀率 1/4/16，感受野 127，抓波形连续性与时序结构"""

    def __init__(self, c, dilations=(1, 4, 16)):
        super().__init__()
        layers, cin = [], c
        for i, d in enumerate(dilations):
            cout = 2 * c if i == 0 else c
            layers.append(DSConv1d(cin, cout, k=7, d=d))
            cin = cout
        self.net = nn.Sequential(*layers)
        self.dilations = tuple(dilations)

    def forward(self, x):
        return self.net(x)


class GatedFusion(nn.Module):
    """门控融合：g = σ(W_g·[F_s;F_l])，F_fuse = g⊙F_s + (1−g)⊙F_l"""

    def __init__(self, c):
        super().__init__()
        self.wg = nn.Conv1d(2 * c, c, 1)

    def forward(self, f_s, f_l):
        g = torch.sigmoid(self.wg(torch.cat([f_s, f_l], dim=1)))
        return g * f_s + (1.0 - g) * f_l, g


class PulseBranch(nn.Module):
    """脉冲感知分支：p_n = σ(f_imp(F)_n)，逐点干扰置信度"""

    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            DSConv1d(c, max(1, c // 2), k=3, d=1),
            nn.Conv1d(max(1, c // 2), 1, 1))

    def forward(self, f):
        return torch.sigmoid(self.net(f))


class ResidualHead(nn.Module):
    """残差重建头：输出 r̂，最终 ŝ = y − r̂

    cin 允许接入脉冲置信度通道（见 PAMNet 中"脉冲感知辅助融合"的说明）。
    """

    def __init__(self, c, cin=None):
        super().__init__()
        cin = c if cin is None else cin
        self.net = nn.Sequential(
            DSConv1d(cin, max(1, c // 2), k=3, d=1),
            nn.Conv1d(max(1, c // 2), 1, 1))

    def forward(self, f):
        return self.net(f)


class FreqRefine(nn.Module):
    """可选的频域精修分支（幅度掩码 + 相位修正）。

    实现与 v3 的 FrequencyBranch 完全一致；w=2.0 时通道为 (32,64,32,16,8)，
    即阶段0 中胜出的 5,910 参数纯频域模型所用的宽度。
    仅用于检验"时域残差估计 vs 频域掩码"的张力，默认不启用。

    pulse_in=True 时额外把脉冲置信度 p_n 作为第 2 个输入通道送进掩码网络
    （按 hop 做 max-pool 对齐到 STFT 帧率，再沿频率轴广播）。
    这样脉冲感知是【喂进频域分支】而不是事后修形——更符合"辅助特征融合"的本意。
    """

    def __init__(self, n_fft, hop, w=2.0, pulse_in=False):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.pulse_in = pulse_in
        c1, c2, c3 = max(1, round(16 * w)), max(1, round(32 * w)), max(1, round(16 * w))
        p1, p2 = max(1, round(8 * w)), max(1, round(4 * w))
        self.widths = (c1, c2, c3, p1, p2)
        self.mag_mask = nn.Sequential(
            DSConv2d(1 + (1 if pulse_in else 0), c1, 3, 1), DSConv2d(c1, c2, 3, 1),
            DSConv2d(c2, c3, 3, 1), nn.Conv2d(c3, 1, 1), nn.Sigmoid())
        self.ph_corr = nn.Sequential(
            DSConv2d(1, p1, 3, 1), DSConv2d(p1, p2, 3, 1),
            nn.Conv2d(p2, 1, 1), nn.Tanh())

    def forward(self, x, p=None):
        B, _, L = x.shape
        win = torch.hann_window(self.n_fft, device=x.device)
        X = torch.stft(x.squeeze(1), n_fft=self.n_fft, hop_length=self.hop,
                       window=win, return_complex=True)
        mag, ph = torch.abs(X).unsqueeze(1), torch.angle(X)
        if self.pulse_in:
            assert p is not None, "pulse_in=True 时必须提供 p"
            T, Fb = mag.shape[-1], mag.shape[2]
            pp = F.max_pool1d(p, kernel_size=self.hop, stride=self.hop)
            if pp.shape[-1] < T:
                pp = F.pad(pp, (0, T - pp.shape[-1]))
            pp = pp[..., :T].unsqueeze(1)                 # [B,1,T]
            pmap = pp.expand(-1, 1, Fb, -1)               # [B,1,F,T]
            inp = torch.cat([mag, pmap], dim=1)
        else:
            inp = mag
        mask = self.mag_mask(inp)
        dphi = self.ph_corr(ph.unsqueeze(1)).squeeze(1)
        Y = (mag * mask).squeeze(1) * torch.exp(1j * (ph + 0.1 * dphi))
        y = torch.istft(Y, n_fft=self.n_fft, hop_length=self.hop,
                        window=win, length=L)
        return y.unsqueeze(1)


# ============================== 主网络 ====================================
class PAMNet(nn.Module):
    """Pulse-Aware Multi-scale Lightweight Network"""

    def __init__(self, c=24, use_short=True, use_long=True, use_gate=True,
                 use_pulse=True, freq_refine=0.0, n_fft=512, hop=128,
                 pulse_freq=False):
        super().__init__()
        assert use_short or use_long, "至少要启用一条多尺度分支"
        assert not pulse_freq or (use_pulse and freq_refine > 0), \
            "pulse_freq 需要同时启用脉冲感知分支与频域精修"
        self.c = c
        self.use_short, self.use_long = use_short, use_long
        self.use_gate, self.use_pulse = use_gate, use_pulse
        self.pulse_freq = pulse_freq

        self.feat = FeatureExtractor(c)
        if use_short:
            self.short = ShortBranch(c)
        if use_long:
            self.long = LongBranch(c)
        if use_short and use_long and use_gate:
            self.gate = GatedFusion(c)
        if use_pulse:
            self.pulse = PulseBranch(c)
        # 脉冲感知"辅助特征融合"的具体形式：
        #   把 p_n 作为附加通道拼到融合特征后，送入残差估计头。
        # 选用拼接而非乘性调制 (1+p)·F，原因是后者会把 L_imp 的梯度直接
        # 灌进共享主干特征，导致"为了拟合掩码而破坏重建"（首轮实验已观察到
        # 该现象：p 分离度变好的同时验证 SNR 反而下降）。拼接让残差头自行
        # 决定如何使用 p，credit assignment 更干净。
        self.resid = ResidualHead(c, cin=c + (1 if use_pulse else 0))
        if freq_refine and freq_refine > 0:
            self.freq = FreqRefine(n_fft, hop, w=freq_refine, pulse_in=pulse_freq)
        else:
            self.freq = None

    # ------------------------------------------------------------------
    def forward(self, x, return_p=False):
        f0 = self.feat(x)

        if self.use_short and self.use_long:
            f_s, f_l = self.short(f0), self.long(f0)
            if self.use_gate:
                f_fuse, _ = self.gate(f_s, f_l)
            else:
                f_fuse = 0.5 * (f_s + f_l)          # 消融：简单平均代替门控
        elif self.use_short:
            f_fuse = self.short(f0)
        else:
            f_fuse = self.long(f0)

        if self.use_pulse:
            p = self.pulse(f_fuse)                   # [B,1,L]
            r = self.resid(torch.cat([f_fuse, p], dim=1))
        else:
            p = None
            r = self.resid(f_fuse)                   # 噪声残差估计

        s_hat = x - r

        if self.freq is not None:
            s_hat = self.freq(s_hat, p)              # 可选频域精修（可注入 p）

        if return_p:
            return s_hat, p
        return s_hat

    # ------------------------------------------------------------------
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self):
        d = {}
        for name, mod in self.named_children():
            d[name] = sum(p.numel() for p in mod.parameters())
        return d

    def gate_stats(self, x):
        """返回门控权重的均值（诊断用）"""
        if not (self.use_short and self.use_long and self.use_gate):
            return None
        with torch.no_grad():
            f0 = self.feat(x)
            _, g = self.gate(self.short(f0), self.long(f0))
            return float(g.mean()), float(g.std())


# ============================== 损失函数 ==================================
def stft_mag(x, n_fft, hop):
    win = torch.hann_window(n_fft, device=x.device)
    return torch.stft(x.squeeze(1), n_fft=n_fft, hop_length=hop,
                      window=win, return_complex=True).abs()


def huber(e, delta):
    """ρ_δ(e)：|e|<=δ 时 0.5e²，否则 δ(|e|−δ/2)"""
    a = e.abs()
    return torch.where(a <= delta, 0.5 * e * e, delta * (a - 0.5 * delta))


def pam_criterion(s_hat, clean, p, mask, n_fft, hop,
                  delta=0.2, lam_imp=1.0, lam_spec=0.5, loss_kind="huber",
                  return_parts=False):
    """L = L_Huber + λ1·L_imp + λ2·L_spec

    loss_kind='l1' 时把 Huber 换成 L1（消融用，对应阶段0 的既有损失形式）。
    p / mask 为 None 时跳过 L_imp。
    """
    e = s_hat - clean
    if loss_kind == "huber":
        l_hub = huber(e, delta).mean()
    elif loss_kind == "l1":
        l_hub = e.abs().mean()
    else:
        raise ValueError("未知损失类型 " + str(loss_kind))

    l_spec = (stft_mag(s_hat, n_fft, hop) - stft_mag(clean, n_fft, hop)).abs().mean()

    if (p is not None) and (mask is not None):
        l_imp = (p - mask).abs().mean()
    else:
        l_imp = torch.zeros((), device=s_hat.device)

    total = l_hub + lam_imp * l_imp + lam_spec * l_spec
    if return_parts:
        return total, {"huber": float(l_hub.detach()), "imp": float(l_imp.detach()),
                       "spec": float(l_spec.detach())}
    return total
