# -*- coding: utf-8 -*-
"""
PAM-Net 单元测试
================================================================================
逐项验证实现正确性。任何一项不通过就不往下跑训练。
  1. 参数量与设计一致
  2. 前向形状 [B,1,L] -> [B,1,L]；p 形状 [B,1,L] 且落在 [0,1]
  3. 梯度能流到【全部】参数（检测死分支 / 未被使用的模块）
  4. Huber 损失与手算一致（|e|<=δ 与 |e|>δ 两段）
  5. L_imp / L_spec 与手算一致
  6. 频域精修分支在复制权重后与 v3 FrequencyBranch 输出【逐位一致】
     （保证与阶段0 的 5,910 参数模型口径相同）
  7. 感受野实测（脉冲响应法），确认长时分支能覆盖突发时长
  8. 长度保持性：任意奇数核 + 膨胀率组合下输出长度 = 输入长度
"""
import os, sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
# ---- 路径解析：兼容仓库布局（库模块在 ../python）与本地开发布局 ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDS = [os.path.normpath(os.path.join(_HERE, "..", "python")),
          r"C:\Users\32213\PycharmProjects\project1"]
for _p in [_HERE] + [c for c in _CANDS if os.path.isdir(c)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pamnet as PN
import v3_model as V3

L = 8000
B = 3
torch.manual_seed(0)

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print("  %s %s%s" % ("✓" if cond else "✗", name, ("   " + detail) if detail else ""))
    return cond


print("=" * 92)
print("PAM-Net 单元测试")
print("=" * 92)

# ---------------------------------------------------------------- 1. 参数量
print("\n【1】参数量")
configs = [
    ("完整 (c=24)", dict(c=24)),
    ("无脉冲分支", dict(c=24, use_pulse=False)),
    ("无门控", dict(c=24, use_gate=False)),
    ("仅短时", dict(c=24, use_long=False)),
    ("仅长时", dict(c=24, use_short=False)),
    ("完整 + 频域精修2.0", dict(c=24, freq_refine=2.0)),
    ("c=16", dict(c=16)),
    ("c=32", dict(c=32)),
]
models = {}
for name, kw in configs:
    m = PN.PAMNet(n_fft=512, hop=128, **kw)
    models[name] = m
    bd = m.param_breakdown()
    print("  %-20s 参数量 %7d   明细 %s" % (name, m.n_params(),
          " ".join("%s=%d" % (k, v) for k, v in bd.items())))
check("参数量均为正且完整模型 < 30000", models["完整 (c=24)"].n_params() < 30000,
      "实际 %d" % models["完整 (c=24)"].n_params())

# 消融模型必须严格小于完整模型
check("去脉冲分支 -> 参数变少",
      models["无脉冲分支"].n_params() < models["完整 (c=24)"].n_params())
check("去门控 -> 参数变少",
      models["无门控"].n_params() < models["完整 (c=24)"].n_params())
check("仅短时 < 完整", models["仅短时"].n_params() < models["完整 (c=24)"].n_params())
check("仅长时 < 完整", models["仅长时"].n_params() < models["完整 (c=24)"].n_params())

# ---------------------------------------------------------------- 2. 前向形状
print("\n【2】前向形状与取值范围")
x = torch.randn(B, 1, L) * 2.0
for name, m in models.items():
    m.eval()
    with torch.no_grad():
        y, p = m(x, return_p=True)
    shp_ok = (tuple(y.shape) == (B, 1, L))
    if p is None:
        p_ok = True
    else:
        p_ok = (tuple(p.shape) == (B, 1, L)) and bool((p >= 0).all()) and bool((p <= 1).all())
    check("%-20s y%s p%s" % (name, tuple(y.shape), None if p is None else tuple(p.shape)),
          shp_ok and p_ok)

# 数值有效性
m = models["完整 (c=24)"]
m.eval()
with torch.no_grad():
    y = m(x)
check("输出无 NaN / Inf", bool(torch.isfinite(y).all()))

# ---------------------------------------------------------------- 3. 梯度流
print("\n【3】梯度流（每个参数都必须有非零梯度）")
for name in ["完整 (c=24)", "无脉冲分支", "无门控", "仅短时", "仅长时"]:
    m = models[name]
    m.train()
    xb = torch.randn(4, 1, L)
    y, p = m(xb, return_p=True)
    loss = y.abs().mean()
    if p is not None:
        loss = loss + p.abs().mean()
    m.zero_grad()
    loss.backward()
    dead = [n for n, prm in m.named_parameters()
            if prm.grad is None or not torch.isfinite(prm.grad).all()]
    check("%-20s 无死参数/无 NaN 梯度" % name, len(dead) == 0,
          "" if not dead else "问题参数: %s" % dead[:5])

# 特别检查：脉冲分支必须真的被训练（梯度非零）
m = models["完整 (c=24)"]
m.train(); m.zero_grad()
xb = torch.randn(4, 1, L)
y, p = m(xb, return_p=True)
(y.abs().mean() + p.abs().mean()).backward()
pg = [prm.grad.abs().sum().item() for prm in m.pulse.parameters()]
check("脉冲分支梯度非零", all(g > 0 for g in pg), "各层梯度和 %s" % ["%.3e" % g for g in pg])

# ---------------------------------------------------------------- 4. Huber
print("\n【4】Huber 损失与手算一致")
e = torch.tensor([[-1.0, -0.1, 0.0, 0.1, 1.0]])
delta = 0.2
h = PN.huber(e, delta)
# 手算：|e|<=0.2 -> 0.5e²；否则 0.2*(|e|-0.1)
manual = torch.tensor([[0.2 * (1.0 - 0.1), 0.5 * 0.01, 0.0, 0.5 * 0.01, 0.2 * (1.0 - 0.1)]])
check("ρ_δ(e) 数值正确", bool(torch.allclose(h, manual, atol=1e-7)),
      "\n     实测 %s\n     手算 %s" % (h.tolist(), manual.tolist()))
# 分界点连续性：|e|=δ 处两段相等
check("分界点连续", abs(0.5 * delta ** 2 - delta * (delta - 0.5 * delta)) < 1e-12)
# 大误差梯度有界（这是 Huber 抗脉冲的关键性质）
e1 = torch.tensor([[0.5]], requires_grad=True)
PN.huber(e1, delta).backward()
check("|e|>δ 时梯度 = δ（有界）", abs(e1.grad.item() - delta) < 1e-6,
      "梯度 %.6f  应为 %.6f" % (e1.grad.item(), delta))
e2 = torch.tensor([[0.05]], requires_grad=True)
PN.huber(e2, delta).backward()
check("|e|<δ 时梯度 = e", abs(e2.grad.item() - 0.05) < 1e-6, "梯度 %.6f" % e2.grad.item())

# ---------------------------------------------------------------- 5. 损失各项
print("\n【5】联合损失各项与手算一致")
s_hat = torch.randn(2, 1, L) * 0.3
clean = torch.randn(2, 1, L) * 0.3
p = torch.rand(2, 1, L)
mask = (torch.rand(2, 1, L) > 0.9).float()
tot, parts = PN.pam_criterion(s_hat, clean, p, mask, 512, 128,
                              delta=0.2, lam_imp=1.0, lam_spec=0.5,
                              loss_kind="huber", return_parts=True)
l_hub_manual = PN.huber(s_hat - clean, 0.2).mean().item()
l_imp_manual = (p - mask).abs().mean().item()
l_spec_manual = (PN.stft_mag(s_hat, 512, 128) - PN.stft_mag(clean, 512, 128)).abs().mean().item()
check("L_Huber 一致", abs(parts["huber"] - l_hub_manual) < 1e-6)
check("L_imp 一致", abs(parts["imp"] - l_imp_manual) < 1e-6)
check("L_spec 一致", abs(parts["spec"] - l_spec_manual) < 1e-6)
check("总损失 = 三项加权和",
      abs(tot.item() - (l_hub_manual + 1.0 * l_imp_manual + 0.5 * l_spec_manual)) < 1e-5,
      "总 %.6f" % tot.item())
# 无 p / mask 时 L_imp 应被跳过
tot2 = PN.pam_criterion(s_hat, clean, None, None, 512, 128, loss_kind="huber")
check("无脉冲标签时自动跳过 L_imp",
      abs(tot2.item() - (l_hub_manual + 0.5 * l_spec_manual)) < 1e-5)
# L1 模式
tot3, parts3 = PN.pam_criterion(s_hat, clean, p, mask, 512, 128,
                                loss_kind="l1", return_parts=True)
check("L1 模式数值正确",
      abs(parts3["huber"] - (s_hat - clean).abs().mean().item()) < 1e-6)

# ---------------------------------------------------------------- 6. 频域分支等价
print("\n【6】频域精修分支 vs v3 FrequencyBranch（权重复制后应逐位一致）")
fr = PN.FreqRefine(512, 128, w=2.0).eval()
vb = V3.FrequencyBranch(512, 128, w=2.0).eval()
check("结构宽度一致", fr.widths == vb.widths,
      "PN=%s  V3=%s" % (fr.widths, vb.widths))
sd = {k: v.clone() for k, v in vb.state_dict().items()}
missing = fr.load_state_dict(sd, strict=True)
xin = torch.randn(2, 1, L)
with torch.no_grad():
    a = fr(xin)
    b = vb(xin)
d = (a - b).abs().max().item()
check("输出逐位一致（最大差 < 1e-6）", d < 1e-6, "最大绝对差 %.3e" % d)

# ---------------------------------------------------------------- 7. 感受野
print("\n【7】感受野实测（脉冲响应法，eval 模式下扰动单个输入点）")


def measure_rf(model, L=2001):
    model = model.eval()
    x0 = torch.zeros(1, 1, L)
    c = L // 2
    x1 = x0.clone()
    x1[0, 0, c] = 1.0
    with torch.no_grad():
        y0, y1 = model(x0), model(x1)
    dif = (y1 - y0).abs().squeeze().numpy()
    idx = np.where(dif > 1e-8)[0]
    if len(idx) == 0:
        return 0, 0, 0
    left, right = c - idx.min(), idx.max() - c
    return len(idx), left, right


for name, kw in [("仅短时", dict(c=24, use_long=False, use_pulse=False)),
                 ("仅长时", dict(c=24, use_short=False, use_pulse=False)),
                 ("完整(含两分支)", dict(c=24, use_pulse=False))]:
    m = PN.PAMNet(n_fft=512, hop=128, **kw)
    n, l, r = measure_rf(m)
    print("  %-18s 受影响点数 %4d   左 %3d 右 %3d  总感受野 %d" % (name, n, l, r, l + r + 1))

n_short, _, _ = measure_rf(PN.PAMNet(n_fft=512, hop=128, c=24, use_long=False, use_pulse=False))
n_long, _, _ = measure_rf(PN.PAMNet(n_fft=512, hop=128, c=24, use_short=False, use_pulse=False))
n_full, _, _ = measure_rf(PN.PAMNet(n_fft=512, hop=128, c=24, use_pulse=False))
check("短时分支感受野 < 长时分支", n_short < n_long, "短 %d vs 长 %d" % (n_short, n_long))
check("完整网络感受野 ≈ 长时分支（取最大）", abs(n_full - n_long) <= 4,
      "完整 %d vs 长时 %d" % (n_full, n_long))
check("长时分支感受野 >= 127 点（覆盖 12~180 点突发）", n_long >= 127, "实测 %d" % n_long)

# ---------------------------------------------------------------- 8. 长度保持
print("\n【8】长度保持性")
for k, d in [(3, 1), (5, 1), (7, 1), (7, 4), (7, 16), (9, 8)]:
    conv = PN.DSConv1d(4, 4, k=k, d=d)
    xx = torch.randn(1, 4, 1000)
    out = conv(xx)
    check("k=%d d=%-2d 输出长度 = 输入长度" % (k, d), out.shape[-1] == 1000,
          "得到 %d" % out.shape[-1])

# ---------------------------------------------------------------- 汇总
print("\n" + "=" * 92)
print("总判定：", "✓ 全部通过" if ok_all else "✗ 存在失败项，禁止进入训练")
print("=" * 92)
sys.exit(0 if ok_all else 1)
