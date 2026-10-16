# -*- coding: utf-8 -*-
"""
PAMNet 纯频域模式（freq-only）的架构等价性验证
================================================================================
目的：确认 PAMNet(use_short=0, use_long=0, use_pulse=0, freq_refine=2.0)
      与阶段0 胜出的 v3.DenoiseNet(use_time=False, use_freq=True, fch=2.0)
      【架构完全等价】。只有这样，"统一协议下重训纯频域模型" 才能把
      "架构贡献" 与 "训练协议贡献" 分离开。

检查项：
  1. 参数量一致（应为 5,910，即阶段0 胜出配置）
  2. 权重复制后前向输出逐位一致
  3. 梯度可回传（无死分支）
  4. return_p 返回 None 而不报错
  5. 默认（时域通路开启）仍能严格加载既有 checkpoint —— 确认改动无副作用
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

OK = True


def check(name, cond, extra=""):
    global OK
    print("  %s %-52s %s" % ("✓" if cond else "✗", name, extra))
    OK &= bool(cond)


print("=" * 92)
print("PAMNet 纯频域模式 · 架构等价性验证")
print("=" * 92)

# ---------------- 1. 参数量 ----------------
print("\n[1] 参数量")
fo = PN.PAMNet(c=24, use_short=False, use_long=False, use_gate=False,
               use_pulse=False, freq_refine=2.0, norm="bn",
               n_fft=512, hop=128)
ref = V3.DenoiseNet(512, 128, hidden=64, use_time=False, use_freq=True, fch=2.0)
n_fo, n_ref = fo.n_params(), ref.n_params()
check("freq-only 参数量 == 阶段0 纯频域模型", n_fo == n_ref, "%d vs %d" % (n_fo, n_ref))
check("参数量 == 5,910（阶段0 胜出配置）", n_fo == 5910, "%d" % n_fo)
check("use_time 标记为 False", fo.use_time is False)
check("时域子模块均为 None",
      all(getattr(fo, k) is None for k in ("feat", "short", "long", "gate", "pulse", "resid")))

# ---------------- 2. 权重复制 + 逐位一致 ----------------
print("\n[2] 权重复制后前向输出逐位一致")
sd = {}
for k, v in ref.state_dict().items():
    if k.startswith("freq."):
        sd[k] = v.clone()
missing, unexpected = fo.load_state_dict(sd, strict=False)
check("权重键全部匹配", len(missing) == 0 and len(unexpected) == 0,
      "missing=%d unexpected=%d" % (len(missing), len(unexpected)))

torch.manual_seed(0)
x = torch.randn(3, 1, 8000)
fo.eval(); ref.eval()
with torch.no_grad():
    y_fo = fo(x)
    y_ref = ref(x)
d = (y_fo - y_ref).abs().max().item()
rel = d / max(y_ref.abs().max().item(), 1e-12)
check("前向输出最大绝对差 < 1e-6", d < 1e-6, "%.3e（相对 %.3e）" % (d, rel))
check("输出形状 [B,1,L]", tuple(y_fo.shape) == (3, 1, 8000), str(tuple(y_fo.shape)))
check("输出有限", bool(torch.isfinite(y_fo).all()))

# ---------------- 3. return_p ----------------
# 注意：必须在任何 train() 前向之前做。BN 的 running stats 会被 train 前向更新，
# 之后再比较 eval 输出就会不一致（这是 BN 的固有行为，不是本模式的问题）。
print("\n[3] return_p 行为")
fo.eval()
with torch.no_grad():
    y2 = fo(x)
check("return_p=False 与 [2] 的 eval 输出一致",
      bool(torch.allclose(y2, y_fo, atol=1e-6)))
with torch.no_grad():
    y3, p3 = fo(x, return_p=True)
check("return_p=True 时 p 为 None", p3 is None)
check("return_p=True 与 False 输出一致", bool(torch.allclose(y3, y2, atol=1e-6)))

# ---------------- 4. 梯度 ----------------
print("\n[4] 梯度可回传")
fo.train()
yy = fo(x)
loss = yy.abs().mean()
fo.zero_grad(); loss.backward()
grads = [(n, p.grad.abs().sum().item()) for n, p in fo.named_parameters()]
dead = [n for n, g in grads if g == 0]
check("无零梯度参数（无死分支）", len(dead) == 0, "dead=%s" % (dead[:4] if dead else "无"))
check("梯度组数与参数张量数一致",
      len(grads) == len(list(fo.parameters())), "%d 组" % len(grads))
fo.eval()

# ---------------- 5. 默认配置与既有 checkpoint 兼容 ----------------
print("\n[5] 默认配置（时域通路开启）与既有 checkpoint 兼容性")
CK = [
    ("model_pamnet.pth",
     dict(c=24, n_fft=512, hop=128)),
    ("model_pamnet_freq_pf_gn.pth",
     dict(c=24, n_fft=512, hop=128, freq_refine=2.0, pulse_freq=True, norm="gn")),
    ("model_pamnet_freq_nopulse_gn.pth",
     dict(c=24, n_fft=512, hop=128, freq_refine=2.0, use_pulse=False, norm="gn")),
    ("model_pamnet_gn.pth",
     dict(c=24, n_fft=512, hop=128, norm="gn")),
    ("model_pamnet_nopulse_gn.pth",
     dict(c=24, n_fft=512, hop=128, use_pulse=False, norm="gn")),
]
for name, kw in CK:
    path = os.path.join(r"D:/project/output_pamnet", name)
    if not os.path.exists(path):
        print("  -  跳过（文件不存在）%s" % name); continue
    m = PN.PAMNet(**kw)
    sd2 = torch.load(path, map_location="cpu")
    try:
        m.load_state_dict(sd2, strict=True)
        check("strict 加载 %s" % name, True, "%d 参数" % m.n_params())
    except Exception as e:
        check("strict 加载 %s" % name, False, str(e)[:70].replace("\n", " "))

# ---------------- 6. 参数明细 ----------------
print("\n[6] freq-only 参数明细")
bd = fo.param_breakdown()
print("      " + "  ".join("%s=%d" % (k, v) for k, v in bd.items()))
print("      频域分支宽度 (c1,c2,c3,p1,p2) = %s" % (fo.freq.widths,))
print()
print("=" * 92)
print("总判定: %s" % ("✓ 全部通过 —— 纯频域模式与阶段0 架构逐位等价" if OK else "✗ 存在失败项"))
print("=" * 92)
sys.exit(0 if OK else 1)
