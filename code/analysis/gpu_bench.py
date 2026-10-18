# -*- coding: utf-8 -*-
"""
GPU 冒烟测试与速度基准
================================================================================
确认三件事：
 1. 模型（含频域精修分支的 STFT/ISTFT、复数张量）在 CUDA 上能跑通且输出有限
 2. 前向+反向速度相对 CPU 的提升倍数
 3. .npy 缓存相对 .mat 的数据加载提速

用法：python gpu_bench.py [--bs 16] [--iters 20]
"""
import os, sys, time, argparse, glob
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDS = [os.path.normpath(os.path.join(_HERE, "..", "python")),
          r"C:\Users\32213\PycharmProjects\project1"]
for _p in [_HERE] + [c for c in _CANDS if os.path.isdir(c)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                      # noqa: E402
import v3_model as V3             # noqa: E402
import pamnet as PN               # noqa: E402
import pam_metrics as PM          # noqa: E402

WORK = r"E:/workbuddy/store/2026-10-12-22-50-08"
OUTD = os.path.join(WORK, "sens_out")
CKPT = r"D:/project/output_pamnet"
DATA = r"E:/project/data/ds_sens/a15_r10"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20)
    A = ap.parse_args()

    print("=" * 74)
    print("环境")
    print("=" * 74)
    print("torch          :", torch.__version__)
    print("V3.DEVICE      :", V3.DEVICE)
    print("cuda available :", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("GPU            : %s  (%.2f GB, cc %d.%d, %d SM)"
              % (p.name, p.total_memory / 1e9, p.major, p.minor, p.multi_processor_count))
    print()

    arch = dict(use_short=True, use_long=True, use_gate=True, use_pulse=True,
                freq_refine=2.0, pulse_freq=True)
    model = PN.PAMNet(c=24, n_fft=V3.CFG["n_fft"], hop=V3.CFG["hop_length"],
                      norm="gn", **arch)
    print("参数量: {:,}".format(sum(p.numel() for p in model.parameters())))

    # ---- 冒烟测试：前向 + 反向 ----
    print()
    print("=" * 74)
    print("冒烟测试（前向 + 反向 + STFT/ISTFT 分支）")
    print("=" * 74)
    ok = True
    for dev in (torch.device("cpu"), V3.DEVICE):
        m = PN.PAMNet(c=24, n_fft=V3.CFG["n_fft"], hop=V3.CFG["hop_length"],
                      norm="gn", **arch).to(dev)
        x = torch.randn(A.bs, 1, V3.CFG["seg_len"], device=dev)
        try:
            y = m(x)
            loss = y.abs().mean()
            loss.backward()
            finite = bool(torch.isfinite(y).all()) and bool(torch.isfinite(loss))
            print("  %-12s 输出 %s  有限值=%s  loss=%.5f" % (dev.type, tuple(y.shape), finite, float(loss)))
            ok = ok and finite
        except Exception as e:
            print("  %-12s 失败: %s" % (dev.type, e))
            ok = False
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print("  结论:", "✓ 两个设备都跑通" if ok else "✗ 有问题")

    # ---- 速度基准 ----
    print()
    print("=" * 74)
    print("速度基准（batch=%d，%d 次迭代取均值）" % (A.bs, A.iters))
    print("=" * 74)
    res = {}
    for dev in (torch.device("cpu"), V3.DEVICE):
        m = PN.PAMNet(c=24, n_fft=V3.CFG["n_fft"], hop=V3.CFG["hop_length"],
                      norm="gn", **arch).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        x = torch.randn(A.bs, 1, V3.CFG["seg_len"], device=dev)
        tgt = torch.randn(A.bs, 1, V3.CFG["seg_len"], device=dev)
        # 预热
        for _ in range(3):
            opt.zero_grad(); (m(x) - tgt).abs().mean().backward(); opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(A.iters):
            opt.zero_grad()
            (m(x) - tgt).abs().mean().backward()
            opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / A.iters
        res[dev.type] = dt
        print("  %-5s  %.1f ms/step  ->  %.0f step/s" % (dev.type, dt * 1e3, 1 / dt))
    if "cpu" in res and "cuda" in res:
        print("  加速比: %.1f×" % (res["cpu"] / res["cuda"]))

    # ---- 数据加载基准 ----
    print()
    print("=" * 74)
    print("数据加载（train 全集读一遍）")
    print("=" * 74)
    import train_pam as TP
    for label, folder in (("mat", DATA + "/train"), ("npy", DATA + "/train")):
        ds = TP.PAMDataset(folder, V3.CFG["seg_len"], V3.CFG["downsample"])
        # 强制走 / 不走 npy
        if label == "mat":
            saved, ds.npy = ds.npy, None
        t0 = time.time()
        for i in range(0, len(ds), 500):
            _ = ds[i]
        dt = time.time() - t0
        n_probe = len(range(0, len(ds), 500))
        if label == "mat":
            ds.npy = saved
        print("  %-4s 抽 %d 个样本 %.2fs  -> 全量 %d 估算 %.1fs"
              % (label, n_probe, dt, len(ds), dt / n_probe * len(ds)))

    print()
    print("=" * 74)
    print("提示：CPU→GPU 会改变浮点归约顺序，与既有 CPU 存档不再逐位一致。")
    print("      报告已记录聚合指标容差 ≤0.13 dB；切换后应先跑一个配置核对验证指标。")
    print("=" * 74)


if __name__ == "__main__":
    main()
