# -*- coding: utf-8 -*-
"""输入尺度归一化的决定性验证：新 checkpoint 是否对输入尺度不再敏感

做法：把 base 条件 testA 的全部样本按 s ∈ {0.25, 0.5, 1, 2, 4} 整体缩放
（缩放不改变 SNR，纯粹改变输入尺度），分别用
  A. 老 checkpoint（未归一化训练）→ 预期性能随 s 剧烈变化
  B. 新 checkpoint（x/rms(x) 训练的 inm_*）→ 预期各 s 下几乎不变
评估干扰帧增益。若 B 平坦而 A 剧烈，即证明归一化解决了部署级风险。

用法：python scale_invariance_test.py [--nmax 300]
"""
import os, sys, json, argparse, glob
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.dirname(os.path.abspath(__file__))
# 多候选路径：探测 code/python（仓库布局），否则回退本地开发目录
_CANDS = [os.path.normpath(os.path.join(HERE, "..", "python")),
          r"C:\Users\32213\PycharmProjects\project1"]
for _p in [HERE] + [c for c in _CANDS if os.path.isdir(c)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pam_metrics as PM
import v3_model as V3
import pamnet as PN

OUTD = os.path.join(HERE, "sens_out")
CKPT = r"D:/project/output_pamnet"
FS, SEG = 6000, 8000
SCALES = [0.25, 0.5, 1.0, 2.0, 4.0]

R5_ARCH = dict(use_short=True, use_long=True, use_gate=True, use_pulse=True,
               freq_refine=2.0, pulse_freq=True)
R4_ARCH = dict(use_short=True, use_long=True, use_gate=True, use_pulse=False,
               freq_refine=2.0, pulse_freq=False)


def load_npy(split):
    """直接从 _npy 缓存读 testA/testC（与 PAMDataset 同一口径，已是定长 float32）"""
    base = r"D:/project/dataset_pam/alphastable/_npy"
    p = [os.path.join(base, "%s_%s.npy" % (split, t)) for t in ("n", "c", "m")]
    if not all(os.path.exists(x) for x in p):
        return None
    a = [np.load(x, mmap_mode="r") for x in p]
    return a[0], a[1], a[2]


def build(arch, ckpt):
    m = PN.PAMNet(c=24, n_fft=V3.CFG["n_fft"], hop=V3.CFG["hop_length"],
                  norm="gn", **arch)
    m.load_state_dict(torch.load(os.path.join(CKPT, ckpt), map_location="cpu"),
                      strict=True)
    return m.to(V3.DEVICE).eval()


@torch.no_grad()
def gains_on_scale(model, N, C, M, s, rms_norm=False, bs=48):
    """把输入整体乘 s（SNR 不变），返回逐样本的 frmgain_imp 数组

    rms_norm=True 用于归一化训练的 checkpoint：前向按 x/rms(x·s)，
    输出乘回 rms(x·s)。注意这是恒等式 x·s/rms(x·s) = x/rms(x)，
    即归一化模型在正确前向下【严格】尺度不变 —— 缩放被完全消掉。
    """
    out = []
    for i in range(0, len(N), bs):
        raw = (N[i:i + bs] * s).astype(np.float32)
        if rms_norm:
            r = np.sqrt((raw.astype(np.float64) ** 2).mean(axis=1)) + 1e-12
            nb = (raw / r[:, None]).astype(np.float32)
        else:
            nb = raw
        e = model(torch.from_numpy(nb).unsqueeze(1).to(V3.DEVICE)
                  ).squeeze(1).cpu().numpy().astype(np.float64)
        if rms_norm:
            e = e * r[:, None]           # 反归一化回 raw 尺度
        for k in range(e.shape[0]):
            j = i + k
            # 三个量必须同一尺度：est 已反归一化到 raw 尺度（= N·s 尺度），
            # 因此 ref 与 noisy 也都要乘 s。此前 ref 漏乘 s 导致 est/ref/noisy
            # 尺度错位，指标整体错误（这正是"归一化模型仍敏感"假象的来源）。
            st = PM._stats_1d(e[k], (C[j] * s).astype(np.float64),
                              (N[j] * s).astype(np.float64),
                              PM.FRAME, M[j].astype(np.float64))
            out.append(st["frmgain_imp"])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=300, help="样本数上限，<=0 表示全量")
    A = ap.parse_args()

    d = load_npy("testA")
    if d is None:
        print("缺 _npy 缓存，先跑 make_npy_cache.py"); return
    N, C, M = d[0], d[1], d[2]
    nm = len(N) if not A.nmax or A.nmax <= 0 else A.nmax
    N = np.array(N[:nm], dtype=np.float32)
    C = np.array(C[:nm], dtype=np.float32)
    M = np.array(M[:nm], dtype=np.float32)
    print("样本 %d，缩放倍率 %s" % (len(N), SCALES))
    print()

    # 待测 checkpoint：老的 R5 + 新训的 R5（归一化）
    ckpts = []
    if os.path.exists(os.path.join(CKPT, "model_pamnet_freq_pf_gn.pth")):
        ckpts.append(("老 R5（未归一化训练）", R5_ARCH, "model_pamnet_freq_pf_gn.pth", False))
    for tag in ["inm_pf", "inm_pf_s2", "inm_pf_s3"]:
        p = os.path.join(CKPT, "model_%s.pth" % tag)
        if os.path.exists(p):
            ckpts.append(("新 R5（%s，归一化训练）" % tag, R5_ARCH, "model_%s.pth" % tag, True))
    if not ckpts:
        print("还没有 inm_pf checkpoint，等训练完成"); return

    res = {}
    for name, arch, ck, is_norm in ckpts:
        model = build(arch, ck)
        print("=" * 92)
        print(name)
        print("=" * 92)
        print("  %6s %14s %12s %12s" % ("尺度 s", "干扰帧增益(dB)", "相对 s=1", "有效SNR(dB)"))
        row = {}
        for s in SCALES:
            g = gains_on_scale(model, N, C, M, s, rms_norm=is_norm)
            fin = g[np.isfinite(g)]
            v = float(np.nanmean(fin))
            row[s] = v
            rel = v - row.get(1.0, v)
            # 有效 SNR 用 snr_db 重算：缩放会改 ssi_sdr 的口径吗？这里只看 frmgain
            print("  %6.2f %14.3f %12s %12s" % (s, v, "—" if s == 1.0 else "%+.3f" % rel,
                                                "—"))
        rng = max(row.values()) - min(row.values())
        print("  -> 跨尺度极差 %.3f dB" % rng)
        print()
        res[name] = dict(gains=row, range_db=rng)

    print("=" * 92)
    print("判定")
    print("=" * 92)
    for name, r in res.items():
        verdict = "✅ 尺度不敏感" if r["range_db"] < 1.0 else (
            "⚠ 中等敏感" if r["range_db"] < 5.0 else "❌ 仍然高度敏感")
        print("  %-34s 跨尺度极差 %8.3f dB   %s" % (name, r["range_db"], verdict))
    if len(res) >= 2:
        ks = list(res.keys())
        old, new = res[ks[0]]["range_db"], res[ks[-1]]["range_db"]
        print()
        print("  老化极差 %.3f dB → 新 %.3f dB，改善 %.1f 倍"
              % (old, new, (old / new) if new > 1e-9 else float("inf")))

    os.makedirs(OUTD, exist_ok=True)
    out = os.path.join(OUTD, "scale_invariance.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(scales=SCALES, n=len(N), res=res), f,
                  ensure_ascii=False, indent=1)
    print("\n->", out)


if __name__ == "__main__":
    main()
