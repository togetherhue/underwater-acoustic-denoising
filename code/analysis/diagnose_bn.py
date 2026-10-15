# -*- coding: utf-8 -*-
"""
诊断训练振荡的来源：BatchNorm 的 running statistics 是否失配
================================================================================
假设：输入是重尾的（α 稳定背景 + 20×rms 脉冲），逐 batch 的均值/方差差异极大，
      BN 的 running statistics 因此漂移，导致 eval 模式（用 running stats）
      与 train 模式（用 batch stats）表现差异巨大，且逐轮抖动。

验证三件事：
  1. 同一 checkpoint，eval 模式（running stats）vs train 模式（batch stats）的
     验证集帧增益差多少。差得多 => BN 统计失配。
  2. BN 的 running_var 在训练过程中漂移多大（读 checkpoint 里的 running 统计，
     并与训练集上的真实统计比较）。
  3. 补救措施是否有效：用「BN 重校准」（冻结权重，在训练集上跑几百个 batch
     让 running stats 收敛）后重新评估，看是否显著变好/变稳。

若 1 的差距很小，则振荡另有原因（优化本身不稳），需要换方向排查。
"""
import os, sys, glob, argparse
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly
import torch

PROJ = r"C:\Users\32213\PycharmProjects\project1"
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
import v3_model as V3          # noqa: E402
import pamnet as PN            # noqa: E402
import pam_metrics as PM       # noqa: E402
from train_pam import PAMDataset   # noqa: E402

FRAME = PM.FRAME


def fg_of(est, ref, noisy, mask, frame=FRAME):
    st = PM._stats_1d(est, ref, noisy, frame, mask)
    return st


def run_eval_mode(model, files, train_mode, n=120):
    """在给定模式下跑一批验证样本，返回帧增益均值"""
    acc = {"frmgain_med": [], "frmgain_clean": [], "frmgain_imp": [], "frmgain_imp_min": []}
    ds = PAMDataset(os.path.dirname(files[0]), V3.CFG["seg_len"], V3.CFG["downsample"])
    idx = list(range(min(n, len(ds))))
    if train_mode:
        model.train()
    else:
        model.eval()
    with torch.no_grad():
        for i in idx:
            nz, cl, mk = ds[i]
            x = nz.view(1, 1, -1).to(V3.DEVICE)
            est = model(x)[0, 0].cpu().numpy().astype(np.float64)
            st = fg_of(est, cl.numpy().astype(np.float64),
                       nz.numpy().astype(np.float64), mk.numpy())
            for k in acc:
                v = st.get(k)
                if v is not None and not np.isnan(v):
                    acc[k].append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}, len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"D:/project/output_pamnet/model_pamnet.pth")
    ap.add_argument("--data", default=r"D:/project/dataset_pam/alphastable")
    ap.add_argument("--n", type=int, default=120)
    A = ap.parse_args()

    model = PN.PAMNet(c=24, n_fft=V3.CFG["n_fft"], hop=V3.CFG["hop_length"]).to(V3.DEVICE)
    model.load_state_dict(torch.load(A.ckpt, map_location=V3.DEVICE), strict=True)

    files = sorted(glob.glob(os.path.join(A.data, "val", "*.mat")))
    print("=" * 96)
    print("BatchNorm 诊断  |  ckpt = %s" % A.ckpt)
    print("=" * 96)

    ev, n = run_eval_mode(model, files, False, A.n)
    tr, _ = run_eval_mode(model, files, True, A.n)
    print("\n【1】eval 模式（running stats） vs train 模式（batch stats）  n=%d" % n)
    print("  %-16s %12s %12s %10s" % ("指标", "eval模式", "train模式", "差"))
    for k in ["frmgain_med", "frmgain_clean", "frmgain_imp", "frmgain_imp_min"]:
        print("  %-16s %12.2f %12.2f %+10.2f" % (k, ev[k], tr[k], tr[k] - ev[k]))
    d = tr["frmgain_med"] - ev["frmgain_med"]
    print("\n  判定：帧增益差 %+.2f dB -> %s"
          % (d, "BN 统计失配是重要原因" if abs(d) > 1.0 else
             "BN 失配不明显，振荡另有原因"))

    # ---------- 2. running stats 漂移 ----------
    print("\n【2】BN running_var 的分布（与输入尺度对比）")
    rv = []
    for name, buf in model.named_buffers():
        if name.endswith("running_var"):
            rv.append((name, float(buf.mean()), float(buf.min()), float(buf.max())))
    print("  %-46s %12s %12s %12s" % ("层", "均值", "最小", "最大"))
    for name, m, lo, hi in rv[:12]:
        print("  %-46s %12.4g %12.4g %12.4g" % (name[:46], m, lo, hi))
    if len(rv) > 12:
        print("  ...（共 %d 个 BN 层）" % len(rv))

    # ---------- 3. BN 重校准 ----------
    print("\n【3】BN 重校准：冻结权重，在训练集上重跑若干 batch 让 running stats 收敛")
    import copy
    m2 = copy.deepcopy(model)
    for mod in m2.modules():
        if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            mod.reset_running_stats()
            mod.momentum = None          # 累积平均
    m2.train()
    tr_ds = PAMDataset(os.path.join(A.data, "train"), V3.CFG["seg_len"], V3.CFG["downsample"])
    bs = 16
    n_batch = min(200, len(tr_ds) // bs)
    with torch.no_grad():
        for b in range(n_batch):
            xs = torch.stack([tr_ds[b * bs + j][0] for j in range(bs)]).unsqueeze(1).to(V3.DEVICE)
            m2(xs)
    print("  已用 %d 个 batch（%d 样本）重估 running stats" % (n_batch, n_batch * bs))
    ev2, _ = run_eval_mode(m2, files, False, A.n)
    print("  %-16s %12s %12s %10s" % ("指标", "原eval", "重校准后", "差"))
    for k in ["frmgain_med", "frmgain_clean", "frmgain_imp", "frmgain_imp_min"]:
        print("  %-16s %12.2f %12.2f %+10.2f" % (k, ev[k], ev2[k], ev2[k] - ev[k]))
    print("\n  判定：重校准带来 %+.2f dB -> %s"
          % (ev2["frmgain_med"] - ev["frmgain_med"],
             "BN 统计确实是瓶颈，值得改归一化方式" if abs(ev2["frmgain_med"] - ev["frmgain_med"]) > 1.0
             else "重校准无效，问题不在 BN running stats"))


if __name__ == "__main__":
    main()
