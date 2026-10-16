# -*- coding: utf-8 -*-
"""
脉冲感知分支诊断（跨条件）
================================================================================
回答：脉冲感知分支的【判别能力】是否随噪声统计条件变化？
与 eval_sensitivity.py 的性能矩阵合起来，就能把
    「能识别」与「能提升性能」两条曲线放在同一个横轴上比较 ——
这是本项目最重要的区分（见阶段1 报告第 5.5 节）。

指标：
  auc_mean    逐样本 AUC 的均值（p_n 作为脉冲检测器，秩统计量，无需选阈值）
  p_imp       干扰位置 p_n 均值
  p_clean     干净位置 p_n 均值
  sep         两者之差

用法：python diagnose_pulse_sens.py --conds base a12_r5 ... --max 300
"""
import os, sys, json, argparse
import numpy as np
import torch

# ---- 路径解析：兼容仓库布局（库模块在 ../python）与本地开发布局 ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDS = [os.path.normpath(os.path.join(_HERE, "..", "python")),
          r"C:\Users\32213\PycharmProjects\project1"]
for _p in [_HERE] + [c for c in _CANDS if os.path.isdir(c)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import v3_model as V3          # noqa: E402
import pamnet as PN            # noqa: E402

WORK = r"E:/workbuddy/store/2026-10-12-22-50-08"
OUTD = os.path.join(WORK, "sens_out")
CKPT = r"D:/project/output_pamnet"
CONDS = ["base", "a12_r5", "a18_r5", "a20_r5", "a15_r25", "a15_r10", "gauss"]

# 含脉冲感知分支的两个配置（只有它们才有 p_n）
PULSE_MODELS = {
    "R5_freqrefine_pulse": dict(
        arch=dict(use_short=True, use_long=True, use_gate=True, use_pulse=True,
                  freq_refine=2.0, pulse_freq=True),
        ckpts=["model_pamnet_freq_pf_gn.pth", "model_pamnet_freq_pf_gn_s2.pth",
               "model_pamnet_freq_pf_gn_s3.pth", "model_pamnet_freq_pf_gn_s4.pth",
               "model_pamnet_freq_pf_gn_s5.pth"]),
    "P_time_pulse": dict(
        arch=dict(use_short=True, use_long=True, use_gate=True, use_pulse=True,
                  freq_refine=0.0, pulse_freq=False),
        ckpts=["model_pamnet_gn.pth", "model_pamnet_gn_s2.pth", "model_pamnet_gn_s3.pth"]),
}


def auc_rank(score, label):
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


@torch.no_grad()
def diag(model, noisy, mask, bs=32):
    aucs, pi, pc = [], [], []
    for s in range(0, len(noisy), bs):
        x = torch.from_numpy(noisy[s:s + bs]).unsqueeze(1)
        mb = mask[s:s + bs].astype(bool)
        _, p = model(x, return_p=True)
        p = p[:, 0].numpy()
        for k in range(p.shape[0]):
            m = mb[k]
            if m.sum() == 0 or (~m).sum() == 0:
                continue
            aucs.append(auc_rank(p[k], m))
            pi.append(p[k][m]); pc.append(p[k][~m])
    pi = np.concatenate(pi) if pi else np.array([np.nan])
    pc = np.concatenate(pc) if pc else np.array([np.nan])
    return dict(auc_mean=float(np.mean(aucs)), auc_median=float(np.median(aucs)),
                auc_p10=float(np.percentile(aucs, 10)),
                frac_auc_gt08=float(np.mean(np.array(aucs) > 0.8)),
                p_imp=float(pi.mean()), p_clean=float(pc.mean()),
                p_imp_median=float(np.median(pi)), p_clean_median=float(np.median(pc)),
                sep=float(pi.mean() - pc.mean()), n=int(len(aucs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", nargs="*", default=CONDS)
    ap.add_argument("--models", nargs="*", default=list(PULSE_MODELS.keys()))
    ap.add_argument("--seeds", type=int, default=1, help="每个配置用前 k 个种子（默认 1，够用且快）")
    ap.add_argument("--max", type=int, default=300, help="每条件最多用多少样本")
    A = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

    out = {}
    for cond in A.conds:
        cp = os.path.join(OUTD, "cache_%s.npz" % cond)
        if not os.path.exists(cp):
            print("跳过 %s（无缓存）" % cond); continue
        z = np.load(cp)
        noisy, mask = z["testA_n"][:A.max], z["testA_m"][:A.max]
        for mn in A.models:
            cfg = PULSE_MODELS[mn]
            for si in range(min(A.seeds, len(cfg["ckpts"]))):
                model = PN.PAMNet(c=24, n_fft=V3.CFG["n_fft"], hop=V3.CFG["hop_length"],
                                  norm="gn", **cfg["arch"]).to(V3.DEVICE).eval()
                model.load_state_dict(torch.load(os.path.join(CKPT, cfg["ckpts"][si]),
                                                 map_location=V3.DEVICE), strict=True)
                d = diag(model, noisy, mask)
                out.setdefault(cond, {}).setdefault(mn, {})["s%d" % si] = d
                print("[%s|%s|s%d] AUC 均值 %.4f 中位 %.4f (>0.8 占 %.0f%%)  "
                      "p_imp %.4f  p_clean %.4f  分离度 %.4f"
                      % (cond, mn, si, d["auc_mean"], d["auc_median"],
                         100 * d["frac_auc_gt08"], d["p_imp"], d["p_clean"], d["sep"]),
                      flush=True)
    p = os.path.join(OUTD, "pulse_diag_sens.json")
    old = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    for k, v in out.items():
        for k2, v2 in v.items():
            old.setdefault(k, {}).setdefault(k2, {}).update(v2)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(old, f, ensure_ascii=False, indent=1)
    print("-> %s" % p)


if __name__ == "__main__":
    main()
