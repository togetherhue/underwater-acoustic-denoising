# -*- coding: utf-8 -*-
"""
阶段1：PAM-Net 训练与消融
================================================================================
数据管线与评估口径【完全复用 train_pam.py】—— 同一个 PAMDataset、同一个
run_eval、同一套 pam_metrics 指标，因此与阶段0 的基线（干净帧 14.13 /
干扰帧 25.09 / 干扰帧最差 13.54）严格可比。

新增内容只有两件：
  1. 模型换成 pamnet.PAMNet（导师方案的脉冲感知多尺度轻量化网络）
  2. 损失换成 L = L_Huber + λ1·L_imp + λ2·L_spec

消融开关（对应导师要求的"验证创新性"）：
  --short/--long   多尺度双分支的有无
  --gate           门控融合 vs 简单平均
  --pulse          脉冲感知分支的有无   ← 核心创新点
  --loss           稳健损失（huber）vs 普通 L1
  --freq-refine    附加频域精修（检验"时域残差估计 vs 频域掩码"的张力）

诊断量：
  p_imp / p_clean  脉冲感知分支在【干扰位置】与【干净位置】的平均置信度。
                   若 p_imp 明显高于 p_clean，说明该分支确实学到了"哪里是脉冲"，
                   而不是输出一个常数。这是导师方案创新点的直接证据。

用法：
  python train_pamnet.py --tag pamnet --data D:/project/dataset_pam/alphastable \
      --epochs 15
  python train_pamnet.py --tag pamnet_nopulse --data ... --pulse 0
"""
import os, sys, time, argparse, csv
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import v3_model as V3          # noqa: E402
import pamnet as PN            # noqa: E402
import pam_metrics as PM       # noqa: E402
from train_pam import PAMDataset, build_loader, run_eval, COLS   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", required=True, help="数据集根目录（含 train/val/testA/testC）")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--c", type=int, default=24, help="基础通道数")
    ap.add_argument("--norm", default="bn", choices=["bn", "gn", "in"],
                    help="归一化方式。bn=BatchNorm（依赖 batch 统计）；"
                         "gn=GroupNorm / in=InstanceNorm（不依赖 batch，train/eval 一致）。"
                         "诊断显示 BN 的 running stats 能把验证指标推动 5.3 dB，"
                         "是逐轮振荡的主要来源之一")
    ap.add_argument("--short", type=int, default=1, help="启用短时尺度分支")
    ap.add_argument("--long", type=int, default=1, help="启用长时尺度分支")
    ap.add_argument("--gate", type=int, default=1, help="门控融合（0=简单平均）")
    ap.add_argument("--pulse", type=int, default=1, help="脉冲感知分支")
    ap.add_argument("--freq-refine", type=float, default=0.0, dest="freq_refine",
                    help=">0 时附加频域精修分支，取值即通道宽度系数（2.0 = 阶段0 胜出配置）")
    ap.add_argument("--pulse-freq", type=int, default=0, dest="pulse_freq",
                    help="把脉冲置信度 p_n 作为附加通道【喂进频域掩码分支】"
                         "（需要同时开启 --pulse 与 --freq-refine）")
    ap.add_argument("--loss", default="huber", choices=["huber", "l1"])
    ap.add_argument("--select", default="fg", choices=["fg", "val_loss"],
                    help="模型选择准则。fg=验证集逐帧降噪增益中位（默认，与评估口径一致）；"
                         "val_loss=最小化验证损失（阶段0 的旧做法，用于受控对比时复现其协议）")
    ap.add_argument("--delta", type=float, default=0.2, help="Huber 的 δ")
    ap.add_argument("--lam-imp", type=float, default=1.0, dest="lam_imp")
    ap.add_argument("--lam-spec", type=float, default=0.05, dest="lam_spec",
                    help="谱保真项权重。默认 0.05 由 calibrate_loss.py 确定："
                         "torch.stft 未归一化的幅度尺度约为时域残差的 200 倍，"
                         "若沿用 v3 的 0.5 会让谱项压倒性主导、淹没 Huber 的稳健性")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--input-norm", type=int, default=0, dest="input_norm",
                    help="输入尺度归一化 x/rms(x)（训练与评估同步）。"
                         "开启后模型对输入整体尺度不敏感，实测数据可直接用")
    ap.add_argument("--outdir", default=r"D:/project/output_pamnet")
    A = ap.parse_args()
    os.makedirs(A.outdir, exist_ok=True)
    SEED = V3.CFG["seed"] if A.seed is None else A.seed

    d_tr = os.path.join(A.data, "train")
    d_va = os.path.join(A.data, "val")
    tests = [("testA", os.path.join(A.data, "testA")),
             ("testC", os.path.join(A.data, "testC"))]
    for d in [d_tr, d_va] + [p for _, p in tests]:
        assert os.path.isdir(d), "缺少目录 " + d

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 100)
    print("阶段1：PAM-Net（脉冲感知多尺度轻量化网络）  |  tag = %s" % A.tag)
    print("=" * 100)
    print("数据集      :", A.data)
    print("设备        :", V3.DEVICE)
    print("组件        : 短时=%d 长时=%d 门控=%d 脉冲感知=%d  频域精修=%s  p注入频域=%d"
          % (A.short, A.long, A.gate, A.pulse,
             A.freq_refine if A.freq_refine > 0 else "关", A.pulse_freq))
    print("架构        : %s" % ("纯频域掩码（无时域分支，用于与阶段0 受控对比）"
                                if not (A.short or A.long) else "PAM-Net 时域+可选频域精修"))
    print("归一化      : %s" % A.norm)
    print("模型选择    : %s" % ("验证帧增益中位（fg）" if A.select == "fg"
                                else "验证损失最小（val_loss，阶段0 旧做法）"))
    print("损失        : %s  δ=%.3f  λ_imp=%.2f  λ_spec=%.2f  seed=%d"
          % (A.loss, A.delta, A.lam_imp, A.lam_spec, SEED))

    tr = build_loader(d_tr, True, A.bs, A.workers, A.input_norm)
    va = build_loader(d_va, False, A.bs, A.workers, A.input_norm)
    print("训练样本 %d   验证样本 %d   每轮 batch %d" %
          (len(tr.dataset), len(va.dataset), len(tr)))

    model = PN.PAMNet(c=A.c, use_short=bool(A.short), use_long=bool(A.long),
                      use_gate=bool(A.gate), use_pulse=bool(A.pulse),
                      freq_refine=A.freq_refine, pulse_freq=bool(A.pulse_freq),
                      norm=A.norm,
                      n_fft=V3.CFG["n_fft"], hop=V3.CFG["hop_length"]).to(V3.DEVICE)
    bd = model.param_breakdown()
    print("参数量      : {:,}   明细 {}".format(
        model.n_params(), " ".join("%s=%d" % (k, v) for k, v in bd.items())))
    print("=" * 100)

    opt = torch.optim.Adam(model.parameters(), lr=A.lr, weight_decay=V3.CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=A.epochs)
    score_of = ((lambda vl, fg: fg) if A.select == "fg" else (lambda vl, fg: -vl))
    best_score, best_path = -float("inf"), os.path.join(A.outdir, "model_%s.pth" % A.tag)

    logf = open(os.path.join(A.outdir, "train_log_%s.txt" % A.tag), "w", encoding="utf-8")
    logf.write("epoch,train_loss,val_loss,l_hub,l_imp,l_spec,val_snr_in,val_snr_out,"
               "p_imp,p_clean,val_fg_med,val_fg_clean,val_fg_imp,val_fg_impmin,lr\n")

    for ep in range(1, A.epochs + 1):
        # ---------------- 训练 ----------------
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        acc = {"huber": 0.0, "imp": 0.0, "spec": 0.0}
        for noisy, clean, mask in tr:
            noisy = noisy.unsqueeze(1).to(V3.DEVICE)
            clean = clean.unsqueeze(1).to(V3.DEVICE)
            mask = mask.unsqueeze(1).to(V3.DEVICE)
            opt.zero_grad()
            est, p = model(noisy, return_p=True)
            loss, parts = PN.pam_criterion(
                est, clean, p if A.pulse else None, mask if A.pulse else None,
                V3.CFG["n_fft"], V3.CFG["hop_length"],
                delta=A.delta, lam_imp=A.lam_imp, lam_spec=A.lam_spec,
                loss_kind=A.loss, return_parts=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item()); nb += 1
            for k in acc:
                acc[k] += parts[k]
        sched.step()
        trl = tot / max(nb, 1)
        acc = {k: v / max(nb, 1) for k, v in acc.items()}

        # ---------------- 验证 ----------------
        # 注意：模型选择【不用 val_loss】。val_loss 被谱保真项主导，且全局类指标在
        # 脉冲噪声下会被个别脉冲主导（阶段0 已证明全局 SNR 会把"漏掉脉冲"判成更好）。
        # 这里改用与评估口径一致的【逐帧降噪增益中位数】作为选择准则。
        model.eval()
        vtot, vn, si, so = 0.0, 0, [], []
        p_imp_sum, p_clean_sum, n_imp_px, n_clean_px = 0.0, 0.0, 0, 0
        fg = {"frmgain_med": [], "frmgain_clean": [], "frmgain_imp": [], "frmgain_imp_min": []}
        with torch.no_grad():
            for noisy, clean, mask in va:
                noisy = noisy.unsqueeze(1).to(V3.DEVICE)
                clean = clean.unsqueeze(1).to(V3.DEVICE)
                mask = mask.unsqueeze(1).to(V3.DEVICE)
                est, p = model(noisy, return_p=True)
                vtot += PN.pam_criterion(
                    est, clean, p if A.pulse else None, mask if A.pulse else None,
                    V3.CFG["n_fft"], V3.CFG["hop_length"],
                    delta=A.delta, lam_imp=A.lam_imp, lam_spec=A.lam_spec,
                    loss_kind=A.loss).item()
                vn += 1
                si.append(V3.snr_db(noisy.squeeze(1), clean.squeeze(1)))
                so.append(V3.snr_db(est.squeeze(1), clean.squeeze(1)))
                if p is not None:
                    m = mask.bool()
                    p_imp_sum += float(p[m].sum()); n_imp_px += int(m.sum())
                    p_clean_sum += float(p[~m].sum()); n_clean_px += int((~m).sum())
                for k in range(noisy.shape[0]):
                    st = PM._stats_1d(est[k, 0].cpu().numpy().astype(np.float64),
                                      clean[k, 0].cpu().numpy().astype(np.float64),
                                      noisy[k, 0].cpu().numpy().astype(np.float64),
                                      PM.FRAME, mask[k, 0].cpu().numpy())
                    for key in fg:
                        v = st.get(key)
                        if v is not None and not np.isnan(v):
                            fg[key].append(v)
        vl = vtot / max(vn, 1)
        vsi, vso = torch.cat(si).mean().item(), torch.cat(so).mean().item()
        p_imp = p_imp_sum / n_imp_px if n_imp_px else float("nan")
        p_clean = p_clean_sum / n_clean_px if n_clean_px else float("nan")
        fg_med = float(np.mean(fg["frmgain_med"]))
        fg_clean = float(np.mean(fg["frmgain_clean"]))
        fg_imp = float(np.mean(fg["frmgain_imp"]))
        fg_impmin = float(np.mean(fg["frmgain_imp_min"]))
        lr = opt.param_groups[0]["lr"]

        print("[epoch %3d/%d] train=%.4f val=%.4f (hub %.4f imp %.4f spec %.4f)  "
              "valSNR %+.2f->%+.2f  p_imp=%.3f p_clean=%.3f  "
              "val帧增益 中位 %+.2f 干净 %+.2f 干扰 %+.2f (最差 %+.2f)  %.0fs"
              % (ep, A.epochs, trl, vl, acc["huber"], acc["imp"], acc["spec"],
                 vsi, vso, p_imp, p_clean, fg_med, fg_clean, fg_imp, fg_impmin,
                 time.time() - t0), flush=True)
        logf.write("%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.6f,%.6f,"
                   "%.4f,%.4f,%.4f,%.4f,%.3e\n"
                   % (ep, trl, vl, acc["huber"], acc["imp"], acc["spec"],
                      vsi, vso, p_imp, p_clean, fg_med, fg_clean, fg_imp, fg_impmin, lr))
        logf.flush()
        sc = score_of(vl, fg_med)
        if sc > best_score:
            best_score = sc
            torch.save(model.state_dict(), best_path)
    logf.close()
    print("最佳验证%s %.4f -> %s"
          % ("帧增益中位(dB)" if A.select == "fg" else "损失(取负)", best_score, best_path))

    # ---------------- 评估（口径与阶段0 完全一致）----------------
    model.load_state_dict(torch.load(best_path, map_location=V3.DEVICE))
    model = model.to(V3.DEVICE).eval()
    print("\n" + "=" * 100)
    print("评估（帧长 %d 点；增益 = 该帧输入噪声/输出噪声）" % PM.FRAME)
    print("=" * 100)
    print("%-7s %7s %8s %8s | %9s %9s | %9s %9s %9s %9s"
          % ("测试集", "n", "全局SNR", "SI-SDR", "帧SNR入", "帧SNR出",
             "增益中位", "增益干净", "增益干扰", "干扰最差"))
    print("-" * 100)
    summary = []
    for name, folder in tests:
        R = run_eval(model, folder, A.bs, A.workers, A.input_norm)
        np.savetxt(os.path.join(A.outdir, "result_%s_%s.csv" % (A.tag, name)), R,
                   delimiter=",", header=",".join(COLS), comments="")
        m = np.nanmean(R, axis=0)
        summary.append((name, len(R)) + tuple(m))
        nn = int(np.isnan(R[:, 8]).sum())
        print("%-7s %7d %8.2f %8.2f | %9.2f %9.2f | %9.2f %9.2f %9.2f %9.2f%s"
              % (name, len(R), m[1], m[3], m[4], m[5], m[6], m[7], m[8], m[9],
                 "" if nn == 0 else "   (干扰帧指标有 %d 个样本不可算)" % nn))
    print("-" * 100)
    for row in summary:
        m = row[2:]
        print("  [%s] 全局SNR提升 %+.2f   帧SNR中位提升 %+.2f   帧增益中位 %+.2f"
              "   干扰帧增益 %+.2f (最差 %+.2f)"
              % (row[0], m[1] - m[0], m[5] - m[4], m[6], m[8], m[9]))

    with open(os.path.join(A.outdir, "summary_%s.csv" % A.tag), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tag", "testset", "n", "params", "c", "norm", "use_short", "use_long",
                    "use_gate", "use_pulse", "freq_refine", "pulse_freq", "loss", "delta",
                    "lam_imp", "lam_spec", "select", "epochs", "seed", "input_norm"] + COLS)
        for row in summary:
            w.writerow([A.tag, row[0], row[1], model.n_params(), A.c, A.norm, A.short, A.long,
                        A.gate, A.pulse, A.freq_refine, A.pulse_freq, A.loss, A.delta,
                        A.lam_imp, A.lam_spec, A.select, A.epochs, SEED]
                       + [A.input_norm] + ["%.4f" % v for v in row[2:]])
    print("结果 -> %s/summary_%s.csv" % (A.outdir, A.tag))


if __name__ == "__main__":
    main()
