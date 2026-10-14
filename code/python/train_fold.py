# -*- coding: utf-8 -*-
"""
3 折信道留出实验：训练 + 评估（seen / unseen 同波形对比）

设计（见 make_folds.m）：
  每折把 8-9 个信道留出。模型在剩余信道上训练，然后在【同一批测试波形】
  （chirp 18-20）上分别评估：
     test_seen    见过的信道
     test_unseen  没见过的信道
  两者之差 = 纯粹的信道新颖度代价。因为折是【按时延扩展分层】的，
  两边难度分布天然匹配。

模型与指标完全复用 大创_v3.py，避免实现不一致。

用法：
    python train_fold.py --fold 0 --epochs 25
    python train_fold.py --fold 1 --epochs 25 --tag fold1_full
"""
import os, sys, time, argparse, csv, json, shutil
import numpy as np
import scipy.io as sio
import torch

# ---- 复用 大创_v3.py 的模型/数据集/指标/损失 ----
# 注意：不能用 importlib 动态加载——DataLoader 的多进程 spawn 无法 pickle
#       动态模块里的类。这里自动同步一份同目录下的 v3_model.py 再正常导入。
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_HERE, "大创_v3.py")
_DST  = os.path.join(_HERE, "v3_model.py")
if (not os.path.exists(_DST)) or (os.path.getmtime(_SRC) > os.path.getmtime(_DST)):
    shutil.copy2(_SRC, _DST)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import v3_model as V3   # noqa: E402

ROOT = r"D:/project/dataset_folds"


def build_loader(folder, shuffle, bs, workers):
    ds = V3.DenoiseDataset(folder, V3.CFG["seg_len"], V3.CFG["downsample"])
    return torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=shuffle, num_workers=workers,
        drop_last=shuffle, pin_memory=(V3.DEVICE.type == "cuda"))


def run_eval(model, folder, bs, workers):
    loader = build_loader(folder, False, bs, workers)
    rows = []
    with torch.no_grad():
        for noisy, clean in loader:
            n = noisy.unsqueeze(1).to(V3.DEVICE)
            c = clean.unsqueeze(1).to(V3.DEVICE)
            e = model(n)
            for k in range(n.shape[0]):
                rows.append((V3.snr_db(n[k, 0], c[k, 0]).item(),
                             V3.snr_db(e[k, 0], c[k, 0]).item(),
                             V3.si_sdr_db(n[k, 0], c[k, 0]).item(),
                             V3.si_sdr_db(e[k, 0], c[k, 0]).item()))
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, help="1/2/3（与 dataset_folds/foldN 对应）")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--fch", type=float, default=1.0)
    ap.add_argument("--time", type=int, default=1)
    ap.add_argument("--freq", type=int, default=1)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--outdir", default=r"D:/project/output_folds")
    A = ap.parse_args()
    tag = A.tag or ("fold%d" % A.fold)
    os.makedirs(A.outdir, exist_ok=True)

    fdir = os.path.join(ROOT, "fold%d" % A.fold)
    d_tr = os.path.join(fdir, "train")
    d_va = os.path.join(fdir, "val")
    d_sn = os.path.join(fdir, "test_seen")
    d_un = os.path.join(fdir, "test_unseen")
    for d in (d_tr, d_va, d_sn, d_un):
        assert os.path.isdir(d), "缺少目录 " + d

    print("=" * 78)
    print("3 折信道留出实验  |  fold %d  |  tag = %s" % (A.fold, tag))
    print("=" * 78)
    print("设备        :", V3.DEVICE)
    print("分支        : 时域=%d 频域=%d fch=%.2f" % (A.time, A.freq, A.fch))

    tr = build_loader(d_tr, True, A.bs, A.workers)
    va = build_loader(d_va, False, A.bs, A.workers)
    print("训练样本 %d   验证样本 %d   每轮 batch %d" %
          (len(tr.dataset), len(va.dataset), len(tr)))

    torch.manual_seed(V3.CFG["seed"]); np.random.seed(V3.CFG["seed"])
    model = V3.DenoiseNet(V3.CFG["n_fft"], V3.CFG["hop_length"], A.hidden,
                          use_time=bool(A.time), use_freq=bool(A.freq),
                          fch=A.fch).to(V3.DEVICE)
    print("参数量      : {:,}".format(model.n_params()))
    print("=" * 78)

    opt = torch.optim.Adam(model.parameters(), lr=V3.CFG["lr"],
                           weight_decay=V3.CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=A.epochs)
    best_val, best_path = float("inf"), os.path.join(A.outdir, "model_%s.pth" % tag)
    logp = os.path.join(A.outdir, "train_log_%s.txt" % tag)
    logf = open(logp, "w", encoding="utf-8")
    logf.write("epoch,train_loss,val_loss,val_snr_in,val_snr_out,lr\n")

    for ep in range(1, A.epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for noisy, clean in tr:
            noisy = noisy.unsqueeze(1).to(V3.DEVICE)
            clean = clean.unsqueeze(1).to(V3.DEVICE)
            opt.zero_grad()
            loss = V3.criterion(model(noisy), clean, V3.CFG["n_fft"], V3.CFG["hop_length"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        trl = tot / max(nb, 1)

        model.eval()
        vtot, vn, si, so = 0.0, 0, [], []
        with torch.no_grad():
            for noisy, clean in va:
                noisy = noisy.unsqueeze(1).to(V3.DEVICE)
                clean = clean.unsqueeze(1).to(V3.DEVICE)
                est = model(noisy)
                vtot += V3.criterion(est, clean, V3.CFG["n_fft"], V3.CFG["hop_length"]).item()
                vn += 1
                si.append(V3.snr_db(noisy.squeeze(1), clean.squeeze(1)))
                so.append(V3.snr_db(est.squeeze(1), clean.squeeze(1)))
        vl = vtot / max(vn, 1)
        vsi, vso = torch.cat(si).mean().item(), torch.cat(so).mean().item()
        print("[epoch %3d/%d] train=%.4f val=%.4f  valSNR %+.2f->%+.2f (%+.2f)  %.0fs"
              % (ep, A.epochs, trl, vl, vsi, vso, vso - vsi, time.time() - t0), flush=True)
        logf.write("%d,%.6f,%.6f,%.4f,%.4f,%.3e\n" % (ep, trl, vl, vsi, vso,
                                                      opt.param_groups[0]["lr"]))
        logf.flush()
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), best_path)
    logf.close()
    print("最佳验证损失 %.4f -> %s" % (best_val, best_path))

    # ---------- 评估：同一批波形，seen vs unseen ----------
    model.load_state_dict(torch.load(best_path, map_location=V3.DEVICE))
    model = model.to(V3.DEVICE).eval()
    print("\n" + "=" * 78)
    print("同波形（chirp 18-20）对比：见过的信道 vs 没见过的信道")
    print("=" * 78)
    res = {}
    for nm, d in (("seen", d_sn), ("unseen", d_un)):
        r = run_eval(model, d, A.bs, A.workers)
        res[nm] = r
        np.savetxt(os.path.join(A.outdir, "result_%s_%s.csv" % (tag, nm)), r,
                   delimiter=",", header="snr_in,snr_out,sisdr_in,sisdr_out", comments="")
        print("%-8s n=%4d  输入 %6.2f  输出 %6.2f  提升 %+6.2f   SI-SDR提升 %+6.2f"
              % (nm, len(r), r[:, 0].mean(), r[:, 1].mean(),
                 (r[:, 1] - r[:, 0]).mean(), (r[:, 3] - r[:, 2]).mean()))

    gs = (res["seen"][:, 1] - res["seen"][:, 0]).mean()
    gu = (res["unseen"][:, 1] - res["unseen"][:, 0]).mean()
    print("-" * 78)
    print("★ 本折的信道新颖度代价 = %+.2f - (%+.2f) = %+.2f dB" % (gu, gs, gu - gs))

    with open(os.path.join(A.outdir, "summary_%s.csv" % tag), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tag", "fold", "arch_time", "arch_freq", "fch", "params",
                    "n_seen", "n_unseen", "gain_seen", "gain_unseen", "penalty_db"])
        w.writerow([tag, A.fold, A.time, A.freq, A.fch, model.n_params(),
                    len(res["seen"]), len(res["unseen"]),
                    "%.4f" % gs, "%.4f" % gu, "%.4f" % (gu - gs)])
    print("结果 -> %s" % os.path.join(A.outdir, "summary_%s.csv" % tag))


if __name__ == "__main__":
    main()
