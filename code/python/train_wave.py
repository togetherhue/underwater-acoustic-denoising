# -*- coding: utf-8 -*-
"""
波形调制泛化实验：LFM（线性调频）vs HFM（双曲调频）

背景：导师 README 写明目标是「实测 HFM 信号」，但既有数据集全部用 LFM 生成。
本脚本训练三种模型，再在【同一批信道】上分别用 LFM / HFM 波形测试：

    lfmonly : 只用 LFM 训练
    hfmonly : 只用 HFM 训练
    mixed   : LFM + HFM 混合训练

测试集（chirp 18-20，训练从未出现）：
    LFM/testA  新 LFM 波形 + 见过的信道（1-18）
    LFM/testC  新 LFM 波形 + 新信道（22-25）
    HFM/testA  新 HFM 波形 + 见过的信道（1-18）
    HFM/testC  新 HFM 波形 + 新信道（22-25）

关键对照：LFM/testA 与 HFM/testA 用【完全相同的信道和 chirp 参数】，
只有调频规律不同 => 两者之差就是纯粹的「调制类型代价」。
经典带通滤波器是调制无关的，可作为参照。

模型与指标完全复用 大创_v3.py。

用法：
    python train_wave.py --tag lfmonly --train D:/project/dataset_wave/LFM/train \
        --val D:/project/dataset_wave/LFM/val --epochs 15 --time 0 --fch 2.0
    python train_wave.py --tag mixed \
        --train D:/project/dataset_wave/LFM/train,D:/project/dataset_wave/HFM/train \
        --val   D:/project/dataset_wave/LFM/val,D:/project/dataset_wave/HFM/val \
        --epochs 15 --time 0 --fch 2.0
"""
import os, sys, time, argparse, csv, shutil
import numpy as np
import torch
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "大创_v3.py")
_DST = os.path.join(_HERE, "v3_model.py")
if (not os.path.exists(_DST)) or (os.path.getmtime(_SRC) > os.path.getmtime(_DST)):
    shutil.copy2(_SRC, _DST)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import v3_model as V3   # noqa: E402

WAVE = r"D:/project/dataset_wave"
DEFAULT_TESTS = [("LFM_testA", os.path.join(WAVE, "LFM", "testA")),
                 ("LFM_testC", os.path.join(WAVE, "LFM", "testC")),
                 ("HFM_testA", os.path.join(WAVE, "HFM", "testA")),
                 ("HFM_testC", os.path.join(WAVE, "HFM", "testC"))]


class MultiDir(Dataset):
    """把多个目录拼成一个数据集（混合训练用）。放在模块顶层，保证可被 spawn pickle。"""

    def __init__(self, folders, seg_len, ds):
        self.dss = [V3.DenoiseDataset(f, seg_len, ds) for f in folders]
        self.lens = [len(d) for d in self.dss]
        self.cum = np.cumsum(self.lens)

    def __len__(self):
        return int(self.cum[-1])

    def __getitem__(self, i):
        k = int(np.searchsorted(self.cum, i, side="right"))
        off = i - (self.cum[k - 1] if k > 0 else 0)
        return self.dss[k][off]


def build_loader(folders, shuffle, bs, workers):
    ds = MultiDir(folders, V3.CFG["seg_len"], V3.CFG["downsample"])
    return torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=shuffle, num_workers=workers,
        drop_last=shuffle, pin_memory=(V3.DEVICE.type == "cuda"))


def run_eval(model, folder, bs, workers):
    loader = build_loader([folder], False, bs, workers)
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
    ap.add_argument("--tag", required=True)
    ap.add_argument("--train", required=True, help="逗号分隔的训练目录")
    ap.add_argument("--val", required=True, help="逗号分隔的验证目录")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--fch", type=float, default=2.0)
    ap.add_argument("--time", type=int, default=0)
    ap.add_argument("--freq", type=int, default=1)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None, help="缺省用 V3.CFG['seed']")
    ap.add_argument("--outdir", default=r"D:/project/output_wave")
    A = ap.parse_args()
    os.makedirs(A.outdir, exist_ok=True)
    SEED = V3.CFG["seed"] if A.seed is None else A.seed

    tr_dirs = [d.strip() for d in A.train.split(",") if d.strip()]
    va_dirs = [d.strip() for d in A.val.split(",") if d.strip()]
    for d in tr_dirs + va_dirs:
        assert os.path.isdir(d), "缺少目录 " + d

    print("=" * 82)
    print("波形调制泛化实验  |  tag = %s" % A.tag)
    print("=" * 82)
    print("设备        :", V3.DEVICE)
    print("分支        : 时域=%d 频域=%d fch=%.2f   seed=%d" % (A.time, A.freq, A.fch, SEED))
    print("训练目录    :", tr_dirs)
    print("验证目录    :", va_dirs)

    tr = build_loader(tr_dirs, True, A.bs, A.workers)
    va = build_loader(va_dirs, False, A.bs, A.workers)
    print("训练样本 %d   验证样本 %d   每轮 batch %d" %
          (len(tr.dataset), len(va.dataset), len(tr)))

    torch.manual_seed(SEED); np.random.seed(SEED)
    model = V3.DenoiseNet(V3.CFG["n_fft"], V3.CFG["hop_length"], A.hidden,
                          use_time=bool(A.time), use_freq=bool(A.freq),
                          fch=A.fch).to(V3.DEVICE)
    print("参数量      : {:,}".format(model.n_params()))
    print("=" * 82)

    opt = torch.optim.Adam(model.parameters(), lr=V3.CFG["lr"],
                           weight_decay=V3.CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=A.epochs)
    best_val, best_path = float("inf"), os.path.join(A.outdir, "model_%s.pth" % A.tag)
    logf = open(os.path.join(A.outdir, "train_log_%s.txt" % A.tag), "w", encoding="utf-8")
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

    # ---------- 评估：四个测试集 ----------
    model.load_state_dict(torch.load(best_path, map_location=V3.DEVICE))
    model = model.to(V3.DEVICE).eval()
    print("\n" + "=" * 82)
    print("四个测试集（同一批信道，LFM / HFM 波形）")
    print("=" * 82)
    print("%-12s %6s %8s %8s %9s %10s" %
          ("测试集", "n", "输入SNR", "输出SNR", "SNR提升", "SI-SDR提升"))
    print("-" * 82)
    rows = []
    for name, folder in DEFAULT_TESTS:
        if not os.path.isdir(folder):
            print("%-12s  (目录不存在)" % name); continue
        r = run_eval(model, folder, A.bs, A.workers)
        np.savetxt(os.path.join(A.outdir, "result_%s_%s.csv" % (A.tag, name)), r,
                   delimiter=",", header="snr_in,snr_out,sisdr_in,sisdr_out", comments="")
        g, gs = (r[:, 1] - r[:, 0]).mean(), (r[:, 3] - r[:, 2]).mean()
        rows.append((name, len(r), r[:, 0].mean(), r[:, 1].mean(), g, gs))
        print("%-12s %6d %8.2f %8.2f %+9.2f %+10.2f" %
              (name, len(r), r[:, 0].mean(), r[:, 1].mean(), g, gs))

    with open(os.path.join(A.outdir, "summary_%s.csv" % A.tag), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tag", "testset", "n", "snr_in", "snr_out", "snr_gain",
                    "sisdr_in", "sisdr_out", "sisdr_gain", "params", "fch",
                    "arch_time", "arch_freq", "train_dirs"])
        for name, n, si_, so_, g, gs in rows:
            w.writerow([A.tag, name, n, "%.4f" % si_, "%.4f" % so_, "%.4f" % g,
                        "", "", "%.4f" % gs, model.n_params(), A.fch,
                        A.time, A.freq, "|".join(tr_dirs)])
    print("\n结果 -> %s/summary_%s.csv" % (A.outdir, A.tag))


if __name__ == "__main__":
    main()
