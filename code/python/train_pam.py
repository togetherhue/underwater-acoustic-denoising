# -*- coding: utf-8 -*-
"""
阶段0：现有模型在【α稳定 + 突发干扰】噪声下的表现 + 时域分支复检

目的（回答三个问题）：
  ① 现有模型从高斯噪声换到脉冲噪声，掉多少？
  ② 时域分支是否终于有用了？（高斯下测得 w_time 从 0.297 衰减到 0.006）
  ③ 全局 SNR 与稳健指标的差异有多大？

本脚本【不改网络、不改损失】，只换数据集与评估指标，因此与既有高斯结果
严格可比。这正是阶段0 的意义：拿到干净基线，再决定后续改动方向。

数据集的每个样本含 imp_mask（干扰位置掩码），用于把指标分解成
"干净帧"与"干扰帧"两部分 —— 这是 PAM-Net 的核心关注点。

用法：
  python train_pam.py --tag pam_fch2 --data D:/project/dataset_pam/alphastable \
      --epochs 15 --time 0 --fch 2.0
  python train_pam.py --tag pam_full --data D:/project/dataset_pam/alphastable \
      --epochs 15 --time 1 --freq 1
"""
import os, sys, time, argparse, csv, glob, shutil
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly
import torch
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "大创_v3.py")
_DST = os.path.join(_HERE, "v3_model.py")
if (not os.path.exists(_DST)) or (os.path.getmtime(_SRC) > os.path.getmtime(_DST)):
    shutil.copy2(_SRC, _DST)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import v3_model as V3          # noqa: E402
import pam_metrics as PM       # noqa: E402


class PAMDataset(Dataset):
    """带干扰掩码的数据集。放在模块顶层以保证多进程 spawn 可 pickle。

    优先使用同目录上一级的 `_npy/<split>_*.npy` 缓存（由 make_npy_cache.py 生成）：
    缓存与 .mat 直接解析【逐位一致】（已校验差 0.000e+00），但读取快一个数量级，
    且用 mmap 避免多 worker 各拷一份。没有缓存时自动回退到 .mat。
    """

    def __init__(self, folder, seg_len, ds, rms_norm=False):
        self.files = sorted(glob.glob(os.path.join(folder, "*.mat")))
        if not self.files:
            raise FileNotFoundError(folder + " 里没有 .mat 文件")
        self.seg_len, self.ds = seg_len, ds
        # 输入尺度归一化：把 noisy / clean 同除以 noisy 的 rms。
        # 训练数据在 MATLAB 侧把 rms(clean) 归一化为 1，模型对输入尺度极度敏感，
        # 而实测数据只能按 noisy 归一化 —— 开启后模型即不受输入尺度影响。
        self.rms_norm = bool(rms_norm)
        # ---- .npy 缓存探测 ----
        self.npy = None
        split = os.path.basename(os.path.normpath(folder))
        cache_dir = os.path.join(os.path.dirname(os.path.normpath(folder)), "_npy")
        cand = [os.path.join(cache_dir, "%s_%s.npy" % (split, t))
                for t in ("n", "c", "m")]
        if all(os.path.exists(p) for p in cand):
            try:
                arrs = [np.load(p, mmap_mode="r") for p in cand]
                if all(a.shape[0] == len(self.files) and a.shape[1] == seg_len
                       for a in arrs):
                    self.npy = arrs
            except Exception:
                self.npy = None

    def __len__(self):
        return len(self.files)

    def _fix(self, x):
        L = self.seg_len
        return x[:L] if len(x) >= L else np.pad(x, (0, L - len(x)))

    def _maybe_norm(self, n, c):
        """rms_norm 开启时同除 noisy 的 rms；否则原样返回。"""
        if not self.rms_norm:
            return n, c
        s = float(np.sqrt(np.mean(np.asarray(n, np.float64) ** 2)))
        if not np.isfinite(s) or s <= 0:
            return n, c
        return n / s, c / s

    def __getitem__(self, i):
        if self.npy is not None:
            n_arr, c_arr, m_arr = self.npy
            n = np.array(n_arr[i], dtype=np.float32)
            c = np.array(c_arr[i], dtype=np.float32)
            k = np.array(m_arr[i], dtype=np.float32)
            n, c = self._maybe_norm(n, c)
            return (torch.from_numpy(n), torch.from_numpy(c), torch.from_numpy(k))
        m = sio.loadmat(self.files[i])
        n = np.asarray(m["noisy_signal"]).squeeze().astype(np.float32)
        c = np.asarray(m["clean_signal"]).squeeze().astype(np.float32)
        k = np.asarray(m["imp_mask"]).squeeze().astype(np.float64)
        if self.ds > 1:
            n = resample_poly(n, 1, self.ds).astype(np.float32)
            c = resample_poly(c, 1, self.ds).astype(np.float32)
            k = resample_poly(k, 1, self.ds)
        k = (self._fix(k) > 0.3).astype(np.float32)
        # 注意：必须【先定长再算 rms】，与 _npy 路径（本就是定长数组）保持同一尺度定义，
        # 否则换数据源会导致训练/测试的输入尺度系统性偏差 sqrt(有效长度/8000)。
        n, c = self._maybe_norm(self._fix(n), self._fix(c))
        return (torch.from_numpy(n), torch.from_numpy(c), torch.from_numpy(k))


def build_loader(folder, shuffle, bs, workers, rms_norm=False):
    ds = PAMDataset(folder, V3.CFG["seg_len"], V3.CFG["downsample"], rms_norm)
    return torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=shuffle, num_workers=workers,
        drop_last=shuffle, pin_memory=(V3.DEVICE.type == "cuda"))


COLS = ["snr_in", "snr_out", "sisdr_in", "sisdr_out",
        "fmsnr_med_in", "fmsnr_med_out", "frmgain_med", "frmgain_clean",
        "frmgain_imp", "frmgain_imp_min", "fmsnr_frac_bad_out", "frac_imp"]


@torch.no_grad()
def run_eval(model, folder, bs, workers, rms_norm=False):
    """逐样本计算全部指标，返回 dict(列名 -> np.array)

    rms_norm 必须与训练时一致：若是，则输入与目标同除 noisy 的 rms，
    指标在归一化域里计算 —— 由于各帧增益/SNR 都是尺度不变量，结果与
    在原尺度上评估等价，但模型实际见到的输入分布与训练时相同。
    """
    loader = build_loader(folder, False, bs, workers, rms_norm)
    rows = []
    for noisy, clean, mask in loader:
        n = noisy.unsqueeze(1).to(V3.DEVICE)
        c = clean.unsqueeze(1).to(V3.DEVICE)
        e = model(n)
        for k in range(n.shape[0]):
            ei, ri, ni = e[k, 0].cpu(), c[k, 0].cpu(), n[k, 0].cpu()
            mi = mask[k].numpy()
            si = PM.snr_db(ni, ri).item()
            so = PM.snr_db(ei, ri).item()
            di = PM.si_sdr_db(ni, ri).item()
            do = PM.si_sdr_db(ei, ri).item()
            a = PM._stats_1d(ei.numpy().astype(np.float64), ri.numpy().astype(np.float64),
                             ni.numpy().astype(np.float64), PM.FRAME, mi)
            b = PM._stats_1d(ni.numpy().astype(np.float64), ri.numpy().astype(np.float64),
                             ni.numpy().astype(np.float64), PM.FRAME, mi)
            rows.append([si, so, di, do, b["fmsnr_med"], a["fmsnr_med"],
                         a["frmgain_med"], a["frmgain_clean"], a["frmgain_imp"],
                         a["frmgain_imp_min"], a["fmsnr_frac_bad"], a["frac_imp"]])
    return np.array(rows, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", required=True, help="数据集根目录（含 train/val/testA/testC）")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--fch", type=float, default=2.0)
    ap.add_argument("--time", type=int, default=0)
    ap.add_argument("--freq", type=int, default=1)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--outdir", default=r"D:/project/output_pam")
    A = ap.parse_args()
    os.makedirs(A.outdir, exist_ok=True)
    SEED = V3.CFG["seed"] if A.seed is None else A.seed

    d_tr = os.path.join(A.data, "train")
    d_va = os.path.join(A.data, "val")
    tests = [("testA", os.path.join(A.data, "testA")),
             ("testC", os.path.join(A.data, "testC"))]
    for d in [d_tr, d_va] + [p for _, p in tests]:
        assert os.path.isdir(d), "缺少目录 " + d

    print("=" * 92)
    print("阶段0：脉冲噪声下的现有模型  |  tag = %s" % A.tag)
    print("=" * 92)
    print("数据集      :", A.data)
    print("设备        :", V3.DEVICE)
    print("架构        : 时域=%d 频域=%d fch=%.2f   seed=%d" % (A.time, A.freq, A.fch, SEED))

    tr = build_loader(d_tr, True, A.bs, A.workers)
    va = build_loader(d_va, False, A.bs, A.workers)
    print("训练样本 %d   验证样本 %d   每轮 batch %d" %
          (len(tr.dataset), len(va.dataset), len(tr)))

    torch.manual_seed(SEED); np.random.seed(SEED)
    model = V3.DenoiseNet(V3.CFG["n_fft"], V3.CFG["hop_length"], A.hidden,
                          use_time=bool(A.time), use_freq=bool(A.freq),
                          fch=A.fch).to(V3.DEVICE)
    print("参数量      : {:,}".format(model.n_params()))
    print("=" * 92)

    opt = torch.optim.Adam(model.parameters(), lr=V3.CFG["lr"],
                           weight_decay=V3.CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=A.epochs)
    best_val, best_path = float("inf"), os.path.join(A.outdir, "model_%s.pth" % A.tag)
    logf = open(os.path.join(A.outdir, "train_log_%s.txt" % A.tag), "w", encoding="utf-8")
    logf.write("epoch,train_loss,val_loss,val_snr_in,val_snr_out,w_time,lr\n")

    for ep in range(1, A.epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for noisy, clean, _ in tr:
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
            for noisy, clean, _ in va:
                noisy = noisy.unsqueeze(1).to(V3.DEVICE)
                clean = clean.unsqueeze(1).to(V3.DEVICE)
                est = model(noisy)
                vtot += V3.criterion(est, clean, V3.CFG["n_fft"], V3.CFG["hop_length"]).item()
                vn += 1
                si.append(V3.snr_db(noisy.squeeze(1), clean.squeeze(1)))
                so.append(V3.snr_db(est.squeeze(1), clean.squeeze(1)))
        vl = vtot / max(vn, 1)
        vsi, vso = torch.cat(si).mean().item(), torch.cat(so).mean().item()

        wt = float("nan")
        if hasattr(model, "w_time") and hasattr(model, "w_freq"):
            w = torch.stack([model.w_time, model.w_freq]).softmax(0)
            wt = w[0].item()

        print("[epoch %3d/%d] train=%.4f val=%.4f  valSNR %+.2f->%+.2f (%+.2f)  "
              "w_time=%.4f  %.0fs"
              % (ep, A.epochs, trl, vl, vsi, vso, vso - vsi, wt, time.time() - t0), flush=True)
        logf.write("%d,%.6f,%.6f,%.4f,%.4f,%.6f,%.3e\n" %
                   (ep, trl, vl, vsi, vso, wt, opt.param_groups[0]["lr"]))
        logf.flush()
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), best_path)
    logf.close()
    print("最佳验证损失 %.4f -> %s" % (best_val, best_path))

    # ---------------- 评估 ----------------
    model.load_state_dict(torch.load(best_path, map_location=V3.DEVICE))
    model = model.to(V3.DEVICE).eval()
    wt = float("nan")
    if hasattr(model, "w_time") and hasattr(model, "w_freq"):
        w = torch.stack([model.w_time, model.w_freq]).softmax(0)
        wt = w[0].item()
    print("\n" + "=" * 92)
    print("评估（帧长 %d 点；增益 = 该帧输入噪声/输出噪声）" % PM.FRAME)
    print("=" * 92)
    print("%-7s %7s %8s %8s | %9s %9s | %9s %9s %9s %9s"
          % ("测试集", "n", "全局SNR", "SI-SDR", "帧SNR入", "帧SNR出",
             "增益中位", "增益干净", "增益干扰", "干扰最差"))
    print("-" * 92)
    summary = []
    for name, folder in tests:
        R = run_eval(model, folder, A.bs, A.workers)
        np.savetxt(os.path.join(A.outdir, "result_%s_%s.csv" % (A.tag, name)), R,
                   delimiter=",", header=",".join(COLS), comments="")
        m = np.nanmean(R, axis=0)   # 个别样本干扰帧全落在零填充区 -> 用 nanmean
        summary.append((name, len(R)) + tuple(m))
        nn = int(np.isnan(R[:, 8]).sum())
        print("%-7s %7d %8.2f %8.2f | %9.2f %9.2f | %9.2f %9.2f %9.2f %9.2f%s"
              % (name, len(R), m[1], m[3], m[4], m[5], m[6], m[7], m[8], m[9],
                 "" if nn == 0 else "   (干扰帧指标有 %d 个样本不可算)" % nn))
    print("-" * 92)
    for row in summary:
        m = row[2:]
        print("  [%s] 全局SNR提升 %+.2f   帧SNR中位提升 %+.2f   帧增益中位 %+.2f"
              "   干扰帧增益 %+.2f (最差 %+.2f)"
              % (row[0], m[1] - m[0], m[5] - m[4], m[6], m[8], m[9]))

    with open(os.path.join(A.outdir, "summary_%s.csv" % A.tag), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tag", "testset", "n", "params", "fch", "arch_time", "arch_freq",
                    "w_time", "epochs", "seed"] + COLS)
        for row in summary:
            w.writerow([A.tag, row[0], row[1], model.n_params(), A.fch,
                        A.time, A.freq, "%.4f" % wt, A.epochs, SEED] + ["%.4f" % v for v in row[2:]])
    print("结果 -> %s/summary_%s.csv" % (A.outdir, A.tag))


if __name__ == "__main__":
    main()
