# -*- coding: utf-8 -*-
"""
轻量化水声信号降噪模型 v3 —— 2x2 因子泛化实验 + 消融
============================================================================
和 v2 的区别
----------------------------------------------------------------------------
1. 数据集换成 dataset_v3（信道与波形【同时】分组），一次训练后在 3 个测试集评估：
       testA  新波形 + 见过的信道   ← 和 v2 的 19.85 dB 对照
       testB  见过的波形 + 新信道   ← v2 缺失的「信道泛化」
       testC  新波形 + 新信道       ← 最严格
2. 内置消融开关（--time / --freq），可跑「只用时域」「只用频域」两组对照
3. 命令行参数化，默认无参数运行 = 完整模型，和 v2 行为一致

【运行】
    python 大创_v3.py                       # 完整模型
    python 大创_v3.py --tag timeonly --freq 0
    python 大创_v3.py --tag freqonly --time 0
============================================================================
"""

import os, glob, time, argparse, csv
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = r"D:/project/dataset_v3"

CFG = {
    "train_dir":  os.path.join(ROOT, "train"),
    "val_dir":    os.path.join(ROOT, "val"),
    "test_dirs":  [("testA_新波形_见信道", os.path.join(ROOT, "testA_seench_newchirp")),
                   ("testB_见波形_新信道", os.path.join(ROOT, "testB_newch_seenchirp")),
                   ("testC_新波形_新信道", os.path.join(ROOT, "testC_newch_newchirp"))],

    "fs_native":  24000,
    "downsample": 4,
    "seg_len":    8000,
    "n_fft":      512,
    "hop_length": 128,
    "hidden":     64,

    "batch_size":  16,
    "lr":          1e-3,
    "weight_decay": 1e-4,
    "num_epochs":  25,
    "num_workers": 4,
    "seed":        20261013,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================== 数据集 ===================================
class DenoiseDataset(Dataset):
    def __init__(self, folder, seg_len, ds):
        self.files = sorted(glob.glob(os.path.join(folder, "*.mat")))
        if not self.files:
            raise FileNotFoundError(f"{folder} 里没有 .mat 文件")
        self.seg_len, self.ds = seg_len, ds

    def __len__(self):
        return len(self.files)

    def _fix_len(self, x):
        L = self.seg_len
        return x[:L] if len(x) >= L else np.pad(x, (0, L - len(x)))

    def __getitem__(self, idx):
        m = sio.loadmat(self.files[idx])
        noisy = np.asarray(m["noisy_signal"]).squeeze().astype(np.float32)
        clean = np.asarray(m["clean_signal"]).squeeze().astype(np.float32)
        if self.ds > 1:
            noisy = resample_poly(noisy, 1, self.ds).astype(np.float32)
            clean = resample_poly(clean, 1, self.ds).astype(np.float32)
        return (torch.from_numpy(self._fix_len(noisy)),
                torch.from_numpy(self._fix_len(clean)))


# ========================= 深度可分离卷积模块 =============================
class DSConv1d(nn.Module):
    def __init__(self, cin, cout, k=3, p=1):
        super().__init__()
        self.dw = nn.Conv1d(cin, cin, k, padding=p, groups=cin)
        self.pw = nn.Conv1d(cin, cout, 1)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))))


class DSConv2d(nn.Module):
    def __init__(self, cin, cout, k=3, p=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, padding=p, groups=cin)
        self.pw = nn.Conv2d(cin, cout, 1)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))))


# ============================== 频域分支 =================================
class FrequencyBranch(nn.Module):
    """频域分支：预测幅度掩码 + 相位修正，输出干净信号的估计。

    w: 通道宽度系数。w=1.0 时通道为 16/32/16 与 8/4，
       与 v3 原始实现【逐位一致】—— 这一点由 verify_refactor 验证。
    """

    def __init__(self, n_fft, hop, w=1.0):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        c1, c2, c3 = max(1, round(16*w)), max(1, round(32*w)), max(1, round(16*w))
        p1, p2 = max(1, round(8*w)), max(1, round(4*w))
        self.widths = (c1, c2, c3, p1, p2)
        self.mag_mask = nn.Sequential(
            DSConv2d(1, c1, 3, 1), DSConv2d(c1, c2, 3, 1),
            DSConv2d(c2, c3, 3, 1), nn.Conv2d(c3, 1, 1), nn.Sigmoid())
        self.ph_corr = nn.Sequential(
            DSConv2d(1, p1, 3, 1), DSConv2d(p1, p2, 3, 1),
            nn.Conv2d(p2, 1, 1), nn.Tanh())

    def forward(self, x):
        B, _, L = x.shape
        win = torch.hann_window(self.n_fft, device=x.device)
        X = torch.stft(x.squeeze(1), n_fft=self.n_fft, hop_length=self.hop,
                       window=win, return_complex=True)
        mag, ph = torch.abs(X).unsqueeze(1), torch.angle(X)
        mask = self.mag_mask(mag)
        dphi = self.ph_corr(ph.unsqueeze(1)).squeeze(1)
        Y = (mag * mask).squeeze(1) * torch.exp(1j * (ph + 0.1 * dphi))
        y = torch.istft(Y, n_fft=self.n_fft, hop_length=self.hop,
                        window=win, length=L)
        return y.unsqueeze(1)


# ============================== 时域分支 =================================
class TimeBranch(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv1d(1, hidden, 1), nn.ReLU(), nn.Dropout(0.1))
        self.core = nn.ModuleList([
            nn.Sequential(DSConv1d(hidden, hidden*2, 3, 1),
                          DSConv1d(hidden*2, hidden, 3, 1),
                          nn.Dropout(0.1)) for _ in range(3)])
        self.out = nn.Sequential(DSConv1d(hidden, hidden//2, 3, 1),
                                 nn.Conv1d(hidden//2, 1, 1))

    def forward(self, x):
        f = self.inp(x)
        for layer in self.core:
            f = layer(f) + 0.5 * f
        return self.out(f)


# ============================== 主网络 ===================================
class DenoiseNet(nn.Module):
    def __init__(self, n_fft=512, hop=128, hidden=64, use_time=True, use_freq=True, fch=1.0):
        super().__init__()
        assert use_time or use_freq, "至少要启用一个分支"
        self.use_time, self.use_freq = use_time, use_freq
        self.fch = fch
        if use_time:
            self.time = TimeBranch(hidden)
            nn.init.zeros_(self.time.out[-1].weight)
            nn.init.zeros_(self.time.out[-1].bias)
        if use_freq:
            self.freq = FrequencyBranch(n_fft, hop, w=fch)
        if use_time and use_freq:
            self.w_time = nn.Parameter(torch.tensor(0.0))
            self.w_freq = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        if self.use_time and self.use_freq:
            d_t = self.time(x)
            d_f = self.freq(x) - x
            w = torch.softmax(torch.stack([self.w_time, self.w_freq]), dim=0)
            return x + w[0] * d_t + w[1] * d_f
        if self.use_time:
            return x + self.time(x)
        return self.freq(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def branch_weights(self):
        if self.use_time and self.use_freq:
            w = torch.softmax(torch.stack([self.w_time, self.w_freq]), dim=0)
            return float(w[0]), float(w[1])
        return None


# ============================== 损失 =====================================
def stft_mag(x, n_fft, hop):
    win = torch.hann_window(n_fft, device=x.device)
    return torch.stft(x.squeeze(1), n_fft=n_fft, hop_length=hop,
                      window=win, return_complex=True).abs()


def criterion(est, clean, n_fft, hop):
    return (F.l1_loss(est.squeeze(1), clean.squeeze(1))
            + 0.5 * F.l1_loss(stft_mag(est, n_fft, hop), stft_mag(clean, n_fft, hop)))


# ============================== 指标 =====================================
@torch.no_grad()
def snr_db(est, ref):
    return 10 * torch.log10(ref.pow(2).sum(dim=-1) /
                            ((est - ref).pow(2).sum(dim=-1) + 1e-12))


@torch.no_grad()
def si_sdr_db(est, ref):
    ref = ref - ref.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)
    a = (est * ref).sum(dim=-1, keepdim=True) / (ref.pow(2).sum(dim=-1, keepdim=True) + 1e-12)
    proj = a * ref
    noise = est - proj
    return 10 * torch.log10(proj.pow(2).sum(dim=-1) / (noise.pow(2).sum(dim=-1) + 1e-12))


# ============================== 训练 / 评估 ==============================
def build_loader(folder, shuffle):
    ds = DenoiseDataset(folder, CFG["seg_len"], CFG["downsample"])
    return DataLoader(ds, batch_size=CFG["batch_size"], shuffle=shuffle,
                      num_workers=CFG["num_workers"], drop_last=shuffle,
                      pin_memory=(DEVICE.type == "cuda"))


def run_eval(model, folder):
    loader = build_loader(folder, False)
    rows = []
    with torch.no_grad():
        for noisy, clean in loader:
            n = noisy.unsqueeze(1).to(DEVICE)
            c = clean.unsqueeze(1).to(DEVICE)
            e = model(n)
            for k in range(n.shape[0]):
                rows.append((snr_db(n[k, 0], c[k, 0]).item(), snr_db(e[k, 0], c[k, 0]).item(),
                             si_sdr_db(n[k, 0], c[k, 0]).item(), si_sdr_db(e[k, 0], c[k, 0]).item()))
    return np.array(rows)


def train(ARGS):
    outdir = ARGS.outdir
    os.makedirs(outdir, exist_ok=True)
    tag = ARGS.tag
    print("=" * 78)
    print("轻量化水声信号降噪模型 v3  |  变体 = %s" % tag)
    print("=" * 78)
    print(f"设备        : {DEVICE}")
    print(f"有效采样率  : {CFG['fs_native']//CFG['downsample']} Hz")
    print(f"分支        : 时域={'开' if ARGS.time else '关'}  频域={'开' if ARGS.freq else '关'}")

    tr = build_loader(CFG["train_dir"], True)
    va = build_loader(CFG["val_dir"], False)
    print(f"训练样本    : {len(tr.dataset)}   验证样本: {len(va.dataset)}")
    print(f"batch_size  : {CFG['batch_size']}   每轮 batch 数: {len(tr)}")

    model = DenoiseNet(CFG["n_fft"], CFG["hop_length"], ARGS.hidden,
                       use_time=bool(ARGS.time), use_freq=bool(ARGS.freq),
                       fch=ARGS.fch).to(DEVICE)
    print(f"频域分支宽度系数 : {ARGS.fch}  通道 {getattr(model.freq, 'widths', None) if ARGS.freq else '（未启用）'}")
    print(f"参数量      : {model.n_params():,}")
    print("=" * 78)

    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ARGS.epochs)

    best_val, best_path = float("inf"), os.path.join(outdir, f"model_{tag}.pth")
    logf = open(os.path.join(outdir, f"train_log_{tag}.txt"), "w", encoding="utf-8")
    logf.write("epoch,train_loss,val_loss,val_snr_in,val_snr_out,val_sisdr_in,val_sisdr_out,lr\n")

    for epoch in range(1, ARGS.epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for noisy, clean in tr:
            noisy = noisy.unsqueeze(1).to(DEVICE)
            clean = clean.unsqueeze(1).to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(noisy), clean, CFG["n_fft"], CFG["hop_length"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        tr_loss = tot / max(nb, 1)

        model.eval()
        vtot, vn, sin_, sout, din, dout = 0.0, 0, [], [], [], []
        with torch.no_grad():
            for noisy, clean in va:
                noisy = noisy.unsqueeze(1).to(DEVICE); clean = clean.unsqueeze(1).to(DEVICE)
                est = model(noisy)
                vtot += criterion(est, clean, CFG["n_fft"], CFG["hop_length"]).item(); vn += 1
                sin_.append(snr_db(noisy.squeeze(1), clean.squeeze(1)))
                sout.append(snr_db(est.squeeze(1), clean.squeeze(1)))
                din.append(si_sdr_db(noisy.squeeze(1), clean.squeeze(1)))
                dout.append(si_sdr_db(est.squeeze(1), clean.squeeze(1)))
        v_loss = vtot / max(vn, 1)
        v_sin, v_sout = torch.cat(sin_).mean().item(), torch.cat(sout).mean().item()
        v_din, v_dout = torch.cat(din).mean().item(), torch.cat(dout).mean().item()
        lr = opt.param_groups[0]["lr"]

        extra = ""
        bw = model.branch_weights()
        if bw: extra = "  w_time=%.3f w_freq=%.3f" % bw
        print(f"[epoch {epoch:3d}/{ARGS.epochs}] train={tr_loss:.4f} val={v_loss:.4f} "
              f"SNR {v_sin:+.2f}->{v_sout:+.2f} ({v_sout-v_sin:+.2f}) "
              f"SI-SDR {v_din:+.2f}->{v_dout:+.2f} lr={lr:.2e} {time.time()-t0:.0f}s{extra}",
              flush=True)
        logf.write(f"{epoch},{tr_loss:.6f},{v_loss:.6f},{v_sin:.4f},{v_sout:.4f},"
                   f"{v_din:.4f},{v_dout:.4f},{lr:.3e}\n")
        logf.flush()
        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), best_path)
            print("  ✓ 保存最佳模型 (val=%.4f)" % best_val)

    logf.close()
    print(f"\n训练完成，最佳验证损失 {best_val:.4f} -> {best_path}")
    return model, best_path


def evaluate_all(model, ARGS):
    print("\n" + "=" * 78)
    print("2x2 泛化实验：三个测试集")
    print("=" * 78)
    print("%-22s %7s %10s %10s %10s" % ("测试集", "样本", "输入SNR", "输出SNR", "提升"))
    print("-" * 78)
    summary = []
    for name, folder in CFG["test_dirs"]:
        if not os.path.isdir(folder):
            print("%-22s  (目录不存在，跳过)" % name); continue
        r = run_eval(model, folder)
        summary.append((name, len(r), r[:, 0].mean(), r[:, 1].mean(),
                        r[:, 1].mean() - r[:, 0].mean(), r[:, 3].mean() - r[:, 2].mean()))
        print("%-22s %7d %10.2f %10.2f %+10.2f" % (name, len(r), r[:, 0].mean(),
                                                   r[:, 1].mean(), r[:, 1].mean() - r[:, 0].mean()))
        np.savetxt(os.path.join(ARGS.outdir, f"result_{ARGS.tag}_{name.split('_')[0]}.csv"),
                   r, delimiter=",", header="snr_in,snr_out,sisdr_in,sisdr_out", comments="")
    print("-" * 78)
    npar = model.n_params()
    with open(os.path.join(ARGS.outdir, f"summary_{ARGS.tag}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tag", "testset", "n", "snr_in", "snr_out", "snr_gain", "sisdr_gain",
                    "params", "fch", "use_time", "use_freq"])
        for s in summary:
            w.writerow([ARGS.tag] + list(s) + [npar, ARGS.fch, ARGS.time, ARGS.freq])
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag",    default="full")
    ap.add_argument("--epochs", type=int, default=CFG["num_epochs"])
    ap.add_argument("--hidden", type=int, default=CFG["hidden"])
    ap.add_argument("--fch",    type=float, default=1.0, help="频域分支通道宽度系数")
    ap.add_argument("--time",   type=int, default=1)
    ap.add_argument("--freq",   type=int, default=1)
    ap.add_argument("--workers", type=int, default=CFG["num_workers"])
    ap.add_argument("--outdir", default=r"D:/project/output_v3")
    ARGS = ap.parse_args()
    CFG["num_workers"] = ARGS.workers

    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])
    t = time.time()
    model, path = train(ARGS)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    evaluate_all(model, ARGS)
    print(f"\n总耗时 {(time.time()-t)/60:.1f} 分钟")
