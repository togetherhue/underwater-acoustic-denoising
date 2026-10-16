# -*- coding: utf-8 -*-
"""
轻量化水声信号降噪模型 —— 适配新数据集（v2）
============================================================================
数据集：D:/project/dataset/{train,val,test}  共 10000 样本
       每个 .mat 含 noisy_signal / clean_signal / true_snr / chirp_id / chan_id

相比旧版 大创.py 的改动
----------------------------------------------------------------------------
【必须改，否则跑不起来】
 1. 路径指向新数据集；按【变量名】读取，不再用 real_keys[0]
 2. 删掉 LayerNorm(18549) —— 新样本长度是 22399~31999 变化的，
    硬编码长度会直接崩；而且它占了全模型 37.5% 的参数
 3. 样本降采样 4 倍（24000 -> 6000 Hz）。信号频带只有 798~1500 Hz，
    Nyquist 3000 Hz 足够，长度从 32000 降到 8000，训练快 4 倍
 4. 统一长度到 seg_len，可正常 batch

【模型修正】
 5. 去掉 final_fusion 的 tanh —— 它把输出锁死在 (-1,1)，并且把残差路径压平
 6. 改成残差学习 output = x + delta，两个分支的末层零初始化
 7. 修掉 _init_weights 覆盖 signal_preservation 恒等初始化的 bug（该层已删）
 8. 频域分支去掉冗余的两级 Sigmoid 级联

【评估修正】
 9. 用 clean_signal 直接算真值 SNR 和 SI-SDR，不再用那个有 +6 dB 偏差的
    自造分段 SNR 估计

【运行】
    python 大创_v2.py
============================================================================
"""

import os
import glob
import time
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================== 配置 =====================================
CFG = {
    # ---- 路径 ----
    "train_dir": r"D:/project/dataset/train",
    "val_dir":   r"D:/project/dataset/val",
    "test_dir":  r"D:/project/dataset/test",
    "output_dir": r"D:/project/output_v2",
    "model_save_path": r"D:/project/denoise_model_v2.pth",

    # ---- 信号 ----
    "fs_native":  24000,     # 数据集原生采样率
    "downsample": 4,         # 降采样倍数 -> 有效 fs = 6000 Hz
    "seg_len":    8000,      # 降采样后统一长度（原 22399~31999 -> 5600~8000）

    # ---- 频域分支 ----
    "n_fft":      512,
    "hop_length": 128,

    # ---- 模型 ----
    "hidden":     64,

    # ---- 训练 ----
    "batch_size":  16,
    "lr":          1e-3,
    "weight_decay": 1e-4,
    "num_epochs":  50,
    "num_workers": 4,
    "seed":        20261013,
    "max_batches_per_epoch": 0,   # >0 时限制每轮 batch 数（调试用）
}

os.makedirs(CFG["output_dir"], exist_ok=True)
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================== 数据集 ===================================
class DenoiseDataset(Dataset):
    """读 .mat -> 降采样 -> 统一长度"""

    def __init__(self, folder, seg_len, ds):
        self.files = sorted(glob.glob(os.path.join(folder, "*.mat")))
        if not self.files:
            raise FileNotFoundError(f"{folder} 里没有 .mat 文件")
        self.seg_len = seg_len
        self.ds = ds

    def __len__(self):
        return len(self.files)

    def _fix_len(self, x):
        L = self.seg_len
        if len(x) >= L:
            return x[:L]
        return np.pad(x, (0, L - len(x)))

    def __getitem__(self, idx):
        m = sio.loadmat(self.files[idx])
        # ★ 按变量名读，不能用 real_keys[0]
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
    """预测幅度掩码 + 相位修正，输出干净信号的估计"""

    def __init__(self, n_fft, hop):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop

        # 幅度掩码（单个 Sigmoid，不再两级级联）
        self.mag_mask = nn.Sequential(
            DSConv2d(1, 16, 3, 1),
            DSConv2d(16, 32, 3, 1),
            DSConv2d(32, 16, 3, 1),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )
        # 相位修正（幅度受限）
        self.ph_corr = nn.Sequential(
            DSConv2d(1, 8, 3, 1),
            DSConv2d(8, 4, 3, 1),
            nn.Conv2d(4, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        """x: [B,1,L] -> 估计信号 [B,1,L]"""
        B, _, L = x.shape
        win = torch.hann_window(self.n_fft, device=x.device)
        X = torch.stft(x.squeeze(1), n_fft=self.n_fft, hop_length=self.hop,
                       window=win, return_complex=True)          # [B,F,T]
        mag = torch.abs(X).unsqueeze(1)                          # [B,1,F,T]
        ph = torch.angle(X)

        mask = self.mag_mask(mag)                                # [B,1,F,T]
        dphi = self.ph_corr(ph.unsqueeze(1)).squeeze(1)          # [B,F,T]

        Y = (mag * mask).squeeze(1) * torch.exp(1j * (ph + 0.1 * dphi))
        y = torch.istft(Y, n_fft=self.n_fft, hop_length=self.hop,
                        window=win, length=L)
        return y.unsqueeze(1)


# ============================== 时域分支 =================================
class TimeBranch(nn.Module):
    """预测残差 delta，使 output = x + delta"""

    def __init__(self, hidden=64):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Conv1d(1, hidden, 1), nn.ReLU(), nn.Dropout(0.1))
        self.core = nn.ModuleList([
            nn.Sequential(
                DSConv1d(hidden, hidden * 2, 3, 1),
                DSConv1d(hidden * 2, hidden, 3, 1),
                nn.Dropout(0.1),
            ) for _ in range(3)
        ])
        self.out = nn.Sequential(
            DSConv1d(hidden, hidden // 2, 3, 1),
            nn.Conv1d(hidden // 2, 1, 1),
        )   # ★ 不加 tanh

    def forward(self, x):
        f = self.inp(x)
        for layer in self.core:
            f = layer(f) + 0.5 * f
        return self.out(f)


# ============================== 主网络 ===================================
class DenoiseNet(nn.Module):
    def __init__(self, n_fft=512, hop=128, hidden=64):
        super().__init__()
        self.time = TimeBranch(hidden)
        self.freq = FrequencyBranch(n_fft, hop)
        self.w_time = nn.Parameter(torch.tensor(0.0))
        self.w_freq = nn.Parameter(torch.tensor(0.0))

        # ★ 零初始化末层 -> 训练开始时网络等价于恒等映射，收敛更稳
        nn.init.zeros_(self.time.out[-1].weight)
        nn.init.zeros_(self.time.out[-1].bias)

    def forward(self, x):
        """x: [B,1,L] 含噪 -> [B,1,L] 降噪"""
        d_time = self.time(x)                       # 时域残差
        e_freq = self.freq(x)                       # 频域估计
        d_freq = e_freq - x                         # 转成残差

        w = torch.softmax(torch.stack([self.w_time, self.w_freq]), dim=0)
        return x + w[0] * d_time + w[1] * d_freq

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================== 损失 =====================================
def stft_mag(x, n_fft, hop):
    win = torch.hann_window(n_fft, device=x.device)
    return torch.stft(x.squeeze(1), n_fft=n_fft, hop_length=hop,
                      window=win, return_complex=True).abs()


def criterion(est, clean, n_fft, hop):
    """时域 L1 + 幅度谱 L1"""
    l_time = F.l1_loss(est.squeeze(1), clean.squeeze(1))
    l_spec = F.l1_loss(stft_mag(est, n_fft, hop), stft_mag(clean, n_fft, hop))
    return l_time + 0.5 * l_spec


# ============================== 指标 =====================================
@torch.no_grad()
def snr_db(est, ref):
    """10*log10(||ref||^2 / ||est-ref||^2)；支持 [B,L] 或 [L]"""
    e = (est - ref).pow(2).sum(dim=-1)
    s = ref.pow(2).sum(dim=-1)
    return 10 * torch.log10(s / (e + 1e-12))


@torch.no_grad()
def si_sdr_db(est, ref):
    """尺度不变信噪比；支持 [B,L] 或 [L]"""
    ref = ref - ref.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)
    a = (est * ref).sum(dim=-1, keepdim=True) / (ref.pow(2).sum(dim=-1, keepdim=True) + 1e-12)
    proj = a * ref
    noise = est - proj
    return 10 * torch.log10(proj.pow(2).sum(dim=-1) / (noise.pow(2).sum(dim=-1) + 1e-12))


# ============================== 训练 =====================================
def build_loader(folder, shuffle):
    ds = DenoiseDataset(folder, CFG["seg_len"], CFG["downsample"])
    return DataLoader(ds, batch_size=CFG["batch_size"], shuffle=shuffle,
                      num_workers=CFG["num_workers"], drop_last=shuffle,
                      pin_memory=(DEVICE.type == "cuda"))


def train():
    print("=" * 70)
    print("轻量化水声信号降噪模型 v2")
    print("=" * 70)
    print(f"设备        : {DEVICE}")
    if DEVICE.type == "cpu":
        print("  ⚠ 当前是 CPU 版 PyTorch，GPU 用不上。想加速请装 CUDA 版：")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cu124")
    print(f"有效采样率  : {CFG['fs_native']//CFG['downsample']} Hz "
          f"(原生 {CFG['fs_native']} Hz ÷ {CFG['downsample']})")
    print(f"统一长度    : {CFG['seg_len']} 点")

    tr = build_loader(CFG["train_dir"], True)
    va = build_loader(CFG["val_dir"], False)
    print(f"训练样本    : {len(tr.dataset)}   验证样本: {len(va.dataset)}")
    print(f"batch_size  : {CFG['batch_size']}   每轮 batch 数: {len(tr)}")

    model = DenoiseNet(CFG["n_fft"], CFG["hop_length"], CFG["hidden"]).to(DEVICE)
    print(f"参数量      : {model.n_params():,}")
    print("=" * 70)

    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"],
                           weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["num_epochs"])

    best_val = float("inf")
    log_path = os.path.join(CFG["output_dir"], "train_log.txt")
    logf = open(log_path, "w", encoding="utf-8")
    logf.write("epoch,train_loss,val_loss,val_snr_in,val_snr_out,val_sisdr_in,val_sisdr_out,lr\n")

    for epoch in range(1, CFG["num_epochs"] + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for i, (noisy, clean) in enumerate(tr):
            if CFG["max_batches_per_epoch"] and i >= CFG["max_batches_per_epoch"]:
                break
            noisy = noisy.unsqueeze(1).to(DEVICE)
            clean = clean.unsqueeze(1).to(DEVICE)
            opt.zero_grad()
            est = model(noisy)
            loss = criterion(est, clean, CFG["n_fft"], CFG["hop_length"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
            if (i + 1) % 50 == 0:
                print(f"  epoch {epoch:3d}  batch {i+1:4d}/{len(tr)}  "
                      f"loss={tot/nb:.4f}  {time.time()-t0:.0f}s", flush=True)
        sched.step()
        tr_loss = tot / max(nb, 1)

        # ---- 验证 ----
        model.eval()
        vtot, vn = 0.0, 0
        sin_, sout, din, dout = [], [], [], []
        with torch.no_grad():
            for noisy, clean in va:
                noisy = noisy.unsqueeze(1).to(DEVICE)
                clean = clean.unsqueeze(1).to(DEVICE)
                est = model(noisy)
                vtot += criterion(est, clean, CFG["n_fft"], CFG["hop_length"]).item()
                vn += 1
                sin_.append(snr_db(noisy.squeeze(1), clean.squeeze(1)))
                sout.append(snr_db(est.squeeze(1), clean.squeeze(1)))
                din.append(si_sdr_db(noisy.squeeze(1), clean.squeeze(1)))
                dout.append(si_sdr_db(est.squeeze(1), clean.squeeze(1)))
        v_loss = vtot / max(vn, 1)
        v_sin, v_sout = torch.cat(sin_).mean().item(), torch.cat(sout).mean().item()
        v_din, v_dout = torch.cat(din).mean().item(), torch.cat(dout).mean().item()

        lr = opt.param_groups[0]["lr"]
        print(f"[epoch {epoch:3d}/{CFG['num_epochs']}] "
              f"train={tr_loss:.4f}  val={v_loss:.4f}  "
              f"SNR {v_sin:+.2f}->{v_sout:+.2f} dB ({v_sout-v_sin:+.2f})  "
              f"SI-SDR {v_din:+.2f}->{v_dout:+.2f} dB ({v_dout-v_din:+.2f})  "
              f"lr={lr:.2e}  {time.time()-t0:.0f}s", flush=True)
        logf.write(f"{epoch},{tr_loss:.6f},{v_loss:.6f},{v_sin:.4f},{v_sout:.4f},"
                   f"{v_din:.4f},{v_dout:.4f},{lr:.3e}\n")
        logf.flush()

        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), CFG["model_save_path"])
            print(f"  ✓ 保存最佳模型 (val={best_val:.4f})")

    logf.close()
    print(f"\n训练完成，最佳验证损失 {best_val:.4f}")
    print(f"模型 -> {CFG['model_save_path']}")
    print(f"日志 -> {log_path}")
    return model


# ============================== 测试 =====================================
@torch.no_grad()
def evaluate(model=None):
    print("\n" + "=" * 70)
    print("测试集评估")
    print("=" * 70)
    if model is None:
        model = DenoiseNet(CFG["n_fft"], CFG["hop_length"], CFG["hidden"])
        model.load_state_dict(torch.load(CFG["model_save_path"], map_location=DEVICE))
    model = model.to(DEVICE).eval()

    loader = build_loader(CFG["test_dir"], False)
    rows = []
    for noisy, clean in loader:
        n = noisy.unsqueeze(1).to(DEVICE)
        c = clean.unsqueeze(1).to(DEVICE)
        est = model(n)
        for k in range(n.shape[0]):
            rows.append((
                snr_db(n[k, 0], c[k, 0]).item(),
                snr_db(est[k, 0], c[k, 0]).item(),
                si_sdr_db(n[k, 0], c[k, 0]).item(),
                si_sdr_db(est[k, 0], c[k, 0]).item(),
            ))
    r = np.array(rows)
    print(f"{'指标':<14}{'输入':>10}{'输出':>10}{'提升':>10}")
    print("-" * 46)
    print(f"{'SNR (dB)':<14}{r[:,0].mean():>10.2f}{r[:,1].mean():>10.2f}{r[:,1].mean()-r[:,0].mean():>+10.2f}")
    print(f"{'SI-SDR (dB)':<14}{r[:,2].mean():>10.2f}{r[:,3].mean():>10.2f}{r[:,3].mean()-r[:,2].mean():>+10.2f}")
    print(f"\n样本数 {len(r)}")

    out = os.path.join(CFG["output_dir"], "test_results.csv")
    np.savetxt(out, r, delimiter=",", header="snr_in,snr_out,sisdr_in,sisdr_out", comments="")
    print(f"逐样本结果 -> {out}")
    return r


if __name__ == "__main__":
    t = time.time()
    model = train()
    evaluate(model)
    print(f"\n总耗时 {(time.time()-t)/60:.1f} 分钟")
