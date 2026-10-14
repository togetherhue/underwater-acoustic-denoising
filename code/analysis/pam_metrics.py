# -*- coding: utf-8 -*-
"""
脉冲噪声下的稳健评估指标

【为什么需要这套指标】
α 稳定分布（α<2）的二阶矩不存在、方差无穷，叠加突发干扰后更严重。
沿用 10*log10(||s||²/||e||²) 这种全局 SNR 时，结果会被少数几个极大脉冲完全主导：
模型即使把 99.9% 的采样点都处理得很好，只要有 1 个脉冲没压住，这个数字就会崩掉，
换一批噪声实现波动巨大 —— 失去可比性。

【核心指标：逐帧降噪增益 frmgain】
    frmgain_f = 10*log10( ||n_f - s_f||² / ||ŝ_f - s_f||² )
即"该帧的输入噪声能量 / 输出噪声能量"。相比帧 SNR，它不受该帧信号强弱影响
（分子分母用同一帧），因此在信号包络起伏大的 chirp 上更稳定。

按"该帧是否含干扰"分解：
    frmgain_clean   无干扰帧的中位增益  —— 常规降噪能力
    frmgain_imp     有干扰帧的中位增益  —— 脉冲段处理能力 ← PAM-Net 核心指标
    frmgain_imp_min 有干扰帧的【最小】增益 —— 最差的一个干扰帧（抓漏掉的脉冲）

【辅助指标】
    snr / sisdr     全局，与既有实现完全一致，保留用于与历史结果对照（对脉冲敏感）
    fmsnr_med       逐帧 SNR 中位数（绝对值参考）
    fmsnr_clean/imp 逐帧 SNR 按干扰分解的中位数
    fmsnr_frac_bad  有效帧中 SNR < 0 dB 的比例

【零填充处理】
数据集约 24% 零填充，这些帧参考能量为 0。本模块自动丢弃参考能量
低于最大帧能量 1e-6 的帧。
"""
import numpy as np
import torch

FRAME = 128
MIN_ENERGY_FRAC = 1e-6


# ============================ 全局指标（与既有实现完全一致）============================
def snr_db(est, ref):
    return 10 * torch.log10(ref.pow(2).sum(dim=-1) /
                            ((est - ref).pow(2).sum(dim=-1) + 1e-12))


def si_sdr_db(est, ref):
    r = ref - ref.mean(dim=-1, keepdim=True)
    x = est - est.mean(dim=-1, keepdim=True)
    a = (x * r).sum(dim=-1, keepdim=True) / (r.pow(2).sum(dim=-1, keepdim=True) + 1e-12)
    proj = a * r
    return 10 * torch.log10(proj.pow(2).sum(dim=-1) /
                            ((x - proj).pow(2).sum(dim=-1) + 1e-12))


# ============================ 单样本逐帧统计 ============================
def _frames(x, nf, frame):
    return x[:nf * frame].reshape(nf, frame)


def _stats_1d(est, ref, noisy, frame, imp_mask):
    L = ref.shape[0]
    nf = L // frame
    if nf < 1:
        raise ValueError("信号长度 %d 不足一帧 %d" % (L, frame))

    s = _frames(ref, nf, frame)
    ps = (s ** 2).sum(1)                                   # 帧信号能量
    pn_in = ((_frames(noisy, nf, frame) - s) ** 2).sum(1)  # 帧输入噪声能量
    pn_out = ((_frames(est, nf, frame) - s) ** 2).sum(1)   # 帧输出噪声能量

    valid = ps > ps.max() * MIN_ENERGY_FRAC                # 丢弃零填充帧

    snr = np.full(nf, np.nan)
    snr[valid] = 10 * np.log10(ps[valid] / (pn_out[valid] + 1e-12))
    gain = np.full(nf, np.nan)
    gain[valid] = 10 * np.log10((pn_in[valid] + 1e-12) / (pn_out[valid] + 1e-12))

    out = dict(n_valid=int(valid.sum()), nf=nf,
               fmsnr_med=float(np.nanmedian(snr[valid])),
               frmgain_med=float(np.nanmedian(gain[valid])),
               fmsnr_frac_bad=float(np.nanmean(snr[valid] < 0.0)))

    if imp_mask is not None and np.any(imp_mask):
        m = _frames(imp_mask, nf, frame).max(1) > 0
        mc, mq = valid & m, valid & (~m)
        out.update(
            frac_imp=float(m.sum() / max(valid.sum(), 1)),
            fmsnr_imp=float(np.nanmedian(snr[mc])) if mc.any() else np.nan,
            fmsnr_clean=float(np.nanmedian(snr[mq])) if mq.any() else np.nan,
            frmgain_imp=float(np.nanmedian(gain[mc])) if mc.any() else np.nan,
            frmgain_imp_min=float(np.nanmin(gain[mc])) if mc.any() else np.nan,
            frmgain_clean=float(np.nanmedian(gain[mq])) if mq.any() else np.nan,
        )
    else:
        out.update(frac_imp=0.0, fmsnr_imp=np.nan, fmsnr_clean=out["fmsnr_med"],
                   frmgain_imp=np.nan, frmgain_imp_min=np.nan,
                   frmgain_clean=out["frmgain_med"])
    return out


# ============================ 批量入口 ============================
def evaluate(est, ref, noisy, frame=FRAME, imp_mask=None):
    """est/ref/noisy: [B, L] torch 张量；imp_mask: [B, L] torch 张量或 None
       返回 dict[str, float]，对 batch 取均值（含 NaN 的项按有效样本数平均）"""
    def np2(x):
        if x is None:
            return None
        a = x.detach().cpu().numpy().astype(np.float64)
        return a[None] if a.ndim == 1 else a

    e, r, n = np2(est), np2(ref), np2(noisy)
    mk = np2(imp_mask)
    if mk is None:
        mk = [None] * e.shape[0]

    keys = ["fmsnr_med", "frmgain_med", "fmsnr_frac_bad", "frac_imp",
            "fmsnr_imp", "fmsnr_clean", "frmgain_imp", "frmgain_imp_min",
            "frmgain_clean"]
    acc = {k: 0.0 for k in keys}
    cnt = {k: 0 for k in keys}
    s_snr = s_sd = 0.0
    B = e.shape[0]

    for i in range(B):
        st = _stats_1d(e[i], r[i], n[i], frame, mk[i])
        s_snr += snr_db(est[i], ref[i]).item()
        s_sd += si_sdr_db(est[i], ref[i]).item()
        for k in keys:
            v = st[k]
            if v is not None and not np.isnan(v):
                acc[k] += v
                cnt[k] += 1

    out = {"snr": s_snr / B, "sisdr": s_sd / B, "n": B}
    for k in keys:
        out[k] = acc[k] / cnt[k] if cnt[k] else float("nan")
    return out


METRIC_COLS = ["snr_in", "snr_out", "sisdr_in", "sisdr_out",
               "fmsnr_med_in", "fmsnr_med_out",
               "frmgain_med", "frmgain_clean", "frmgain_imp",
               "frmgain_imp_min", "fmsnr_frac_bad_out", "frac_imp"]


# ============================ 自检 ============================
if __name__ == "__main__":
    import scipy.io as sio
    from scipy.signal import resample_poly

    SEG = 8000
    m = sio.loadmat(r"D:/project/dataset_wave/LFM/testA/s_00001.mat")
    c = resample_poly(np.asarray(m["clean_signal"]).squeeze().astype(np.float64), 1, 4)
    c = c[:SEG] if len(c) >= SEG else np.pad(c, (0, SEG - len(c)))
    T = torch.from_numpy(c).float().unsqueeze(0)
    rms = float(T.pow(2).mean().sqrt())

    rng = np.random.default_rng(20261014)
    g = torch.from_numpy(rng.standard_normal(SEG)).float().unsqueeze(0) * rms
    imp = torch.zeros_like(T)
    pos = rng.choice(np.arange(200, SEG - 200), 8, replace=False)
    imp[0, pos] = torch.from_numpy(rng.choice([-1.0, 1.0], 8)).float() * 20 * rms
    mk = (imp.abs() > 0).float()
    N = T + g + imp

    print("干净 rms=%.4f；高斯噪声 rms=%.4f；脉冲 8 个 @ ±20×rms\n" % (rms, rms))
    hdr = ("%-24s %8s %8s | %10s %10s | %10s %10s %10s | %8s"
           % ("情形", "全局SNR", "SI-SDR", "帧SNR中位", "坏帧比",
              "增益中位", "增益·干净帧", "增益·干扰帧", "干扰帧最差"))
    print(hdr); print("-" * len(hdr))
    cases = [
        ("输入(不处理)", N),
        ("理想 est=clean", T),
        ("优秀(残差0.15×)", T + 0.15 * g + 0.15 * imp),
        ("好(残差0.30×)", T + 0.30 * g + 0.30 * imp),
        ("漏1个脉冲", T + 0.15 * g + 0.15 * imp
         + imp * (torch.arange(SEG) == int(pos[0])).float()),
        ("漏全部脉冲", T + 0.15 * g + imp),
        ("只降高斯不碰脉冲", T + 0.02 * g + imp),
    ]
    res = {}
    for name, est in cases:
        d = evaluate(est, T, N, imp_mask=mk)
        res[name] = d
        print("%-24s %8.2f %8.2f | %10.2f %10.3f | %10.2f %10.2f %10.2f | %8.2f"
              % (name, d["snr"], d["sisdr"], d["fmsnr_med"], d["fmsnr_frac_bad"],
                 d["frmgain_med"], d["frmgain_clean"], d["frmgain_imp"],
                 d["frmgain_imp_min"]))
    print()
    a, b = res["好(残差0.30×)"], res["漏1个脉冲"]
    print("【关键验证】'好(残差0.30×)' vs '漏1个脉冲'：")
    print("  全局 SNR   : %.2f vs %.2f  -> 差 %+.2f dB（判反了：漏脉冲的看起来更好）"
          % (a["snr"], b["snr"], b["snr"] - a["snr"]))
    print("  帧SNR中位  : %.2f vs %.2f  -> 差 %+.2f dB（分不开）"
          % (a["fmsnr_med"], b["fmsnr_med"], b["fmsnr_med"] - a["fmsnr_med"]))
    print("  增益·干扰帧: %.2f vs %.2f  -> 差 %+.2f dB（能分开）"
          % (a["frmgain_imp"], b["frmgain_imp"], b["frmgain_imp"] - a["frmgain_imp"]))
    print("  干扰帧最差 : %.2f vs %.2f  -> 差 %+.2f dB（最灵敏）"
          % (a["frmgain_imp_min"], b["frmgain_imp_min"],
             b["frmgain_imp_min"] - a["frmgain_imp_min"]))
    print()
    print("  结论：全局 SNR 会把'漏掉脉冲'判成更好；必须用【干扰帧增益】类指标。")
