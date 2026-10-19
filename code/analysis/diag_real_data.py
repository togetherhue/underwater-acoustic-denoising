# -*- coding: utf-8 -*-
"""实测数据（_smoke/test.wav）端到端复查 + 合成对照

回答三个问题：
1. out_of_band 0.316 → 0.0008 是「降噪」还是「窄带化」？
   → 对照：干净信号本身带外比是多少（训练数据带限 798-1500 Hz ⇒ 应≈0）
2. 模型的带内信号有没有被破坏？
   → 合成对照：同一模型在 testA 上的带内保真（有 clean 参考）
3. 无重叠分帧会不会引入拼接click？
   → 帧边界幅度/梯度检验
"""
import os, sys, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_CANDS = [os.path.normpath(os.path.join(HERE, "..", "python")),
          r"C:\Users\32213\PycharmProjects\project1"]
for _p in [HERE] + [c for c in _CANDS if os.path.isdir(c)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import torch
import pamnet as PN
import v3_model as V3

FS_NET = 6000.0
SEG = 8000
BAND = (780.0, 1520.0)
CKPT_DIR = r"E:/project/data/output_pamnet"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def band_ratio(x, lo=BAND[0], hi=BAND[1]):
    """返回 (带内占比, 带外占比)"""
    X = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / FS_NET)
    tot = X.sum()
    if tot <= 0:
        return float("nan"), float("nan")
    inb = X[(f >= lo) & (f <= hi)].sum()
    return float(inb / tot), float((tot - inb) / tot)


def band_energy_profile(x, nb=24):
    """按频率分 nb 段，返回每段能量占比 —— 看带外是结构化的还是宽带噪声"""
    X = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / FS_NET)
    tot = X.sum()
    edges = np.linspace(0, FS_NET / 2, nb + 1)
    out = []
    for i in range(nb):
        m = (f >= edges[i]) & (f < edges[i + 1])
        out.append(float(X[m].sum() / tot) if tot > 0 else 0.0)
    return np.array(out)


def wav_read(path):
    from scipy.io import wavfile
    fs, raw = wavfile.read(path)
    raw = np.asarray(raw).astype(np.float64)
    if np.issubdtype(raw.dtype, np.integer):
        info = np.iinfo(np.asarray(raw).dtype)
        raw = raw / max(abs(info.min), info.max)
    return raw.ravel(), fs


def load_model(ckpt, arch="R5"):
    ARCHS = {"R5": dict(use_short=True, use_long=True, use_gate=True, use_pulse=True,
                        freq_refine=2.0, pulse_freq=True)}
    m = PN.PAMNet(c=24, n_fft=V3.CFG["n_fft"], hop=V3.CFG["hop_length"],
                  norm="gn", **ARCHS[arch])
    m.load_state_dict(torch.load(os.path.join(CKPT_DIR, ckpt), map_location="cpu"), strict=True)
    return m.to(DEV).eval()


@torch.no_grad()
def run(model, frames, rms_norm=True):
    outs = []
    for i in range(0, len(frames), 32):
        nb = torch.from_numpy(frames[i:i + 32].astype(np.float32)).to(DEV).unsqueeze(1)
        if rms_norm:
            rms = torch.sqrt((nb ** 2).mean(dim=-1, keepdim=True) + 1e-12)
            out = model(nb / rms) * rms
        else:
            out = model(nb)
        outs.append(out.squeeze(1).cpu().numpy().astype(np.float64))
    return np.concatenate(outs, 0)


def main():
    rng = np.random.default_rng(0)
    print("=" * 70)
    print("A. 实测数据复查")
    print("=" * 70)
    x, fs0 = wav_read(os.path.join(HERE, "_smoke", "test.wav"))
    from scipy.signal import resample_poly
    sig4 = resample_poly(x, 1, 4)
    nf = int(np.ceil(len(sig4) / SEG))
    frames = np.zeros((nf, SEG))
    for i in range(nf):
        seg = sig4[i * SEG:(i + 1) * SEG]
        frames[i, :len(seg)] = seg
    est = np.load(os.path.join(HERE, "results", "real", "test_enhanced.npy"))
    print("输入 %d 点 @%.0f Hz → 降采样 %d 点 → %d 帧；输出形状 %s"
          % (len(x), fs0, len(sig4), nf, est.shape))

    print("\n-- 带外能量按频率分段的占比（输入）--")
    prof_in = np.zeros(24)
    for i in range(nf):
        prof_in += band_energy_profile(frames[i]) / nf
    prof_out = np.zeros(24)
    for i in range(nf):
        prof_out += band_energy_profile(est[i]) / nf
    edges = np.linspace(0, FS_NET / 2, 25)
    for i in range(24):
        bar = "#" * int(prof_in[i] * 120)
        tag = ""
        if edges[i] >= BAND[0] and edges[i + 1] <= BAND[1]:
            tag = "←信号带内"
        elif edges[i] < 1000:
            tag = "←低于信号带"
        print("  %5.0f-%5.0f Hz  输入 %6.3f  输出 %7.5f  %s %s"
              % (edges[i], edges[i + 1], prof_in[i], prof_out[i], bar, tag))

    print("\n-- 带外衰减是否均匀（若带外各段同比例下降 ⇒ 宽带噪声；若只有某段被压 ⇒ 结构）--")
    ob_mask = (edges[:-1] < BAND[0]) | (edges[1:] > BAND[1])
    ratio = prof_out[ob_mask] / np.maximum(prof_in[ob_mask], 1e-12)
    print("  带外各段 输出/输入 比：min %.2e  max %.2e  中位 %.2e"
          % (ratio.min(), ratio.max(), np.median(ratio)))

    print("\n-- 帧边界的拼接痕迹（无重叠分帧 ⇒ 检查 samples 8000/16000 附近）--")
    full = np.concatenate([est[0], est[1], est[2]])
    for b in (7998, 7999, 8000, 8001, 16000):
        if b + 2 < len(full):
            print("  n=%d: %+.5f %+.5f %+.5f" % (b, full[b], full[b + 1], full[b + 2]))
    d = np.abs(np.diff(full))
    n_inner = np.concatenate([np.arange(4000, 7900), np.arange(12000, 15800)])
    n_bound = np.array([7997, 7998, 7999, 8000, 8001, 15997, 15998, 15999, 16000, 16001])
    print("  帧内 |Δx| 均值 %.5f（p99 %.5f）  vs 边界 |Δx| 均值 %.5f（max %.5f）"
          % (d[n_inner - 1].mean(), np.quantile(d[n_inner - 1], .99),
             d[n_bound].mean(), d[n_bound].max()))

    print()
    print("=" * 70)
    print("B. 合成对照：同一模型在有 clean 参考的 testA 上（关键基准）")
    print("=" * 70)
    root = r"E:/project/data/dataset_pam/alphastable/_npy"
    model = load_model("model_inm_pf.pth")
    # base 条件 = α=1.5, rate=5/s（dataset_pam/alphastable）
    for cand in (r"E:/project/data/dataset_pam/alphastable/_npy",
                 r"E:/project/data/dataset_pam/gauss/_npy",
                 r"E:/project/data/ds_sens/a12_r5/_npy"):
        if os.path.isdir(cand):
            root = cand
            break
    print("数据:", root)
    n_ = np.load(os.path.join(root, "testA_n.npy"), mmap_mode="r")
    c_ = np.load(os.path.join(root, "testA_c.npy"), mmap_mode="r")
    print("testA 样本数", len(n_))
    idx = rng.choice(len(n_), 60, replace=False)
    rows = {"clean": [], "noisy": [], "est": []}
    F = []
    for i in idx:
        F.append(np.array(n_[i], dtype=np.float64))
    F = np.array(F)
    E = run(model, F)
    C = np.array([c_[i] for i in idx], dtype=np.float64)
    for nm, arr in (("clean", C), ("noisy", F), ("est", E)):
        ib = [band_ratio(a)[0] for a in arr]
        ob = [band_ratio(a)[1] for a in arr]
        rows[nm] = (np.mean(ib), np.mean(ob))
        print("  %-6s 带内占比 %.4f  带外占比 %.4f" % (nm, np.mean(ib), np.mean(ob)))
    # 带内保真：est 相对 clean 的时域误差（同尺度）
    def snr(a, b):
        return 10 * np.log10((b ** 2).sum() / max(((a - b) ** 2).sum(), 1e-20))
    s_ec = np.mean([snr(E[i], C[i]) for i in range(len(E))])
    s_nc = np.mean([snr(F[i], C[i]) for i in range(len(E))])
    print("  时域 SNR: noisy vs clean %+.2f dB → est vs clean %+.2f dB（提升 %+.2f dB）"
          % (s_nc, s_ec, s_ec - s_nc))
    # clean 的带外能量（这是「带外应≈0」的直接证据）
    print("  ⇒ 干净信号自身带外占比仅 %.5f，故带外降为该量级是降噪而非窄带化"
          % rows["clean"][1])
    # 干净信号的频段结构：是否严格带限
    cprof = np.zeros(24)
    for a in C:
        cprof += band_energy_profile(a) / len(C)
    ob = (np.arange(24) * 250 + 125)
    print("  clean 在 >1520 Hz 各段能量占比 max %.2e；在 <780 Hz 各段 max %.2e"
          % (cprof[(edges[:-1] > 1520)].max(), cprof[(edges[1:] < 780)].max()))

    json.dump({"real": {"prof_in": prof_in.tolist(), "prof_out": prof_out.tolist()},
               "synth": {"clean_ib": rows["clean"][0], "clean_ob": rows["clean"][1],
                         "noisy_ib": rows["noisy"][0], "noisy_ob": rows["noisy"][1],
                         "est_ib": rows["est"][0], "est_ob": rows["est"][1],
                         "snr_noisy": s_nc, "snr_est": s_ec}},
              open(os.path.join(HERE, "sens_out", "real_data_diag.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n-> sens_out/real_data_diag.json")


if __name__ == "__main__":
    main()
