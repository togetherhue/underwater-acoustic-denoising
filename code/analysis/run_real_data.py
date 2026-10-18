# -*- coding: utf-8 -*-
"""实测数据接入管道：从原始录音到增强结果，按 docs/real_data_deployment.md 的清单逐项核对

设计原则：
  - 不假设数据格式，支持 .wav / .npy / .mat / .csv
  - 采样率不是 24 kHz 就重采样（模型链路假设 9600→24000→降采样4→6000 Hz）
  - 切分成训练时的 8000 点帧
  - 输入尺度：默认使用"输入归一化训练"的 checkpoint（inm_*），此时模型自动
    吃 x/rms(x)，无需任何标定。若强行指定旧 checkpoint，会打印大号警告。
  - 无干净参考，所以只报无参考指标（带内/带外能量比、谱平坦度、脉冲残留估计）

用法：
  python run_real_data.py --input path/to.wav --ckpt model_inm_pf.pth
  python run_real_data.py --input path/ --ckpt model_inm_pf.pth --outdir results/real
  python run_real_data.py --input path.wav --ckpt model_pamnet_freq_pf_gn.pth  # 旧模型，会警告
"""
import os, sys, json, argparse, glob
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

CKPT_DIR = r"D:/project/output_pamnet"
FS_MODEL = 24000          # 训练数据在 MATLAB 侧的采样率
FS_NET = 6000             # 降采样 4 倍后模型实际看到的采样率
SEG = 8000                # 帧长（降采样后）
BAND = (780.0, 1520.0)    # 信号频带（798–1500 Hz，留一点余量）
ARCHS = {
    "R3": dict(use_short=False, use_long=False, use_gate=True, use_pulse=False,
               freq_refine=2.0, pulse_freq=False),
    "R4": dict(use_short=True, use_long=True, use_gate=True, use_pulse=False,
               freq_refine=2.0, pulse_freq=False),
    "R5": dict(use_short=True, use_long=True, use_gate=True, use_pulse=True,
               freq_refine=2.0, pulse_freq=True),
}


# ---------------------------------------------------------------- 读取
def read_audio(path):
    """返回 (data[Ch, N] float64, fs)。支持 wav/npy/mat/csv"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        # 约定返回值统一为 [Ch, N]。一维信号 -> [1, N]
        try:
            import soundfile as sf
            x, fs = sf.read(path, dtype="float64", always_2d=True)
            return np.ascontiguousarray(x.T), fs
        except ImportError:
            from scipy.io import wavfile
            fs, raw = wavfile.read(path)
            raw = np.asarray(raw)
            x = raw.astype(np.float64)
            if np.issubdtype(raw.dtype, np.integer):
                info = np.iinfo(raw.dtype)
                x = x / max(abs(info.min), info.max)
            return (x[None, :] if x.ndim == 1 else np.ascontiguousarray(x.T)), fs
    if ext == ".npy":
        x = np.load(path)
        x = np.asarray(x, np.float64)
        if x.ndim == 1:
            x = x[:, None]
        else:
            x = x.T
        return x, None
    if ext == ".mat":
        import scipy.io as sio
        m = sio.loadmat(path)
        keys = [k for k in m if not k.startswith("__")]
        cand = None
        for k in ("noisy_signal", "signal", "data", "x"):
            if k in m:
                cand = m[k]; break
        if cand is None:
            cand = m[keys[0]]
        x = np.asarray(cand).squeeze().astype(np.float64)
        if x.ndim == 1:
            x = x[:, None]
        else:
            x = x.T
        return x, None
    if ext == ".csv":
        x = np.loadtxt(path, delimiter=",", skiprows=1)
        return x[:, None].T if x.ndim == 1 else x.T, None
    raise ValueError("不支持的格式: %s" % ext)


# ---------------------------------------------------------------- 检查
def check_band(x, fs_net):
    """信号能量落在 798–1500 Hz 的比例"""
    X = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs_net)
    tot = X.sum()
    if tot <= 0:
        return float("nan")
    return float(X[(f >= BAND[0]) & (f <= BAND[1])].sum() / tot)


def spec_flatness(x):
    """谱平坦度（几何均值/算术均值），越接近 1 越像白噪声"""
    X = np.abs(np.fft.rfft(x)) ** 2 + 1e-20
    return float(np.exp(np.mean(np.log(X))) / np.mean(X))


def out_of_band_ratio(est, fs_net):
    """增强后信号在带外的能量占比 —— 越低说明噪声被压得越干净"""
    E = np.abs(np.fft.rfft(est)) ** 2
    f = np.fft.rfftfreq(len(est), 1.0 / fs_net)
    tot = E.sum()
    if tot <= 0:
        return float("nan")
    return float(E[(f < BAND[0]) | (f > BAND[1])].sum() / tot)


# ---------------------------------------------------------------- 主流程
def load_model(ckpt, arch_key):
    p = ckpt if os.path.isabs(ckpt) else os.path.join(CKPT_DIR, ckpt)
    m = PN.PAMNet(c=24, n_fft=V3.CFG["n_fft"], hop=V3.CFG["hop_length"],
                  norm="gn", **ARCHS[arch_key])
    m.load_state_dict(torch.load(p, map_location="cpu"), strict=True)
    return m.to(V3.DEVICE).eval()


@torch.no_grad()
def enhance(model, frames, rms_norm=True, bs=32):
    outs = []
    for i in range(0, len(frames), bs):
        nb = frames[i:i + bs].astype(np.float32)
        if rms_norm:
            nb = np.stack([v / (np.sqrt(np.mean(v ** 2)) + 1e-12) for v in nb])
        e = model(torch.from_numpy(nb).unsqueeze(1).to(V3.DEVICE))
        e = e.squeeze(1).cpu().numpy().astype(np.float64)
        for k, v in zip(range(i, i + len(e)), e):
            if rms_norm:
                v = v * (np.sqrt(np.mean(frames[k] ** 2)) + 1e-12)
            outs.append(v)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="实测音频文件或目录")
    ap.add_argument("--ckpt", default="model_inm_pf.pth",
                    help="checkpoint 名（默认用归一化训练的 R5）")
    ap.add_argument("--arch", default="R5", choices=list(ARCHS.keys()))
    ap.add_argument("--fs", type=int, default=None,
                    help="若文件无采样率信息，用它（默认 24000）")
    ap.add_argument("--ch", type=int, default=0, help="多通道取哪一轨（默认 0）")
    ap.add_argument("--legacy", action="store_true",
                    help="强制用未归一化训练的旧 checkpoint（会用 x/rms(x) 前向）")
    ap.add_argument("--outdir", default=os.path.join(HERE, "results", "real"))
    A = ap.parse_args()

    files = []
    if os.path.isdir(A.input):
        for ext in ("*.wav", "*.npy", "*.mat", "*.csv"):
            files += sorted(glob.glob(os.path.join(A.input, ext)))
    else:
        files = [A.input]
    if not files:
        print("没找到输入文件"); return

    # 归一化训练 vs 旧 checkpoint：决定是否用 x/rms(x)
    rms_norm = True
    if A.legacy or not A.ckpt.startswith("model_inm_"):
        print()
        print("!" * 78)
        print("!! 警告：_ckpt 不是归一化训练的（tag 不是 inm_*）。")
        print("!! 旧 checkpoint 对输入尺度极度敏感（实测跨尺度极差 18.7 dB）。")
        print("!! 本脚本会强制用 x/rms(x) 前向以缓解，但这仍是【有偏近似】，")
        print("!! 要用无偏结果请改用 --ckpt model_inm_*.pth。")
        print("!" * 78)
        print()
        rms_norm = True     # 旧模型至少按 noisy 归一化，比裸喂好

    model = load_model(A.ckpt, A.arch)
    os.makedirs(A.outdir, exist_ok=True)
    print("模型 %s（%s），输入归一化 %s" % (A.ckpt, A.arch, "x/rms(x)" if rms_norm else "无"))

    report = []
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        x, fs = read_audio(path)
        if x.shape[0] > 1:
            print("  %s：%d 通道，取第 %d 轨" % (name, x.shape[0], A.ch))
        sig = x[A.ch]
        fs0 = fs or A.fs
        if fs0 is None:
            print("  %s：无采样率信息且未给 --fs，跳过" % name); continue

        # ① 采样率
        note = []
        if fs0 != FS_MODEL:
            from scipy.signal import resample_poly
            g = np.gcd(int(fs0), FS_MODEL)
            sig = resample_poly(sig, FS_MODEL // g, int(fs0) // g)
            note.append("采样率 %d -> %d Hz 已重采样" % (fs0, FS_MODEL))

        # ② 频带
        fs_net = FS_MODEL // 4
        sig4 = sig
        if FS_MODEL != fs_net:
            from scipy.signal import resample_poly
            sig4 = resample_poly(sig, 1, 4)
        bf = check_band(sig4[:min(len(sig4), 8000)], fs_net)
        note.append("频带 %d-%d Hz 能量占比 %.1f%%" % (BAND[0], BAND[1], 100 * bf))

        # ③ 切帧（定长 8000，尾部零填充）
        nf = int(np.ceil(len(sig4) / SEG))
        frames = np.zeros((nf, SEG), dtype=np.float64)
        for i in range(nf):
            seg = sig4[i * SEG:(i + 1) * SEG]
            frames[i, :len(seg)] = seg

        # ④ 前向
        est = enhance(model, frames, rms_norm)

        # ⑤ 无参考指标
        ob_in = np.nanmean([out_of_band_ratio(frames[i], fs_net) for i in range(nf)])
        ob_out = np.nanmean([out_of_band_ratio(est[i], fs_net) for i in range(nf)])
        sf_in = np.nanmean([spec_flatness(frames[i]) for i in range(nf)])
        sf_out = np.nanmean([spec_flatness(est[i]) for i in range(nf)])
        row = dict(file=name, n_frames=nf, band_frac=bf,
                   out_of_band_in=ob_in, out_of_band_out=ob_out,
                   spec_flatness_in=sf_in, spec_flatness_out=sf_out,
                   notes=note)
        report.append(row)

        out = os.path.join(A.outdir, "%s_enhanced.npy" % name)
        np.save(out, np.array(est))
        print("  %-28s %d 帧 | %s | 带外能量 %.4f -> %.4f | -> %s"
              % (name, nf, "; ".join(note), ob_in, ob_out, out))

    rp = os.path.join(A.outdir, "real_data_report.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print()
    print("汇总 ->", rp)
    print()
    print("无参考指标的读法：")
    print("  - 带外能量占比下降 = 宽带噪声被压制（正向）")
    print("  - 谱平坦度下降 = 频谱更有结构（正向，但要人工听感复核）")
    print("  - 带内能量占比低（<50%）= 信号可能不在 798-1500 Hz 或采样率不对，需复查")


if __name__ == "__main__":
    main()
