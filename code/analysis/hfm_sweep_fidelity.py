# -*- coding: utf-8 -*-
"""扫频保真检验：在【无干净参考】的条件下判断降噪是否破坏了目标的扫频结构

背景
----
导师方案的目标是实测 HFM（双曲调频）信号。实测数据拿不到 clean，
所以常规 SNR 类指标全部不可用。但扫频信号有一个可利用的结构先验：
它的瞬时频率轨迹 f(t) 是已知形状 ——

  LFM : f(t) = f0 + (f1-f0)·t/T          ⇒  f 对 t 线性
  HFM : f(t) = f0·f1·T / (f1·T-(f1-f0)·t) ⇒  1/f 对 t 线性

于是可以只从「信号自身」估计轨迹，比较处理前后的轨迹是否一致。
如果降噪把某个频段吃掉、或把轨迹扭了，轨迹对比会直接暴露。

判据（三个层次）
----------------
1. **是否真有扫频**：轨迹拟合的 R²。R² 低 ⇒ 没有可辨的扫频结构，
   后面的保真度指标<b>无意义</b>（这是防止在噪声上强行解读的闸门）。
2. **调制类型**：1/f~t 与 f~t 谁拟合得更好。HFM 应 1/f 胜出，LFM 反之。
3. **保真度**：
   - 轨迹相关系数 r（时间对齐后，输出轨迹 vs 输入轨迹）
   - 轨迹 RMS 偏差（Hz）
   - 起止频率的相对误差
   - 覆盖范围收缩率（输出轨迹覆盖的频段 / 输入覆盖的频段）

工具有效性验证（本脚本 --selftest）
----------------------------------
在**合成** HFM 上做闭环：加脉冲噪声 → 用模型降噪 → 期望保真度高；
再人为制造四种破坏 → 期望保真度显著下降。只有这一步通过了，
把这个工具用到实测数据上的结果才值得相信。

用法
----
  python hfm_sweep_fidelity.py --selftest            # 先验证工具本身
  python hfm_sweep_fidelity.py --input a.wav --ckpt model_inm_pf.pth
  python hfm_sweep_fidelity.py --input dir/ --ckpt model_inm_pf.pth --outdir results/real
"""
import os, sys, json, argparse, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_CANDS = [os.path.normpath(os.path.join(HERE, "..", "python")),
          r"C:\Users\32213\PycharmProjects\project1"]
for _p in [HERE] + [c for c in _CANDS if os.path.isdir(c)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

FS_NET = 6000.0           # 模型实际看到的采样率
SEG = 8000                # 训练帧长（@6 kHz = 1.33 s）
BAND = (780.0, 1520.0)    # 目标信号频带（798-1500 留余量）
CKPT_DIR = r"E:/project/data/output_pamnet"
NPERSET = 256             # STFT 窗长（@6 kHz = 42.7 ms）
NOVER = 224               # 87.5% 重叠，保证轨迹连续

ARCHS = {"R5": dict(use_short=True, use_long=True, use_gate=True, use_pulse=True,
                    freq_refine=2.0, pulse_freq=True)}


# ============================================================ 信号生成
def gen_chirp(fs, T, f0, f1, modtype="HFM", env="rect", phi=0.0, A=1.0):
    """与 gen_chirp.m 逐位一致（cumtrapz 用梯形积分实现）"""
    n = int(round(T * fs))
    t = np.arange(n) / fs
    if modtype.upper() == "LFM":
        f = f0 + (f1 - f0) * t / T
    elif modtype.upper() == "HFM":
        f = (f0 * f1 * T) / (f1 * T - (f1 - f0) * t)
    else:
        raise ValueError(modtype)
    # MATLAB cumtrapz：首元素为 0，之后梯形累加
    ph = np.zeros(n)
    ph[1:] = np.cumsum((f[1:] + f[:-1]) * 0.5 / fs) * 2 * np.pi
    if env == "hann":
        e = np.hanning(n)
    elif env == "gauss":
        e = np.exp(-0.5 * ((t - t.mean()) / (0.25 * T)) ** 2)
    else:
        e = np.ones(n)
    return A * e * np.cos(ph + phi), t


def add_impulsive(x, fs, rate=5.0, alpha=1.5, snr_db=-3.0, rng=None):
    """加 α 稳定背景 + 突发干扰（与训练数据同一套生成逻辑的轻量版）"""
    rng = rng or np.random.default_rng(0)
    n = len(x)
    # α 稳定背景：用 Chambers-Mallows-Stuck 的对称情形
    V = rng.uniform(-np.pi / 2, np.pi / 2, n)
    W = rng.exponential(1.0, n)
    if abs(alpha - 1.0) < 1e-9:
        bg = np.tan(V)
    else:
        bg = (np.sin(alpha * V) / np.cos(V) ** (1.0 / alpha)) * \
             (np.cos((1 - alpha) * V) / W) ** ((1 - alpha) / alpha)
    bg = bg - bg.mean()
    bg = bg / (np.std(bg) + 1e-12)
    # 突发干扰：泊松到达的矩形包络 × 带内噪声（简化为随机的带内脉冲段）
    nb = int(round(rate * n / fs))
    imp = np.zeros(n)
    for _ in range(nb):
        L = int(rng.integers(int(0.012 * fs), int(0.18 * fs)))
        s = int(rng.integers(0, max(n - L, 1)))
        amp = rng.uniform(2.0, 8.0)
        imp[s:s + L] += amp * rng.standard_normal(L)
    # 按目标 SNR 定标（以 x 的能量为基准）
    px = np.mean(x ** 2)
    pn = np.mean(bg ** 2)
    g = np.sqrt(px / (pn * 10 ** (snr_db / 10.0)))
    return x + g * (bg + imp), bg * g, imp * g


# ============================================================ 轨迹估计
def stft_power(x, fs, band=BAND, nperseg=NPERSET, noverlap=NOVER):
    """返回带内功率图 (P[F,T] dB, fb[F], t[T])"""
    from scipy.signal import stft
    f, t, Z = stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap,
                   window="hann", boundary=None, padded=False)
    P = np.abs(Z) ** 2
    inb = (f >= band[0]) & (f <= band[1])
    return 10 * np.log10(np.maximum(P[inb], 1e-20)), f[inb], t


def track_dp(P, max_jump=6, lam=0.6):
    """动态规划轨迹跟踪（Viterbi 式），返回 (path_bins, score)

    为什么必须用 DP：逐帧独立取最大峰时，SNR=-3 dB 的脉冲噪声会让峰值
    在几百 Hz 间乱跳，拟合 R² 直接掉到 0.15 —— 于是"轨迹保真度"测的其实是
    噪声，不是扫频。DP 用「相邻帧频率只能小步移动」的先验把路径约束成平滑曲线，
    这正是扫频的物理约束。

    P     : [F, T] 对数功率
    lam   : 每帧能量的权重（<1 时更看重路径平滑，抗噪更强）
    返回 path[ T ]（bin 索引）、score（最优路径平均每帧 dB）
    """
    F, T = P.shape
    acc = P[:, 0].copy()
    back = np.zeros((F, T), np.int32)
    idx = np.arange(F)
    for j in range(1, T):
        best = np.full(F, -1e18)
        bestk = np.zeros(F, np.int32)
        for d in range(-max_jump, max_jump + 1):
            src = idx - d
            ok = (src >= 0) & (src < F)
            cand = np.full(F, -1e18)
            cand[ok] = acc[src[ok]]
            upd = cand > best
            best[upd] = cand[upd]
            bestk[upd] = src[upd]
        acc = best + lam * P[:, j]
        back[:, j] = bestk
    path = np.zeros(T, np.int32)
    path[-1] = int(np.argmax(acc))
    for j in range(T - 1, 0, -1):
        path[j - 1] = back[path[j], j]
    return path, float(np.max(acc) / T)


def track_ifreq(x, fs, band=BAND, nperseg=NPERSET, noverlap=NOVER, max_jump=6):
    """DP 轨迹跟踪。返回 (t, f, snr, conc, along_db)

    snr      : 沿轨迹功率 vs 带内每帧中位功率（dB）—— 只反映"该帧有没有峰"
    conc     : 沿轨迹能量 / 带内总能量 —— 信号被削弱时直接下降
    along_db : 沿轨迹**绝对**功率（相对全段最大值，dB）——
               这才是判断"这一帧有没有信号"的正确量。
               snr 不能用于此：纯噪声帧里 DP 找到的也是该帧最大值，
               相对中位数依然很高（实测：纯噪声段 snr 仍有 10+ dB）。
    """
    P, fb, t = stft_power(x, fs, band, nperseg, noverlap)
    if fb.size < 4 or t.size < 4:
        return np.array([]), np.array([]), np.array([]), float("nan"), np.array([])
    path, _ = track_dp(P, max_jump=max_jump)
    df = fb[1] - fb[0]
    fpk = np.zeros(t.size)
    for i, ki in enumerate(path):
        if 0 < ki < len(fb) - 1:
            y0, y1, y2 = P[ki - 1, i], P[ki, i], P[ki + 1, i]
            den = y0 - 2 * y1 + y2
            d = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
            d = float(np.clip(d, -0.5, 0.5))
        else:
            d = 0.0
        fpk[i] = fb[ki] + d * df
    p_lin = 10 ** (P / 10.0)
    along = p_lin[path, np.arange(t.size)]
    tot = p_lin.sum(axis=0)
    med = np.median(p_lin, axis=0)
    snr = 10 * np.log10(np.maximum(along, 1e-30) / np.maximum(med, 1e-30))
    conc = float(along.sum() / max(tot.sum(), 1e-30))
    along_db = 10 * np.log10(np.maximum(along, 1e-30) / max(along.max(), 1e-30))
    return t, fpk, snr, conc, along_db


def dedup_track(t, f, along_db, drop_db=-35.0):
    """只保留「沿轨迹确有信号能量」的帧。

    判据是**绝对**功率（相对全段峰值 −35 dB），不是相对中位数。
    这一点是自检里发现的：用 snr（相对中位数）做门限时，
    补零段/纯噪声段的 DP 路径同样会显著高于中位数，
    于是 2.67 s 的补零信号里 0.99 的帧都"有效"，
    噪声段的随机游走把轨迹拟合 R² 从 0.999 拉到 0.07。
    """
    if len(f) < 5:
        return t, f
    keep = along_db >= drop_db
    if keep.sum() < 5:
        keep = np.ones(len(f), bool)
    return t[keep], f[keep]


def fit_track(t, f, kind):
    """拟合 f(t) 或 1/f(t) 对 t 的线性模型。

    HFM 的判定依据就是 1/f 对 t 线性：
      LFM : f   = f0 + (f1-f0)·t/T          ⇒ f   对 t 线性
      HFM : 1/f = 1/f0 - (1/f1-1/f0)·t/T    ⇒ 1/f 对 t 线性
    kind='lin' → f = a·t + b；kind='inv' → 1/f = a·t + b
    返回 (R², a, b)
    """
    if len(t) < 4:
        return float("nan"), float("nan"), float("nan")
    y = f if kind == "lin" else 1.0 / np.maximum(f, 1e-9)
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else float("nan")
    return r2, float(coef[0]), float(coef[1])


def fit_track_ransac(t, f, kind, n_iter=200, k_mad=2.5, seed=0, min_inlier=8):
    """RANSAC 鲁棒拟合，返回 (R², a, b, inlier_frac)

    【为什么必须鲁棒】输入里本来就有突发干扰。DP 会跟到残留突发的谱峰上，
    产生一段与扫频无关的轨迹。普通最小二乘会被这些点带偏 ——
    实测：合成 HFM 经模型降噪后，普通拟合 R²=0.147，
    而只看真实信号段是 0.967。差的 0.82 全来自干扰帧，不是扫频被破坏。
    RANSAC 只用"多数点认同的那条直线"，把干扰帧当离群点排除。

    inlier_frac 是独立信息：它回答"有多大比例的帧服从这条主轨迹"。
    R² 高 + inlier_frac 低 = 局部丢失（例如后半段没了），必须与 R² 一起读。
    """
    n = len(t)
    if n < min_inlier:
        r2, a, b = fit_track(t, f, kind)
        return r2, a, b, float("nan")
    y = f if kind == "lin" else 1.0 / np.maximum(f, 1e-9)
    rng = np.random.default_rng(seed)
    # 内点容差用 y 的稳健尺度
    scale = 1.4826 * np.median(np.abs(y - np.median(y)))
    if scale <= 0:
        scale = float(np.std(y)) or 1e-12
    tol = k_mad * scale * 0.15        # 15% 的稳健尺度，宽松但足以排除干扰段
    best = None
    best_cnt = -1
    for _ in range(n_iter):
        i, j = rng.choice(n, 2, replace=False)
        if abs(t[i] - t[j]) < 1e-9:
            continue
        a = (y[i] - y[j]) / (t[i] - t[j])
        b = y[i] - a * t[i]
        cnt = int(np.sum(np.abs(y - (a * t + b)) <= tol))
        if cnt > best_cnt:
            best_cnt, best = cnt, (a, b)
    if best is None:
        r2, a, b = fit_track(t, f, kind)
        return r2, a, b, float("nan")
    a, b = best
    inl = np.abs(y - (a * t + b)) <= tol
    if inl.sum() >= min_inlier:
        a2, b2 = np.polyfit(t[inl], y[inl], 1)
        a, b = float(a2), float(b2)
        inl = np.abs(y - (a * t + b)) <= tol
    r2, _, _ = fit_track(t[inl] if inl.sum() >= min_inlier else t,
                         f[inl] if inl.sum() >= min_inlier else f, kind)
    return r2, a, b, float(inl.mean())


def sweep_report(x, fs, band=BAND, label=""):
    """对一个信号给出扫频结构报告（无参考）"""
    t, f, snr, conc, along_db = track_ifreq(x, fs, band)
    if len(f) < 8:
        return dict(label=label, ok=False, note="轨迹点不足",
                    valid_frac=0.0, conc=float("nan"), r2_lin=float("nan"),
                    r2_inv=float("nan"), f_lo=float("nan"), f_hi=float("nan"),
                    snr_med=float("nan"), t=np.array([]), f=np.array([]))
    td, fd = dedup_track(t, f, along_db)
    r2_lin, _, _, inl_lin = fit_track_ransac(td, fd, "lin")
    r2_inv, _, _, inl_inv = fit_track_ransac(td, fd, "inv")
    return dict(label=label, ok=True,
                n_pts=int(len(fd)), n_all=int(len(f)),
                valid_frac=float(len(fd) / len(f)),
                inlier_frac=float(max(inl_inv, inl_lin)),
                conc=float(conc),
                # 覆盖范围用有效段，不用全部帧 —— 全部帧含噪声段随机游走，会把范围虚宽
                f_lo=float(np.percentile(fd, 2)), f_hi=float(np.percentile(fd, 98)),
                f_med=float(np.median(fd)),
                r2_lin=float(r2_lin), r2_inv=float(r2_inv),
                snr_med=float(np.median(snr)),
                t=np.asarray(td), f=np.asarray(fd))


def compare_sweep(a, b):
    """比较输入(a)与输出(b)的轨迹保真度。

    先在时间上插值对齐（STM 帧时刻不同则取交集），再算：
      r        : 轨迹相关系数
      rms_hz   : 轨迹 RMS 偏差
      f0/f1 误差: 用各自拟合的 HFM 端点频率比较
      cover    : 输出覆盖频段宽度 / 输入覆盖频段宽度
    """
    if not (a.get("ok") and b.get("ok")):
        return dict(ok=False, note="轨迹不足")
    tt = np.intersect1d(np.round(a["t"], 6), np.round(b["t"], 6))
    if tt.size < 8:
        # 帧网格通常一致；若不一致则在 a 的时间轴上插值 b
        tt = a["t"]
        fb = np.interp(tt, b["t"], b["f"]) if tt[-1] <= b["t"][-1] else None
        if fb is None:
            return dict(ok=False, note="时间轴不可对齐")
    else:
        fa = np.interp(tt, a["t"], a["f"])
        fb = np.interp(tt, b["t"], b["f"])
    fa = np.interp(tt, a["t"], a["f"])
    fb = np.interp(tt, b["t"], b["f"])
    r = float(np.corrcoef(fa, fb)[0, 1]) if len(fa) > 2 else float("nan")
    rms = float(np.sqrt(np.mean((fa - fb) ** 2)))
    ca = a["f_hi"] - a["f_lo"]
    cb = b["f_hi"] - b["f_lo"]
    return dict(ok=True, r=r, rms_hz=rms,
                cover=(cb / ca if ca > 1e-9 else float("nan")),
                f_lo_err=float(b["f_lo"] - a["f_lo"]),
                f_hi_err=float(b["f_hi"] - a["f_hi"]))


# ============================================================ 模型
def load_model(ckpt, arch="R5"):
    import torch
    import pamnet as PN
    import v3_model as V3
    m = PN.PAMNet(c=24, n_fft=V3.CFG["n_fft"], hop=V3.CFG["hop_length"],
                  norm="gn", **ARCHS[arch])
    p = ckpt if os.path.isabs(ckpt) else os.path.join(CKPT_DIR, ckpt)
    m.load_state_dict(torch.load(p, map_location="cpu"), strict=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return m.to(dev).eval(), dev


def enhance_long(x, fs, model, dev, rms_norm=True):
    """整段增强：按 SEG 分帧（末帧补零），返回同长度输出

    注意：训练是逐帧独立前向，所以整段处理必然有帧间不连续。
    本函数保持与 run_real_data.py 完全一致的路径，不额外做交叉淡化，
    以便下游指标反映真实部署行为。
    """
    import torch
    n = len(x)
    nf = int(np.ceil(n / SEG))
    F = np.zeros((nf, SEG), np.float32)
    for i in range(nf):
        s = x[i * SEG:(i + 1) * SEG]
        F[i, :len(s)] = s
    outs = []
    with torch.no_grad():
        for i in range(0, nf, 16):
            nb = torch.from_numpy(F[i:i + 16]).to(dev).unsqueeze(1)
            if rms_norm:
                rms = torch.sqrt((nb ** 2).mean(dim=-1, keepdim=True) + 1e-12)
                y = model(nb / rms) * rms
            else:
                y = model(nb)
            outs.append(y.squeeze(1).cpu().numpy())
    Y = np.concatenate(outs, 0)
    return np.concatenate([Y[i, :len(x[i * SEG:(i + 1) * SEG])] for i in range(nf)])


# ============================================================ 自检
def selftest():
    print("=" * 74)
    print("工具有效性验证：合成 HFM → 加噪 → 降噪 → 检验能否识别「保住」与「破坏」")
    print("=" * 74)
    rng = np.random.default_rng(0)
    fs = FS_NET
    T = 1.0
    clean, t = gen_chirp(fs, T, 798.0, 1500.0, "HFM", "rect", 0.0, 1.0)
    n = len(clean)
    pad = np.zeros(SEG * 2)
    pad[:n] = clean
    clean_full = pad
    noisy, bg, imp = add_impulsive(clean_full, fs, rate=5.0, alpha=1.5,
                                    snr_db=-3.0, rng=rng)

    ref = sweep_report(clean_full, fs, label="clean")
    print("\n[1] 干净 HFM 的轨迹（工具应能正确识别为 HFM 且 R² 很高）")
    print("    拟合 R²: 1/f~t（HFM 模型） %.4f   f~t（LFM 模型） %.4f  ⇒ 判定 %s"
          % (ref["r2_inv"], ref["r2_lin"],
             "HFM" if ref["r2_inv"] > ref["r2_lin"] else "LFM"))
    print("    轨迹点 %d，覆盖 %.0f–%.0f Hz，峰值 SNR 中位 %.1f dB"
          % (ref["n_pts"], ref["f_lo"], ref["f_hi"], ref["snr_med"]))

    model, dev = load_model("model_inm_pf.pth")
    est = enhance_long(noisy, fs, model, dev)

    print("\n[2] 加噪后 → 模型降噪 → 轨迹保真（期望：R² 仍高、r 接近 1）")
    rows = []
    cases = {
        "noisy（降噪前）": noisy,
        "est（模型输出）": est,
    }
    for nm, sig in cases.items():
        rep = sweep_report(sig, fs, label=nm)
        cmp_ = compare_sweep(ref, rep)
        rows.append((nm, rep, cmp_))
        print("    %-14s R²(HFM) %.3f  R²(LFM) %.3f  轨迹r %+.3f  RMS %4.0f Hz  "
              "覆盖 %.2f  内点 %.2f  有效段 %.2f  集中度 %.3f"
              % (nm, rep["r2_inv"], rep["r2_lin"],
                 cmp_.get("r", float("nan")), cmp_.get("rms_hz", float("nan")),
                 cmp_.get("cover", float("nan")), rep["inlier_frac"],
                 rep["valid_frac"], rep["conc"]))

    print("\n[3] 人为破坏：工具必须能把它们与「正常降噪」区分开")
    breaks = {}
    # ① 带通削边：把 780-900 与 1400-1520 砍掉
    from scipy.signal import butter, sosfiltfilt
    sos = butter(8, [900 / (fs / 2), 1400 / (fs / 2)], btype="band", output="sos")
    breaks["削边（900–1400 带通）"] = sosfiltfilt(sos, est)
    # ② 频率压缩 5%：轨迹整体被扭
    breaks["频率压缩 5%"] = np.interp(np.arange(len(est)),
                                   np.arange(len(est)) / 1.05, est)
    # ③ 时移 100 ms：轨迹整体错位
    sh = int(0.1 * fs)
    breaks["时移 100 ms"] = np.roll(est, sh)
    # ④ 半段置零：后半段扫频丢失
    half = est.copy()
    half[len(half) // 2:] = 0
    breaks["后半段丢失"] = half

    print("    %-20s %8s %8s %8s %8s %8s" %
          ("破坏方式", "轨迹r", "RMS(Hz)", "覆盖", "内点", "集中度"))
    bstats = {}
    for nm, sig in breaks.items():
        rep = sweep_report(sig, fs, label=nm)
        cmp_ = compare_sweep(ref, rep)
        bstats[nm] = dict(r=cmp_.get("r"), rms=cmp_.get("rms_hz"),
                          cover=cmp_.get("cover"), valid=rep["valid_frac"],
                          inl=rep["inlier_frac"], conc=rep["conc"],
                          r2=rep["r2_inv"])
        print("    %-20s %+8.3f %8.0f %8.2f %8.2f %8.3f"
              % (nm, cmp_.get("r", float("nan")), cmp_.get("rms_hz", float("nan")),
                 cmp_.get("cover", float("nan")), rep["inlier_frac"], rep["conc"]))

    # ---- 判据 ----
    est_rep, est_cmp = rows[1][1], rows[1][2]
    c1 = ref["r2_inv"] > 0.95 and ref["r2_inv"] > ref["r2_lin"]      # 能识别 HFM
    c2 = est_cmp.get("r", 0) > 0.95                                   # 降噪后轨迹位置保住
    c3 = est_rep["r2_inv"] > 0.90                                     # 降噪后轨迹形状保住
    # 每种破坏至少要在一个指标上明显偏离 est
    est_rms = est_cmp.get("rms_hz") or 1e-9
    def flagged(b):
        return ((b["r"] is not None and b["r"] < est_cmp.get("r", 1) - 0.04) or
                (b["rms"] is not None and b["rms"] > est_rms * 1.4) or
                (b["cover"] is not None and abs(b["cover"] - 1) > 0.05) or
                (b["valid"] < est_rep["valid_frac"] - 0.10) or
                (b["inl"] < est_rep["inlier_frac"] - 0.10) or
                (b["conc"] < est_rep["conc"] * 0.7))
    c4 = all(flagged(v) for v in bstats.values())
    print("\n[判定] 识别 HFM %s | 降噪后 轨迹r=%.3f %s | 轨迹R²=%.3f %s | 四种破坏全被标记 %s"
          % ("✓" if c1 else "✗", est_cmp.get("r", float("nan")), "✓" if c2 else "✗",
             est_rep["r2_inv"], "✓" if c3 else "✗", "✓" if c4 else "✗"))
    for nm, v in bstats.items():
        print("       %-20s %s" % (nm, "被标记" if flagged(v) else "**未检出**"))

    return dict(ref=dict(r2_inv=ref["r2_inv"], r2_lin=ref["r2_lin"],
                         f_lo=ref["f_lo"], f_hi=ref["f_hi"], n_pts=ref["n_pts"]),
                est=dict(r=est_cmp.get("r"), rms=est_cmp.get("rms_hz"),
                         cover=est_cmp.get("cover"),
                         valid=est_rep["valid_frac"], conc=est_rep["conc"],
                         inl=est_rep["inlier_frac"],
                         r2_inv=est_rep["r2_inv"], r2_lin=est_rep["r2_lin"]),
                noisy=dict(r=rows[0][2].get("r"), rms=rows[0][2].get("rms_hz"),
                           valid=rows[0][1]["valid_frac"], conc=rows[0][1]["conc"],
                           inl=rows[0][1]["inlier_frac"],
                           r2_inv=rows[0][1]["r2_inv"], r2_lin=rows[0][1]["r2_lin"]),
                breaks=bstats,
                tool_ok=bool(c1 and c2 and c3 and c4),
                checks=dict(ident=c1, fidelity=c2, coverage=c3, detect=c4))


# ============================================================ 实测
def run_real(path, ckpt, outdir):
    from scipy.io import wavfile
    fs0, raw = wavfile.read(path)
    x = np.asarray(raw).astype(np.float64)
    if np.issubdtype(np.asarray(raw).dtype, np.integer):
        info = np.iinfo(np.asarray(raw).dtype)
        x = x / max(abs(info.min), info.max)
    x = x.ravel()
    from scipy.signal import resample_poly
    if abs(fs0 - 24000) < 1:
        s = resample_poly(x, 1, 4)
    elif abs(fs0 - FS_NET) < 1:
        s = x
    else:
        g = int(np.gcd(int(fs0), int(FS_NET)))
        s = resample_poly(x, int(FS_NET) // g, int(fs0) // g)

    model, dev = load_model(ckpt)
    est = enhance_long(s, FS_NET, model, dev)

    a = sweep_report(s, FS_NET, label="input")
    b = sweep_report(est, FS_NET, label="output")
    c = compare_sweep(a, b)

    print("\n实测文件 %s（%.2f s @%d Hz → %.0f Hz）" % (os.path.basename(path),
                                                 len(x) / fs0, fs0, FS_NET))
    print("  输入轨迹: 点 %d  R²(1/f) %.3f  R²(f) %.3f  覆盖 %.0f–%.0f Hz  峰值SNR %.1f dB"
          % (a["n_pts"], a["r2_inv"], a["r2_lin"], a["f_lo"], a["f_hi"], a["snr_med"]))
    print("  输出轨迹: 点 %d  R²(1/f) %.3f  R²(f) %.3f  覆盖 %.0f–%.0f Hz  峰值SNR %.1f dB"
          % (b["n_pts"], b["r2_inv"], b["r2_lin"], b["f_lo"], b["f_hi"], b["snr_med"]))
    gate = max(a["r2_inv"], a["r2_lin"])
    print("  轨迹保真: r %+.3f  RMS %.0f Hz  覆盖比 %.2f"
          % (c.get("r", float("nan")), c.get("rms_hz", float("nan")),
             c.get("cover", float("nan"))))
    if gate < 0.6:
        print("  ⚠ 闸门未过：输入轨迹的最好拟合 R² 仅 %.3f < 0.6 ⇒ "
              "【该信号没有可辨的扫频结构】，上面的保真度指标无意义。" % gate)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "hfm_fidelity_%s.json" % os.path.splitext(os.path.basename(path))[0])
    json.dump(dict(file=os.path.basename(path), fs_in=int(fs0), fs_net=FS_NET,
                   dur_s=len(x) / fs0,
                   input=dict(n_pts=a["n_pts"], r2_inv=a["r2_inv"], r2_lin=a["r2_lin"],
                              f_lo=a["f_lo"], f_hi=a["f_hi"], snr_med=a["snr_med"]),
                   output=dict(n_pts=b["n_pts"], r2_inv=b["r2_inv"], r2_lin=b["r2_lin"],
                               f_lo=b["f_lo"], f_hi=b["f_hi"], snr_med=b["snr_med"]),
                   compare=dict(r=c.get("r"), rms_hz=c.get("rms_hz"),
                                cover=c.get("cover"), f_lo_err=c.get("f_lo_err"),
                                f_hi_err=c.get("f_hi_err")),
                   gate_r2=gate, gate_pass=bool(gate >= 0.6),
                   verdict=("无扫频结构，保真度不可判" if gate < 0.6
                            else "存在扫频结构，可读保真度")),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("  -> %s" % out)
    return dict(file=os.path.basename(path), gate=gate, gate_pass=bool(gate >= 0.6),
                r=c.get("r"), rms=c.get("rms_hz"), cover=c.get("cover"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--input")
    ap.add_argument("--ckpt", default="model_inm_pf.pth")
    ap.add_argument("--outdir", default=os.path.join(HERE, "results", "real"))
    A = ap.parse_args()

    res = {}
    if A.selftest:
        res["selftest"] = selftest()
    if A.input:
        files = ([A.input] if os.path.isfile(A.input)
                 else sorted(glob.glob(os.path.join(A.input, "*.wav"))))
        res["real"] = [run_real(f, A.ckpt, A.outdir) for f in files]
    if not A.selftest and not A.input:
        ap.print_help()
        return
    p = os.path.join(HERE, "sens_out", "hfm_fidelity.json")
    old = {}
    if os.path.exists(p):
        old = json.load(open(p, encoding="utf-8"))
    old.update(res)
    json.dump(old, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n-> sens_out/hfm_fidelity.json")


if __name__ == "__main__":
    main()
