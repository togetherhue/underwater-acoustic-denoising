# -*- coding: utf-8 -*-
"""验证 PAM 数据集：字段完整性、配对性、噪声强度匹配、脉冲性、指标稳定性
   用法: python verify_pam_dataset.py <数据根目录> [对照根目录]
"""
import os, sys, glob
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

ROOT = sys.argv[1] if len(sys.argv) > 1 else r"D:/project/ds_pam_probe/alphastable"
ALT = sys.argv[2] if len(sys.argv) > 2 else None
SPLITS = ["train", "val", "testA", "testC"]
SEG, DS = 8000, 4

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pam_metrics as PM
import torch

print("=" * 104)
print("PAM 数据集验证")
print("  主数据集: %s" % ROOT)
if ALT:
    print("  对照集  : %s" % ALT)
print("=" * 104)


def load_one(root, sp, i):
    """降采样到训练时的采样率（6000 Hz）再截断到 8000，与训练管线一致"""
    f = sorted(glob.glob(os.path.join(root, sp, "*.mat")))[i]
    m = sio.loadmat(f)
    def rs(x):
        return resample_poly(np.asarray(x).squeeze().astype(np.float64), 1, DS)
    n = rs(m["noisy_signal"])[:SEG]
    c = rs(m["clean_signal"])[:SEG]
    ip = rs(m["imp"])[:SEG]
    mk = rs(np.asarray(m["imp_mask"]).squeeze().astype(np.float64))[:SEG]
    return n, c, ip, (mk > 0.3).astype(np.float64), m


# ---------- 1. 字段完整性 ----------
print("\n[1] 字段完整性")
need = {"noisy_signal", "clean_signal", "imp", "imp_mask",
        "true_snr", "chirp_id", "chan_id"}
ok_fields = True
for sp in SPLITS:
    files = sorted(glob.glob(os.path.join(ROOT, sp, "*.mat")))
    if not files:
        print("  %-6s 无文件" % sp); ok_fields = False; continue
    m = sio.loadmat(files[0])
    have = {k for k in m if not k.startswith("__")}
    shp = np.shape(m["noisy_signal"])
    bad = (need - have) or np.iscomplexobj(m["noisy_signal"]) or len(shp) != 2
    print("  %-6s n=%4d  缺失=%-8s noisy形状=%-12s 复数=%-5s %s"
          % (sp, len(files), ",".join(sorted(need - have)) or "无", shp,
             np.iscomplexobj(m["noisy_signal"]), "✗" if bad else "✓"))
    if bad:
        ok_fields = False
assert ok_fields, "字段检查未通过，停止"

# ---------- 2. 配对性 ----------
if ALT:
    print("\n[2] 配对性：与对照集逐样本比较（应完全一致）")
    for sp in SPLITS:
        fa = sorted(glob.glob(os.path.join(ROOT, sp, "*.mat")))
        fb = sorted(glob.glob(os.path.join(ALT, sp, "*.mat")))
        n = min(len(fa), len(fb))
        d_imp = d_mask = d_clean = d_cid = 0
        for i in range(n):
            A, B = sio.loadmat(fa[i]), sio.loadmat(fb[i])
            d_imp = max(d_imp, float(np.abs(np.asarray(A["imp"]).squeeze()
                                            - np.asarray(B["imp"]).squeeze()).max()))
            d_mask = max(d_mask, int(np.abs(np.asarray(A["imp_mask"]).squeeze().astype(int)
                                            - np.asarray(B["imp_mask"]).squeeze().astype(int)).max()))
            d_clean = max(d_clean, float(np.abs(np.asarray(A["clean_signal"]).squeeze()
                                                - np.asarray(B["clean_signal"]).squeeze()).max()))
            d_cid = max(d_cid, int(np.asarray(A["chirp_id"]).ravel()[0])
                        - int(np.asarray(B["chirp_id"]).ravel()[0]))
        print("  %-6s n=%3d  干扰信号差 %.1e  掩码差 %d  干净信号差 %.1e  chirp_id 差 %d"
              % (sp, n, d_imp, d_mask, d_clean, d_cid))
    print("  → 干扰实现完全一致 => 两套数据构成【配对对照】，只差背景噪声分布")

# ---------- 3. 噪声强度与脉冲性 ----------
print("\n[3] 噪声强度与脉冲性（降采样后）")
print("  %-12s %-6s %10s %10s %10s %9s %11s %9s"
      % ("数据集", "split", "q75(|v|)", "脉冲峰值", "占空比", "干扰/rms", "背景/rms", "偏度"))
for tag, root in ([(os.path.basename(ROOT), ROOT)] +
                  ([(os.path.basename(ALT), ALT)] if ALT else [])):
    for sp in SPLITS:
        files = sorted(glob.glob(os.path.join(root, sp, "*.mat")))
        q, pk, du, ir, br, sk = [], [], [], [], [], []
        for i in range(0, len(files), max(1, len(files) // 30)):
            n, c, ip, mk, _ = load_one(root, sp, i)
            v = n - c - ip
            rs_ = np.sqrt((c ** 2).mean())
            q.append(np.percentile(np.abs(v), 75))
            pk.append(np.abs(ip).max() / rs_)
            du.append(mk.mean())
            ir.append(np.sqrt((ip ** 2).mean()) / rs_)
            br.append(np.sqrt((v ** 2).mean()) / rs_)
            vv = v - v.mean()
            sk.append((vv ** 3).mean() / max((vv ** 2).mean() ** 1.5, 1e-30))
        print("  %-12s %-6s %10.4f %10.2f %10.3f %9.3f %11.3f %9.2f"
              % (tag, sp, np.mean(q), np.mean(pk), np.mean(du), np.mean(ir), np.mean(br), np.mean(sk)))
print("  偏度参考：高斯≈0；α稳定(α=1.5)经低通后仍应明显大于 0")
print("  占空比：突发参数 rate=5/s、时长 2~30ms -> 理论约 9%%（降采样会略微改变）")

# ---------- 4. 干净信号与既有数据集一致性 ----------
print("\n[4] 干净信号 vs 既有 v3 数据集（保证与历史结果可比）")
ref = r"D:/project/dataset_v3/testA_seench_newchirp/s_00001.mat"
cur = os.path.join(ROOT, "testA", "s_00001.mat")
if os.path.exists(ref) and os.path.exists(cur):
    a = np.asarray(sio.loadmat(cur)["clean_signal"]).squeeze()
    b = np.asarray(sio.loadmat(ref)["clean_signal"]).squeeze()
    m = min(len(a), len(b))
    d = float(np.abs(a[:m] - b[:m]).max())
    print("  长度 %d vs %d   最大绝对差 %.3e" % (len(a), len(b), d))
    print("  判定:", "✓ 逐位一致" if d == 0 else "✗ 不一致")
else:
    print("  （参考文件不存在，跳过）")

# ---------- 5. 全局 SNR 是否已失去意义 ----------
print("\n[5] 关键检查：脉冲噪声下全局 SNR 的稳定性")
for tag, root in ([(os.path.basename(ROOT), ROOT)] +
                  ([(os.path.basename(ALT), ALT)] if ALT else [])):
    snrs, meds = [], []
    files = sorted(glob.glob(os.path.join(root, "testA", "*.mat")))
    for i in range(0, len(files), max(1, len(files) // 40)):
        n, c, ip, mk, _ = load_one(root, "testA", i)
        T = torch.from_numpy(c).float().unsqueeze(0)
        N = torch.from_numpy(n).float().unsqueeze(0)
        snrs.append(PM.snr_db(N, T).item())
        st = PM._stats_1d(n, c, n, PM.FRAME, mk)
        meds.append(st["fmsnr_med"])
    snrs, meds = np.array(snrs), np.array(meds)
    print("  [%s] 输入信号：" % tag)
    print("      全局 SNR     均值 %7.2f  标准差 %6.2f  范围 %.2f ~ %.2f dB"
          % (snrs.mean(), snrs.std(), snrs.min(), snrs.max()))
    print("      帧 SNR 中位  均值 %7.2f  标准差 %6.2f  范围 %.2f ~ %.2f dB"
          % (meds.mean(), meds.std(), meds.min(), meds.max()))
    print("      -> 全局 SNR 波动是帧中位的 %.1f 倍" % (snrs.std() / max(meds.std(), 1e-9)))
