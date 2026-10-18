# -*- coding: utf-8 -*-
"""
把 .mat 数据集预处理成 .npy 缓存（加速训练数据管线）
================================================================================
【为什么】
PAMDataset 每次 __getitem__ 都 loadmat + resample_poly。5040 个训练样本 × 每轮
= 每轮重复做完全相同的降采样。GPU 提速后这将成为新瓶颈。

本脚本一次性把预处理结果存成 .npy，之后用 mmap 读取，加载成本降一个数量级。

【口径必须与 PAMDataset 完全一致】
    n = resample_poly(loadmat['noisy_signal'].squeeze().astype(float32), 1, ds).astype(float32)
    c = resample_poly(loadmat['clean_signal'].squeeze().astype(float32), 1, ds).astype(float32)
    k = resample_poly(loadmat['imp_mask'].squeeze().astype(float64), 1, ds)
    fix(x) = x[:seg_len] if len>=seg_len else pad
    mask = (fix(k) > 0.3).astype(float32)
注意顺序：**先 astype(float32) 再 resample_poly** —— 反过来会产生 1e-7 相对差，
经时域通路放大成 0.1 dB 级差异（报告局限里记录过）。

【输出】
    <root>/_npy/<split>_n.npy   float32 [N, seg_len]
    <root>/_npy/<split>_c.npy   float32 [N, seg_len]
    <root>/_npy/<split>_m.npy   float32 [N, seg_len]  (0/1)

用法：
    python make_npy_cache.py --data E:/project/data/ds_sens/a15_r10
    python make_npy_cache.py --data ... --verify   # 转换后逐元素校验
"""
import os, sys, glob, argparse, time
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

SEG_DEFAULT = 8000
DS_DEFAULT = 4


def fix(x, L):
    return x[:L] if len(x) >= L else np.pad(x, (0, L - len(x)))


def load_one(path, seg_len, ds):
    m = sio.loadmat(path)
    n = np.asarray(m["noisy_signal"]).squeeze().astype(np.float32)
    c = np.asarray(m["clean_signal"]).squeeze().astype(np.float32)
    k = np.asarray(m["imp_mask"]).squeeze().astype(np.float64)
    if ds > 1:
        n = resample_poly(n, 1, ds).astype(np.float32)
        c = resample_poly(c, 1, ds).astype(np.float32)
        k = resample_poly(k, 1, ds)
    k = (fix(k, seg_len) > 0.3).astype(np.float32)
    return fix(n, seg_len), fix(c, seg_len), k


def convert_split(folder, outdir, split, seg_len, ds, verbose=True):
    files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    if not files:
        print("  [跳过] %s 里没有 .mat" % folder); return None
    t0 = time.time()
    N = len(files)
    n_arr = np.zeros((N, seg_len), dtype=np.float32)
    c_arr = np.zeros((N, seg_len), dtype=np.float32)
    m_arr = np.zeros((N, seg_len), dtype=np.float32)
    for i, f in enumerate(files):
        a, b, k = load_one(f, seg_len, ds)
        n_arr[i], c_arr[i], m_arr[i] = a, b, k
        if verbose and (i + 1) % 500 == 0:
            print("    %s %d/%d  (%.0fs)" % (split, i + 1, N, time.time() - t0), flush=True)
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, "%s_n.npy" % split), n_arr)
    np.save(os.path.join(outdir, "%s_c.npy" % split), c_arr)
    np.save(os.path.join(outdir, "%s_m.npy" % split), m_arr)
    mb = n_arr.nbytes * 2.25 / 1e6
    print("  %-6s %5d 样本 -> %.0f MB  (%.1fs)" % (split, N, mb, time.time() - t0))
    return N


def verify(folder, outdir, split, seg_len, ds, ncheck=40):
    """随机抽 ncheck 个样本，逐元素比对 .npy 与直接解析 .mat 的结果"""
    files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    n_arr = np.load(os.path.join(outdir, "%s_n.npy" % split), mmap_mode="r")
    c_arr = np.load(os.path.join(outdir, "%s_c.npy" % split), mmap_mode="r")
    m_arr = np.load(os.path.join(outdir, "%s_m.npy" % split), mmap_mode="r")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(files), size=min(ncheck, len(files)), replace=False)
    dn = dc = dm = 0.0
    for i in idx:
        a, b, k = load_one(files[i], seg_len, ds)
        dn = max(dn, float(np.abs(a - n_arr[i]).max()))
        dc = max(dc, float(np.abs(b - c_arr[i]).max()))
        dm = max(dm, float(np.abs(k - m_arr[i]).max()))
    ok = (dn == 0.0 and dc == 0.0 and dm == 0.0)
    print("  [校验] %s  抽查 %d 个  最大差 noisy %.3e / clean %.3e / mask %.3e  %s"
          % (split, len(idx), dn, dc, dm, "✓ 逐位一致" if ok else "✗ 不一致"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="数据集根目录（含 train/val/testA/testC）")
    ap.add_argument("--seg", type=int, default=SEG_DEFAULT)
    ap.add_argument("--ds", type=int, default=DS_DEFAULT)
    ap.add_argument("--splits", default="train,val,testA,testC")
    ap.add_argument("--verify", action="store_true")
    A = ap.parse_args()

    root = A.data.replace("\\", "/").rstrip("/")
    outdir = os.path.join(root, "_npy")
    splits = [s.strip() for s in A.splits.split(",") if s.strip()]
    print("数据集 %s -> %s" % (root, outdir))
    for sp in splits:
        folder = os.path.join(root, sp)
        if not os.path.isdir(folder):
            print("  [跳过] 无目录 %s" % folder); continue
        if A.verify:
            verify(folder, outdir, sp, A.seg, A.ds)
        else:
            convert_split(folder, outdir, sp, A.seg, A.ds)
    if not A.verify:
        print("\n转换完成。用 --verify 可逐元素校验。")


if __name__ == "__main__":
    main()
