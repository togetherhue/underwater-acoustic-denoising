"""验证 大创.py 的预处理与 SNR 指标，纯 Python 实现（无 numpy）。"""
import sys, os, math, cmath
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mat_probe import load


def gv(path):
    return {m['name']: list(m['data']) for m in load(path)}


def pct99(v):
    a = sorted(abs(x) for x in v)
    n = len(a)
    pos = 0.99 * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return a[lo] * (1 - frac) + a[hi] * frac


def rms(v):
    return math.sqrt(sum(t * t for t in v) / len(v))


def corr(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a) / n
    vb = sum((x - mb) ** 2 for x in b) / n
    if va <= 0 or vb <= 0:
        return 0.0
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / n / math.sqrt(va * vb)


def reprocess(noisy_path, clean_path, L=18549):
    """复刻 DenoiseDataset.__getitem__ 的逻辑"""
    noisy = gv(noisy_path)
    noisy = list(noisy.values())[0]
    clean = gv(clean_path)['chirp_signal']

    if len(clean) < L:
        clean_final = clean + [0.0] * (L - len(clean))
    else:
        clean_final = clean[:L]

    if len(noisy) < L:
        pad_len = L - len(noisy)
        if len(noisy) > pad_len:
            pad = noisy[-pad_len:]
        else:
            pad = [0.0] * pad_len
        noisy_final = noisy + pad
    else:
        noisy_final = noisy[:L]

    mn = pct99(noisy_final)
    noisy_arr = [x / mn for x in noisy_final] if mn > 0 else list(noisy_final)
    noisy_arr = [max(-1.5, min(1.5, x)) for x in noisy_arr]

    mc = pct99(clean_final)
    clean_arr = [x / mc for x in clean_final] if mc > 0 else list(clean_final)
    clean_arr = [max(-1.5, min(1.5, x)) for x in clean_arr]
    return noisy_arr, clean_arr


def pseudo_snr(signal, intervals, fs):
    """复刻 calculate_snr_intervals"""
    out = []
    for s0, s1, n0, n1 in intervals:
        a, b = round(s0 * fs), round(s1 * fs)
        c, d = round(n0 * fs), round(n1 * fs)
        sp = signal[a:b]
        npr = signal[c:d]
        if len(npr) == 0:
            continue
        rf = int(math.ceil(len(sp) / len(npr)))
        rep = (npr * rf)[:len(sp)]
        sig_only = [sp[i] - rep[i] for i in range(len(sp))]
        npow = sum(x * x for x in npr) / len(npr)
        if npow > 1e-10:
            out.append(10 * math.log10((sum(x * x for x in sig_only) / len(sig_only)) / npow))
    return sum(out) / len(out) if out else float('-inf')


def true_snr(ref, test):
    ps = sum(x * x for x in ref) / len(ref)
    pn = sum((test[i] - ref[i]) ** 2 for i in range(len(ref))) / len(ref)
    return 10 * math.log10(ps / pn) if pn > 0 else float('inf')


print('=' * 74)
print('1) 复刻预处理，与 output/final_denoised_signal_1.mat 对比')
print('=' * 74)
noisy_r, clean_r = reprocess('D:/project/te_signal/test_sample_1.mat', 'D:/project/chirp_signal.mat')
saved = gv('D:/project/output/final_denoised_signal_1.mat')
print('  复刻 noisy : 长度=%d min=%.6f max=%.6f rms=%.6f' % (len(noisy_r), min(noisy_r), max(noisy_r), rms(noisy_r)))
print('  存档 noisy : 长度=%d min=%.6f max=%.6f rms=%.6f' %
      (len(saved['noisy_signal']), min(saved['noisy_signal']), max(saved['noisy_signal']), rms(saved['noisy_signal'])))
print('  两者相关系数 = %.6f' % corr(noisy_r, saved['noisy_signal']))
print()
print('  复刻 clean : 长度=%d min=%.6f max=%.6f rms=%.6f' % (len(clean_r), min(clean_r), max(clean_r), rms(clean_r)))
print('  存档 clean : 长度=%d min=%.6f max=%.6f rms=%.6f' %
      (len(saved['clean_signal']), min(saved['clean_signal']), max(saved['clean_signal']), rms(saved['clean_signal'])))
print('  两者相关系数 = %.6f' % corr(clean_r, saved['clean_signal']))

print()
print('=' * 74)
print('2) 指标可信度：代码自报 SNR vs 真实 SNR  (intervals=[(0.10,0.30,0.50,0.70)], fs=24000)')
print('=' * 74)
IV = [(0.10, 0.30, 0.50, 0.70)]
FS = 24000
rows = []
for i in range(1, 12):
    p = 'D:/project/output/final_denoised_signal_%d.mat' % i
    if not os.path.exists(p):
        continue
    d = gv(p)
    N, D, C = d['noisy_signal'], d['denoised_signal'], d['clean_signal']
    pn = pseudo_snr(N, IV, FS)
    pd = pseudo_snr(D, IV, FS)
    tn = true_snr(C, N)
    td = true_snr(C, D)
    rows.append((i, pn, pd, tn, td, float(d['original_snr'][0]), float(d['final_snr'][0])))
print('%-4s %10s %10s %10s %10s %12s %12s' % ('file', '伪SNR_输入', '伪SNR_输出', '真SNR_输入', '真SNR_输出', '存档报输入', '存档报输出'))
for r in rows:
    print('%-4d %10.2f %10.2f %10.2f %10.2f %12.2f %12.2f' % r)
print()
print('  伪SNR 与真SNR 的偏差(输入): 平均 %.2f dB' %
      (sum(r[1] - r[3] for r in rows) / len(rows)))
print('  伪SNR 与真SNR 的偏差(输出): 平均 %.2f dB' %
      (sum(r[2] - r[4] for r in rows) / len(rows)))
print('  真实平均增益 %.2f dB  |  伪SNR 平均增益 %.2f dB  |  存档平均增益 %.2f dB' %
      (sum(r[4] - r[3] for r in rows) / len(rows),
       sum(r[2] - r[1] for r in rows) / len(rows),
       sum(r[6] - r[5] for r in rows) / len(rows)))

print()
print('=' * 74)
print('3) 干净 chirp 结构（判断 fs 与区间设置是否合理）')
print('=' * 74)
clean = gv('D:/project/chirp_signal.mat')['chirp_signal']
print('  长度 = %d 点' % len(clean))
env = [abs(clean[i:i + 200]) for i in range(0, len(clean) - 200, 200)]
envmax = [max(e) for e in env]
thr = 0.05 * max(envmax)
on = [i for i, v in enumerate(envmax) if v > thr]
print('  能量显著区间: 采样点 %d ~ %d' % (on[0] * 200, on[-1] * 200 + 200))


def dominant(seg, fs, lo, hi, step):
    best, bf = -1, 0
    f = lo
    while f <= hi:
        w = 2 * math.pi * f / fs
        re = sum(seg[i] * math.cos(w * i) for i in range(0, len(seg), 2))
        im = sum(seg[i] * math.sin(w * i) for i in range(0, len(seg), 2))
        p = re * re + im * im
        if p > best:
            best, bf = p, f
        f += step
    return bf


seg_a = clean[on[0] * 200:on[0] * 200 + 1500]
seg_b = clean[max(0, on[-1] * 200 - 1500):on[-1] * 200]
for fs in (12000, 24000, 48000):
    fa = dominant(seg_a, fs, 100, fs / 2 - 100, 1)
    fb = dominant(seg_b, fs, 100, fs / 2 - 100, 1)
    dur = len(clean) / fs
    print('  假设 fs=%5d Hz → 总时长 %.3f s, 起始段主频 %6.1f Hz, 结束段主频 %6.1f Hz' % (fs, dur, fa, fb))
