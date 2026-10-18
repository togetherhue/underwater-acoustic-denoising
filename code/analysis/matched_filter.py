"""匹配滤波检验：含噪信号里到底有没有 chirp？"""
import sys, os, math, cmath
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mat_probe import load


def gv(p, k=None):
    d = {m['name']: list(m['data']) for m in load(p)}
    return d[k] if k else list(d.values())[0]


def fft(a):
    n = len(a)
    if n == 1:
        return a
    ev = fft(a[0::2])
    od = fft(a[1::2])
    T = [cmath.exp(-2j * math.pi * k / n) * od[k] for k in range(n // 2)]
    return [ev[k] + T[k] for k in range(n // 2)] + [ev[k] - T[k] for k in range(n // 2)]


def ifft(a):
    n = len(a)
    y = fft([x.conjugate() for x in a])
    return [x.conjugate() / n for x in y]


def matched(x, tmpl, N):
    """互相关 x 与 tmpl（零延迟对齐到 x[0]）"""
    X = fft([complex(v) for v in x[:N]] + [0j] * (N - len(x[:N])))
    T = fft([complex(v) for v in tmpl] + [0j] * (N - len(tmpl)))
    C = [X[i] * T[i].conjugate() for i in range(N)]
    c = ifft(C)
    return [z.real for z in c]


def rms(v):
    return math.sqrt(sum(t * t for t in v) / len(v))


print('=' * 78)
print('匹配滤波：用干净 chirp 与含噪信号做互相关')
print('=' * 78)
chirp = gv('D:/project/chirp_signal.mat', 'chirp_signal')
N = 32768
cases = [
    ('te_signal/test_sample_1', 'D:/project/te_signal/test_sample_1.mat'),
    ('te_signal/test_sample_2', 'D:/project/te_signal/test_sample_2.mat'),
    ('tr_signal/chrip_with_noise1000', 'D:/project/tr_signal/chrip_with_noise1000.mat'),
    ('tr_signal/chrip_with_noise1500', 'D:/project/tr_signal/chrip_with_noise1500.mat'),
    ('tr_signal/chirp_with_noise_1 (4月)', 'D:/project/tr_signal/chirp_with_noise_1.mat'),
]
for name, path in cases:
    v = gv(path)
    c = matched(v, chirp, N)
    m = max(abs(z) for z in c[:len(v)])
    r = rms(c[:len(v)])
    peak_idx = max(range(len(v)), key=lambda i: abs(c[i]))
    # 噪声底：去掉峰值邻域后的 rms
    masked = [c[i] for i in range(len(v)) if abs(i - peak_idx) > 200]
    nr = rms(masked) if masked else 1e-30
    print('  %-34s 峰值=%.3e @n=%5d   相关后底噪rms=%.3e   峰值/底噪=%6.1f (%.1f dB)'
          % (name, m, peak_idx, nr, m / nr, 20 * math.log10(m / nr)))

print()
print('=' * 78)
print('多个 channel_H 文件的频带（归一化 cycles/sample）')
print('=' * 78)
import glob
files = sorted(glob.glob('E:/project/data/信道产生/data/channel_H_*.mat'))[:8]
for p in files:
    H = gv(p, 'channel_H')
    N2 = 8192
    x = [complex(t) for t in H[:N2]] + [0j] * max(0, N2 - len(H))
    X = fft(x)
    P = [abs(X[k]) ** 2 for k in range(N2 // 2 + 1)]
    tot = sum(P)
    acc = 0.0
    lo = hi = None
    for k, pv in enumerate(P):
        acc += pv
        if lo is None and acc >= 0.005 * tot:
            lo = k
        if acc >= 0.995 * tot:
            hi = k
            break
    print('  %-26s 长度=%d  99%%能量带 = %.5f ~ %.5f  (bin %d~%d)'
          % (os.path.basename(p), len(H), lo / N2, hi / N2, round(lo / N2 * 8000), round(hi / N2 * 8000)))
