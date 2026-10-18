"""决定性检验：含噪信号里到底有没有 chirp 的成分？各频带能量对比。"""
import sys, os, math, cmath
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mat_probe import load


def gv(p, key=None):
    d = {m['name']: list(m['data']) for m in load(p)}
    return d[key] if key else list(d.values())[0]


def fft(a):
    n = len(a)
    if n == 1:
        return a
    ev = fft(a[0::2])
    od = fft(a[1::2])
    T = [cmath.exp(-2j * math.pi * k / n) * od[k] for k in range(n // 2)]
    return [ev[k] + T[k] for k in range(n // 2)] + [ev[k] - T[k] for k in range(n // 2)]


def power_spectrum(v, N):
    x = [complex(t) for t in v[:N]] + [0j] * max(0, N - len(v))
    X = fft(x)
    return [abs(X[k]) ** 2 for k in range(N // 2 + 1)]


def band_energy(P, N, f0, f1):
    k0, k1 = int(round(f0 * N)), int(round(f1 * N))
    return sum(P[k0:k1 + 1])


BANDS = [
    ('chirp 频带 (800-1500Hz@fs9600)', 0.08337, 0.15607),
    ('教师原代码信道带 (bin266-500)', 0.03325, 0.06250),
    ('你的信道带 (bin80-150)', 0.00989, 0.01868),
    ('对照: 0.20-0.28', 0.20, 0.28),
    ('对照: 0.30-0.38', 0.30, 0.38),
    ('对照: 0.42-0.48', 0.42, 0.48),
]

print('=' * 82)
print('各频带能量占比 (%) —— 若各带大致相等，说明含噪信号基本是白噪声，chirp 根本没进来')
print('=' * 82)
files = [
    ('te_signal/test_sample_1', 'D:/project/te_signal/test_sample_1.mat', 32768),
    ('tr_signal/chrip_with_noise1000', 'D:/project/tr_signal/chrip_with_noise1000.mat', 32768),
    ('tr_signal/chrip_with_noise1500', 'D:/project/tr_signal/chrip_with_noise1500.mat', 32768),
]
for name, path, N in files:
    v = gv(path)
    P = power_spectrum(v, N)
    tot = sum(P)
    print()
    print('  【%s】 长度=%d 总能量=%.4e' % (name, len(v), tot))
    for label, f0, f1 in BANDS:
        e = band_energy(P, N, f0, f1)
        w = (f1 - f0)
        print('    %-36s 能量占比 %6.3f%%   谱密度 %.3f' % (label, e / tot * 100, e / tot / w))

print()
print('=' * 82)
print('参考：干净 chirp 自己的频带分布')
print('=' * 82)
chirp = gv('D:/project/chirp_signal.mat', 'chirp_signal')
P = power_spectrum(chirp, 16384)
tot = sum(P)
for label, f0, f1 in BANDS:
    e = band_energy(P, 16384, f0, f1)
    print('    %-36s 能量占比 %6.3f%%   谱密度 %.3f' % (label, e / tot * 100, e / tot / (f1 - f0)))

print()
print('=' * 82)
print('参考：用户生成的 channel_H_1 自己的频带分布')
print('=' * 82)
H = gv('E:/project/data/信道产生/data/channel_H_1.mat', 'channel_H')
P = power_spectrum(H, 8192)
tot = sum(P)
for label, f0, f1 in BANDS:
    e = band_energy(P, 8192, f0, f1)
    print('    %-36s 能量占比 %6.3f%%   谱密度 %.3f' % (label, e / tot * 100, e / tot / (f1 - f0)))
